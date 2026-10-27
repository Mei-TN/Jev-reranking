"""Evaluate native decision responses with separate, never transmitted labels.

For JevBench, Kev, and MC11 paper results, use the benchmark runners
documented in docs/EVALUATION.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


SCHEMA = "jev-native-evaluation-v1"
IDENTITY_KEYS = ("model", "probability_source", "head_config", "head_kind")
SERVICE_KEYS = IDENTITY_KEYS + (
    "manifest_sha256", "head_sha256", "dtype", "head_dtype", "merge",
    "temperature", "lora_scale", "prefix_cache", "strict", "max_state",
    "max_branch", "max_packed", "bundle_loader",
)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def _decode(value: str | bytes):
    return json.loads(value, object_pairs_hook=_object, parse_constant=_constant)


def _dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            try:
                value = _decode(line)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            records.append(value)
    if not records:
        raise ValueError(f"empty JSONL file: {path}")
    return records


def _index(rows, kind, allowed):
    result = {}
    for row in rows:
        if set(row) - allowed or not {"id"} <= set(row):
            raise ValueError(f"unexpected {kind} row fields")
        rid = row["id"]
        if not isinstance(rid, str) or not rid or rid in result:
            raise ValueError(f"{kind} ids must be unique nonempty strings")
        result[rid] = row
    return result


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("request contains a nonfinite number")
    if isinstance(value, dict):
        for part in value.values():
            _finite_json(part)
    elif isinstance(value, list):
        for part in value:
            _finite_json(part)


def _keys(question):
    if not isinstance(question, dict) or set(question) - {"type", "instructions", "criteria"}:
        raise ValueError("question accepts only type, instructions and criteria")
    if not {"type", "instructions"} <= set(question):
        raise ValueError("question requires type and instructions")
    qtype, criteria = question["type"], question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
            raise ValueError("choice criteria must be an object with 1..255 options")
        if any(not key for key in criteria):
            raise ValueError("choice option keys must be nonempty")
        return list(criteria)
    if qtype == "noul":
        if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"false", "true"}):
            raise ValueError("noul criteria must contain only false/true descriptions")
        return ["false", "true"]
    if qtype == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 255:
            raise ValueError("score criteria must be a list of 2..255 ordered levels")
        return [str(i) for i in range(len(criteria))]
    raise ValueError(f"unsupported question type: {qtype}")


def _target(value, qtype, keys):
    if qtype == "noul" and type(value) is bool:
        value = "true" if value else "false"
    elif qtype == "score" and type(value) is int:
        value = str(value)
    if not isinstance(value, str) or value not in keys:
        raise ValueError("target must name one of the native response option keys")
    return value


def load_inputs(requests, labels, model=None):
    """Validate the complete input population before any HTTP request is made."""
    request_rows = _index(_jsonl(requests), "request", {"id", "request"})
    label_rows = _index(_jsonl(labels), "label", {"id", "targets"})
    if set(request_rows) != set(label_rows):
        raise ValueError("labels must cover exactly the request ids")
    bodies, targets, counts = {}, {}, Counter()
    models = set()
    for rid, row in request_rows.items():
        body = row.get("request")
        if not isinstance(body, dict) or set(body) - {"model", "state", "questions"} or not {"state", "questions"} <= set(body):
            raise ValueError(f"{rid}: request requires state/questions and optional model; labels/metadata are forbidden")
        if "model" in body:
            if not isinstance(body["model"], str) or not body["model"]:
                raise ValueError(f"{rid}: model must be a nonempty string")
            models.add(body["model"])
        _finite_json(body)
        questions = body["questions"]
        if not isinstance(questions, dict) or not questions or any(not qid for qid in questions):
            raise ValueError(f"{rid}: questions must be a nonempty object with nonempty ids")
        expected = label_rows[rid].get("targets")
        if not isinstance(expected, dict) or set(expected) != set(questions):
            raise ValueError(f"{rid}: targets must cover exactly the question ids")
        targets[rid] = {}
        for qid, question in questions.items():
            keys = _keys(question)
            targets[rid][qid] = _target(expected[qid], question["type"], keys)
            counts[question["type"]] += 1
        bodies[rid] = dict(body)
    if model is not None:
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a nonempty string")
        models.add(model)
    if len(models) > 1:
        raise ValueError("requests and --model name different models")
    selected = next(iter(models)) if models else None
    if selected:
        for body in bodies.values():
            body["model"] = selected
    return bodies, targets, dict(counts), selected


def question_metric(distribution, target, qtype):
    """Compute per-question metrics using request option order.

    This stdlib port retains its 1e-12 NLL floor, binary Brier convention and
    normalized ordinal RPS. It does not apply benchmark selection or aggregation.
    """
    keys, ps = list(distribution), list(distribution.values())
    gold = keys.index(target)
    guess = max(range(len(ps)), key=ps.__getitem__)
    result = dict(accuracy=float(guess == gold), nll=-math.log(max(1e-12, ps[gold])),
                  brier=sum((p - (i == gold)) ** 2 for i, p in enumerate(ps)))
    if qtype == "noul":
        result["brier"] = (distribution["true"] - (target == "true")) ** 2
    if qtype == "score":
        cumulative = rps = 0.0
        for i, p in enumerate(ps[:-1]):
            cumulative += p
            rps += (cumulative - (gold <= i)) ** 2
        error = sum(i * p for i, p in enumerate(ps)) - gold
        result.update(expected_grade_mae=abs(error), expected_grade_mse=error ** 2,
                      argmax_grade_mae=abs(guess - gold), rps=rps / (len(ps) - 1))
    return keys[guess], result


def _identity(response, model, source, expected=None):
    if not isinstance(response, dict) or response.get("model") != model:
        raise ValueError("response model identity mismatch")
    if response.get("probability_source") != source:
        raise ValueError("response probability_source mismatch")
    if source == "classification_head":
        head = response.get("head_config")
        if (not isinstance(head, dict) or not isinstance(head.get("kind"), str)
                or not head["kind"] or response.get("head_kind") != head["kind"]):
            raise ValueError("response must identify its classification head")
    identity = {key: response.get(key) for key in IDENTITY_KEYS}
    if expected is not None and identity != expected:
        raise ValueError("response head identity changed during evaluation")
    return identity


def _score_one(body, target, response, model, source, expected=None):
    identity = _identity(response, model, source, expected)
    probabilities = response.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(body["questions"]):
        raise ValueError("response question coverage mismatch")
    scores = {}
    for qid, question in body["questions"].items():
        keys = _keys(question)
        values = probabilities[qid]
        if not isinstance(values, dict) or set(values) != set(keys):
            raise ValueError(f"{qid}: response option coverage mismatch")
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values.values()):
            raise ValueError(f"{qid}: invalid or nonfinite probability")
        if abs(sum(values.values()) - 1) > 1e-5:
            raise ValueError(f"{qid}: probability distribution does not sum to one")
        # The request's source-slot order decides ties, independent of response JSON order.
        distribution = {key: values[key] for key in keys}
        predicted, metrics = question_metric(distribution, target[qid], question["type"])
        scores[qid] = dict(type=question["type"], target=target[qid], prediction=predicted, **metrics)
    return identity, scores


def _summary(scored, counts, model, identity, protocol):
    grouped = defaultdict(list)
    for questions in scored.values():
        for value in questions.values():
            grouped[value["type"]].append(value)
    if {kind: len(rows) for kind, rows in grouped.items()} != counts:
        raise ValueError("completed question counts differ from fixed input denominators")
    by_type = {}
    for kind, n in counts.items():
        rows = grouped[kind]
        fields = ("accuracy", "nll", "brier") + (("expected_grade_mae", "expected_grade_mse", "argmax_grade_mae", "rps") if kind == "score" else ())
        by_type[kind] = dict(n=n, correct=sum(row["accuracy"] == 1 for row in rows),
                             **{key: math.fsum(row[key] for row in rows) / n for key in fields})
    n = sum(counts.values())
    correct = sum(row["correct"] for row in by_type.values())
    return dict(schema=SCHEMA, status="COMPLETE", protocol=protocol, model=model,
                service_identity=identity, requests=len(scored), questions=n,
                denominators=counts, correct=correct, accuracy=correct / n, by_type=by_type,
                aggregation="question micro accuracy; per-type question means; no benchmark macro applied",
                tie_rule="first maximum in request option order", automatic_retry=False)


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def _provenance(requests, labels, predictions=None):
    result = {kind: dict(path=str(Path(path).resolve()), sha256=_sha(path))
              for kind, path in (("requests", requests), ("labels", labels))}
    if predictions is not None:
        result["predictions"] = dict(path=str(Path(predictions).resolve()), sha256=_sha(predictions))
    return result


def score_files(requests, labels, predictions, output, model=None, *, protocol="generic_native",
                probability_source="classification_head"):
    """Score complete native {id,response} JSONL files without network/model imports.

    Return SCORES.json's content. Invalid inputs raise ValueError and produce no
    aggregate; an existing output directory is always refused.
    """
    destination = Path(output)
    if probability_source not in ("classification_head", "pointer_head"):
        raise ValueError("probability_source must be classification_head or pointer_head")
    if destination.exists():
        raise ValueError("output already exists; choose a new evaluation directory")
    bodies, targets, counts, selected = load_inputs(requests, labels, model)
    predicted = _index(_jsonl(predictions), "prediction", {"id", "response", "elapsed_seconds"})
    if set(predicted) != set(bodies):
        raise ValueError("predictions must cover exactly the request ids")
    if selected is None:
        first_response = predicted[next(iter(bodies))].get("response")
        selected = first_response.get("model") if isinstance(first_response, dict) else None
    if not isinstance(selected, str) or not selected:
        raise ValueError("model must be supplied in requests, responses or --model")
    identity, scored = None, {}
    for rid, body in bodies.items():
        body["model"] = selected
        identity, scored[rid] = _score_one(body, targets[rid], predicted[rid].get("response"),
                                            selected, probability_source, identity)
    summary = _summary(scored, counts, selected, identity, protocol)
    provenance = _provenance(requests, labels, predictions)
    destination.mkdir(parents=True, exist_ok=False)
    _write(destination / "PLAN.json", dict(schema=SCHEMA, mode="offline", protocol=protocol,
        model=selected, planned_requests=len(bodies), denominators=counts, inputs=provenance))
    shutil.copyfile(requests, destination / "requests.jsonl")
    shutil.copyfile(labels, destination / "labels.jsonl")
    shutil.copyfile(predictions, destination / "predictions.jsonl")
    with (destination / "question_scores.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for rid, scores in scored.items():
            stream.write(_dump(dict(id=rid, questions=scores)) + "\n")
    _write(destination / "SCORES.json", summary)
    _write(destination / "RESULT.json", dict(schema=SCHEMA, status="COMPLETE", mode="offline",
        model=selected, completed_requests=len(scored), inputs=provenance,
        scores_sha256=_sha(destination / "SCORES.json"), predictions_sha256=_sha(predictions)))
    return summary


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("decision endpoint redirects are not allowed")


def _endpoint(value):
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("endpoint must be an HTTP(S) origin without path, credentials, query or fragment")
    return value.rstrip("/")


def _http(opener, url, timeout, headers, body=None):
    request = urllib.request.Request(url, data=None if body is None else _dump(body).encode("utf-8"), headers=headers)
    with opener.open(request, timeout=timeout) as response:
        return _decode(response.read())


def evaluate_endpoint(requests, labels, endpoint, output, model=None, *, timeout=300,
                      protocol="generic_native", probability_source="classification_head", api_key_env=None):
    """Send each input occurrence once, preserving raw native responses and failures."""
    destination = Path(output)
    if probability_source not in ("classification_head", "pointer_head"):
        raise ValueError("probability_source must be classification_head or pointer_head")
    if destination.exists():
        raise ValueError("output already exists; choose a new evaluation directory")
    endpoint = _endpoint(endpoint)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    bodies, targets, counts, selected = load_inputs(requests, labels, model)
    if selected is None:
        raise ValueError("endpoint evaluation requires --model or a model in every request population")
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        secret = os.environ.get(api_key_env)
        if not secret:
            raise ValueError("API key environment variable is empty or absent")
        headers["Authorization"] = "Bearer " + secret
    opener = urllib.request.build_opener(_NoRedirect())
    health = _http(opener, endpoint + "/health", timeout, headers)
    identity = _identity(health, selected, probability_source)
    bound_service = {key: health.get(key) for key in SERVICE_KEYS}
    provenance = _provenance(requests, labels)
    destination.mkdir(parents=True, exist_ok=False)
    _write(destination / "PLAN.json", dict(schema=SCHEMA, mode="endpoint", endpoint=endpoint,
        protocol=protocol, model=selected, planned_requests=len(bodies), denominators=counts,
        inputs=provenance, service_health=health, timeout_seconds=timeout, automatic_retry=False))
    shutil.copyfile(requests, destination / "requests.jsonl")
    shutil.copyfile(labels, destination / "labels.jsonl")
    scored = {}
    attempted = 0
    try:
        with (destination / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as predictions_stream, (destination / "question_scores.jsonl").open("x", encoding="utf-8", newline="\n") as scores_stream:
            for rid, body in bodies.items():
                attempted += 1
                tick = time.monotonic()
                response = _http(opener, endpoint + "/v1/decision", timeout, headers, body)
                # Retain a returned body even when validation fails. It is never normalized.
                predictions_stream.write(_dump(dict(id=rid, response=response, elapsed_seconds=time.monotonic() - tick)) + "\n")
                predictions_stream.flush()
                identity, scores = _score_one(body, targets[rid], response, selected, probability_source, identity)
                scores_stream.write(_dump(dict(id=rid, questions=scores)) + "\n")
                scores_stream.flush()
                scored[rid] = scores
        final_health = _http(opener, endpoint + "/health", timeout, headers)
        if {key: final_health.get(key) for key in SERVICE_KEYS} != bound_service:
            raise ValueError("service configuration changed during evaluation")
        summary = _summary(scored, counts, selected, identity, protocol)
        _write(destination / "SCORES.json", summary)
        _write(destination / "RESULT.json", dict(schema=SCHEMA, status="COMPLETE", mode="endpoint",
            completed_requests=len(scored), attempted_requests=attempted, inputs=provenance,
            scores_sha256=_sha(destination / "SCORES.json"), predictions_sha256=_sha(destination / "predictions.jsonl")))
        return summary
    except Exception as error:
        _write(destination / "RESULT.json", dict(schema=SCHEMA, status="FAILED", mode="endpoint",
            completed_requests=len(scored), attempted_requests=attempted, planned_requests=len(bodies),
            denominators=counts, automatic_retry=False, failure=dict(type=type(error).__name__, message=str(error))))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--endpoint", help="native service origin, e.g. http://127.0.0.1:32100")
    source.add_argument("--predictions", type=Path, help="offline native {id,response} JSONL file")
    parser.add_argument("--model", help="required unless input population identifies one model")
    parser.add_argument("--protocol", default="generic_native", help="descriptive protocol label; does not select a benchmark scorer")
    parser.add_argument("--probability-source", choices=("classification_head", "pointer_head"), default="classification_head")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--api-key-env", help="environment variable name, never the key itself")
    args = parser.parse_args(argv)
    try:
        common = dict(model=args.model, protocol=args.protocol, probability_source=args.probability_source)
        if args.predictions:
            result = score_files(args.requests, args.labels, args.predictions, args.output, **common)
        else:
            result = evaluate_endpoint(args.requests, args.labels, args.endpoint, args.output,
                timeout=args.timeout, api_key_env=args.api_key_env, **common)
    except (ValueError, OSError, urllib.error.URLError) as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
