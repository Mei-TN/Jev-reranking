"""Resume a training pipeline from recorded command results and verified outputs."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import socket
import uuid

from . import evaluate, workflow
from .config import read, safe_id, sha, write_new
from .recipes import options


SCHEMA = "jev.pipeline.v1"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("missing pipeline input/artifact: " + str(path))
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def _source_files(source):
    return {path.relative_to(source).as_posix(): sha(path) for path in sorted(Path(source).rglob("*.py"))
            if "__pycache__" not in path.parts}


def _pins(paths):
    return {str(Path(path).resolve()): _file(path) for path in paths}


def _inventory(folder):
    return [path for path in sorted(Path(folder).rglob("*")) if path.is_file()
            and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")]


def _check_pins(pins):
    for name, expected in pins.items():
        if _file(name) != expected:
            raise ValueError("completed pipeline artifact changed: " + name)


@contextmanager
def _claim(path):
    """Hold an OS lifecycle lock; lock-file existence alone does not indicate a running process."""
    path = Path(path)
    stream = path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("another pipeline controller holds the lifecycle claim") from error
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("another pipeline controller holds the lifecycle claim") from error
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _native(logs, stage, required=None, interpreter=None):
    command_path, result_path = Path(logs) / (stage + ".command.json"), Path(logs) / (stage + ".result.json")
    command, result = read(command_path), read(result_path)
    if (command.get("schema") != "jev.stage.v1" or result.get("schema") != "jev.stage.v1"
            or command.get("stage") != stage or result.get("stage") != stage
            or command.get("argv") != result.get("argv") or type(result.get("exit_code")) is not int):
        raise ValueError("invalid native command/terminal receipt for " + stage)
    argv = result["argv"]
    if not isinstance(argv, list) or not argv or (interpreter and argv[0] != interpreter):
        raise ValueError("native stage used another interpreter: " + stage)
    for flag, expected in (required or {}).items():
        if argv.count(flag) != 1 or argv.index(flag) + 1 == len(argv) or argv[argv.index(flag) + 1] != str(expected):
            raise ValueError("native " + stage + " command binding differs: " + flag)
    if result["exit_code"] != 0:
        raise ValueError(f"native {stage} did not complete successfully (exit {result['exit_code']})")
    return [command_path, result_path]


def _same_argument(actual, expected):
    if isinstance(actual, bool):
        return actual is expected or str(int(actual)) == str(expected)
    if type(actual) in (int, float):
        try:
            return float(expected) == actual
        except (ValueError, TypeError):
            return False
    if actual == expected:
        return True
    try:
        return json.loads(actual) == json.loads(expected)
    except (ValueError, TypeError):
        return False


def _head(config):
    arguments = config.get("args", {})
    value = config.get("head_config") or arguments.get("head_config")
    if isinstance(value, str):
        value = json.loads(value)
    return value or dict(version=1, kind="pointer", dim=arguments.get("head_dim", 256), readout="end")


def _training_proof(plan, source_identity):
    checkpoint = Path(plan["output"]).resolve()
    run = checkpoint.parent
    if read(run / "execution.json") != plan:
        raise ValueError("training execution record differs from the owned pipeline plan")
    if read(run / "source_files.json") != source_identity or _source_files(Path(plan["source_root"])) != source_identity:
        raise ValueError("training source identity differs from the pipeline")
    native = _native(run, "train", interpreter=plan["site"]["python"]["training"])
    if read(native[0])["argv"] != plan["argv"]:
        raise ValueError("native training command differs from the pipeline plan")
    verified = read(run / "verified_checkpoint.json")
    metrics = read(checkpoint / "training_metrics.json")
    if (verified.get("metrics") != metrics or metrics.get("status") != "complete"
            or metrics.get("stopped_early") is not False):
        raise ValueError("training has no proven full-horizon completed checkpoint")
    for key in ("optimizer_steps", "planned_optimizer_steps"):
        if key in plan["expected"] and metrics.get(key) != plan["expected"][key]:
            raise ValueError("training horizon differs: " + key)
    if "source_records" in plan["expected"] and metrics.get("source_records_seen") != plan["expected"]["source_records"]:
        raise ValueError("training source-record count differs")
    artifacts = verified.get("artifacts_sha256")
    if not isinstance(artifacts, dict) or not {"training_config.json", "training_metrics.json"} <= set(artifacts):
        raise ValueError("checkpoint verification lacks artifact hashes")
    for name, expected in artifacts.items():
        if PurePosixPath(name).name != name or sha(checkpoint / name) != expected:
            raise ValueError("verified checkpoint artifact changed: " + name)
    config = read(checkpoint / "training_config.json")
    for flag, expected in options(plan["argv"]).items():
        if not _same_argument(config.get("args", {}).get(flag[2:]), expected):
            raise ValueError("native training configuration differs: " + flag)
    paths = native + [run / name for name in ("execution.json", "source_files.json", "verified_checkpoint.json")]
    paths += [checkpoint / name for name in artifacts]
    return dict(checkpoint=str(checkpoint), head_config=_head(config)), _pins(paths)


def _bundle_proof(bundle, checkpoint, plan, examples, model_name):
    bundle, checkpoint = Path(bundle), Path(checkpoint)
    logs = bundle.parent / (bundle.name + ".export")
    native = _native(logs, "export", {"--run": checkpoint, "--out": bundle, "--name": model_name,
        "--base": plan["site"]["base_model"], "--verification-examples": examples}, plan["site"]["python"]["training"])
    manifest_path = bundle / "model_manifest.json"
    manifest = read(manifest_path)
    if (manifest.get("name") != model_name or manifest.get("probability_source") != "classification_head"
            or manifest.get("source_checkpoint") != str(checkpoint)
            or manifest.get("verification_input_sha256") != sha(examples)
            or manifest.get("head_config") != _head(read(checkpoint / "training_config.json"))):
        raise ValueError("bundle identity does not bind this checkpoint, head and verification population")
    source = Path(plan["source_root"])
    expected_sources = {path.relative_to(source).as_posix(): sha(path)
        for package in ("kev", "runtime") for path in sorted((source / package).glob("*.py"))}
    if manifest.get("source_hashes") != expected_sources:
        raise ValueError("bundle exporter used a different selected source implementation")
    for name in ("head.pt", "adapter_model.safetensors", "adapter_config.json"):
        if manifest.get("checkpoint_hashes", {}).get(name) != sha(checkpoint / name):
            raise ValueError("bundle checkpoint hash differs: " + name)
    members = manifest.get("files")
    if (not isinstance(members, dict) or not {"classification_head.safetensors", "head_config.json"} <= set(members)
            or not any(name.startswith("backbone/") for name in members)):
        raise ValueError("bundle manifest lacks actual backbone/head member identities")
    for name, expected in members.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("unsafe bundle manifest member")
        if sha(bundle.joinpath(*relative.parts)) != expected:
            raise ValueError("bundle file differs from its manifest: " + name)
    report = read(bundle / "validation_report.json")
    delta = report.get("max_probability_delta")
    if (report.get("status") != "PASS" or report.get("manifest_sha256") != sha(manifest_path)
            or report.get("head_config") != manifest["head_config"] or report.get("argmax_equal") is not True
            or report.get("source_unchanged") is not True or report.get("checkpoint_unchanged") is not True
            or type(report.get("requests")) is not int or report["requests"] <= 0
            or type(delta) not in (int, float) or not math.isfinite(delta) or not 0 <= delta <= 1e-5):
        raise ValueError("bundle lacks a bound successful fresh-reload verification")
    return dict(bundle=str(bundle), model_name=model_name), _pins(native + _inventory(bundle))


def _prediction_content(requests, labels, predictions, model_name, head):
    bodies, targets, counts, _ = evaluate.load_inputs(requests, labels, model_name)
    rows = evaluate._index(evaluate._jsonl(predictions), "prediction", {"id", "response", "elapsed_seconds"})
    if set(rows) != set(bodies):
        raise ValueError("native predictions do not cover the complete pipeline request population")
    identity, scored = None, {}
    for rid, body in bodies.items():
        response = rows[rid].get("response")
        identity, scored[rid] = evaluate._score_one(body, targets[rid], response, model_name, "classification_head", identity)
        if response.get("head_config") != head:
            raise ValueError("prediction head differs from the trained checkpoint")
    return evaluate._summary(scored, counts, model_name, identity, "generic_native")


def _prediction_proof(predictions, checkpoint, bundle, plan, requests, labels, model_name):
    predictions = Path(predictions)
    required = {"--requests": requests, "--output": predictions}
    if bundle:
        required["--bundle"] = bundle
    else:
        required.update({"--checkpoint": checkpoint, "--source-root": plan["source_root"], "--model-name": model_name,
                         "--base": plan["site"]["base_model"]})
    native = _native(predictions.parent / (predictions.name + ".logs"), "predict", required,
                     plan["site"]["python"]["serving"])
    summary = _prediction_content(requests, labels, predictions, model_name,
                                  _head(read(Path(checkpoint) / "training_config.json")))
    return dict(predictions=str(predictions), model_name=model_name, requests=summary["requests"]), _pins(native + [predictions])


def _evaluation_proof(output, predictions, checkpoint, requests, labels, model_name):
    output = Path(output)
    result, scores = read(output / "RESULT.json"), read(output / "SCORES.json")
    expected = _prediction_content(requests, labels, predictions, model_name,
                                   _head(read(Path(checkpoint) / "training_config.json")))
    if (result.get("status") != "COMPLETE" or result.get("mode") != "offline"
            or result.get("completed_requests") != expected["requests"] or result.get("scores_sha256") != sha(output / "SCORES.json")
            or result.get("predictions_sha256") != sha(predictions) or scores != expected):
        raise ValueError("offline evaluation lacks complete matching native response scores")
    for name, original in (("requests", requests), ("labels", labels), ("predictions", predictions)):
        if result.get("inputs", {}).get(name, {}).get("sha256") != sha(original) or sha(output / (name + ".jsonl")) != sha(original):
            raise ValueError("evaluation input binding differs: " + name)
    return dict(evaluation_dir=str(output), summary=scores), _pins(_inventory(output))


def _attempts(control, stage):
    folder = control / "attempts"
    return sorted(folder.glob(stage + "-[0-9][0-9][0-9][0-9]")) if folder.is_dir() else []


def _new_attempt(control, stage, binding_sha):
    attempts = _attempts(control, stage)
    index = int(attempts[-1].name.rsplit("-", 1)[1]) + 1 if attempts else 1
    destination = control / "attempts" / f"{stage}-{index:04d}"
    destination.mkdir(parents=True, exist_ok=False)
    write_new(destination / "START.json", dict(schema=SCHEMA, stage=stage, binding_sha256=binding_sha, started_utc=_now()))
    return destination


def _terminal_native(attempt, stage, model_name):
    if stage == "export":
        logs, native = attempt / (model_name + ".export"), "export"
    else:
        logs, native = attempt / "predictions.jsonl.logs", "predict"
    result = logs / (native + ".result.json")
    if result.is_file():
        row = read(result)
        command = read(logs / (native + ".command.json"))
        if (row.get("schema") == command.get("schema") == "jev.stage.v1"
                and row.get("stage") == command.get("stage") == native
                and isinstance(command.get("argv"), list) and command["argv"] == row.get("argv")
                and type(row.get("exit_code")) is int):
            return row["exit_code"]
        raise ValueError("invalid native terminal receipt: " + str(result))
    if (logs / (native + ".command.json")).exists():
        raise RuntimeError("previous external stage has no native terminal receipt; it may still be running. "
            "Observe that owned job to terminal completion before resuming; no retry was launched: " + str(attempt))
    return None


def _stage(control, stage, binding_sha, model_name, invoke, prove, resume):
    for attempt in _attempts(control, stage):
        started = read(attempt / "START.json")
        if started.get("binding_sha256") != binding_sha or started.get("stage") != stage:
            raise ValueError("pipeline attempt identity changed")
        complete = attempt / "COMPLETE.json"
        if complete.is_file():
            receipt = read(complete)
            if receipt.get("binding_sha256") != binding_sha:
                raise ValueError("completed stage belongs to another pipeline binding")
            _check_pins(receipt["files"])
            data, _ = prove(attempt)
            if receipt.get("artifacts") != data:
                raise ValueError("completed stage artifact identity differs")
            return data
        if not resume:
            raise ValueError("pipeline has previous attempts; use --resume")
        terminal = _terminal_native(attempt, stage, model_name) if stage != "evaluate" else None
        call = attempt / "CALL_TERMINAL.json"
        if terminal == 0 or stage == "evaluate":
            try:
                data, pins = prove(attempt)
            except (ValueError, OSError, KeyError, TypeError):
                # Native failure/partial local scoring stays intact; a new
                # attempt is used below. External command-without-exit was
                # already refused by _terminal_native.
                pass
            else:
                write_new(complete, dict(schema=SCHEMA, stage=stage, binding_sha256=binding_sha,
                    artifacts=data, files=pins, adopted_after_terminal_verification=True, completed_utc=_now()))
                return data
        if not call.exists():
            write_new(call, dict(schema=SCHEMA, stage=stage, status="INCOMPLETE_PREVIOUS_ATTEMPT",
                native_exit=terminal, same_process_stage=stage == "evaluate", observed_utc=_now()))
    attempt = _new_attempt(control, stage, binding_sha)
    try:
        invoke(attempt)
        data, pins = prove(attempt)
        write_new(attempt / "CALL_TERMINAL.json", dict(schema=SCHEMA, stage=stage, status="RETURNED",
            native_exit=None if stage == "evaluate" else 0, same_process_stage=stage == "evaluate", observed_utc=_now()))
        write_new(attempt / "COMPLETE.json", dict(schema=SCHEMA, stage=stage, binding_sha256=binding_sha,
            artifacts=data, files=pins, completed_utc=_now()))
        return data
    except BaseException as error:
        if not (attempt / "CALL_TERMINAL.json").exists():
            write_new(attempt / "CALL_TERMINAL.json", dict(schema=SCHEMA, stage=stage, status="FAILED_CALL",
                failure=dict(type=type(error).__name__, message=str(error)), observed_utc=_now()))
        raise


def run_pipeline(plan, gpu, requests, labels, verification_examples, inference_format="auto",
                 allow_environment_drift=False, resume=False):
    """Train once, then resume export, prediction, and offline scoring.

    A failed or unproven training is never restarted. Failed later stages may
    be retried only in new attempt directories. An external native command
    lacking a terminal receipt is not assumed stopped from a state/lock file.
    """
    plan = json.loads(json.dumps(plan))
    source = Path(plan["source_root"]).resolve()
    checkpoint = Path(plan["output"]).resolve()
    model_name = safe_id(plan["run_id"])
    if inference_format not in ("auto", "bundle", "checkpoint"):
        raise ValueError("inference_format must be auto, bundle or checkpoint")
    supports_bundle = all((source / "runtime" / name).is_file() for name in ("export_bundle.py", "load_bundle.py"))
    selected = "bundle" if inference_format == "auto" and supports_bundle else "checkpoint" if inference_format == "auto" else inference_format
    if selected == "bundle" and not supports_bundle:
        raise ValueError("selected source has no complete native bundle runtime; use checkpoint inference")
    if not (source / "kev/train.py").is_file():
        raise ValueError("selected source_root lacks kev/train.py")
    requests, labels = Path(requests).resolve(), Path(labels).resolve()
    evaluate.load_inputs(requests, labels, model_name)
    examples = Path(verification_examples).resolve() if verification_examples else None
    if selected == "bundle" and (examples is None or not examples.is_file()):
        raise ValueError("bundle pipeline requires real verification examples")
    arguments = options(plan["argv"])
    source_identity = _source_files(source)
    inputs = {"requests": _file(requests), "labels": _file(labels), "training_data": _file(arguments["--data"])}
    if examples is not None:
        inputs["verification_examples"] = _file(examples)
    base_config = Path(arguments["--base"]) / "config.json"
    inputs["base_config"] = _file(base_config)
    binding = dict(schema=SCHEMA, plan=plan, inputs=inputs, source_root=str(source), source_files=source_identity,
        controller_sources={name: sha(path) for name, path in (("pipeline", __file__), ("workflow", workflow.__file__), ("evaluate", evaluate.__file__))},
        model_name=model_name, inference_format_requested=inference_format, inference_format=selected,
        allow_environment_drift=bool(allow_environment_drift))
    binding_sha = _digest(binding)
    run = checkpoint.parent
    control = run.with_name(run.name + ".pipeline")
    if control.exists() and not resume:
        raise ValueError("pipeline records already exist; use --resume")
    if not control.exists() and resume:
        raise ValueError("no owned pipeline records exist to resume")
    if run.exists() and not resume:
        raise ValueError("training run directory already exists; this pipeline cannot overwrite or adopt an unowned run")
    control.mkdir(parents=True, exist_ok=True)
    with _claim(control / "controller.lock"):
        pipeline_plan = control / "PIPELINE_PLAN.json"
        if pipeline_plan.exists():
            if not resume or read(pipeline_plan) != dict(binding=binding, binding_sha256=binding_sha):
                raise ValueError("resumed pipeline plan/input/source identity drift; choose a new experiment output")
        elif resume:
            raise ValueError("pipeline has no sealed initial binding; cannot adopt existing outputs")
        else:
            write_new(pipeline_plan, dict(binding=binding, binding_sha256=binding_sha))
        owner = control / "controllers" / uuid.uuid4().hex
        owner.mkdir(parents=True)
        write_new(owner / "OWNER.json", dict(schema=SCHEMA, pid=os.getpid(), host=socket.gethostname(),
            started_utc=_now(), gpu=gpu, binding_sha256=binding_sha, resume=bool(resume)))
        def unchanged():
            _check_pins({item["path"]: item for item in inputs.values()})
            if _source_files(source) != source_identity:
                raise ValueError("selected pipeline source changed during execution")
        def prove(callback, attempt):
            unchanged()
            return callback(attempt)
        def validate_model_inputs():
            unchanged()
            problems = workflow.validate_inputs(plan)
            if problems:
                raise ValueError("\n".join(problems))
        try:
            if resume:
                validate_model_inputs()
            train_complete = control / "TRAIN_COMPLETE.json"
            if train_complete.exists():
                receipt = read(train_complete)
                if receipt.get("binding_sha256") != binding_sha:
                    raise ValueError("training completion belongs to another pipeline")
                _check_pins(receipt["files"])
                trained, _ = _training_proof(plan, source_identity)
            elif run.exists():
                try:
                    trained, pins = _training_proof(plan, source_identity)
                except (ValueError, OSError, KeyError, TypeError) as error:
                    raise RuntimeError("existing training is running, failed or lacks proven completion; it will not be restarted: " + str(error)) from error
                write_new(train_complete, dict(schema=SCHEMA, binding_sha256=binding_sha, artifacts=trained, files=pins,
                    adopted_after_native_verification=True, completed_utc=_now()))
            else:
                if (control / "TRAIN_CALL_TERMINAL.json").exists() or (control / "TRAIN_STARTED.json").exists():
                    raise RuntimeError("previous training call has no proven checkpoint; training is never restarted by --resume")
                write_new(control / "TRAIN_STARTED.json", dict(schema=SCHEMA, binding_sha256=binding_sha, started_utc=_now()))
                try:
                    unchanged()
                    actual = workflow.train(plan, gpu, allow_environment_drift=allow_environment_drift)
                    if Path(actual).resolve() != checkpoint:
                        raise ValueError("trainer returned another checkpoint path")
                    unchanged()
                    trained, pins = _training_proof(plan, source_identity)
                    write_new(train_complete, dict(schema=SCHEMA, binding_sha256=binding_sha, artifacts=trained, files=pins, completed_utc=_now()))
                    write_new(control / "TRAIN_CALL_TERMINAL.json", dict(schema=SCHEMA, status="RETURNED", observed_utc=_now()))
                except BaseException as error:
                    write_new(control / "TRAIN_CALL_TERMINAL.json", dict(schema=SCHEMA, status="FAILED_CALL",
                        failure=dict(type=type(error).__name__, message=str(error)), observed_utc=_now()))
                    raise
            bundle = None
            if selected == "bundle":
                def export(attempt):
                    validate_model_inputs()
                    return workflow.export_checkpoint(checkpoint, attempt / model_name, plan["site"],
                        tree=plan["tree"], gpu=gpu, examples=examples, source_root=source)
                exported = _stage(control, "export", binding_sha, model_name,
                    export, lambda attempt: prove(lambda current: _bundle_proof(current / model_name,
                        checkpoint, plan, examples, model_name), attempt), resume)
                bundle = Path(exported["bundle"])
            device = "cpu" if arguments.get("--device") == "cpu" else "cuda"
            def predict(attempt):
                unchanged()
                path = attempt / "predictions.jsonl"
                if bundle:
                    return workflow.predict(bundle, requests, path, plan["site"], gpu=gpu, device=device)
                validate_model_inputs()
                return workflow.predict_checkpoint(checkpoint, source, requests, path, plan["site"], gpu=gpu,
                                                   device=device, model_name=model_name)
            predicted = _stage(control, "predict", binding_sha, model_name, predict,
                lambda attempt: prove(lambda current: _prediction_proof(current / "predictions.jsonl",
                    checkpoint, bundle, plan, requests, labels, model_name), attempt), resume)
            predictions = Path(predicted["predictions"])
            evaluated = _stage(control, "evaluate", binding_sha, model_name,
                lambda attempt: evaluate.score_files(requests, labels, predictions, attempt / "evaluation", model=model_name),
                lambda attempt: prove(lambda current: _evaluation_proof(current / "evaluation",
                    predictions, checkpoint, requests, labels, model_name), attempt), resume)
            result = dict(schema=SCHEMA, status="COMPLETE", checkpoint=str(checkpoint), bundle=str(bundle) if bundle else None,
                predictions=str(predictions), evaluation=evaluated["summary"], evaluation_dir=evaluated["evaluation_dir"],
                pipeline_dir=str(control), model_name=model_name, inference_format=selected, binding_sha256=binding_sha)
            unchanged()
            complete_path = control / "PIPELINE_COMPLETE.json"
            if complete_path.exists():
                if read(complete_path) != result:
                    raise ValueError("pipeline completion record differs from verified artifacts")
            else:
                write_new(complete_path, result)
            write_new(owner / "RESULT.json", dict(status="RETURNED", completed_utc=_now(), result=result))
            return result
        except BaseException as error:
            write_new(owner / "RESULT.json", dict(status="FAILED_CALL", ended_utc=_now(),
                failure=dict(type=type(error).__name__, message=str(error))))
            raise
