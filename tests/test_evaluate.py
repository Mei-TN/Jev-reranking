"""Test HTTP and offline evaluation."""
import ast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest

from jev.evaluate import evaluate_endpoint, question_metric, score_files
from jev.config import root


HEAD = {"version": 1, "kind": "pointer", "dim": 256, "readout": "last_span", "rng_anchor_dim": 256}
IDENTITY = dict(model="fixture-model", probability_source="classification_head", head_config=HEAD, head_kind="pointer")


def jsonl(path, values):
    Path(path).write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


class NativeEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.requests = self.root / "requests.jsonl"
        self.labels = self.root / "labels.jsonl"
        self.predictions = self.root / "predictions.jsonl"
        self.request_rows = [{"id": "case1", "request": {"model": "fixture-model", "state": "fixture state", "questions": {
            "c": {"type": "choice", "instructions": "Choose", "criteria": {"first": "a", "second": "b"}},
            "b": {"type": "noul", "instructions": "True?"},
            "s": {"type": "score", "instructions": "Grade", "criteria": ["low", "middle", "high"]},
        }}}]
        self.label_rows = [{"id": "case1", "targets": {"c": "first", "b": True, "s": 1}}]
        self.response = dict(IDENTITY, probabilities={"c": {"second": 0.5, "first": 0.5},
            "b": {"false": 0.2, "true": 0.8}, "s": {"0": 0.2, "1": 0.6, "2": 0.2}})
        self.write_inputs()

    def write_inputs(self):
        jsonl(self.requests, self.request_rows)
        jsonl(self.labels, self.label_rows)
        jsonl(self.predictions, [{"id": "case1", "response": self.response}])

    def test_offline_exact_counts_raw_responses_and_native_metrics(self):
        result = score_files(self.requests, self.labels, self.predictions, self.root / "scores")
        self.assertEqual(result["denominators"], {"choice": 1, "noul": 1, "score": 1})
        self.assertEqual((result["questions"], result["correct"], result["accuracy"]), (3, 3, 1.0))
        self.assertAlmostEqual(result["by_type"]["noul"]["brier"], 0.04)
        self.assertAlmostEqual(result["by_type"]["score"]["rps"], 0.04)
        self.assertEqual(result["by_type"]["score"]["expected_grade_mae"], 0.0)
        self.assertEqual(json.loads((self.root / "scores/RESULT.json").read_text())["status"], "COMPLETE")
        # Scoring never rewrites or rounds the prediction file.
        self.assertEqual(json.loads(self.predictions.read_text())["response"], self.response)

    def test_missing_duplicate_nonfinite_and_wrong_identity_are_not_averaged(self):
        cases = [
            [],
            [{"id": "other", "response": self.response}],
            [{"id": "case1", "response": self.response}] * 2,
            [{"id": "case1", "response": dict(self.response, model="other")}],
            [{"id": "case1", "response": dict(self.response, probabilities={"c": {"first": float("nan"), "second": 0.5}})}],
            [{"id": "case1", "response": dict(self.response, probabilities={})}],
        ]
        for index, records in enumerate(cases):
            jsonl(self.predictions, records)
            output = self.root / f"invalid{index}"
            with self.assertRaises(ValueError):
                score_files(self.requests, self.labels, self.predictions, output)
            self.assertFalse(output.exists())

    def test_labels_and_unknown_fields_are_never_accepted_as_request_inputs(self):
        self.request_rows[0]["request"]["label"] = "secret-gold"
        self.write_inputs()
        with self.assertRaises(ValueError):
            score_files(self.requests, self.labels, self.predictions, self.root / "labels-leaked")
        self.request_rows[0]["request"].pop("label")
        self.label_rows[0]["targets"].pop("s")
        self.write_inputs()
        with self.assertRaises(ValueError):
            score_files(self.requests, self.labels, self.predictions, self.root / "missing-label")

    def test_duplicate_json_keys_are_rejected_before_scoring(self):
        self.requests.write_text('{"id":"case1","id":"case1","request":{}}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            score_files(self.requests, self.labels, self.predictions, self.root / "duplicate")

    def service(self, response=None):
        owner = self
        owner.calls = []
        served = self.response if response is None else response

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, value):
                payload = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                self.reply(dict(IDENTITY, strict=True, temperature=1, prefix_cache=False))

            def do_POST(self):
                owner.calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.reply(served)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_native_endpoint_and_offline_scoring_agree_without_transmitting_gold(self):
        endpoint = self.service()
        live = evaluate_endpoint(self.requests, self.labels, endpoint, self.root / "live")
        offline = score_files(self.requests, self.labels, self.root / "live/predictions.jsonl", self.root / "offline")
        self.assertEqual(live, offline)
        self.assertEqual(self.calls, [self.request_rows[0]["request"]])
        self.assertNotIn("targets", json.dumps(self.calls))
        record = json.loads((self.root / "live/predictions.jsonl").read_text())
        self.assertEqual(record["response"], self.response)

    def test_failed_native_response_retains_denominators_and_never_retries(self):
        endpoint = self.service(dict(self.response, probabilities={}))
        output = self.root / "failed"
        with self.assertRaises(ValueError):
            evaluate_endpoint(self.requests, self.labels, endpoint, output)
        result = json.loads((output / "RESULT.json").read_text())
        self.assertEqual((result["status"], result["completed_requests"], result["attempted_requests"]), ("FAILED", 0, 1))
        self.assertEqual(result["denominators"], {"choice": 1, "noul": 1, "score": 1})
        self.assertFalse(result["automatic_retry"])
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((output / "SCORES.json").exists())
        self.assertEqual(json.loads((output / "predictions.jsonl").read_text())["response"]["probabilities"], {})

    def test_all_input_labels_validated_before_contacting_endpoint(self):
        endpoint = self.service()
        self.label_rows[0]["targets"]["s"] = 99
        self.write_inputs()
        with self.assertRaises(ValueError):
            evaluate_endpoint(self.requests, self.labels, endpoint, self.root / "invalid-live")
        self.assertEqual(self.calls, [])

    def test_metric_port_matches_archived_native_arithmetic(self):
        # Extract only the original pure function so its bare sibling imports
        # cannot collide with the portable package or import model dependencies.
        source = root() / "vendor/evaluation/jk_audit.py"
        node = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "question_metric")
        namespace = {"math": math}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        for kind, dist, target in (
            ("choice", {"b": 0.5, "a": 0.5}, "a"),
            ("choice", {"only": 1.0}, "only"),
            ("noul", {"false": 1.0, "true": 0.0}, "true"),
            ("score", {"0": 0.1, "1": 0.3, "2": 0.6}, "1"),
        ):
            self.assertEqual(question_metric(dist, target, kind)[1], namespace["question_metric"](dist, target, kind))


if __name__ == "__main__":
    unittest.main()
