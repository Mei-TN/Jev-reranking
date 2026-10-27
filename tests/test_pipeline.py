"""Test pipeline state transitions and retry handling."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jev import evaluate, pipeline
from jev.config import read, root, sha, write_new
from jev.recipes import command, options


HEAD = dict(version=1, kind="pointer", dim=256, readout="end", rng_anchor_dim=256)


class PipelineContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.source = self.folder / "selected-source"
        (self.source / "kev").mkdir(parents=True)
        (self.source / "kev/train.py").write_text("# orchestration fixture\n", encoding="utf-8")
        (self.source / "runtime").mkdir()
        for name in ("export_bundle.py", "load_bundle.py"):
            (self.source / "runtime" / name).write_text("# orchestration fixture\n", encoding="utf-8")
        base = self.folder / "base"
        base.mkdir()
        write_new(base / "config.json", dict(model_type="fixture"))
        data = self.folder / "data.jsonl"
        data.write_text('{"fixture":"training input identity only"}\n', encoding="utf-8")
        self.checkpoint = self.folder / "runs/custom/fixture-model/output"
        self.site = dict(base_model=str(base), python=dict(training="fixture-training-python", serving="fixture-serving-python"), environment={})
        args = {"--base": str(base), "--data": str(data), "--out": str(self.checkpoint), "--device": "cuda", "--epochs": "1"}
        self.plan = dict(schema="jev.execution.v1", study="custom", run_id="fixture-model", tree="selected-tree",
            custom_source=True, source_root=str(self.source), argv=command(self.site["python"]["training"], args),
            output=str(self.checkpoint), site=self.site, environment={}, scientific_overrides=True,
            expected=dict(optimizer_steps=1, planned_optimizer_steps=1, source_records=1), references=[],
            input_pins={str(data): sha(data), str(base / "config.json"): sha(base / "config.json")})
        self.requests, self.labels, self.examples = [self.folder / name for name in ("requests.jsonl", "labels.jsonl", "verification.json")]
        self.requests.write_text(json.dumps(dict(id="request-1", request=dict(state="fixture", questions={
            "choice": dict(type="choice", instructions="fixture", criteria={"a": "first", "b": "second"})}))) + "\n", encoding="utf-8")
        self.labels.write_text(json.dumps(dict(id="request-1", targets={"choice": "a"})) + "\n", encoding="utf-8")
        self.examples.write_text('{"fixture":"unlabelled export verification"}\n', encoding="utf-8")
        self.calls = dict(train=0, export=0, predict=0, checkpoint_predict=0, evaluate=0)
        self.fail = {}
        self.score_files = evaluate.score_files
        for name, replacement in (("train", self.train), ("export_checkpoint", self.export),
                                  ("predict", self.predict), ("predict_checkpoint", self.predict_checkpoint)):
            current = patch.object(pipeline.workflow, name, side_effect=replacement)
            current.start()
            self.addCleanup(current.stop)
        current = patch.object(pipeline.evaluate, "score_files", side_effect=self.score)
        current.start()
        self.addCleanup(current.stop)

    @property
    def control(self):
        run = self.checkpoint.parent
        return run.with_name(run.name + ".pipeline")

    def native(self, logs, stage, argv, exit_code=0):
        Path(logs).mkdir(parents=True, exist_ok=True)
        record = dict(schema="jev.stage.v1", stage=stage, argv=argv)
        write_new(Path(logs) / (stage + ".command.json"), record)
        write_new(Path(logs) / (stage + ".result.json"), dict(record, exit_code=exit_code))

    def train(self, plan, gpu, allow_environment_drift=False):
        self.calls["train"] += 1
        self.assertEqual(Path(plan["source_root"]), self.source)
        run = self.checkpoint.parent
        run.mkdir(parents=True, exist_ok=False)
        write_new(run / "execution.json", plan)
        write_new(run / "source_files.json", pipeline._source_files(self.source))
        code = 7 if self.fail.get("train") else 0
        self.native(run, "train", plan["argv"], code)
        if code:
            raise RuntimeError("fixture native training exit 7")
        self.checkpoint.mkdir()
        write_new(self.checkpoint / "training_config.json", dict(args={key[2:]: value for key, value in options(plan["argv"]).items()}, head_config=HEAD))
        metrics = dict(status="complete", stopped_early=False, optimizer_steps=1, planned_optimizer_steps=1, source_records_seen=1)
        write_new(self.checkpoint / "training_metrics.json", metrics)
        for name in ("head.pt", "adapter_model.safetensors", "adapter_config.json"):
            (self.checkpoint / name).write_text("fixture checkpoint bytes " + name, encoding="utf-8")
        write_new(run / "verified_checkpoint.json", dict(metrics=metrics,
            artifacts_sha256={path.name: sha(path) for path in self.checkpoint.iterdir()}))
        return self.checkpoint

    def export(self, checkpoint, destination, site, tree="e11", gpu=None, examples=None, source_root=None):
        self.calls["export"] += 1
        self.assertEqual(Path(source_root), self.source)
        self.assertEqual(tree, self.plan["tree"])
        destination = Path(destination)
        destination.mkdir(parents=True)
        argv = [site["python"]["training"], "-P", "-m", "runtime.export_bundle", "--run", str(checkpoint),
            "--out", str(destination), "--name", destination.name, "--base", site["base_model"], "--verification-examples", str(examples)]
        code = 7 if self.fail.get("export") else 0
        self.native(destination.parent / (destination.name + ".export"), "export", argv, code)
        (destination / "partial.txt").write_text("preserve all attempt bytes", encoding="utf-8")
        if code:
            raise RuntimeError("fixture native export exit 7")
        (destination / "backbone").mkdir()
        (destination / "backbone/weights.safetensors").write_text("fixture backbone identity", encoding="utf-8")
        (destination / "classification_head.safetensors").write_text("fixture head identity", encoding="utf-8")
        write_new(destination / "head_config.json", HEAD)
        manifest = dict(name=destination.name, probability_source="classification_head", head_config=HEAD,
            source_checkpoint=str(checkpoint), verification_input_sha256=sha(examples),
            source_hashes={path.relative_to(self.source).as_posix(): sha(path) for package in ("kev", "runtime")
                for path in (self.source / package).glob("*.py")},
            checkpoint_hashes={name: sha(Path(checkpoint) / name) for name in ("head.pt", "adapter_model.safetensors", "adapter_config.json")},
            files={path.relative_to(destination).as_posix(): sha(path) for path in destination.rglob("*") if path.is_file()})
        write_new(destination / "model_manifest.json", manifest)
        write_new(destination / "validation_report.json", dict(status="PASS", manifest_sha256=sha(destination / "model_manifest.json"),
            head_config=HEAD, argmax_equal=True, source_unchanged=True, checkpoint_unchanged=True, requests=1, max_probability_delta=0.0))
        return destination

    def prediction(self, output, argv, model_name):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        code = 7 if self.fail.get("predict") else 0
        self.native(output.parent / (output.name + ".logs"), "predict", argv, code)
        if code:
            output.write_text("partial failed prediction", encoding="utf-8")
            raise RuntimeError("fixture native prediction exit 7")
        response = dict(model=model_name, probability_source="classification_head", head_kind="pointer", head_config=HEAD,
            probabilities={"choice": {"a": 0.7, "b": 0.3}})
        output.write_text(json.dumps(dict(id="request-1", response=response)) + "\n", encoding="utf-8")
        return output

    def predict(self, bundle, requests, output, site, gpu=None, device="cuda"):
        self.calls["predict"] += 1
        argv = [site["python"]["serving"], "predict_worker.py", "--bundle", str(bundle), "--requests", str(requests), "--output", str(output)]
        return self.prediction(output, argv, Path(bundle).name)

    def predict_checkpoint(self, checkpoint, source, requests, output, site, gpu=None, device="cuda", model_name=None):
        self.calls["checkpoint_predict"] += 1
        self.assertEqual(Path(source), self.source)
        argv = [site["python"]["serving"], "predict_worker.py", "--checkpoint", str(checkpoint), "--source-root", str(source),
            "--model-name", model_name, "--base", site["base_model"], "--requests", str(requests), "--output", str(output)]
        return self.prediction(output, argv, model_name)

    def score(self, requests, labels, predictions, output, model=None):
        self.calls["evaluate"] += 1
        if self.fail.get("evaluate"):
            Path(output).mkdir(parents=True)
            (Path(output) / "partial.txt").write_text("partial scoring", encoding="utf-8")
            raise ValueError("fixture offline scoring failure")
        return self.score_files(requests, labels, predictions, output, model=model)

    def run_pipeline(self, **kwargs):
        return pipeline.run_pipeline(self.plan, "0", self.requests, self.labels, self.examples, **kwargs)

    def test_complete_pipeline_reuses_all_proven_stages_on_resume(self):
        result = self.run_pipeline()
        self.assertEqual((result["status"], result["model_name"], result["inference_format"]), ("COMPLETE", "fixture-model", "bundle"))
        calls = dict(self.calls)
        resumed = self.run_pipeline(resume=True)
        self.assertEqual(resumed, result)
        self.assertEqual(self.calls, calls)
        with self.assertRaisesRegex(ValueError, "records already exist"):
            self.run_pipeline()

    def test_failed_export_preserves_attempt_and_reuses_training_in_new_attempt(self):
        self.fail["export"] = True
        with self.assertRaisesRegex(RuntimeError, "export exit 7"):
            self.run_pipeline()
        preserved = self.control / "attempts/export-0001/fixture-model/partial.txt"
        before = preserved.read_bytes()
        self.fail.clear()
        result = self.run_pipeline(resume=True)
        self.assertEqual(self.calls["train"], 1)
        self.assertEqual(self.calls["export"], 2)
        self.assertIn("export-0002", result["bundle"])
        self.assertEqual(preserved.read_bytes(), before)

    def test_failed_prediction_preserves_attempt_and_reuses_proven_bundle(self):
        self.fail["predict"] = True
        with self.assertRaisesRegex(RuntimeError, "prediction exit 7"):
            self.run_pipeline()
        previous = self.control / "attempts/predict-0001/predictions.jsonl"
        before = previous.read_bytes()
        self.fail.clear()
        result = self.run_pipeline(resume=True)
        self.assertEqual((self.calls["train"], self.calls["export"], self.calls["predict"]), (1, 1, 2))
        self.assertIn("predict-0002", result["predictions"])
        self.assertEqual(previous.read_bytes(), before)

    def test_failed_offline_scoring_reuses_predictions_and_preserves_old_attempt(self):
        self.fail["evaluate"] = True
        with self.assertRaisesRegex(ValueError, "offline scoring failure"):
            self.run_pipeline()
        self.fail.clear()
        result = self.run_pipeline(resume=True)
        self.assertEqual((self.calls["train"], self.calls["export"], self.calls["predict"], self.calls["evaluate"]), (1, 1, 1, 2))
        self.assertIn("evaluate-0002", result["evaluation_dir"])
        self.assertTrue((self.control / "attempts/evaluate-0001/evaluation/partial.txt").is_file())

    def test_failed_training_is_never_restarted(self):
        self.fail["train"] = True
        with self.assertRaisesRegex(RuntimeError, "training exit 7"):
            self.run_pipeline()
        self.fail.clear()
        with self.assertRaisesRegex(RuntimeError, "will not be restarted"):
            self.run_pipeline(resume=True)
        self.assertEqual(self.calls["train"], 1)
        self.assertEqual(self.calls["export"], 0)

    def test_missing_external_native_exit_does_not_imply_a_stopped_job(self):
        self.fail["predict"] = True
        with self.assertRaises(RuntimeError):
            self.run_pipeline()
        native = self.control / "attempts/predict-0001/predictions.jsonl.logs/predict.result.json"
        native.unlink()  # Simulate an interrupted observer with a command but no terminal evidence.
        self.fail.clear()
        with self.assertRaisesRegex(RuntimeError, "may still be running"):
            self.run_pipeline(resume=True)
        self.assertEqual(self.calls["predict"], 1)

    def test_completed_checkpoint_byte_changes_are_rejected_without_launch(self):
        self.run_pipeline()
        calls = dict(self.calls)
        (self.checkpoint / "head.pt").write_text("changed completed weights", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.run_pipeline(resume=True)
        self.assertEqual(self.calls, calls)

    def test_input_source_and_plan_drift_are_rejected(self):
        self.run_pipeline()
        calls = dict(self.calls)
        labels = self.labels.read_bytes()
        self.labels.write_text(json.dumps(dict(id="request-1", targets={"choice": "b"})) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity drift"):
            self.run_pipeline(resume=True)
        self.labels.write_bytes(labels)
        source = (self.source / "kev/train.py").read_bytes()
        (self.source / "kev/train.py").write_text("# changed implementation\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity drift"):
            self.run_pipeline(resume=True)
        (self.source / "kev/train.py").write_bytes(source)
        self.plan["site"]["environment"]["CUSTOM_SETTING"] = "different"
        with self.assertRaisesRegex(ValueError, "identity drift"):
            self.run_pipeline(resume=True)
        self.assertEqual(self.calls, calls)

    def test_native_complete_stages_can_be_adopted_without_state_only_success(self):
        result = self.run_pipeline()
        calls = dict(self.calls)
        (self.control / "TRAIN_COMPLETE.json").unlink()
        for stage in ("export", "predict", "evaluate"):
            (self.control / f"attempts/{stage}-0001/COMPLETE.json").unlink()
        self.assertEqual(self.run_pipeline(resume=True), result)
        self.assertEqual(self.calls, calls)
        self.assertTrue(read(self.control / "TRAIN_COMPLETE.json")["adopted_after_native_verification"])

    def test_auto_uses_exact_legacy_source_without_silently_substituting_e11(self):
        self.source = root() / "vendor/original28"
        self.assertTrue((self.source / "kev/train.py").is_file())
        self.assertFalse((self.source / "runtime/export_bundle.py").is_file())
        self.plan["source_root"] = str(self.source)
        self.plan["custom_source"] = False
        self.plan["tree"] = "original28"
        result = self.run_pipeline()
        self.assertEqual(result["inference_format"], "checkpoint")
        self.assertIsNone(result["bundle"])
        self.assertEqual((self.calls["export"], self.calls["predict"], self.calls["checkpoint_predict"]), (0, 0, 1))

    def test_lifecycle_claim_refuses_another_controller(self):
        self.run_pipeline()
        with pipeline._claim(self.control / "controller.lock"):
            with self.assertRaisesRegex(RuntimeError, "another pipeline controller"):
                self.run_pipeline(resume=True)

    def test_invalid_population_is_rejected_before_training_or_pipeline_state(self):
        self.labels.write_text('{"id":"request-1","targets":{}}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "targets must cover"):
            self.run_pipeline()
        self.assertEqual(self.calls["train"], 0)
        self.assertFalse(self.control.exists())


if __name__ == "__main__":
    unittest.main()
