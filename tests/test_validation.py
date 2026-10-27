"""Stdlib checks for asset/reference binding and fresh-interpreter diagnostics."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jev import validation
from jev.config import read, root, sha
from jev.recipes import command


class ValidationContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)

    def test_base_pins_include_both_weight_shards_and_tokenizer_members(self):
        model = self.folder / "assets/model"
        pins = validation.base_asset_pins({"base_model": str(model)})
        manifest = read(root() / validation.BASE_MANIFEST)
        expected = {str(model / item["path"][5:]): item["sha256"]
                    for item in manifest["assets"].values() if item["path"].startswith("base/")}
        self.assertEqual(pins, expected)
        self.assertEqual(len(pins), 10)
        for member in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            self.assertIn(str(model / member), pins)
        with self.assertRaisesRegex(ValueError, "base_revision differs"):
            validation.base_asset_pins({"base_model": str(model), "base_revision": "different-revision"})

    def test_protocol_byte_tampering_and_unregistered_paths_are_rejected(self):
        protocol = self.folder / "provenance/plans/fixed.json"
        protocol.parent.mkdir(parents=True)
        protocol.write_text('{"fixed":true}\n', encoding="utf-8")
        entry = dict(path="provenance/plans/fixed.json", sha256=sha(protocol), bytes=protocol.stat().st_size, origin="fixture-source")
        manifest = self.folder / "provenance/PROTOCOLS.json"
        manifest.write_text(json.dumps(dict(schema="jev.protocols.v1", files=[entry])), encoding="utf-8")
        with patch.object(validation, "root", return_value=self.folder):
            self.assertEqual(validation.verify_protocol(entry["path"]), entry)
            self.assertEqual(validation.verify_protocol(protocol), entry)
            protocol.write_text('{"fixed":null}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protocol bytes differ"):
                validation.verify_protocol(protocol)
            with self.assertRaisesRegex(ValueError, "exactly one pinned"):
                validation.verify_protocol("unregistered.json")
            with self.assertRaisesRegex(ValueError, "outside"):
                validation.verify_protocol(self.folder.parent / "outside.json")

    def reference_plan(self):
        output = self.folder / "reference/output"
        output.mkdir(parents=True)
        base, data = str(self.folder / "base"), str(self.folder / "data/F2.train.jsonl")
        config = output / "training_config.json"
        config.write_text(json.dumps(dict(args=dict(base=base, data=data))), encoding="utf-8")
        return dict(argv=command(sys.executable, {"--base": base, "--data": data}),
                    references=[dict(run_id="E458_F2s44_POOL_ALL", path=str(output))]), config

    def test_reference_locations_require_native_exact_paths_without_rewriting_metadata(self):
        plan, config = self.reference_plan()
        self.assertEqual(validation.validate_reference_locations(plan), [])
        saved = read(config)
        saved["args"]["base"] = "/old-deployment/model"
        saved["args"]["data"] = "/old-deployment/F2.train.jsonl"
        config.write_text(json.dumps(saved), encoding="utf-8")
        before = config.read_bytes()
        problems = validation.validate_reference_locations(plan)
        self.assertEqual(len(problems), 2)
        self.assertIn("saved args.base", problems[0])
        self.assertIn("saved args.data", problems[1])
        self.assertTrue(all("Create fresh prerequisite references" in problem for problem in problems))
        self.assertTrue(all("Do not rewrite historical training_config.json" in problem for problem in problems))
        self.assertEqual(config.read_bytes(), before)

    def test_missing_reference_training_config_is_an_actionable_problem(self):
        plan, config = self.reference_plan()
        config.unlink()
        problems = validation.validate_reference_locations(plan)
        self.assertEqual(len(problems), 1)
        self.assertIn("genuine prerequisite reference", problems[0])

    def observed(self):
        recorded_python, versions = validation._recorded_environment()
        return dict(python=recorded_python, executable="lexical-role-python", prefix="fixture-role-env",
            packages=versions, native_imports={name: dict(status="PASS", file="fixture") for name in validation.NATIVE_IMPORTS},
            torch_build=dict(version=versions["torch"], cuda="12.8"))

    def probe(self, observed, code=0, stdout_prefix=""):
        return SimpleNamespace(returncode=code, stdout=stdout_prefix + "JEV_ENVIRONMENT_REPORT=" + json.dumps(observed) + "\n", stderr="fixture stderr")

    def test_fresh_probe_matches_all_recorded_versions_and_preserves_lexical_interpreter(self):
        executable = self.folder / "role/bin/python"
        executable.parent.mkdir(parents=True)
        try:
            executable.symlink_to(Path(sys.executable).resolve())
        except OSError:
            executable.write_text("fixture interpreter", encoding="utf-8")
        observed = self.observed()
        with patch.object(validation.subprocess, "run", return_value=self.probe(observed, stdout_prefix="native import diagnostic\n")) as run:
            report = validation.environment_report(str(executable), self.folder / "custom-source", {"PRIVATE_TEST_TOKEN": "never-record-this", "PYTHONPATH": "wrong-source", "CUDA_VISIBLE_DEVICES": "0"})
        self.assertEqual(report["status"], "MATCH_RECORDED_VERSIONS")
        self.assertEqual(len(report["packages"]), len(observed["packages"]))
        self.assertFalse(report["exact_closure_verified"])
        self.assertFalse(report["gpu_execution"])
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], str(executable))
        self.assertEqual(argv[1:3], ["-P", "-c"])
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["PYTHONPATH"], str(self.folder / "custom-source"))
        self.assertNotIn("never-record-this", json.dumps(report))

    def test_missing_mismatched_native_and_python_versions_are_classified(self):
        observed = self.observed()
        observed["packages"]["transformers"] = "different"
        observed["packages"]["peft"] = None
        observed["native_imports"]["fla"] = dict(status="IMPORT_FAILED", message="missing kernel library")
        observed["python"] = "3.13.0"
        with patch.object(validation.subprocess, "run", return_value=self.probe(observed)):
            report = validation.environment_report("python-role", self.folder)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertIn("peft", report["missing_packages"])
        self.assertIn("transformers", report["mismatched_packages"])
        self.assertIn("fla", report["failed_native_imports"])
        self.assertEqual(report["python"]["status"], "MISMATCH")
        observed["packages"]["peft"] = validation._recorded_environment()[1]["peft"]
        observed["native_imports"]["fla"]["status"] = "PASS"
        with patch.object(validation.subprocess, "run", return_value=self.probe(observed)):
            report = validation.environment_report("python-role", self.folder)
        self.assertEqual(report["status"], "MISMATCH")

    def test_torch_cuda_suffix_normalization_requires_matching_imported_build(self):
        observed = self.observed()
        observed["packages"]["torch"] = "2.8.0"
        with patch.object(validation.subprocess, "run", return_value=self.probe(observed)):
            report = validation.environment_report("python-role", self.folder)
        self.assertEqual(report["status"], "MATCH_WITH_TORCH_CUDA_SUFFIX_NORMALIZATION")
        self.assertEqual(report["packages"]["torch"]["status"], "CUDA_SUFFIX_NORMALIZED")
        for version, build in (("2.8.0+cpu", dict(version="2.8.0+cpu", cuda=None)),
                               ("2.8.0+cu126", dict(version="2.8.0+cu126", cuda="12.6")),
                               ("2.8.0", dict(version="2.8.0+cu128", cuda="12.6")),
                               ("2.8.0+cu128", dict(version="2.8.0+cpu", cuda=None))):
            observed["packages"]["torch"] = version
            observed["torch_build"] = build
            with self.subTest(version=version, build=build), patch.object(validation.subprocess, "run", return_value=self.probe(observed)):
                report = validation.environment_report("python-role", self.folder)
                self.assertEqual(report["status"], "MISMATCH")
                self.assertIn("torch", report["mismatched_packages"])

    def test_wrong_native_training_source_and_probe_failures_never_report_a_match(self):
        observed = self.observed()
        observed["native_imports"]["kev.train"]["status"] = "SOURCE_MISMATCH"
        with patch.object(validation.subprocess, "run", return_value=self.probe(observed)):
            self.assertEqual(validation.environment_report("python-role", self.folder)["status"], "INCOMPLETE")
        malformed = self.observed()
        malformed["native_imports"]["torch"] = "not an import record"
        for result in (self.probe({}, code=7), self.probe(malformed), SimpleNamespace(returncode=0, stdout="not a report", stderr="")):
            with patch.object(validation.subprocess, "run", return_value=result):
                report = validation.environment_report("python-role", self.folder)
                self.assertEqual(report["status"], "INCOMPLETE")
                self.assertIn("failure", report)
        with patch.object(validation.subprocess, "run", side_effect=subprocess.TimeoutExpired(["python-role"], 60)):
            report = validation.environment_report("python-role", self.folder)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["failure"]["type"], "TimeoutExpired")


if __name__ == "__main__":
    unittest.main()
