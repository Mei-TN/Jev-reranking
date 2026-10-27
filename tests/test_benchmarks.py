"""Test benchmark preparation and resource configuration."""
import contextlib
import ast
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from jev import benchmarks
from jev.config import read, root, sha, write_new


class BenchmarkHelpContracts(unittest.TestCase):
    def test_help_includes_all_native_arguments_without_site_or_resource_query(self):
        native, _ = benchmarks._load_original()
        options = {option for action in native.parser()._actions for option in action.option_strings}
        for arguments in (["--help"], ["-h"]):
            with self.subTest(arguments=arguments):
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured), \
                        patch.object(benchmarks, "_site", side_effect=AssertionError("help read a site")), \
                        patch.object(benchmarks, "_machine_and_gpu", side_effect=AssertionError("help queried a GPU")), \
                        patch.object(benchmarks.subprocess, "run", side_effect=AssertionError("help launched a process")):
                    self.assertEqual(benchmarks.main(arguments), 0)
                help_text = captured.getvalue()
                for option in options | {"--site"}:
                    self.assertIn(option, help_text)
                self.assertNotIn("--_worker", help_text)

    def test_help_rejects_changed_preserved_runner_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory)
            evaluation = fake_root / "vendor/evaluation"
            evaluation.mkdir(parents=True)
            (evaluation / "run_evaluation.py").write_text("raise AssertionError('must never import')\n", encoding="utf-8")
            captured = io.StringIO()
            with patch.object(benchmarks, "root", return_value=fake_root), contextlib.redirect_stderr(captured):
                self.assertEqual(benchmarks.main(["--help"]), 2)
            self.assertIn("pinned public source", captured.getvalue())


class ControllerCapabilityContracts(unittest.TestCase):
    def test_linux_missing_api_gives_site_correction_before_native_launch(self):
        for missing in ("open", "signal"):
            with self.subTest(missing=missing), patch.object(benchmarks.sys, "platform", "linux"), \
                    patch.object(benchmarks.os, "pidfd_open", None if missing == "open" else lambda pid: 37, create=True), \
                    patch.object(benchmarks.signal, "pidfd_send_signal", None if missing == "signal" else lambda fd, sig: None, create=True):
                with self.assertRaisesRegex(ValueError, "select a Linux controller Python"):
                    benchmarks._controller_capabilities()

    def test_linux_probe_uses_only_self_signal_zero_and_closes_descriptor(self):
        with patch.object(benchmarks.sys, "platform", "linux"), \
                patch.object(benchmarks.os, "pidfd_open", return_value=37, create=True) as opened, \
                patch.object(benchmarks.signal, "pidfd_send_signal", return_value=None, create=True) as sent, \
                patch.object(benchmarks.os, "close") as closed:
            actual = benchmarks._controller_capabilities()
        opened.assert_called_once_with(os.getpid())
        sent.assert_called_once_with(37, 0)
        closed.assert_called_once_with(37)
        self.assertEqual(actual["status"], "SELF_PIDFD_PROBE_PASSED")

    def test_linux_kernel_failure_is_actionable_and_closes_open_descriptor(self):
        with patch.object(benchmarks.sys, "platform", "linux"), \
                patch.object(benchmarks.os, "pidfd_open", return_value=37, create=True), \
                patch.object(benchmarks.signal, "pidfd_send_signal", side_effect=OSError("probe denied"), create=True), \
                patch.object(benchmarks.os, "close") as closed:
            with self.assertRaisesRegex(ValueError, "cannot use the native Linux pidfd APIs"):
                benchmarks._controller_capabilities()
        closed.assert_called_once_with(37)

    def test_non_linux_preparation_reports_its_scope_without_opening_pidfd(self):
        with patch.object(benchmarks.sys, "platform", "win32"), \
                patch.object(benchmarks.os, "pidfd_open", side_effect=AssertionError("opened pidfd"), create=True):
            actual = benchmarks._controller_capabilities()
        self.assertEqual(actual["status"], "NON_LINUX_CPU_PREPARATION_ONLY")


class BenchmarkAdapterContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name).resolve()
        self.output = self.folder / "new-benchmark"
        self.source = root() / "vendor/e12"
        self.checkpoint = self.folder / "checkpoint"
        self.checkpoint.mkdir()
        for name in ("head.pt", "adapter_model.safetensors", "adapter_config.json"):
            (self.checkpoint / name).write_bytes(b"CPU-preparation fixture, never loaded as a model")
        self.native = {}
        for key in ("base", "oc_root", "rw_root", "runtime_cache", "selection_root", "decision_cache", "hf_cache"):
            path = self.folder / key
            path.mkdir()
            self.native[key] = str(path)
        self.native.update(serve_python=sys.executable, oc_python=sys.executable, mc_python=sys.executable,
                           base_revision="56f0584bb40578a2c33b1b40a08ccd17243ad710")
        examples = self.folder / "verification_examples.json"
        write_new(examples, {"purpose": "CPU input-validation fixture only"})
        self.native["verification_examples"] = str(examples)
        for key, suffix in (("base", "config.json"), ("oc_root", "run.py"),
                            ("rw_root", "scripts/openjev/data_revisions.json"),
                            ("rw_root", "src/agentic_evaluate/benchmarks/openjev_adapter.py"),
                            ("runtime_cache", "data/gpqa-official/gpqa_diamond.csv")):
            item = Path(self.native[key]) / suffix
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_text("CPU fixture\n", encoding="utf-8")
        runner, _ = benchmarks._load_original()
        for task, count in runner.JK_COUNTS.items():
            write_new(Path(self.native["selection_root"]) / (task + ".selection.json"),
                      {"selected_ids": list(range(count))})
        self.source_hashes = {p.name: sha(p) for p in sorted((self.source / "kev").glob("*.py"))
                              if not p.name.startswith("._")}
        write_new(self.checkpoint / "training_config.json", dict(
            base_revision=self.native["base_revision"],
            args=dict(base=self.native["base"], epochs=1, weights_dtype="fp32", full_finetune=False),
            dse=dict(planned_optimizer_steps=2, source_sha256=self.source_hashes),
            head_config=dict(version=1, kind="pointer", dim=256, readout="end", rng_anchor_dim=256)))
        write_new(self.checkpoint / "training_metrics.json", dict(status="complete", stopped_early=False,
                    planned_optimizer_steps=2, optimizer_steps=2))
        self.guard = dict(hostnames=["explicit-test-host"], accounts=["explicit-test-account"], uids=[1234],
                          gpu_name_contains="H100", idle_memory_mib_below=500)
        self.site = self.folder / "benchmark-site.json"
        write_new(self.site, dict(schema=benchmarks.SCHEMA, label="CPU fixture", controller_python=sys.executable,
                                 native_options=self.native, guard=self.guard))
        self.forwarded = ["--checkpoint", str(self.checkpoint), "--code-tree", str(self.source),
                          "--run-id", "cpu-preparation-fixture", "--gpu", "0", "--output", str(self.output)]

    def parsed_site(self):
        return benchmarks._site(self.site)

    def test_unchanged_native_preparation_has_no_gpu_query_or_output(self):
        original, identities = benchmarks._verify_original()
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), patch.object(benchmarks.subprocess, "run", side_effect=AssertionError("CPU preparation queried a subprocess")):
            self.assertEqual(benchmarks._worker(*self.parsed_site(), self.forwarded + ["--prepare-only"]), 0)
        report = json.loads(captured.getvalue())
        self.assertEqual(report["status"], "CPU_PREPARED_NOT_LAUNCHED")
        self.assertEqual(report["plan"]["jk_records"], 6667)
        self.assertEqual(report["plan"]["mc_calls"], 382552)
        self.assertEqual(report["plan"]["replicas"], 4)
        self.assertEqual(report["plan"]["code_tree"], str(self.source.resolve()))
        self.assertEqual(report["adapter"]["modified_hooks"], ["machine_and_gpu"])
        self.assertEqual(report["adapter"]["checkpoint_source_continuity"]["kev_source_sha256"], self.source_hashes)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.folder.glob("new-benchmark.*")))
        self.assertEqual(sha(original), benchmarks.ORIGINAL_RUNNER_SHA256)
        self.assertEqual(benchmarks._verify_original()[1], identities)

    def test_metadata_source_drift_and_two_epoch_gate_fail_before_resource_query(self):
        config_path = self.checkpoint / "training_config.json"
        original = read(config_path)
        bad = json.loads(json.dumps(original))
        bad["dse"]["source_sha256"]["model.py"] = "0" * 64
        config_path.write_text(json.dumps(bad), encoding="utf-8")
        with patch.object(benchmarks, "_machine_and_gpu", side_effect=AssertionError("queried GPU")):
            with self.assertRaisesRegex(ValueError, "source hashes"):
                benchmarks._worker(*self.parsed_site(), self.forwarded + ["--prepare-only"])
        original["args"]["epochs"] = 2
        config_path.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "one full epoch"):
            benchmarks._worker(*self.parsed_site(), self.forwarded + ["--prepare-only"])
        self.assertFalse(self.output.exists())

    def test_original_runner_tamper_is_rejected_before_import(self):
        fake_root = self.folder / "tampered-package"
        evaluation = fake_root / "vendor/evaluation"
        evaluation.mkdir(parents=True)
        (evaluation / "run_evaluation.py").write_text("raise RuntimeError('must never import')\n", encoding="utf-8")
        with patch.object(benchmarks, "root", return_value=fake_root):
            with self.assertRaisesRegex(ValueError, "pinned public source"):
                benchmarks._load_original()

    def gpu_context(self, gpu="0, GPU-0123, NVIDIA H100 80GB HBM3, 12\n", apps="GPU-other, 99\n"):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(benchmarks.sys, "platform", "linux"))
        stack.enter_context(patch.object(benchmarks.socket, "gethostname", return_value="explicit-test-host"))
        stack.enter_context(patch.object(benchmarks.getpass, "getuser", return_value="explicit-test-account"))
        stack.enter_context(patch.object(benchmarks.os, "geteuid", return_value=1234, create=True))
        stack.enter_context(patch.object(benchmarks, "_boot_id", return_value="actual-test-boot"))
        query = stack.enter_context(patch.object(benchmarks.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 0, stdout=gpu), subprocess.CompletedProcess([], 0, stdout=apps)]))
        return stack, query

    def test_guard_records_actual_h100_with_exact_native_return_keys(self):
        stack, query = self.gpu_context()
        with stack:
            actual = benchmarks._machine_and_gpu("0", self.guard)
        self.assertEqual(set(actual), {"index", "uuid", "name", "memory_mib", "host", "boot_id", "account",
                                      "sampled_epoch", "gpu_query", "compute_query"})
        self.assertEqual(actual["name"], "NVIDIA H100 80GB HBM3")
        self.assertEqual(actual["host"], "explicit-test-host")
        self.assertEqual(actual["memory_mib"], 12)
        self.assertEqual(query.call_args_list[0].args[0][1], "--id=0")

    def test_guard_rejects_wrong_host_wrong_hardware_and_either_kind_of_occupancy(self):
        with patch.object(benchmarks.sys, "platform", "linux"), patch.object(socket, "gethostname", return_value="outside-site"), patch.object(benchmarks.subprocess, "run") as query:
            with self.assertRaisesRegex(ValueError, "host/account"):
                benchmarks._machine_and_gpu("0", self.guard)
            query.assert_not_called()
        for gpu, apps, message in (("0, GPU-0123, NVIDIA H20, 12\n", "", "hardware"),
                                   ("0, GPU-0123, NVIDIA H100, 500\n", "", "occupied"),
                                   ("0, GPU-0123, NVIDIA H100, 0\n", "GPU-0123, 87\n", "compute process")):
            with self.subTest(message=message):
                stack, _ = self.gpu_context(gpu, apps)
                with stack, self.assertRaisesRegex(ValueError, message):
                    benchmarks._machine_and_gpu("0", self.guard)

    def test_site_preserves_lexical_venv_python_and_requires_explicit_guards(self):
        site = read(self.site)
        site["controller_python"] = "roles/controller/bin/python"
        site["native_options"]["serve_python"] = "roles/model/bin/python"
        self.site.write_text(json.dumps(site), encoding="utf-8")
        _, parsed = self.parsed_site()
        self.assertEqual(parsed["controller_python"], os.path.abspath(self.folder / "roles/controller/bin/python"))
        self.assertEqual(parsed["native_options"]["serve_python"], os.path.abspath(self.folder / "roles/model/bin/python"))
        site["guard"]["uids"] = []
        self.site.write_text(json.dumps(site), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "explicitly list"):
            self.parsed_site()

    def test_native_execution_replaces_only_resource_callback_and_records_hardware(self):
        native, identities = benchmarks._load_original()
        before = {name: getattr(native, name) for name in ("prepare", "checkpoint_metadata", "input_identity", "evaluate", "machine_and_gpu")}
        actual = dict(index=0, uuid="GPU-0123", name="NVIDIA H100", memory_mib=0, host="explicit-test-host",
                      boot_id="boot", account="explicit-test-account", sampled_epoch=1, gpu_query="raw", compute_query="")
        def execute(plan):
            self.assertEqual(native.machine_and_gpu("0"), actual)
            write_new(Path(plan["output"]) / "RESULT.json", {"status": "orchestration fixture, no scientific results"})
            return 7
        with patch.object(benchmarks, "_load_original", return_value=(native, identities)), patch.object(native, "evaluate", side_effect=execute), patch.object(benchmarks, "_machine_and_gpu", return_value=actual):
            self.assertEqual(benchmarks._worker(*self.parsed_site(), self.forwarded), 7)
            self.assertIs(native.prepare, before["prepare"])
            self.assertIs(native.checkpoint_metadata, before["checkpoint_metadata"])
            self.assertIs(native.input_identity, before["input_identity"])
        terminal = read(benchmarks._sidecar(self.output, "SITE_ADAPTER_RESULT.json"))
        self.assertEqual(terminal["native_function_return"], 7)
        self.assertEqual(terminal["actual_gpu_observations"], [actual])
        self.assertTrue(terminal["native_sources_unchanged"])
        self.assertEqual(terminal["native_result_sha256"], sha(self.output / "RESULT.json"))

    def test_launcher_uses_fresh_selected_python_and_preserves_native_failure_receipt(self):
        commands = []
        initial_site_digest = sha(self.site)
        def failed(command, **kwargs):
            commands.append((command, kwargs))
            kwargs["stderr"].write(b"native fixture failure\n")
            self.site.write_text(self.site.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 7)
        with patch.object(benchmarks.subprocess, "run", side_effect=failed), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(benchmarks.main(["--site", str(self.site), *self.forwarded]), 7)
        command, kwargs = commands[0]
        self.assertEqual(command[:4], [sys.executable, "-P", "-m", "jev.benchmarks"])
        self.assertIn("--_worker", command)
        self.assertEqual(kwargs["env"]["PYTHONNOUSERSITE"], "1")
        self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        receipt = read(benchmarks._sidecar(self.output, "CONTROLLER_EXIT.json"))
        self.assertEqual(receipt["native_exit"], 7)
        self.assertEqual(receipt["site_sha256"], initial_site_digest)
        self.assertEqual(receipt["site_sha256_after"], sha(self.site))
        self.assertFalse(receipt["site_file_unchanged"])
        self.assertIsNone(receipt["native_result_sha256"])
        self.assertFalse(self.output.exists())
        self.assertEqual(benchmarks._sidecar(self.output, "controller.stderr.log").read_bytes(), b"native fixture failure\n")

    def test_post_execution_source_failure_still_retains_attempt_receipt(self):
        native, identities = benchmarks._load_original()
        with patch.object(benchmarks, "_load_original", return_value=(native, identities)), \
                patch.object(native, "evaluate", return_value=0), \
                patch.object(benchmarks, "_verify_original", side_effect=ValueError("recorded source changed")):
            with self.assertRaisesRegex(ValueError, "post-execution validation"):
                benchmarks._worker(*self.parsed_site(), self.forwarded)
        receipt = read(benchmarks._sidecar(self.output, "SITE_ADAPTER_RESULT.json"))
        self.assertFalse(receipt["native_sources_unchanged"])
        self.assertEqual(receipt["native_source_validation_error"], "recorded source changed")
        self.assertEqual(receipt["native_function_return"], 0)
        self.assertFalse(self.output.exists())

    def test_fresh_interpreter_cpu_prepare_forwards_paths_without_creating_outputs(self):
        command = [sys.executable, "-P", "-m", "jev.benchmarks", "--site", str(self.site), *self.forwarded, "--prepare-only"]
        env = dict(os.environ, PYTHONPATH=str(Path(benchmarks.__file__).resolve().parents[1]), PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "CPU_PREPARED_NOT_LAUNCHED")
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.folder.glob("new-benchmark.*")))

    def two_epoch_checkpoint(self, fold="F2", readout="end"):
        source = root() / "vendor/e11"
        steps, records = (3864, 30912) if fold == "F2" else (3862, 30884)
        config = read(self.checkpoint / "training_config.json")
        config["args"]["epochs"] = 2
        config["head_config"]["readout"] = readout
        config["dse"].update(planned_optimizer_steps=steps, requested_source_records=records,
            source_sha256={p.name: sha(p) for p in sorted((source / "kev").glob("*.py")) if not p.name.startswith("._")})
        (self.checkpoint / "training_config.json").write_text(json.dumps(config), encoding="utf-8")
        metrics = dict(status="complete", stopped_early=False, optimizer_steps=steps,
                       planned_optimizer_steps=steps, source_records_seen=records, requested_records=records)
        (self.checkpoint / "training_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        self.forwarded[self.forwarded.index("--code-tree") + 1] = str(source)
        return config, metrics

    def fresh_benchmark(self, *extra, protocol="e11"):
        command = [sys.executable, "-P", "-m", "jev.benchmarks", "--site", str(self.site),
                   "--protocol", protocol, *self.forwarded, "--prepare-only", *extra]
        env = dict(os.environ, PYTHONPATH=str(Path(benchmarks.__file__).resolve().parents[1]),
                   PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
        return subprocess.run(command, capture_output=True, text=True, env=env, check=False, timeout=30)

    def test_e11_native_complete_fold_contracts_and_profile_identity(self):
        for fold, readout, steps, records in (("F2", "end", 3864, 30912), ("F3", "mean_span", 3862, 30884)):
            with self.subTest(fold=fold):
                self.two_epoch_checkpoint(fold, readout)
                completed = self.fresh_benchmark("--training-contract", "two_epoch_" + fold)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                report = json.loads(completed.stdout)
                contract = dict(name="two_epoch_" + fold, epochs=2, optimizer_steps=steps, source_records=records)
                self.assertEqual(report["plan"]["training_contract"], contract)
                self.assertEqual(report["adapter"]["native_profile"], dict(protocol="e11", directory="followup/evaluation",
                    runner_sha256=benchmarks.E11_RUNNER_SHA256, training_contract=contract))
                self.assertEqual(report["adapter"]["original_runner_sha256"], benchmarks.E11_RUNNER_SHA256)
                self.assertEqual(report["plan"]["jk_records"], 6667)
                self.assertEqual(report["plan"]["mc_calls"], 382552)
                self.assertEqual(report["plan"]["head_config"]["readout"], readout)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.folder.glob("new-benchmark.*")))

    def test_e11_requires_explicit_two_epoch_contract_and_default_remains_one_epoch(self):
        self.two_epoch_checkpoint()
        for arguments in ((), ("--training-contract", "one_epoch_F2")):
            with self.subTest(arguments=arguments):
                completed = self.fresh_benchmark(*arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertIn("requires the native --training-contract", completed.stderr)
        default = self.fresh_benchmark(protocol="one_epoch")
        self.assertEqual(default.returncode, 2)
        self.assertIn("one full epoch", default.stderr)
        wrong_fold = self.fresh_benchmark("--training-contract", "two_epoch_F3")
        self.assertEqual(wrong_fold.returncode, 2)
        self.assertIn("not the complete two_epoch_F3", wrong_fold.stderr)
        self.assertFalse(self.output.exists())

    def test_e11_refuses_each_incomplete_occurrence_counter_and_f3_ceil_undercount(self):
        config, metrics = self.two_epoch_checkpoint("F3", "mean_span")
        for target, key, value in (("metrics", "source_records_seen", 30883), ("metrics", "requested_records", 30883),
                                   ("dse", "requested_source_records", 30883), ("metrics", "optimizer_steps", 3861),
                                   ("metrics", "stopped_early", True)):
            with self.subTest(key=key):
                actual_config, actual_metrics = json.loads(json.dumps(config)), dict(metrics)
                (actual_metrics if target == "metrics" else actual_config["dse"])[key] = value
                (self.checkpoint / "training_config.json").write_text(json.dumps(actual_config), encoding="utf-8")
                (self.checkpoint / "training_metrics.json").write_text(json.dumps(actual_metrics), encoding="utf-8")
                completed = self.fresh_benchmark("--training-contract", "two_epoch_F3")
                self.assertEqual(completed.returncode, 2, completed.stdout)
                self.assertIn("checkpoint is not", completed.stderr)
        config["dse"]["planned_optimizer_steps"] = 3861
        metrics.update(optimizer_steps=3861, planned_optimizer_steps=3861)
        (self.checkpoint / "training_config.json").write_text(json.dumps(config), encoding="utf-8")
        (self.checkpoint / "training_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        synchronized_undercount = self.fresh_benchmark("--training-contract", "two_epoch_F3")
        self.assertEqual(synchronized_undercount.returncode, 2)
        self.assertIn("not the complete two_epoch_F3", synchronized_undercount.stderr)
        self.assertFalse(self.output.exists())

    def test_e11_preserves_native_fp32_master_lora_gate(self):
        config, _ = self.two_epoch_checkpoint()
        for key, value in (("weights_dtype", "bf16"), ("full_finetune", True)):
            with self.subTest(key=key):
                changed = json.loads(json.dumps(config))
                changed["args"][key] = value
                (self.checkpoint / "training_config.json").write_text(json.dumps(changed), encoding="utf-8")
                completed = self.fresh_benchmark("--training-contract", "two_epoch_F2")
                self.assertEqual(completed.returncode, 2)
                self.assertIn("FP32 master LoRA", completed.stderr)

    def test_e11_help_uses_selected_native_parser_without_site_or_gpu(self):
        for protocol, has_contract in (("one_epoch", False), ("e11", True)):
            command = [sys.executable, "-P", "-m", "jev.benchmarks", "--protocol", protocol, "--help"]
            env = dict(os.environ, PYTHONPATH=str(Path(benchmarks.__file__).resolve().parents[1]), CUDA_VISIBLE_DEVICES="",
                       PYTHONDONTWRITEBYTECODE="1")
            completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("--protocol {e11,one_epoch}", completed.stdout)
            self.assertEqual("--training-contract" in completed.stdout, has_contract)
            self.assertIn("--port-base", completed.stdout)
            self.assertIn("--site", completed.stdout)

    def test_e11_workers_are_frozen_and_only_three_controller_contract_functions_differ(self):
        original, baseline = benchmarks._verify_original()
        followup, selected = benchmarks._verify_original("e11")
        self.assertEqual(selected["run_evaluation.py"], benchmarks.E11_RUNNER_SHA256)
        self.assertEqual({k: v for k, v in selected.items() if k != "run_evaluation.py"},
                         {k: v for k, v in baseline.items() if k != "run_evaluation.py"})
        functions = []
        for path in (original, followup):
            functions.append({node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                              if isinstance(node, (ast.FunctionDef, ast.ClassDef))})
        self.assertEqual({name for name in set(functions[0]) | set(functions[1]) if functions[0].get(name) != functions[1].get(name)},
                         {"checkpoint_metadata", "parser", "prepare"})

    def test_e11_launcher_retains_selected_native_profile_on_native_failure(self):
        with (patch.object(benchmarks.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as run,
              contextlib.redirect_stdout(io.StringIO())):
            code = benchmarks.main(["--site", str(self.site), "--protocol", "e11", *self.forwarded,
                                    "--training-contract", "two_epoch_F2"])
        self.assertEqual(code, 7)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--protocol") + 1], "e11")
        receipt = read(benchmarks._sidecar(self.output, "CONTROLLER_EXIT.json"))
        self.assertEqual(receipt["protocol"], "e11")
        self.assertEqual(receipt["original_runner_sha256"], benchmarks.E11_RUNNER_SHA256)
        self.assertEqual(receipt["native_exit"], 7)
        self.assertIsNone(receipt["native_result_sha256"])


if __name__ == "__main__":
    unittest.main()
