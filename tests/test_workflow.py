"""Test command generation and subprocess handling."""
from contextlib import redirect_stdout
import copy
import io
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jev import cli, recipes, workflow
from jev.config import load_site, read, root, sha


LOCATIONS = {"--base", "--data", "--out", "--e458_reference_run", "--e11_reference_run"}


class WorkflowContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.site = dict(schema="jev.site.v1", work_root=str(self.folder / "work"),
            base_model=str(self.folder / "base"), data_root=str(self.folder / "data"),
            python={"training": sys.executable, "serving": sys.executable},
            data_files={"F2": "F2.train.jsonl", "F3": "F3.train.jsonl"},
            environment={}, reference_outputs={})
        self.site_file = self.folder / "site.json"
        self.site_file.write_text(json.dumps(self.site), encoding="utf-8")

    def test_all_56_paper_commands_preserve_every_scientific_argument(self):
        count = 0
        for study, expected_count in (("e1e2", 28), ("e458", 20), ("e11", 8)):
            archived = read(root() / "provenance/plans" / (study + ".json"))
            self.assertEqual(len(archived["runs"]), expected_count)
            for historical in archived["runs"]:
                with self.subTest(study=study, run_id=historical["run_id"]):
                    plan = recipes.paper_plan(study, self.site, run_id=historical["run_id"])
                    actual, original = recipes.options(plan["argv"]), recipes.options(historical["argv"])
                    self.assertEqual(set(actual), set(original))
                    changed = {key for key in actual if actual[key] != original[key]}
                    self.assertLessEqual(changed, LOCATIONS)
                    self.assertEqual(plan["changed_location_arguments"], sorted(changed))
                    for key in set(original) - LOCATIONS:
                        self.assertEqual(actual[key], original[key], key)
                    self.assertEqual(plan["argv"][:4], [sys.executable, "-P", "-m", "kev.train"])
                    self.assertEqual(actual["--base"], self.site["base_model"])
                    self.assertEqual(actual["--data"], str(Path(self.site["data_root"]) / self.site["data_files"][historical["fold"]]))
                    self.assertEqual(plan["input_pins"][actual["--data"]], historical["data_sha256"])
                    self.assertEqual(plan["input_pins"][str(Path(self.site["base_model"]) / "config.json")], archived["base_config_sha256"])
                    self.assertEqual(plan["expected"], historical["expected"])
                    self.assertEqual(plan["head_config"], historical["head_config"])
                    self.assertFalse(plan["scientific_overrides"])
                    count += 1
        self.assertEqual(count, 56)

    def test_e11_pool_references_actual_e458_pool_and_pointer_references_e12(self):
        for fold, seed in itertools.product(("F2", "F3"), (44, 45)):
            for arm, predecessor_study, predecessor in (
                ("POOL_2EP", "e458", f"E458_{fold}s{seed}_POOL_ALL"),
                ("POINTER_2EP", "e1e2", f"E12_{fold}s{seed}_LL_A"),
            ):
                with self.subTest(fold=fold, seed=seed, arm=arm):
                    plan = recipes.paper_plan("e11", self.site, arm=arm, fold=fold, seed=seed)
                    expected = str(Path(self.site["work_root"]) / "runs" / predecessor_study / predecessor / "output")
                    self.assertEqual(plan["references"], [{"run_id": predecessor, "path": expected}])
                    self.assertEqual(recipes.options(plan["argv"])["--e11_reference_run"], expected)
        site = copy.deepcopy(self.site)
        site["reference_outputs"]["E458_F2s44_POOL_ALL"] = str(self.folder / "retained-model")
        plan = recipes.paper_plan("e11", site, arm="POOL_2EP", fold="F2", seed=44)
        self.assertEqual(plan["references"][0]["path"], site["reference_outputs"]["E458_F2s44_POOL_ALL"])


    def test_custom_json_braces_survive_placeholder_substitution(self):
        recipe = read(root() / "configs/custom.example.json")
        recipe["trainer_args"]["head_config"] = '{"version":1,"kind":"pointer","dim":256,"readout":"last_span","rng_anchor_dim":256}'
        path = self.folder / "recipe.json"
        path.write_text(json.dumps(recipe), encoding="utf-8")
        plan = recipes.custom_plan(path, self.site)
        args = recipes.options(plan["argv"])
        self.assertEqual(json.loads(args["--head_config"])["readout"], "last_span")
        self.assertEqual(args["--head_config"], recipe["trainer_args"]["head_config"])
        self.assertEqual(args["--base"], self.site["base_model"])
        self.assertEqual(args["--data"], self.site["data_root"] + "/custom.train.jsonl")
        self.assertEqual(args["--out"], plan["output"])
        self.assertTrue(plan["scientific_overrides"])

    def test_public_lowrate_recipe_selects_its_distinct_native_entrypoint(self):
        path = root() / "configs/studies/matched_scope_noffn/LOWRATE_NOFFN_F2s44.json"
        recipe = read(path)
        plan = recipes.custom_plan(path, self.site)
        self.assertEqual(plan["argv"][:4], [sys.executable, "-P", "-m", "kev.train_lowrate_noffn"])
        arguments = recipes.options(plan["argv"])
        for name, value in recipe["trainer_args"].items():
            if "{" not in str(value):
                self.assertEqual(arguments["--" + name], str(value))
        self.assertEqual(arguments["--initial_adapter_sha256"],
                         recipe["trainer_args"]["initial_adapter_sha256"])

    def test_all_public_paper_recipes_bind_locations_and_preserve_their_native_module(self):
        paths = sorted((root() / "configs/studies").rglob("*.json"))
        self.assertEqual(len(paths), 84)
        for path in paths:
            recipe = read(path)
            with self.subTest(recipe=path.relative_to(root()).as_posix()):
                plan = recipes.custom_plan(path, self.site)
                self.assertEqual(plan["argv"][3], recipe.get("trainer_module", "kev.train"))
                arguments = recipes.options(plan["argv"])
                self.assertEqual(arguments["--base"], self.site["base_model"])
                self.assertEqual(arguments["--out"], plan["output"])
                for value in [*arguments.values(), *plan["input_pins"]]:
                    if isinstance(value, str):
                        self.assertNotIn("{base_model}", value)
                        self.assertNotIn("{data_root}", value)
                        self.assertNotIn("{output}", value)
                for name, original in recipe["trainer_args"].items():
                    flag = "--" + name.removeprefix("--")
                    if original is False:
                        self.assertNotIn(flag, arguments)
                    elif isinstance(original, bool):
                        self.assertIs(arguments[flag], True)
                    elif not any(token in str(original) for token in ("{base_model}", "{data_root}", "{output}")):
                        self.assertEqual(arguments[flag], str(original))

    def test_active_custom_reference_paths_bind_tokens_and_explanatory_notes_need_no_run_path(self):
        recipe = read(root() / "configs/custom.example.json")
        recipe["references"] = [dict(run_id="prerequisite", path="{data_root}/reference/output")]
        path = self.folder / "recipe.json"
        path.write_text(json.dumps(recipe), encoding="utf-8")
        plan = recipes.custom_plan(path, self.site)
        self.assertEqual(plan["references"][0]["path"], self.site["data_root"] + "/reference/output")
        recipe["references"] = [dict(run_id="notes-only", scope="historical initialization description")]
        path.write_text(json.dumps(recipe), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reference_notes"):
            recipes.custom_plan(path, self.site)

    def test_forked_editable_tree_is_selected_by_relative_custom_source_root(self):
        fork = self.folder / "editable/source"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["fork", "--tree", "e11", "--output", str(fork)]), 0)
        origin = read(fork / "JEV_ORIGIN.json")
        self.assertEqual(origin["tree"], "e11")
        self.assertEqual(origin["files"]["kev/train.py"], sha(root() / "vendor/e11/kev/train.py"))
        with (fork / "kev/train.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# local community extension fixture\n")
        recipe = read(root() / "configs/custom.example.json")
        recipe.pop("tree")
        recipe["source_root"] = "editable/source"
        path = self.folder / "custom.json"
        path.write_text(json.dumps(recipe), encoding="utf-8")
        args = SimpleNamespace(site=self.site_file, recipe=path)
        plan = recipes.build(args)
        self.assertTrue(plan["custom_source"])
        self.assertEqual(Path(plan["source_root"]), fork.resolve())
        self.assertNotEqual(sha(fork / "kev/train.py"), origin["files"]["kev/train.py"])

    def test_failed_subprocess_keeps_native_exit_and_output_evidence(self):
        logs = self.folder / "native-failure"
        argv = [sys.executable, "-c", "import sys; print('fixture stdout'); print('fixture stderr', file=sys.stderr); sys.exit(7)"]
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "native exit 7"):
            workflow.execute(argv, dict(os.environ), logs, "fixture")
        command = read(logs / "fixture.command.json")
        result = read(logs / "fixture.result.json")
        self.assertEqual(command["argv"], argv)
        self.assertEqual(result["exit_code"], 7)
        self.assertGreaterEqual(result["elapsed_seconds"], 0)
        self.assertIn("fixture stdout", (logs / "fixture.stdout.log").read_text())
        self.assertIn("fixture stderr", (logs / "fixture.stderr.log").read_text())

    def test_missing_pinned_inputs_prevent_gpu_query_and_subprocess_launch(self):
        missing = self.folder / "missing-data.jsonl"
        output = self.folder / "never-created/run/output"
        plan = dict(custom_source=True, tree="custom", scientific_overrides=True,
            source_root=str(self.folder / "source"), site=self.site, output=str(output),
            argv=recipes.command(sys.executable, {"--base": str(self.folder / "missing-base"),
                "--data": str(missing), "--out": str(output), "--device": "cuda"}),
            input_pins={str(missing): "0" * 64}, references=[], expected={})
        with patch.object(workflow, "gpu_check") as gpu, patch.object(workflow, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "missing input"):
                workflow.train(plan, "0")
            gpu.assert_not_called()
            execute.assert_not_called()
        self.assertFalse(output.parent.exists())

    def test_changed_pinned_input_bytes_prevent_gpu_query_and_subprocess_launch(self):
        data = self.folder / "changed-data.jsonl"
        data.write_text('{"fixture":"changed"}\n', encoding="utf-8")
        base = self.folder / "base"
        base.mkdir()
        output = self.folder / "never-created/run/output"
        plan = dict(custom_source=True, tree="custom", scientific_overrides=True,
            source_root=str(self.folder / "source"), site=self.site, output=str(output),
            argv=recipes.command(sys.executable, {"--base": str(base), "--data": str(data),
                "--out": str(output), "--device": "cuda"}),
            input_pins={str(data): "0" * 64}, references=[], expected={})
        with patch.object(workflow, "gpu_check") as gpu, patch.object(workflow, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "input bytes differ"):
                workflow.train(plan, "0")
            gpu.assert_not_called()
            execute.assert_not_called()
        self.assertFalse(output.parent.exists())

    def test_site_keeps_lexical_venv_interpreter_path(self):
        executable = self.folder / "venv/bin/python"
        executable.parent.mkdir(parents=True)
        try:
            executable.symlink_to(Path(sys.executable).resolve())
        except OSError as error:
            self.skipTest("filesystem cannot create an interpreter symlink: " + str(error))
        site = copy.deepcopy(self.site)
        site["python"]["training"] = "venv/bin/python"
        self.site_file.write_text(json.dumps(site), encoding="utf-8")
        resolved = load_site(self.site_file)
        self.assertEqual(resolved["python"]["training"], str(executable.absolute()),
            "resolving a venv executable symlink changes sys.prefix and selects the wrong role environment")


if __name__ == "__main__":
    unittest.main()
