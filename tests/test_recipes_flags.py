"""Recipe switches must parse through the preserved trainer's own CLI definitions."""
import argparse
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev.config import root
from jev.recipes import command, custom_plan, options


def native_parser():
    """Load only argparse statements from the actual trainer, without importing Torch."""
    source = root() / "vendor/e11/kev"
    dse = ast.parse((source / "dse_options.py").read_text(encoding="utf-8"))
    add_arguments = next(node for node in dse.body if isinstance(node, ast.FunctionDef) and node.name == "add_arguments")
    trainer = ast.parse((source / "train.py").read_text(encoding="utf-8"))
    main = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    body = [add_arguments]
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ap" for target in node.targets):
            body.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "ap" and call.func.attr == "add_argument"):
                body.append(node)
            elif isinstance(call.func, ast.Name) and call.func.id == "add_arguments":
                body.append(node)
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source / "train.py"), "exec"), namespace)
    return namespace["ap"]


class RecipeSwitchTests(unittest.TestCase):
    def test_true_switches_are_valueless_false_switches_absent_and_numeric_flags_keep_values(self):
        argv = command("python", {"--full_finetune": True, "--head_only": False,
                                  "--smoothing_score": True, "--checkpointing": 0,
                                  "--special_embeddings": 1, "--holdout": "", "--lr": -1.0})
        parsed = native_parser().parse_args(argv[4:])
        self.assertTrue(parsed.full_finetune)
        self.assertFalse(parsed.head_only)
        self.assertTrue(parsed.smoothing_score)
        self.assertEqual((parsed.checkpointing, parsed.special_embeddings), (0, 1))
        self.assertEqual(parsed.holdout, "")
        self.assertEqual(parsed.lr, -1.0)
        self.assertEqual(options(argv), {"--full_finetune": True, "--smoothing_score": True,
                                         "--checkpointing": "0", "--special_embeddings": "1",
                                         "--holdout": "", "--lr": "-1.0"})

    def test_single_trailing_switch_and_adjacent_switches_roundtrip(self):
        for values in ({"--head_only": True}, {"--head_only": True, "--smoothing_score": True}):
            self.assertEqual(options(command("python", values)), values)

    def test_duplicate_or_unflagged_arguments_are_rejected(self):
        prefix = ["python", "-m", "kev.train"]
        for values in (["--head_only", "--head_only"], ["--lr", "0.1", "--lr", "0.2"],
                       ["0.1"], ["--"], ["--lr=0.1"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                options(prefix + values)

    def test_every_paper_command_keeps_its_recorded_scientific_arguments(self):
        for study in ("e1e2", "e458", "e11"):
            plan = json.loads((root() / "provenance/plans" / (study + ".json")).read_text(encoding="utf-8"))
            for run in plan["runs"]:
                with self.subTest(run=run["run_id"]):
                    rebuilt = command(run["argv"][0], options(run["argv"]))
                    self.assertEqual(rebuilt, run["argv"])

    def test_custom_json_booleans_survive_location_binding_and_source_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            location = Path(directory)
            (location / "extension/kev").mkdir(parents=True)
            (location / "extension/kev/train.py").write_text("# external extension fixture\n", encoding="utf-8")
            recipe = dict(schema="jev.recipe.v1", id="switch-test", source_root="extension",
                          trainer_args={"base": "{base_model}", "data": "{data_root}/train.jsonl",
                                        "out": "{output}", "head_only": True,
                                        "smoothing_score": False, "checkpointing": 1})
            path = location / "recipe.json"
            path.write_text(json.dumps(recipe), encoding="utf-8")
            site = dict(work_root=str(location / "work"), base_model=str(location / "base"),
                        data_root=str(location / "data"), python={"training": "python"})
            plan = custom_plan(path, site)
            parsed = native_parser().parse_args(plan["argv"][4:])
            self.assertTrue(parsed.head_only)
            self.assertFalse(parsed.smoothing_score)
            self.assertEqual(parsed.checkpointing, 1)
            self.assertEqual(parsed.base, site["base_model"])
            self.assertTrue(plan["custom_source"])
            self.assertEqual(plan["source_root"], str((location / "extension").resolve()))

    def test_explicit_native_module_is_selected_and_missing_module_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            location = Path(directory)
            package = location / "extension/kev"
            package.mkdir(parents=True)
            (package / "train.py").write_text("# common native trainer fixture\n")
            (package / "train_lowrate_noffn.py").write_text("# distinct native entrypoint fixture\n")
            recipe = dict(schema="jev.recipe.v1", id="native-lowrate", source_root="extension",
                trainer_module="kev.train_lowrate_noffn", trainer_args={"base": "{base_model}",
                    "data": "{data_root}/train.jsonl", "out": "{output}", "lr": "1e-5"})
            path = location / "recipe.json"
            site = dict(work_root=str(location / "work"), base_model=str(location / "base"),
                        data_root=str(location / "data"), python={"training": "python"})
            path.write_text(json.dumps(recipe))
            plan = custom_plan(path, site)
            self.assertEqual(plan["argv"][3], "kev.train_lowrate_noffn")
            self.assertEqual(options(plan["argv"])["--lr"], "1e-5")
            (package / "train_lowrate_noffn.py").unlink()
            with self.assertRaisesRegex(ValueError, "lacks trainer_module"):
                custom_plan(path, site)
            recipe["trainer_module"] = "other_package.train"
            path.write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError, "kev"):
                custom_plan(path, site)


if __name__ == "__main__":
    unittest.main()
