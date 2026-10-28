"""Check zero-effect sign counts against published per-model aggregate scores.

These tests reconstruct summary arithmetic; they do not rescore raw predictions.
"""
import csv
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCORING = ROOT / "vendor/stored_scoring"
VIEWS = SCORING / "summary_views"
SPEC = importlib.util.spec_from_file_location(
    "appendix_verifier", SCORING / "scorers/verify_appendix_from_reconstructed.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
SETTINGS = ["F2s44", "F2s45", "F3s44", "F3s45"]
CASES = [
    ("E1_head_main_effect", "source_chess", "F2s45", 500, (2, 1, 1)),
    ("E1_interaction", "source_GPQA", "F2s45", 198, (3, 1, 0)),
    ("last_content_minus_end", "source_GPQA", "F3s45", 198, (1, 1, 2)),
    ("mean_content_minus_last_content", "source_GPQA", "F2s45", 198, (2, 1, 1)),
]


def rows(name):
    with (VIEWS / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class AppendixSignCounts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = json.loads((SCORING / "expected_aggregates.json").read_text())["core56"]
        registry = json.loads((SCORING / "model_registry.json").read_text())
        cls.by_arm = {}
        core_registry = []
        for model in registry:
            occurrence = model["package_occurrences"][0]
            if occurrence["module"] != "core56":
                continue
            setting = f"{model['fold']}s{model['seed']}"
            arm = model["model_id"].split("_", 2)[2]
            cls.by_arm[setting, arm] = occurrence["id"]
            core_registry.append(dict(scientific_model_id=model["model_id"],
                fold=model["fold"], seed=model["seed"], id=occurrence["id"]))
        for setting in SETTINGS:
            cls.by_arm[setting, "POINTER_1EP"] = cls.by_arm[setting, "LL_A"]
            cls.by_arm[setting, "POOL_1EP"] = cls.by_arm[setting, "POOL_ALL"]
        cls.experiments = {"_model_registry": core_registry}
        cls.coefficients = {}
        for row in rows("appendix_contrast_coefficients.csv"):
            block = cls.experiments.setdefault(row["block"], {"contrast_coefficients": {}})
            coefficients = block["contrast_coefficients"].setdefault(row["contrast_id"], {})
            coefficients[row["arm"]] = float(row["coefficient"])
            cls.coefficients[row["contrast_id"]] = coefficients
        for block in VERIFIER.BLOCKS:
            coefficients = cls.experiments[block]["contrast_coefficients"]
            arms = {arm for values in coefficients.values() for arm in values}
            cls.experiments[block]["settings"] = [dict(
                fold=setting[:2], seed=int(setting[3:]),
                models={arm: cls.by_arm[setting, arm] for arm in arms}) for setting in SETTINGS]
        cls.contrasts = rows("appendix_paired_contrasts.csv")

    def test_four_affected_effects_are_exact_zero_from_integer_correct_counts(self):
        for contrast, source, setting, denominator, _ in CASES:
            tasks = ["chess"] if source == "source_chess" else ["gpqa", "gpqa_fewshot"]
            exact = Fraction(0)
            for arm, coefficient in self.coefficients[contrast].items():
                scores = self.core["results"][self.by_arm[setting, arm]]["tasks_percent"]
                for task in tasks:
                    count = scores[task] * denominator / 100
                    self.assertAlmostEqual(count, round(count), places=10)
                    exact += Fraction(str(coefficient)) * round(count)
            with self.subTest(contrast=contrast, setting=setting):
                self.assertEqual(exact, 0)

    def test_public_csv_has_all_four_corrected_counts(self):
        for contrast, source, _, _, expected in CASES:
            row = next(row for row in self.contrasts
                       if row["contrast_id"] == contrast and row["view"] == source)
            with self.subTest(contrast=contrast):
                self.assertEqual(tuple(int(row[key]) for key in ("positive", "zero", "negative")), expected)
                self.assertEqual(VERIFIER.sign_counts([float(row[s]) for s in SETTINGS]),
                                 dict(zip(("positive", "zero", "negative"), expected)))

    def test_tolerance_preserves_small_real_effects_and_main_four_setting_gains(self):
        epsilon = VERIFIER.SIGN_ZERO_TOLERANCE_PP
        self.assertEqual(VERIFIER.sign_counts([-1e-8, -epsilon, 0, epsilon, 1e-8]),
                         dict(positive=1, zero=3, negative=1))
        real = [float(row[setting]) for row in self.contrasts for setting in SETTINGS
                if abs(float(row[setting])) > epsilon]
        self.assertGreater(min(map(abs, real)), 0.0005)
        self.assertEqual(VERIFIER.sign_counts(real),
                         dict(positive=sum(x > 0 for x in real), zero=0, negative=sum(x < 0 for x in real)))
        for contrast in ("pointer_noffn_minus_all", "pooling_minus_pointer_noffn"):
            row = next(row for row in self.contrasts
                       if row["contrast_id"] == contrast and row["view"] == "MC11")
            self.assertGreater(min(float(row[s]) for s in SETTINGS), 0.33)
            self.assertEqual(VERIFIER.sign_counts([float(row[s]) for s in SETTINGS]),
                             dict(positive=4, zero=0, negative=0))

    def test_full_public_appendix_reconstructs_from_per_model_scores(self):
        report = VERIFIER.main(self.core, self.experiments, VIEWS)
        self.assertEqual(report["status"], "PASS", report["failures"])
        self.assertEqual(report["sign_count_zero_tolerance_pp"], 1e-9)
        self.assertEqual(report["cohort_current_rows_verified"], 16)
        self.assertEqual(report["file_rows"]["appendix_paired_contrasts.csv"], 621)
        self.assertEqual(report["file_rows"]["appendix_loso_complete.csv"], 322)

    def test_verifier_rejects_historical_float_sign_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            for source in VIEWS.glob("*.csv"):
                shutil.copyfile(source, destination / source.name)
            target = destination / "appendix_paired_contrasts.csv"
            with target.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                fields, records = reader.fieldnames, list(reader)
            affected = next(row for row in records if row["contrast_id"] == "E1_head_main_effect"
                            and row["view"] == "source_chess")
            affected.update(zero="0", negative="2")
            with target.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
            report = VERIFIER.main(self.core, self.experiments, destination)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual({failure["check"] for failure in report["failures"]},
                             {"E123/E1_head_main_effect/source_chess/zero",
                              "E123/E1_head_main_effect/source_chess/negative"})


if __name__ == "__main__":
    unittest.main()
