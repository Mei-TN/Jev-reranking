"""Test scoring-package integrity, extraction, and failure handling."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from jev import scoring
from jev.config import read, sha


class StoredScoringContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)

    def archive(self, members, name="fixture.zip"):
        archive = self.folder / name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(archive, "w") as package:
                for member, value in members:
                    package.writestr(member, value)
        return archive

    def fixture_pin(self, archive):
        return dict(bytes=archive.stat().st_size, sha256=sha(archive), origin="unit-test transport fixture")

    def test_original_pin_identifies_the_external_probability_archive(self):
        pin = scoring._archive_pin()
        self.assertEqual(pin["sha256"], "3125ccce2f8127efcf956bf2025367081cb7f0ab3b97e3a33e03826e357d2d45")
        self.assertEqual(pin["bytes"], 200841481)

    def test_checksum_mismatch_stops_before_extraction_or_native_execution(self):
        archive = self.archive([("score_all.py", "raise AssertionError('must not execute')")])
        output = self.folder / "not-created"
        pin = self.fixture_pin(archive)
        pin["sha256"] = "0" * 64
        with patch.object(scoring, "_archive_pin", return_value=pin), patch.object(scoring, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                scoring.reproduce_scores(archive, output)
            execute.assert_not_called()
        self.assertFalse(output.exists())

    def test_unsafe_paths_symlinks_duplicates_and_file_parent_are_refused_before_extract(self):
        symlink = zipfile.ZipInfo("link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        cases = [
            [("../escape", "bad")], [("/absolute", "bad")], [("C:/drive", "bad")],
            [("folder\\..\\escape", "bad")], [("safe/../escape", "bad")],
            [("safe/./alias", "bad")], [("file:stream", "bad")], [("NUL.txt", "bad")],
            [("trailing. /file", "bad")], [(symlink, "../escape")],
            [("same", "first"), ("same", "second")], [("SAME", "first"), ("same", "second")],
            [("parent", "file"), ("parent/child", "bad")],
        ]
        for index, members in enumerate(cases):
            with self.subTest(members=members):
                archive = self.archive(members, f"unsafe-{index}.zip")
                output = self.folder / f"output-{index}"
                with self.assertRaises(ValueError):
                    scoring._safe_extract(archive, output)
                self.assertFalse(output.exists())

    def test_safe_members_are_streamed_without_overwriting_an_existing_output(self):
        archive = self.archive([("top.txt", "top"), ("nested/", ""), ("nested/bytes.bin", b"\x00\xff\x80")])
        output = self.folder / "extracted"
        result = scoring._safe_extract(archive, output)
        self.assertEqual(result, dict(entries=3, files=2, uncompressed_bytes=6))
        self.assertEqual((output / "nested/bytes.bin").read_bytes(), b"\x00\xff\x80")
        with self.assertRaisesRegex(ValueError, "already exists"):
            scoring._safe_extract(archive, output)

    def test_real_native_failure_records_unchanged_exit_and_utf8_logs(self):
        script = "import sys\nprint('UTF8: 复现失败')\nprint('native fixture failure', file=sys.stderr)\nsys.exit(7)\n"
        archive = self.archive([("score_all.py", script)])
        output = self.folder / "failed-run"
        with patch.object(scoring, "_archive_pin", return_value=self.fixture_pin(archive)), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "native exit 7"):
                scoring.reproduce_scores(archive, output)
        native = read(output / "score_all.result.json")
        result = read(output / "REPRODUCTION_RESULT.json")
        self.assertEqual(native["exit_code"], 7)
        self.assertEqual(result["native_exit"], 7)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(native["argv"][:4], [sys.executable, "-X", "utf8", "-P"])
        self.assertIn("复现失败", (output / "score_all.stdout.log").read_text(encoding="utf-8"))
        self.assertIn("native fixture failure", (output / "score_all.stderr.log").read_text(encoding="utf-8"))
        self.assertEqual((output / "score_all.py").read_text(encoding="utf-8"), script)

    def test_zero_exit_without_original_complete_report_is_not_accepted(self):
        archive = self.archive([("score_all.py", "print('does not verify scores')\n")])
        output = self.folder / "no-report"
        with patch.object(scoring, "_archive_pin", return_value=self.fixture_pin(archive)), redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                scoring.reproduce_scores(archive, output)
        result = read(output / "REPRODUCTION_RESULT.json")
        self.assertEqual((result["status"], result["native_exit"]), ("FAILED", 0))

    def test_report_acceptance_requires_original_counts_and_numerical_checks(self):
        original = dict(status="PASS", fresh_directory_verification=True,
            modules=["core56", "qwen8", "auxiliary"], boundaries={"status": "PASS"},
            maximum_aggregate_difference_pp=0.0, maximum_appendix_difference_pp=0.0,
            **scoring.EXPECTED_COUNTS)
        self.assertEqual(scoring._check_report(original), original)
        for key, value in (("independent_checkpoint_models", 71), ("integer_differences", 1),
                           ("maximum_aggregate_difference_pp", float("nan")), ("fresh_directory_verification", False)):
            invalid = dict(original, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                scoring._check_report(invalid)


if __name__ == "__main__":
    unittest.main()
