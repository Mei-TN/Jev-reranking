"""Compressed source releases must expose preserved source to the same audit."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("jev_release_audit", ROOT / "scripts/audit_release.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

# Explicit inert path used only by local scanner fixtures.
LOCATION_FIXTURE = "/" + "data" + "/example/checkpoint"


class CompressedSourceAuditTests(unittest.TestCase):
    def reviewed_fixture(self, root, *, kind="literal_credential_assignment", provider=False):
        value = ("hf_" if provider else "v") + "Ab3" * 10
        content = "password" + " = " + repr(value) + "\n"
        source = root / "fixture.py"
        source.write_text(content, encoding="utf-8")
        (root / "provenance").mkdir()
        row = dict(path="fixture.py", line=1, kind=kind,
                   file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   reason="Explicit test harness sentinel reviewed in its local mock context.")
        (root / "provenance/audit-classifications.json").write_text(
            json.dumps(dict(schema="jev.reviewed-audit-findings.v1", findings=[row])), encoding="utf-8")
        return source, value

    def test_reviewed_assignment_remains_visible_and_requires_exact_file_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, value = self.reviewed_fixture(root)
            report = AUDIT.audit(root)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["counts"]["raw_credential_pattern_matches"], 1)
            self.assertEqual(report["counts"]["reviewed_pattern_matches"], 1)
            self.assertEqual(report["secret_findings"], [])
            self.assertEqual(report["reviewed_findings"][0]["path"], "fixture.py")
            self.assertNotIn(value, json.dumps(report))
            source.write_text(source.read_text() + "# changed published bytes\n", encoding="utf-8")
            changed = AUDIT.audit(root)
            self.assertEqual(changed["status"], "SCAN_ERROR")
            self.assertEqual(changed["counts"]["reviewed_pattern_matches"], 0)
            self.assertTrue(changed["secret_findings"])

    def test_generic_review_cannot_hide_a_provider_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, value = self.reviewed_fixture(root, provider=True)
            report = AUDIT.audit(root)
            self.assertEqual(report["status"], "LITERAL_SECRETS_FOUND")
            self.assertEqual(report["counts"]["reviewed_pattern_matches"], 0)
            self.assertEqual(report["secret_findings"][0]["kind"], "huggingface_token")
            self.assertNotIn(value, json.dumps(report))

    def test_provider_token_kind_cannot_be_added_to_review_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.reviewed_fixture(root, kind="huggingface_token", provider=True)
            report = AUDIT.audit(root)
            self.assertEqual(report["status"], "SCAN_ERROR")
            self.assertEqual(report["counts"]["reviewed_pattern_matches"], 0)
            self.assertTrue(report["secret_findings"])

    def test_release_log_files_are_scanned_as_published_text(self):
        token = "hf_" + "Ab3" * 10
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            line = "token" + " = " + repr(token) + "\n"
            (root / "verification.log").write_text(line, encoding="utf-8")
            report = AUDIT.audit(root)
            self.assertEqual(report["status"], "LITERAL_SECRETS_FOUND")
            self.assertIn({"path": "verification.log", "line": 1, "kind": "huggingface_token"},
                          report["secret_findings"])
            self.assertNotIn(token, json.dumps(report))

    def make_bundle(self, root, members, *, wrong_hash=False):
        root.mkdir()
        (root / "README.md").write_text("Small release fixture.\n", encoding="utf-8")
        files = {}
        with zipfile.ZipFile(root / "vendor.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in members.items():
                archive.writestr(name, content)
                files[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        if wrong_hash:
            next(iter(files.values()))["sha256"] = "0" * 64
        (root / "vendor_index.json").write_text(
            json.dumps({"schema": "jev-vendor-payload-v1", "files": files}), encoding="utf-8")

    def test_compressed_vendor_credentials_and_locations_keep_relative_paths(self):
        token = "hf_" + "Ab3" * 10
        content = ("token = " + repr(token) + "\nlocation = " + repr(LOCATION_FIXTURE) + "\n").encode()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "release"
            self.make_bundle(root, {"vendor/sample/module.py": content})
            with patch.dict(os.environ, {"JEV_RESOURCE_CACHE": str(base / "cache")}):
                report = AUDIT.audit(root)
            self.assertEqual(report["status"], "LITERAL_SECRETS_FOUND")
            self.assertEqual(report["errors"], [])
            self.assertIn("+materialized-vendor-payload", report["selection"])
            self.assertIn({"path": "vendor/sample/module.py", "line": 1,
                           "kind": "huggingface_token"}, report["secret_findings"])
            self.assertIn({"path": "vendor/sample/module.py", "line": 2,
                           "kind": "private_posix_path", "area": "preserved_vendor"},
                          report["location_findings"])
            self.assertNotIn(token, json.dumps(report))
            self.assertGreaterEqual(report["scanned_bytes"], len(content))

    def test_compressed_and_uncompressed_vendor_have_identical_findings(self):
        content = ("location = " + repr(LOCATION_FIXTURE) + "\n").encode()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            compressed = base / "compressed"
            raw = base / "raw"
            self.make_bundle(compressed, {"vendor/sample/module.py": content})
            (raw / "vendor/sample").mkdir(parents=True)
            (raw / "vendor/sample/module.py").write_bytes(content)
            with patch.dict(os.environ, {"JEV_RESOURCE_CACHE": str(base / "cache")}):
                packed_report = AUDIT.audit(compressed)
                raw_report = AUDIT.audit(raw)
            self.assertEqual(packed_report["status"], raw_report["status"])
            self.assertEqual(packed_report["secret_findings"], raw_report["secret_findings"])
            self.assertEqual(packed_report["location_findings"], raw_report["location_findings"])

    def test_invalid_compressed_vendor_is_a_scan_error_not_a_binary_skip_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "release"
            self.make_bundle(root, {"vendor/sample/module.py": b"print('fixture')\n"}, wrong_hash=True)
            output = io.StringIO()
            with patch.dict(os.environ, {"JEV_RESOURCE_CACHE": str(base / "cache")}), redirect_stdout(output):
                native_exit = AUDIT.main(["--root", str(root), "--summary"])
            report = json.loads(output.getvalue())
            self.assertEqual(native_exit, 2)
            self.assertEqual(report["status"], "SCAN_ERROR")
            self.assertEqual(report["errors"], [{"path": "vendor.zip", "kind": "invalid-vendor-payload"}])

    def test_source_checkout_retains_its_original_file_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "vendor/sample").mkdir(parents=True)
            (root / "vendor/sample/module.py").write_text("print('fixture')\n", encoding="utf-8")
            with patch("jev.resources.resource_root", side_effect=AssertionError("unexpected materialization")):
                report = AUDIT.audit(root)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["selection"], "release-directory-fallback")
            self.assertEqual(report["scanned_text_files"], 1)

    def test_nested_release_does_not_inherit_an_enclosing_git_ignore_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            outer = Path(temporary)
            try:
                created = subprocess.run(["git", "init", "--quiet", str(outer)],
                                         capture_output=True, check=False)
            except OSError:
                self.skipTest("Git is unavailable")
            if created.returncode:
                self.skipTest("Git could not initialize the temporary fixture")
            (outer / ".gitignore").write_text("work/\n", encoding="utf-8")
            root = outer / "work/release"
            root.mkdir(parents=True)
            (root / "module.py").write_text("location = " + repr(LOCATION_FIXTURE) + "\n", encoding="utf-8")
            report = AUDIT.audit(root)
            self.assertEqual(report["selection"], "release-directory-fallback")
            self.assertEqual(report["scanned_text_files"], 1)
            self.assertEqual(report["location_findings"], [
                {"path": "module.py", "line": 1, "kind": "private_posix_path", "area": "maintained"}])


if __name__ == "__main__":
    unittest.main()
