"""The public inventory is self-contained and rejects altered or extra source files."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jev import config, workflow


class PublicSourceInventoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.put("vendor/paper/kev/train.py", b"# native paper trainer\n")
        self.protocol = self.put("provenance/plans/fixed.json", b'{"epochs":2}\n')
        self.requirement = self.put("requirements/training-lock.txt", b"# python 3.12.13\ntorch==2.8.0+cu128\n")
        self.put("vendor/SOURCES.json", json.dumps(dict(snapshots={"paper": dict(
            import_root="vendor/paper", files={"kev/train.py": self.identity(self.source)})},
            provenance_files={})).encode())
        self.put("provenance/PROTOCOLS.json", json.dumps(dict(files=[dict(
            path="provenance/plans/fixed.json", **self.identity(self.protocol))])).encode())
        self.put("requirements/manifest.json", json.dumps(dict(files={
            "training-lock.txt": self.identity(self.requirement)})).encode())

    def put(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    @staticmethod
    def identity(path):
        data = path.read_bytes()
        return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def test_public_source_protocol_and_environment_need_no_author_workspace(self):
        with patch.object(workflow, "root", return_value=self.root):
            self.assertEqual(workflow.verify_sources(), 3)

    def test_extra_source_and_changed_paper_protocol_are_rejected(self):
        extra = self.put("vendor/paper/internal_controller.py", b"# not selected for release\n")
        with patch.object(workflow, "root", return_value=self.root):
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                workflow.verify_sources()
            extra.unlink()
            self.protocol.write_bytes(b'{"epochs":1}\n')
            with self.assertRaisesRegex(ValueError, "fixed.json"):
                workflow.verify_sources()

    def test_changed_environment_lock_is_rejected(self):
        self.requirement.write_bytes(b"# altered dependency\n")
        with patch.object(workflow, "root", return_value=self.root):
            with self.assertRaisesRegex(ValueError, "requirements/training-lock.txt"):
                workflow.verify_sources()

    def test_registered_nested_training_root_is_selected_without_directory_aliases(self):
        nested = self.put("vendor/portable_pair/src/kev/train.py", b"# nested native trainer\n")
        manifest = json.loads((self.root / "vendor/SOURCES.json").read_text())
        manifest["snapshots"]["portable_pair"] = dict(import_root="vendor/portable_pair/src",
            files={"kev/train.py": self.identity(nested)})
        (self.root / "vendor/SOURCES.json").write_text(json.dumps(manifest))
        with patch.object(config, "root", return_value=self.root):
            self.assertEqual(config.tree_path("portable_pair"), nested.parent.parent)
            with self.assertRaisesRegex(ValueError, "unknown training tree"):
                config.tree_path("unregistered")
        for unsafe in ("../vendor/paper", "/vendor/paper", "vendor/paper/../paper", "C:/vendor/paper"):
            manifest["snapshots"]["paper"]["import_root"] = unsafe
            (self.root / "vendor/SOURCES.json").write_text(json.dumps(manifest))
            with self.subTest(import_root=unsafe), patch.object(config, "root", return_value=self.root):
                with self.assertRaisesRegex(ValueError, "relative vendor path"):
                    config.tree_path("paper")

    def test_nested_source_and_surrounding_resource_snapshots_each_keep_exact_inventory(self):
        nested = self.put("vendor/portable_pair/src/kev/train.py", b"# nested native trainer\n")
        notice = self.put("vendor/portable_pair/NOTICE", b"Preserved source notice\n")
        manifest = json.loads((self.root / "vendor/SOURCES.json").read_text())
        manifest["snapshots"]["portable_pair"] = dict(import_root="vendor/portable_pair/src",
            files={"kev/train.py": self.identity(nested)})
        manifest["snapshots"]["portable_resources"] = dict(import_root="vendor/portable_pair",
            files={"NOTICE": self.identity(notice)})
        (self.root / "vendor/SOURCES.json").write_text(json.dumps(manifest))
        with patch.object(workflow, "root", return_value=self.root):
            self.assertEqual(workflow.verify_sources(), 5)
            extra = self.put("vendor/portable_pair/src/kev/unlisted.py", b"# undeclared\n")
            with self.assertRaisesRegex(ValueError, "portable_pair: file inventory changed"):
                workflow.verify_sources()
            extra.unlink()
            self.put("vendor/portable_pair/unlisted.json", b"{}\n")
            with self.assertRaisesRegex(ValueError, "portable_resources: file inventory changed"):
                workflow.verify_sources()


if __name__ == "__main__":
    unittest.main()
