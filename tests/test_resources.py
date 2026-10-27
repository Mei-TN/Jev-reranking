"""Test resource extraction, caching, and integrity checks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev import resources


def bundle(folder, members=None):
    folder.mkdir(parents=True)
    metadata = folder / "configs/site.example.json"
    metadata.parent.mkdir()
    metadata.write_text('{"portable":true}\n', encoding="utf-8")
    (folder / "LICENSE").write_text("fixture license\n", encoding="utf-8")
    members = members or {"vendor/e11/kev/train.py": b"# exact source fixture\n",
                          "vendor/e11/runtime/__init__.py": b""}
    index = {}
    with zipfile.ZipFile(folder / "vendor.zip", "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
            index[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "mode": 0o644}
    (folder / "vendor_index.json").write_text(json.dumps({"schema": resources.SCHEMA, "files": index}), encoding="utf-8")
    return folder


class ResourceCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=resources._long_path(Path(tempfile.gettempdir())))
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.environment = patch.dict(os.environ, {"JEV_RESOURCE_CACHE": str(self.folder / "cache")})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_materializes_independent_bundle_and_reuses_unchanged_cache(self):
        installed = bundle(self.folder / "installed-package/_resources")
        first = resources.resource_root(installed)
        self.assertEqual((first / "vendor/e11/kev/train.py").read_bytes(), b"# exact source fixture\n")
        self.assertEqual((first / "configs/site.example.json").read_bytes(), (installed / "configs/site.example.json").read_bytes())
        stamp = (first / resources.RECEIPT).stat().st_mtime_ns
        self.assertEqual(resources.resource_root(installed), first)
        self.assertEqual((first / resources.RECEIPT).stat().st_mtime_ns, stamp)
        self.assertNotEqual(first, installed)

    def test_metadata_license_change_gets_a_new_content_identity(self):
        installed = bundle(self.folder / "installed")
        first = resources.resource_root(installed)
        (installed / "LICENSE").write_text("changed fixture license\n", encoding="utf-8")
        second = resources.resource_root(installed)
        self.assertNotEqual(first, second)
        self.assertEqual((first / "LICENSE").read_text(), "fixture license\n")

    def test_deep_archive_members_work_beyond_windows_max_path(self):
        name = "vendor/" + "/".join(["historical_source_component"] * 12) + "/train.py"
        installed = bundle(self.folder / "deep-installed", {name: b"# long path source\n"})
        target = resources.resource_root(installed)
        member = target.joinpath(*name.split("/"))
        self.assertGreater(len(str(member)), 260)
        self.assertEqual(member.read_bytes(), b"# long path source\n")

    def test_tampered_or_extra_cache_file_is_rejected_without_repair(self):
        installed = bundle(self.folder / "installed")
        target = resources.resource_root(installed)
        source = target / "vendor/e11/kev/train.py"
        original = source.read_bytes()
        tampered = original + b"# tampered\n"
        self.assertNotEqual(tampered, original)
        self.assertNotEqual(len(tampered), len(original))
        source.write_bytes(tampered)
        with self.assertRaisesRegex(ValueError, "differs"):
            resources.resource_root(installed)
        self.assertEqual(source.read_bytes(), tampered)

    def test_archive_traversal_and_symlink_are_rejected_before_publication(self):
        for number, member in enumerate(("../escaped.py", "vendor/../../escaped.py", "/vendor/absolute.py", "vendor\\bad.py")):
            installed = bundle(self.folder / ("unsafe" + str(number)), {member: b"never extracted"})
            with self.subTest(member=member), self.assertRaisesRegex(ValueError, "unsafe"):
                resources.resource_root(installed)
        installed = bundle(self.folder / "symlink")
        name, data = "vendor/link", b"outside"
        info = zipfile.ZipInfo(name)
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(installed / "vendor.zip", "w") as archive:
            archive.writestr(info, data)
        (installed / "vendor_index.json").write_text(json.dumps({"schema": resources.SCHEMA, "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "regular files"):
            resources.resource_root(installed)
        self.assertFalse((self.folder / "escaped.py").exists())

    def test_concurrent_materializers_return_one_complete_cache(self):
        installed = bundle(self.folder / "installed")
        with ThreadPoolExecutor(max_workers=4) as pool:
            roots = list(pool.map(lambda _: resources.resource_root(installed), range(4)))
        self.assertEqual(len(set(roots)), 1)
        self.assertEqual((roots[0] / "vendor/e11/kev/train.py").read_bytes(), b"# exact source fixture\n")
        self.assertEqual(len(list((self.folder / "cache").iterdir())), 1)

    def test_source_layout_is_returned_without_cache_creation(self):
        source = self.folder / "source"
        (source / "vendor").mkdir(parents=True)
        self.assertEqual(resources.resource_root(source), resources._long_path(source))
        self.assertFalse((self.folder / "cache").exists())

    def test_compressed_source_layout_materializes_only_runtime_metadata(self):
        source = bundle(self.folder / "source-sdist")
        for relative in ("src/jev/cli.py", "tests/test_example.py", "work/temporary.txt", "dist/build.txt"):
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("source/build material stays outside the resource cache\n")
        target = resources.resource_root(source, source_layout=True)
        self.assertEqual((target / "LICENSE").read_text(), "fixture license\n")
        self.assertTrue((target / "configs/site.example.json").is_file())
        self.assertTrue((target / "vendor/e11/kev/train.py").is_file())
        self.assertFalse(any((target / name).exists() for name in ("src", "tests", "work", "dist")))
        first = target
        (source / "work/temporary.txt").write_text("unrelated build output changed\n")
        self.assertEqual(resources.resource_root(source, source_layout=True), first)

    def test_archive_validation_checks_bytes_without_extracting(self):
        source = bundle(self.folder / "source")
        index = resources.verify_vendor_payload(source / "vendor.zip", source / "vendor_index.json")
        self.assertEqual(len(index["files"]), 2)
        self.assertFalse((source / "vendor").exists())
        with zipfile.ZipFile(source / "vendor.zip", "w") as archive:
            archive.writestr("vendor/e11/kev/train.py", b"# wrong source bytes!\n")
            archive.writestr("vendor/e11/runtime/__init__.py", b"")
        with self.assertRaisesRegex(ValueError, "differs"):
            resources.verify_vendor_payload(source / "vendor.zip", source / "vendor_index.json")

    def test_ambiguous_source_layout_and_nonobject_index_are_rejected(self):
        source = bundle(self.folder / "source")
        (source / "vendor").mkdir()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resources.resource_root(source, source_layout=True)
        (source / "vendor_index.json").write_text("[]\n")
        with self.assertRaisesRegex(ValueError, "index"):
            resources.verify_vendor_payload(source / "vendor.zip", source / "vendor_index.json")


if __name__ == "__main__":
    unittest.main()
