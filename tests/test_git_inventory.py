"""A Git source release must contain the exact bytes named by its inventories.

Standalone source archives and installed distributions have no Git index, so
this release-closure check skips there. It reads Git metadata without staging,
refreshing, or modifying the caller's checkout.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]


def git(repository: Path, *arguments: str, input_bytes: bytes | None = None):
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repository), *arguments],
        input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment, check=False,
    )


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError("unsafe pinned inventory path: " + repr(value))
    return path.as_posix()


def inventory_pins(repository: Path) -> dict[str, dict]:
    """Include both inventories, all named payloads, and archive provenance."""
    pins = {}

    def add(name, metadata):
        name = relative_path(name)
        pin = {"sha256": metadata["sha256"], "bytes": metadata["bytes"]}
        if name in pins and pins[name] != pin:
            raise ValueError("conflicting inventory pins for " + name)
        pins[name] = pin

    def load(name):
        raw = (repository / name).read_bytes()
        add(name, {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        return json.loads(raw)

    sources = load("vendor/SOURCES.json")
    for snapshot in sources["snapshots"].values():
        root = relative_path(snapshot["import_root"])
        for name, metadata in snapshot["files"].items():
            add(root + "/" + relative_path(name), metadata)
        if "provenance_sha256" in snapshot:
            name = root + "/PROVENANCE.json"
            add(name, {"sha256": snapshot["provenance_sha256"],
                       "bytes": (repository / name).stat().st_size})
    for name, metadata in sources["provenance_files"].items():
        add(name, metadata)
    protocols = load("provenance/PROTOCOLS.json")
    for metadata in protocols["files"]:
        add(metadata["path"], metadata)
    requirements = load("requirements/manifest.json")
    for name, metadata in requirements["files"].items():
        add("requirements/" + relative_path(name), metadata)
    return pins


def indexed_files(repository: Path) -> dict[str, str]:
    result = git(repository, "ls-files", "--stage", "-z")
    if result.returncode:
        raise AssertionError("cannot read Git index: " + result.stderr.decode("utf-8", "replace"))
    files = {}
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        header, raw_name = record.split(b"\t", 1)
        mode, oid, stage = header.split()
        name = raw_name.decode("utf-8")
        if stage != b"0" or mode not in {b"100644", b"100755"}:
            continue
        files[name] = oid.decode("ascii")
    return files


def index_blob_pins(repository: Path, oids: set[str]) -> dict[str, dict]:
    """Read every distinct indexed blob in one invocation, without checkout."""
    ordered = sorted(oids)
    result = git(repository, "cat-file", "--batch",
                 input_bytes="".join(oid + "\n" for oid in ordered).encode("ascii"))
    if result.returncode:
        raise AssertionError("cannot read indexed blobs: " + result.stderr.decode("utf-8", "replace"))
    pins, offset = {}, 0
    for expected_oid in ordered:
        end = result.stdout.find(b"\n", offset)
        header = result.stdout[offset:end].split()
        if end < 0 or len(header) != 3 or header[0].decode("ascii") != expected_oid or header[1] != b"blob":
            raise AssertionError("invalid Git blob response for " + expected_oid)
        size = int(header[2])
        offset = end + 1
        blob = result.stdout[offset:offset + size]
        if len(blob) != size or result.stdout[offset + size:offset + size + 1] != b"\n":
            raise AssertionError("truncated Git blob response for " + expected_oid)
        pins[expected_oid] = {"bytes": size, "sha256": hashlib.sha256(blob).hexdigest()}
        offset += size + 1
    if offset != len(result.stdout):
        raise AssertionError("unexpected trailing Git blob output")
    return pins


class GitInventoryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("git") is None:
            raise unittest.SkipTest("Git is unavailable; source-release index check only")
        result = git(REPOSITORY, "rev-parse", "--show-toplevel")
        if result.returncode:
            raise unittest.SkipTest("not a Git checkout; source-release index check only")
        top = Path(result.stdout.decode("utf-8").strip()).resolve()
        if top != REPOSITORY.resolve():
            raise unittest.SkipTest("not the JEV Git checkout root")

    def test_all_pinned_sources_and_protocols_are_tracked_with_exact_bytes(self):
        pins = inventory_pins(REPOSITORY)
        indexed = indexed_files(REPOSITORY)
        missing = sorted(pins.keys() - indexed.keys())
        self.assertFalse(missing,
            "Pinned release files are absent from the Git index (including ignored files or unresolved entries): "
            + repr(missing[:20]) + (f"; {len(missing)} total" if missing else ""))

        changed = []
        for name, expected in pins.items():
            path = REPOSITORY / name
            if path.is_symlink() or not path.is_file():
                changed.append(name + " (not a regular working-tree file)")
                continue
            raw = path.read_bytes()
            actual = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            if actual != expected:
                changed.append(name)
        self.assertFalse(changed,
            "Working-tree bytes differ from recorded release pins: " + repr(changed[:20]))

        blobs = index_blob_pins(REPOSITORY, {indexed[name] for name in pins})
        changed = sorted(name for name, expected in pins.items() if blobs[indexed[name]] != expected)
        self.assertFalse(changed,
            "Git-index bytes differ from pinned working-tree files; stage exact original bytes and preserve binary attributes: "
            + repr(changed[:20]) + (f"; {len(changed)} total" if changed else ""))


if __name__ == "__main__":
    unittest.main()
