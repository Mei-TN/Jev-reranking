"""Materialize immutable wheel resources without Windows MAX_PATH installation failures.

Only the packaged ZIP and metadata are read. The cache is content-addressed and
contains source/configuration artifacts, never downloaded models or datasets.
An altered existing cache is rejected; it is never silently overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


SCHEMA = "jev-vendor-payload-v1"
RECEIPT = ".jev-resource-cache.json"
_VERIFIED = {}
SOURCE_METADATA = ("configs", "provenance", "docs", "examples", "requirements", "third_party",
                   "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _long_path(path):
    """Windows extended absolute paths work without a machine-wide registry change."""
    value = str(path.absolute())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def _member(name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name
            or relative.as_posix() != name or len(relative.parts) < 2 or relative.parts[0] != "vendor"):
        raise ValueError("unsafe packaged vendor member")
    return relative


def _ignored(path):
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


def _metadata_files(bundled, *, source_layout=False):
    files = {}
    if source_layout:
        paths = []
        for name in SOURCE_METADATA:
            path = bundled / name
            if path.is_symlink():
                raise ValueError("packaged resource metadata must not contain symlinks")
            if path.is_dir():
                paths.extend(path.rglob("*"))
            elif path.is_file():
                paths.append(path)
    else:
        paths = bundled.rglob("*")
    for path in sorted(paths):
        relative = path.relative_to(bundled)
        if _ignored(relative) or relative.as_posix() in {"vendor.zip", "vendor_index.json"}:
            continue
        if path.is_symlink():
            raise ValueError("packaged resource metadata must not contain symlinks")
        if path.is_file():
            files[relative.as_posix()] = {"sha256": _digest(path), "bytes": path.stat().st_size}
    return files


def verify_vendor_payload(payload, index_file, *, verify_bytes=True):
    """Validate a ZIP/index without extracting, including sdist-to-wheel builds."""
    payload, index_file = Path(payload), Path(index_file)
    if payload.is_symlink() or index_file.is_symlink():
        raise ValueError("vendor payload and index must be regular files, not symlinks")
    index = json.loads(Path(index_file).read_text(encoding="utf-8"))
    if not isinstance(index, dict) or index.get("schema") != SCHEMA or not isinstance(index.get("files"), dict) or not index["files"]:
        raise ValueError("invalid JEV vendor payload index")
    for name, entry in index["files"].items():
        _member(name)
        if (not isinstance(entry, dict) or type(entry.get("bytes")) is not int or entry["bytes"] < 0
                or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in entry["sha256"])
                or type(entry.get("mode", 0o644)) is not int or entry.get("mode", 0o644) not in {0o644, 0o755}):
            raise ValueError("invalid packaged vendor file identity")
    with zipfile.ZipFile(payload) as archive:
        members = archive.infolist()
        names = [entry.filename for entry in members]
        if len(names) != len(set(names)) or set(names) != set(index["files"]):
            raise ValueError("vendor ZIP inventory differs from its index")
        for entry in members:
            _member(entry.filename)
            mode = (entry.external_attr >> 16) & 0xFFFF
            if entry.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG}):
                raise ValueError("vendor ZIP must contain only regular files")
            if entry.file_size != index["files"][entry.filename]["bytes"]:
                raise ValueError("vendor ZIP member size differs from its index")
            if verify_bytes:
                with archive.open(entry) as source:
                    actual = hashlib.file_digest(source, "sha256").hexdigest()
                if actual != index["files"][entry.filename]["sha256"]:
                    raise ValueError("vendor ZIP member digest differs from its index")
    return index


def _cache_parent():
    configured = os.environ.get("JEV_RESOURCE_CACHE")
    if configured:
        base = Path(configured).expanduser()
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "JEV/resources"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "jev/resources"
    return _long_path(base)


def _signature(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _validate_cache(destination, expected, identity):
    try:
        receipt = json.loads((destination / RECEIPT).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("JEV resource cache has no valid completion receipt; select a new JEV_RESOURCE_CACHE") from error
    if receipt != {"schema": SCHEMA, "identity": identity}:
        raise ValueError("JEV resource cache identity differs from this installation")
    actual = set()
    for path in destination.rglob("*"):
        relative = path.relative_to(destination)
        if _ignored(relative) or relative.as_posix() == RECEIPT:
            continue
        if path.is_symlink():
            raise ValueError("JEV resource cache contains a symlink")
        if path.is_file():
            actual.add(relative.as_posix())
    if actual != set(expected):
        raise ValueError("JEV resource cache file inventory changed; select a new JEV_RESOURCE_CACHE")
    previous = _VERIFIED.get(str(destination), {})
    signatures = {}
    for name, entry in expected.items():
        path = destination.joinpath(*PurePosixPath(name).parts)
        signature = _signature(path)
        if signature[0] != entry["bytes"] or (previous.get(name) != signature and _digest(path) != entry["sha256"]):
            raise ValueError("JEV resource cache file differs from the packaged bytes: " + name)
        signatures[name] = signature
    _VERIFIED[str(destination)] = signatures


def resource_root(bundled_dir, *, source_layout=False):
    """Return a usable Path for packaged metadata and preserved vendor sources.

    Existing raw source-layout resources are returned directly. Wheels and
    compressed source distributions use a user-writable cache. Source layouts
    copy only SOURCE_METADATA, leaving code, tests and build artifacts in place.
    Concurrent callers publish a complete directory by
    atomic rename; each caller cleans only its own temporary staging directory.
    Each process hashes a cache on first use, then rechecks file inventories and
    metadata and rehashes files whose size/timestamps changed during that process.
    """
    bundled = _long_path(Path(bundled_dir).expanduser().resolve())
    payload, index_file = bundled / "vendor.zip", bundled / "vendor_index.json"
    if payload.is_file() and (bundled / "vendor").exists():
        raise ValueError("ambiguous JEV resources contain both raw vendor and vendor.zip")
    if not payload.is_file():
        if (bundled / "vendor").is_dir():
            return bundled
        raise ValueError("installed JEV resources have no preserved vendor payload")
    index = verify_vendor_payload(payload, index_file, verify_bytes=False)
    expected = _metadata_files(bundled, source_layout=source_layout)
    expected.update(index["files"])
    identity_data = {"schema": SCHEMA, "payload_sha256": _digest(payload), "index_sha256": _digest(index_file),
                     "metadata": {name: value for name, value in expected.items() if not name.startswith("vendor/")}}
    identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cache = _cache_parent()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / identity
    if destination.exists():
        _validate_cache(destination, expected, identity)
        return destination

    with zipfile.ZipFile(payload) as archive:
        with tempfile.TemporaryDirectory(prefix="." + identity + ".", dir=cache) as temporary:
            stage = _long_path(Path(temporary))
            for name in expected:
                target = stage.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if name in index["files"]:
                    with archive.open(name) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    if _digest(target) != expected[name]["sha256"]:
                        raise ValueError("vendor ZIP member digest differs from its index")
                    if os.name != "nt":
                        os.chmod(target, expected[name].get("mode", 0o644) & 0o777)
                else:
                    shutil.copyfile(bundled.joinpath(*PurePosixPath(name).parts), target)
                    if _digest(target) != expected[name]["sha256"]:
                        raise ValueError("packaged metadata changed while materializing")
            (stage / RECEIPT).write_text(json.dumps({"schema": SCHEMA, "identity": identity}, sort_keys=True) + "\n",
                                         encoding="utf-8", newline="\n")
            try:
                stage.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
                # A concurrent process already published this content identity.
            _validate_cache(destination, expected, identity)
    return destination
