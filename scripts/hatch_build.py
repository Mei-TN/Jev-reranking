"""Package preserved vendor sources as one deterministic, installable ZIP payload."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import zipfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def build_vendor_payload(repository: Path, destination: Path):
    vendor = repository / "vendor"
    archived_payload = repository / "vendor.zip"
    archived_index = repository / "vendor_index.json"
    if archived_payload.is_file():
        if vendor.exists():
            raise ValueError("ambiguous build sources contain both raw vendor and vendor.zip")
        spec = importlib.util.spec_from_file_location("jev_build_resources", repository / "src/jev/resources.py")
        resources = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resources)
        resources.verify_vendor_payload(archived_payload, archived_index)
        return archived_payload, archived_index
    if archived_index.exists():
        raise ValueError("vendor_index.json has no matching vendor.zip")
    if vendor.is_symlink() or not vendor.is_dir():
        raise ValueError("a build requires the preserved vendor tree or verified vendor.zip/index")
    inventory = {}
    destination.mkdir(parents=True, exist_ok=True)
    payload = destination / "vendor.zip"
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(vendor.rglob("*"), key=lambda value: value.relative_to(repository).as_posix()):
            relative = path.relative_to(repository)
            if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                raise ValueError("vendor symlinks cannot enter the wheel: " + relative.as_posix())
            if not path.is_file():
                continue
            data = path.read_bytes()
            name = relative.as_posix()
            mode = 0o755 if path.suffix == ".sh" else 0o644
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, data, compresslevel=9)
            inventory[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "mode": mode}
    if not inventory:
        raise ValueError("the wheel requires a nonempty preserved vendor tree")
    index = destination / "vendor_index.json"
    index.write_text(json.dumps({"schema": "jev-vendor-payload-v1", "files": inventory},
                                indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return payload, index


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name not in {"wheel", "sdist"}:
            return
        repository = Path(self.root)
        include = build_data.setdefault("force_include", {})
        if self.target_name == "wheel":
            # Directory force-inclusion bypasses Hatch's VCS/exclude rules.
            # Select each public metadata file before mapping it into the wheel.
            for name in ("configs", "provenance", "docs", "examples", "requirements", "third_party"):
                folder = repository / name
                if folder.is_symlink() or not folder.is_dir():
                    raise ValueError("wheel metadata must be a regular directory: " + name)
                for path in sorted(folder.rglob("*")):
                    relative = path.relative_to(repository).as_posix()
                    if path.is_symlink():
                        raise ValueError("metadata symlinks cannot enter the wheel: " + relative)
                    if path.is_file() and self.build_config.include_path(relative, explicit=True):
                        include[str(path)] = "jev/_resources/" + relative
        self._payload_directory = tempfile.TemporaryDirectory(prefix="jev-build-vendor-")
        payload, index = build_vendor_payload(repository, Path(self._payload_directory.name))
        prefix = "jev/_resources/" if self.target_name == "wheel" else ""
        include[str(payload)] = prefix + "vendor.zip"
        include[str(index)] = prefix + "vendor_index.json"

    def finalize(self, version, build_data, artifact_path):
        temporary = getattr(self, "_payload_directory", None)
        if temporary is not None:
            temporary.cleanup()
