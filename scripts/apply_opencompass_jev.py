"""Apply or verify the small JEV overlay against the fixed public OpenCompass source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

PLUGIN = Path(__file__).resolve().parents[1] / "third_party/opencompass_jev"


def pin(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def run(argv, cwd):
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise ValueError("command failed: " + " ".join(argv) + "\n" + result.stderr.strip())
    return result.stdout.strip()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError("unsafe overlay path")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("overlay path escapes source root")
    for part in [target, *target.parents]:
        if part == root.parent:
            break
        if part.is_symlink():
            raise ValueError("symlink in overlay destination")
    return target


def load_manifest():
    manifest = json.loads((PLUGIN / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "jev.opencompass-plugin.v1":
        raise ValueError("unexpected plugin manifest")
    for name, expected in manifest["overlay_files"].items():
        if pin(safe_path(PLUGIN / "overlay", name)) != expected:
            raise ValueError("shipped overlay differs: " + name)
    patch = manifest["patch"]
    if pin(safe_path(PLUGIN, patch["path"])) != {key: patch[key] for key in ["bytes", "sha256"]}:
        raise ValueError("shipped compatibility patch differs")
    for name, expected in manifest["licenses"].items():
        if pin(safe_path(PLUGIN, name)) != expected:
            raise ValueError("shipped license differs")
    return manifest


def verify_source(root, manifest, applied):
    revision = run(["git", "rev-parse", "HEAD"], root)
    if revision != manifest["upstream"]["revision"]:
        raise ValueError("checkout is not the fixed public OpenCompass revision")
    for name, row in manifest["patched_files"].items():
        if pin(safe_path(root, name)) != row["paper" if applied else "upstream"]:
            raise ValueError("patched interface identity differs: " + name)
    for name, expected in manifest["overlay_files"].items():
        target = safe_path(root, name)
        if applied:
            if pin(target) != expected:
                raise ValueError("installed overlay differs: " + name)
        elif target.exists():
            raise ValueError("overlay destination already exists: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="separate owned OpenCompass checkout")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify pristine inputs and patch applicability without writing")
    mode.add_argument("--apply", action="store_true", help="apply the reviewed patch and copy the 16-file overlay")
    mode.add_argument("--verify", action="store_true", help="verify the applied selected files without writing")
    args = parser.parse_args()
    try:
        source = args.source.expanduser().resolve(strict=True)
        if not source.is_dir() or source == PLUGIN or source.is_relative_to(PLUGIN):
            raise ValueError("provide a separate owned OpenCompass checkout")
        manifest = load_manifest()
        verify_source(source, manifest, args.verify)
        if not args.verify:
            if run(["git", "status", "--porcelain", "--untracked-files=all"], source):
                raise ValueError("use a clean checkout; existing changes are preserved")
            patch = str(PLUGIN / manifest["patch"]["path"])
            run(["git", "-c", "core.autocrlf=false", "apply", "--check", "--whitespace=nowarn", patch], source)
            if args.apply:
                run(["git", "-c", "core.autocrlf=false", "apply", "--whitespace=nowarn", patch], source)
                for name in manifest["overlay_files"]:
                    target = safe_path(source, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as stream:
                        stream.write(safe_path(PLUGIN / "overlay", name).read_bytes())
                verify_source(source, manifest, True)
        print(json.dumps({"status": "SELECTED_FILES_VERIFIED" if args.verify or args.apply else "READY_TO_APPLY",
                          "upstream_revision": manifest["upstream"]["revision"],
                          "overlay_files": len(manifest["overlay_files"]),
                          "patched_files": len(manifest["patched_files"]),
                          "wrote_source": bool(args.apply), "model_or_GPU_operations": 0,
                          "scope": "Selected source byte identity only; no runtime or benchmark-equivalence claim."}))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "FAILED", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
