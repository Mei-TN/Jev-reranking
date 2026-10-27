"""CPU validation of recorded assets, reference locations and role environments."""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

from .config import read, root, sha, worker_env


BASE_MANIFEST = "provenance/portable_pair/ASSET_MANIFEST.json"
TRAINING_ENVIRONMENT = "requirements/training-lock.txt"
NATIVE_IMPORTS = ("torch", "transformers", "peft", "datasets", "safetensors", "fla", "triton", "kev.train")
ENVIRONMENT_SCOPE = (
    "Compares recorded distribution versions, Python version and fresh-interpreter native imports. "
    "Does not establish exact wheel/native-binary, model/data, hardware or numeric reproduction closure."
)


def verify_protocol(path):
    """Verify one registered protocol file and return its pinned manifest entry.

    A relative path is relative to the installed/source resource root. This
    checks both byte count and digest; arbitrary unregistered paths are refused.
    """
    resource_root = root().resolve()
    candidate = Path(path).expanduser()
    candidate = (resource_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        relative = candidate.relative_to(resource_root).as_posix()
    except ValueError as error:
        raise ValueError("protocol path is outside the JEV resource root: " + str(path)) from error
    manifest = read(resource_root / "provenance/PROTOCOLS.json")
    if manifest.get("schema") != "jev.protocols.v1" or not isinstance(manifest.get("files"), list):
        raise ValueError("invalid protocol manifest")
    matches = [entry for entry in manifest["files"] if entry.get("path") == relative]
    if len(matches) != 1:
        raise ValueError("protocol path must have exactly one pinned manifest entry: " + relative)
    entry = matches[0]
    if not candidate.is_file():
        raise ValueError("missing pinned protocol: " + relative)
    if candidate.stat().st_size != entry.get("bytes") or sha(candidate) != entry.get("sha256"):
        raise ValueError("protocol bytes differ from the recorded source: " + relative)
    return dict(entry)


def base_asset_pins(site):
    """Map all recorded NeoHorse base members to the configured model directory.

    The returned path-to-SHA256 mapping can extend a plan's input_pins. It
    includes both safetensor shards and the tokenizer, rather than identifying
    a model by config.json alone. This function does not read weight files.
    """
    verify_protocol(BASE_MANIFEST)
    manifest = read(root() / BASE_MANIFEST)
    if site.get("base_revision") and site["base_revision"] != manifest["base_revision"]:
        raise ValueError("site base_revision differs from the recorded NeoHorse base; use an explicit custom recipe")
    base = Path(site["base_model"]).expanduser().absolute()
    pins, members = {}, set()
    for item in manifest.get("assets", {}).values():
        name = item.get("path", "")
        if not name.startswith("base/"):
            continue
        relative = PurePosixPath(name)
        if ".." in relative.parts or "\\" in name or len(relative.parts) < 2:
            raise ValueError("unsafe base asset path in recorded manifest")
        member = PurePosixPath(*relative.parts[1:])
        digest = item.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("invalid base asset digest: " + name)
        target = str(base.joinpath(*member.parts))
        if target in pins:
            raise ValueError("duplicate base asset member: " + name)
        pins[target] = digest
        members.add(member.as_posix())
    required = {"config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"}
    if not required <= members:
        raise ValueError("recorded base asset manifest lacks full weights/tokenizer identity")
    return pins


def validate_reference_locations(plan):
    """Return actionable mismatches with native reference path comparisons.

    The original trainers compare saved base/data argument strings. A copied
    old reference therefore cannot be made usable by rebinding only the new
    run's locations. Its historical metadata must remain unchanged.
    """
    from .recipes import options

    arguments = options(plan["argv"])
    problems = []
    for reference in plan.get("references", []):
        rid = reference.get("run_id", "unnamed reference")
        path = Path(reference["path"]) / "training_config.json"
        try:
            config = read(path)
            saved = config["args"]
            if not isinstance(saved, dict):
                raise ValueError("args is not an object")
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(f"reference {rid}: cannot read native training arguments at {path}: {error}. "
                "Create the genuine prerequisite reference run at this site; do not reconstruct or rewrite historical metadata.")
            continue
        for name in ("base", "data"):
            expected = arguments.get("--" + name)
            actual = saved.get(name)
            if not isinstance(actual, str) or actual != expected:
                problems.append(f"reference {rid}: saved args.{name}={actual!r} differs from this plan's {expected!r}. "
                    "The native reference contract compares these paths exactly. Create fresh prerequisite references "
                    "with this site's same base/data paths, or select a site matching the genuine reference. "
                    "Do not rewrite historical training_config.json.")
    return problems


def _recorded_environment():
    path = root() / TRAINING_ENVIRONMENT
    requirements = read(root() / "requirements/manifest.json")
    entry = requirements["files"]["training-lock.txt"]
    if not path.is_file() or path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
        raise ValueError("recorded training environment file differs from its source snapshot")
    python, versions = None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("# python "):
            python = line.split()[2]
        elif line and not line.startswith("#"):
            if line.count("==") != 1:
                raise ValueError("recorded environment contains an unsupported requirement: " + line)
            name, version = line.split("==")
            if not name or not version or name in versions:
                raise ValueError("recorded environment contains a duplicate/empty package")
            versions[name] = version
    if not python or not versions:
        raise ValueError("recorded training environment lacks Python/package versions")
    return python, versions


_ENVIRONMENT_PROBE = r'''
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

names, native, source = json.loads(sys.argv[1]), json.loads(sys.argv[2]), Path(sys.argv[3]).resolve()
versions = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
imports = {}
torch_build = {}
for name in native:
    try:
        module = importlib.import_module(name)
        item = dict(status='PASS', file=getattr(module, '__file__', None))
        if name == 'kev.train' and (not item['file'] or not Path(item['file']).resolve().is_relative_to(source)):
            item['status'] = 'SOURCE_MISMATCH'
        if name == 'torch':
            torch_build = dict(version=str(module.__version__), cuda=module.version.cuda)
        imports[name] = item
    except Exception as error:
        imports[name] = dict(status='IMPORT_FAILED', error_type=type(error).__name__, message=str(error))
print('JEV_ENVIRONMENT_REPORT=' + json.dumps(dict(python=platform.python_version(), executable=sys.executable,
    prefix=sys.prefix, packages=versions, native_imports=imports, torch_build=torch_build), allow_nan=False))
'''


def _torch_version_status(recorded, observed, build):
    if observed == recorded:
        return "MATCH"
    expected = re.fullmatch(r"([^+]+)(?:\+cu([0-9]+))?", recorded)
    actual = re.fullmatch(r"([^+]+)(?:\+cu([0-9]+))?", observed or "")
    if not expected or not actual or expected[1] != actual[1]:
        return "MISMATCH"
    # An omitted metadata CUDA suffix is accepted only when the imported
    # torch build independently supplies exactly the recorded CUDA identity.
    if expected[2] and actual[2]:
        return "MISMATCH"
    suffix = expected[2] or actual[2]
    imported = str(build.get("version", ""))
    cuda = str(build.get("cuda") or "").replace(".", "")
    if suffix and imported == expected[1] + "+cu" + suffix and cuda == suffix:
        return "CUDA_SUFFIX_NORMALIZED"
    return "MISMATCH"


def environment_report(python, tree, extraenv=None):
    """Inspect a selected interpreter in a fresh CPU-only child process.

    ``python`` is passed lexically to subprocess, preserving a venv executable
    symlink. All recorded distributions are compared, and actual native imports
    are checked without loading model weights or initializing a GPU job.
    """
    recorded_python, expected = _recorded_environment()
    selected_python = os.fspath(python)
    selected_tree = str(Path(tree).expanduser().absolute())
    command = [selected_python, "-P", "-c", _ENVIRONMENT_PROBE,
               json.dumps(list(expected)), json.dumps(list(NATIVE_IMPORTS)), selected_tree]
    env = worker_env(selected_tree, extra=extraenv)
    env["CUDA_VISIBLE_DEVICES"] = ""
    report = dict(schema="jev.environment-report.v1", status="INCOMPLETE", interpreter=selected_python,
        source_root=selected_tree, recorded_environment=TRAINING_ENVIRONMENT,
        recorded_environment_sha256=sha(root() / TRAINING_ENVIRONMENT), scope=ENVIRONMENT_SCOPE,
        exact_closure_verified=False, gpu_execution=False)
    try:
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error))
        return report
    report["native_exit"] = result.returncode
    if result.returncode:
        report["failure"] = dict(type="ProbeFailed", stderr=result.stderr[-4000:], stdout=result.stdout[-4000:])
        return report
    lines = [line[len("JEV_ENVIRONMENT_REPORT="):] for line in result.stdout.splitlines()
             if line.startswith("JEV_ENVIRONMENT_REPORT=")]
    try:
        if len(lines) != 1:
            raise ValueError("child did not emit exactly one environment report")
        observed = json.loads(lines[0])
        versions = observed["packages"]
        imports = observed["native_imports"]
        if not isinstance(versions, dict) or not isinstance(imports, dict):
            raise ValueError("invalid package/import report")
        if (any(value is not None and not isinstance(value, str) for value in versions.values())
                or any(not isinstance(value, dict) for value in imports.values())
                or not isinstance(observed.get("torch_build", {}), dict)):
            raise ValueError("invalid package version/native build fields")
    except (ValueError, KeyError, TypeError) as error:
        report["failure"] = dict(type="InvalidProbeOutput", message=str(error), stdout=result.stdout[-4000:])
        return report
    comparisons = {}
    for name, version in expected.items():
        actual = versions.get(name)
        status = "MISSING" if actual is None else "MATCH" if actual == version else "MISMATCH"
        if name == "torch" and actual is not None:
            status = _torch_version_status(version, actual, observed.get("torch_build", {}))
        comparisons[name] = dict(recorded=version, observed=actual, status=status)
    expected_torch = expected.get("torch", "")
    expected_cuda = re.fullmatch(r"([^+]+)\+cu([0-9]+)", expected_torch)
    build = observed.get("torch_build", {})
    build_status = "UNRECORDED"
    if expected_cuda:
        build_status = "MATCH" if (build.get("version") == expected_torch and
            str(build.get("cuda") or "").replace(".", "") == expected_cuda[2]) else "MISMATCH"
        if build_status == "MISMATCH" and "torch" in comparisons and comparisons["torch"]["status"] != "MISSING":
            comparisons["torch"]["status"] = "MISMATCH"
    missing = [name for name, item in comparisons.items() if item["status"] == "MISSING"]
    mismatched = [name for name, item in comparisons.items() if item["status"] == "MISMATCH"]
    normalized = [name for name, item in comparisons.items() if item["status"] == "CUDA_SUFFIX_NORMALIZED"]
    failed_imports = [name for name in NATIVE_IMPORTS if imports.get(name, {}).get("status") != "PASS"]
    python_status = "MATCH" if observed.get("python") == recorded_python else "MISMATCH"
    if missing or failed_imports:
        status = "INCOMPLETE"
    elif mismatched or python_status != "MATCH":
        status = "MISMATCH"
    elif normalized:
        status = "MATCH_WITH_TORCH_CUDA_SUFFIX_NORMALIZATION"
    else:
        status = "MATCH_RECORDED_VERSIONS"
    report.update(status=status, python=dict(recorded=recorded_python, observed=observed.get("python"), status=python_status),
        observed_interpreter=observed.get("executable"), observed_prefix=observed.get("prefix"),
        packages=comparisons, native_imports=imports, torch_build=observed.get("torch_build", {}),
        torch_build_comparison=build_status,
        missing_packages=missing, mismatched_packages=mismatched, normalized_packages=normalized,
        failed_native_imports=failed_imports)
    return report
