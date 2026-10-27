"""Run repository checks and save their outputs."""
import argparse
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def check_test_dependencies():
    """Check declared exact test pins and the native schema API in this Python."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = project["project"]["optional-dependencies"]["test"]
    problems = []
    for requirement in requirements:
        name, separator, expected = requirement.partition("==")
        if not separator or not name or not expected:
            problems.append(f"test requirement must use an exact pin: {requirement}")
            continue
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"missing {requirement}")
        else:
            if installed != expected:
                problems.append(f"{name}=={installed} is installed; expected {requirement}")
    if not problems:
        try:
            pydantic = importlib.import_module("pydantic")
            for name in ("BaseModel", "Field", "model_validator"):
                getattr(pydantic, name)
        except (ImportError, AttributeError) as error:
            problems.append(f"Pydantic 2 schema API cannot be imported: {error}")
    if problems:
        print(f"Contract-test dependencies are unavailable in {sys.executable}:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(f'Install with this interpreter: "{sys.executable}" -m pip install -e ".[test]"', file=sys.stderr)
        return 2
    print(f"Contract-test dependencies ready in {sys.executable}: {', '.join(requirements)}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, help="new directory for check outputs and native exit receipts")
    p.add_argument("--check-test-dependencies", action="store_true", help="check this Python's test extra without running tests")
    p.add_argument("--native-python", help="separate Python for optional original CPU head/readout tests; root tests use this script's Python")
    a = p.parse_args()
    if a.check_test_dependencies:
        return check_test_dependencies()
    if a.output is None:
        p.error("--output is required unless --check-test-dependencies is used")
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    checks = [
        ("test_dependencies", [sys.executable, str(ROOT / "scripts/check.py"), "--check-test-dependencies"], env),
        ("sources", [sys.executable, str(ROOT / "scripts/verify_sources.py")], env),
        ("contracts", [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"], env),
        ("release_audit", [sys.executable, str(ROOT / "scripts/audit_release.py"), "--summary"], env),
    ]
    if a.native_python:
        # Scripts/tests remain in the source checkout. Preserved native sources
        # may be in the content-addressed resource cache for a compressed sdist.
        source = str(ROOT / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from jev.config import root
        resources = root()
        for name, tree, pattern in (("original_heads", "head1_v1", "test_heads.py"), ("original_readouts", "e11", "test_last_span_readout.py")):
            source = resources / "vendor" / tree
            native_env = dict(env, PYTHONPATH=str(source), JEV_E458_BASIS_SOURCE=str(resources / "vendor/e12"))
            checks.append((name, [a.native_python, "-m", "unittest", "discover", "-s", str(source / "tests"), "-p", pattern, "-v"], native_env))
    results = []
    for name, command, environment in checks:
        with (output / (name + ".stdout.log")).open("x", encoding="utf-8") as stdout, (output / (name + ".stderr.log")).open("x", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
        results.append(dict(name=name, native_exit=result.returncode, command=command))
        print(json.dumps(dict(check=name, native_exit=result.returncode)), flush=True)
        if name == "test_dependencies" and result.returncode != 0:
            print((output / (name + ".stderr.log")).read_text(encoding="utf-8"), end="", file=sys.stderr)
            break
    report = dict(schema="jev.preparation-checks.v1", status="PASS" if all(r["native_exit"] == 0 for r in results) else "FAILED",
                  checked_utc=datetime.now(timezone.utc).isoformat(), checks=results,
                  scope="Source, control-plane, score arithmetic and optional native CPU fixtures; no pretrained model training/inference or GPU execution")
    (output / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
