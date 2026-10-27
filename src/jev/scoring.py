"""Reproduce the original stored-output scoring supplement in a fresh directory."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import zipfile

from .config import read, root, sha, write_new
from .workflow import execute


ORIGINAL_ARCHIVE_SHA256 = "3125ccce2f8127efcf956bf2025367081cb7f0ab3b97e3a33e03826e357d2d45"
ORIGINAL_ARCHIVE_BYTES = 200841481
ARCHIVE_NAME = "SCORING_SUPPLEMENT_REVISED.zip"
EXPECTED_COUNTS = dict(independent_checkpoint_models=72, execution_modules=96,
    core_model_tasks=1152, auxiliary_model_conditions=200, eligible_question_occurrences=3722928,
    integer_differences=0)


def _archive_pin():
    """Return the expected name, size, and hash of the probability archive."""
    return dict(filename=ARCHIVE_NAME, bytes=ORIGINAL_ARCHIVE_BYTES,
                sha256=ORIGINAL_ARCHIVE_SHA256)



def _verify_archive(archive, pin):
    archive = Path(archive).expanduser().resolve()
    if not archive.is_file():
        raise ValueError("stored-scoring archive is missing: " + str(archive))
    if archive.stat().st_size != pin["bytes"]:
        raise ValueError("stored-scoring archive byte count differs from the original supplement")
    if sha(archive) != pin["sha256"]:
        raise ValueError("stored-scoring archive SHA256 differs from the original supplement")
    return archive


def _member_parts(info):
    name = info.filename
    path = PurePosixPath(name)
    parts = path.parts
    if (not name or "\\" in name or path.is_absolute() or not parts or
            any(part in (".", "..") for part in name.rstrip("/").split("/"))):
        raise ValueError("unsafe ZIP member path: " + name)
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    for part in parts:
        if (":" in part or "\x00" in part or part.rstrip(" .") != part or
                part.split(".")[0].upper() in reserved):
            raise ValueError("nonportable ZIP member path: " + name)
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode) or kind not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError("ZIP symlinks and special files are not allowed: " + name)
    if info.flag_bits & 1:
        raise ValueError("encrypted ZIP members are not supported")
    return parts


def _safe_extract(archive, output):
    """Preflight every member, then stream extraction into a new directory."""
    output = Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError("stored-scoring output already exists; use a fresh directory")
    output = output.resolve()
    with zipfile.ZipFile(archive) as package:
        entries, names, files = [], set(), set()
        for info in package.infolist():
            parts = _member_parts(info)
            key = "/".join(parts).casefold()
            if key in names:
                raise ValueError("duplicate or case-colliding ZIP members: " + info.filename)
            names.add(key)
            if not info.is_dir():
                files.add(key)
            entries.append((info, parts))
        for info, parts in entries:
            if any("/".join(parts[:i]).casefold() in files for i in range(1, len(parts))):
                raise ValueError("ZIP member has a file as its parent: " + info.filename)
        output.mkdir(parents=True, exist_ok=False)
        for info, parts in entries:
            destination = output.joinpath(*parts)
            if not destination.resolve().is_relative_to(output):
                raise ValueError("ZIP member escapes the extraction directory")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with package.open(info) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        return dict(entries=len(entries), files=len(files), uncompressed_bytes=sum(info.file_size for info, _ in entries))


def _check_report(report):
    if not isinstance(report, dict) or report.get("status") != "PASS" or report.get("fresh_directory_verification") is not True:
        raise ValueError("original score_all.py did not produce its successful fresh-directory report")
    for name, expected in EXPECTED_COUNTS.items():
        if type(report.get(name)) is not int or report[name] != expected:
            raise ValueError("stored-score verification count differs from the original supplement: " + name)
    if report.get("modules") != ["core56", "qwen8", "auxiliary"] or report.get("boundaries", {}).get("status") != "PASS":
        raise ValueError("stored-score modules or boundary checks are incomplete")
    for name in ("maximum_aggregate_difference_pp", "maximum_appendix_difference_pp"):
        value = report.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e-9:
            raise ValueError("stored-score reconstruction difference is invalid: " + name)
    return report


def reproduce_scores(archive, output):
    """Verify/extract the exact original ZIP and run its unchanged score_all.py.

    Returns a JSON-serializable object containing ``report_path`` and the full
    original ``summary``. Failed runs retain their command, exit code, and logs.
    """
    destination = Path(output).expanduser()
    if destination.exists() or destination.is_symlink():
        raise ValueError("stored-scoring output already exists; use a fresh directory")
    destination = destination.resolve()
    pin = _archive_pin()
    source_archive = _verify_archive(archive, pin)
    extracted = _safe_extract(source_archive, destination)
    script = destination / "score_all.py"
    if not script.is_file():
        raise ValueError("original supplement lacks score_all.py at its root")
    argv = [sys.executable, "-X", "utf8", "-P", str(script)]
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_AUDIT"):
        env.pop(key, None)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="")
    plan = dict(schema="jev.stored-score-reproduction.v1", archive=str(source_archive),
        archive_sha256=pin["sha256"], archive_bytes=pin["bytes"], output=str(destination),
        extraction=extracted, argv=argv, started_utc=datetime.now(timezone.utc).isoformat(),
        scope="Original stored-output scoring reproduction; no training/model inference reproduction.")
    write_new(destination / "REPRODUCTION_PLAN.json", plan)
    try:
        native = execute(argv, env, destination, "score_all")
        report_path = destination / "verification_outputs/VERIFICATION.json"
        summary = _check_report(read(report_path))
        result = dict(schema=plan["schema"], status="PASS", native_exit=native["exit_code"],
            archive_sha256=pin["sha256"], report_path=str(report_path), report_sha256=sha(report_path),
            summary=summary, scope=plan["scope"])
        write_new(destination / "REPRODUCTION_RESULT.json", result)
        return result
    except Exception as error:
        native_path = destination / "score_all.result.json"
        native_exit = read(native_path).get("exit_code") if native_path.is_file() else None
        write_new(destination / "REPRODUCTION_RESULT.json", dict(schema=plan["schema"], status="FAILED",
            native_exit=native_exit, archive_sha256=pin["sha256"],
            failure=dict(type=type(error).__name__, message=str(error)), scope=plan["scope"]))
        raise
