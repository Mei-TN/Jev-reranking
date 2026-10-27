#!/usr/bin/env python3
"""Read-only release audit for literal credentials and site-specific locations.

Default output is JSON. Findings contain paths, line numbers and categories;
matched credential or location values are never printed. Exit 1 means a likely
literal credential was found; exit 2 means a scan error; locations alone exit 0.
Compressed source vendor payloads are verified and materialized in the resource
cache before their files are scanned with the same relative paths and filters.
No source files are changed, and no network, model or SSH operation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EXCLUDED_DIRECTORIES = {".git", "assets", "work", "runs", "dist", "build", "node_modules",
                        "__pycache__", ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox"}
BINARY_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".zip", ".gz", ".tgz",
                   ".tar", ".7z", ".whl", ".safetensors", ".pt", ".pth", ".bin", ".pyc", ".pyo"}
TOKEN_PATTERNS = {
    "modelscope_token": re.compile(r"\bms-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "openai_token": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key_pem": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
}
ASSIGNMENT = re.compile(
    r'''(?P<name>["']?[A-Za-z_][A-Za-z0-9_.-]*["']?)\s*(?:=|:)\s*'''
    r'''(?:(?P<quote>["'])(?P<quoted>[^"'\r\n]*)(?P=quote)|(?P<bare>[^\s,;#}\])]+))''')
BEARER = re.compile(r"\bBearer\s+([A-Za-z0-9._~+/=-]{16,})", re.IGNORECASE)
URL_CREDENTIAL = re.compile(r"\b(?:https?|ssh)://[^\s/:@]+:([^\s/@]+)@", re.IGNORECASE)
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
HOST_LABEL = re.compile(r"\b(?:WINH?[0-7]|unicom8card[0-9]+sibyl)\b", re.IGNORECASE)
PRIVATE_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:data|home|root|Users|mnt)/[^\s\"'<>),;\]}]+")
PRIVATE_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:[\\/](?:Users|Shared|data|workspace)[\\/][^\s\"'<>),;\]}]+", re.IGNORECASE)
PLACEHOLDER_WORDS = {"", "none", "null", "false", "true", "password", "passwd", "pwd", "secret", "token",
                     "key", "example", "placeholder", "redacted", "changeme", "change_me", "dummy", "test"}


def excluded(relative: Path) -> bool:
    for part in relative.parts[:-1]:
        if part in EXCLUDED_DIRECTORIES or part.startswith(".venv") or part.endswith(".egg-info"):
            return True
    name = relative.name
    return (name.startswith("._") or name.endswith((".pyc", ".pyo", ".local.json"))
            or name == ".DS_Store")


def release_files(root: Path):
    """Prefer Git's tracked/nonignored inventory; fallback supports unpacked releases."""
    try:
        checkout = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                                  capture_output=True, check=False, timeout=30)
        result = None
        if checkout.returncode == 0 and Path(os.fsdecode(checkout.stdout).rstrip("\r\n")).resolve() == root.resolve():
            result = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "."],
                                    capture_output=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0:
        paths = {Path(os.fsdecode(item)) for item in result.stdout.split(b"\0") if item}
        return sorted((root / path for path in paths if not excluded(path)), key=lambda path: path.as_posix()), "git-tracked-and-nonignored"
    paths = []
    for directory, directories, names in os.walk(root):
        directories[:] = sorted(name for name in directories if name not in EXCLUDED_DIRECTORIES
                                and not name.startswith(".venv") and not name.endswith(".egg-info"))
        for name in sorted(names):
            path = Path(directory) / name
            if not excluded(path.relative_to(root)):
                paths.append(path)
    return paths, "release-directory-fallback"


def placeholder(value: str) -> bool:
    value = value.strip()
    lower = value.lower()
    return (lower in PLACEHOLDER_WORDS or lower.startswith(("your_", "your-", "replace_", "replace-", "example_", "dummy_", "test-", "test_"))
            or bool(re.fullmatch(r"<[^>]*>|\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*|%[A-Za-z_][A-Za-z0-9_]*%", value))
            or bool(re.fullmatch(r"[A-Z][A-Z0-9_]*(?:_KEY|_TOKEN|_PASSWORD|_SECRET|_ENV)", value))
            or lower.startswith(("env:", "os.environ", "getenv(", "process.env", "settings.")))


def sensitive_name(name: str) -> bool:
    name = name.strip("\"'").lower().rsplit(".", 1)[-1]
    if name.endswith(("_env", "_name", "_names", "_pattern", "_patterns", "_regex", "_field", "_fields", "_variable")):
        return False
    return (name in {"password", "passwd", "pwd", "token", "secret", "credential", "credentials", "private_key"}
            or name.endswith(("_password", "_passwd", "_pwd", "_api_key", "_access_token", "_auth_token", "_client_secret", "_secret_key"))
            or name in {"apikey", "api_key", "access_token", "auth_token", "client_secret", "secret_key"})


def provider_placeholder(value: str) -> bool:
    suffix = re.sub(r"^(?:sk-(?:proj-|svcacct-)?|hf_|ms-|gh[pousr]_|github_pat_|xox[baprs]-)", "", value)
    return placeholder(suffix) or bool(re.fullmatch(r"[xX0_-]{8,}", suffix))


def line_findings(line: str, *, code=False):
    secrets = set()
    for kind, pattern in TOKEN_PATTERNS.items():
        for match in pattern.finditer(line):
            if kind == "private_key_pem" or not (placeholder(match.group()) or provider_placeholder(match.group())):
                secrets.add(kind)
    for match in ASSIGNMENT.finditer(line):
        if not sensitive_name(match.group("name")):
            continue
        value = match.group("quoted") if match.group("quote") else match.group("bare")
        if code and not match.group("quote"):
            continue  # variable/function references are not string literals
        if placeholder(value) or provider_placeholder(value):
            continue
        if (code and line.lstrip().startswith("def ")
                and match.group("name").strip("\"'").lower() in {"api_key", "apikey"}
                and value.lower() in {"empty", "local", "unused", "not-needed"}):
            continue  # explicit inert API-client default, not a password/token value
        if value.endswith((".pem", ".key", ".pub")):
            continue  # private-key file locations are inventoried separately
        if not secrets:
            secrets.add("literal_credential_assignment")
    for pattern, kind in ((BEARER, "literal_bearer_token"), (URL_CREDENTIAL, "url_embedded_password")):
        for match in pattern.finditer(line):
            if not placeholder(match.group(1)):
                secrets.add(kind)
    locations = set()
    for match in IPV4.finditer(line):
        try:
            address = ipaddress.ip_address(match.group())
        except ValueError:
            continue
        if not address.is_loopback and not address.is_unspecified:
            locations.add("machine_address")
    if HOST_LABEL.search(line):
        locations.add("project_host_label")
    if PRIVATE_POSIX_PATH.search(line):
        locations.add("private_posix_path")
    if PRIVATE_WINDOWS_PATH.search(line):
        locations.add("private_windows_path")
    return sorted(secrets), sorted(locations)


def display_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    for pattern in TOKEN_PATTERNS.values():
        relative = pattern.sub("[redacted]", relative)
    return relative


def reviewed_findings(root, findings, file_hashes):
    """Classify only reviewed generic assignments bound to exact published bytes."""
    registry = root / "provenance/audit-classifications.json"
    if not registry.is_file():
        return findings, [], []
    try:
        if registry.is_symlink():
            raise ValueError("classification registry is a symlink")
        spec = json.loads(registry.read_text(encoding="utf-8"))
        if (not isinstance(spec, dict) or spec.get("schema") != "jev.reviewed-audit-findings.v1"
                or not isinstance(spec.get("findings"), list)):
            raise ValueError("invalid classification schema")
        classified = {}
        for row in spec["findings"]:
            path = row["path"]
            relative = PurePosixPath(path)
            if (not isinstance(path, str) or relative.is_absolute() or ".." in relative.parts
                    or "\\" in path or ":" in path or relative.as_posix() != path):
                raise ValueError("invalid classification path")
            if (type(row.get("line")) is not int or row["line"] < 1
                    or row.get("kind") != "literal_credential_assignment"
                    or not isinstance(row.get("reason"), str) or not row["reason"].strip()
                    or re.fullmatch(r"[0-9a-f]{64}", row.get("file_sha256", "")) is None):
                raise ValueError("only byte-bound, explained generic assignments can be classified")
            if line_findings(row["reason"])[0]:
                raise ValueError("classification reasons must not contain credential patterns")
            if file_hashes.get(path) != row["file_sha256"]:
                raise ValueError("classification file identity differs")
            key = (path, row["line"], row["kind"])
            if key in classified:
                raise ValueError("duplicate classification")
            classified[key] = row
        unresolved, reviewed = [], []
        for finding in findings:
            classification = classified.get((finding["path"], finding["line"], finding["kind"]))
            if classification is None:
                unresolved.append(finding)
            else:
                reviewed.append(dict(finding, reason=classification["reason"],
                                     file_sha256=classification["file_sha256"]))
        return unresolved, reviewed, []
    except (OSError, ValueError, TypeError, KeyError):
        return findings, [], [dict(path="provenance/audit-classifications.json", kind="invalid-reviewed-findings")]


def audit(root: Path) -> dict:
    files, selection = release_files(root)
    secrets, locations, skipped, errors = [], [], [], []
    scan_files = [(path, root) for path in files]
    if (root / "vendor.zip").is_file():
        try:
            from jev.resources import resource_root

            materialized = resource_root(root, source_layout=True)
            vendor = materialized / "vendor"
            if not vendor.is_dir():
                raise ValueError("materialized source bundle has no vendor directory")
            scan_files.extend((path, materialized) for path in sorted(vendor.rglob("*"))
                              if (path.is_file() or path.is_symlink())
                              and not excluded(path.relative_to(materialized)))
            selection += "+materialized-vendor-payload"
        except (OSError, ValueError, ImportError, zipfile.BadZipFile):
            errors.append(dict(path="vendor.zip", kind="invalid-vendor-payload"))
    text_files = total_bytes = 0
    file_hashes = {}
    for path, display_root in scan_files:
        relative = display_path(path, display_root)
        if path.is_symlink():
            skipped.append(dict(path=relative, reason="symlink-not-followed"))
            continue
        try:
            if path.suffix.lower() in BINARY_SUFFIXES:
                skipped.append(dict(path=relative, reason="binary-file-type"))
                continue
            content = path.read_bytes()
            if b"\0" in content[:8192]:
                skipped.append(dict(path=relative, reason="binary-content"))
                continue
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                skipped.append(dict(path=relative, reason="non-utf8-content"))
                continue
            text_files += 1
            total_bytes += len(content)
            file_hashes[relative] = hashlib.sha256(content).hexdigest()
            for number, line in enumerate(text.splitlines(), 1):
                code = path.suffix.lower() in {".py", ".js", ".ts"} or path.name.endswith("_source.txt")
                secret_kinds, location_kinds = line_findings(line, code=code)
                secrets.extend(dict(path=relative, line=number, kind=kind) for kind in secret_kinds)
                area = "preserved_vendor" if relative.startswith("vendor/") else "provenance" if relative.startswith("provenance/") else "maintained"
                locations.extend(dict(path=relative, line=number, kind=kind, area=area) for kind in location_kinds)
        except OSError:
            errors.append(dict(path=relative, kind="unreadable-file"))
    raw_match_count = len(secrets)
    secrets, reviewed, classification_errors = reviewed_findings(root, secrets, file_hashes)
    errors.extend(classification_errors)
    return dict(schema="jev-release-audit-v1", status="SCAN_ERROR" if errors else "LITERAL_SECRETS_FOUND" if secrets else "PASS",
                selection=selection, scanned_text_files=text_files, scanned_bytes=total_bytes,
                secret_findings=secrets, reviewed_findings=reviewed, location_findings=locations, skipped_files=skipped, errors=errors,
                counts=dict(literal_secrets=len(secrets), location_findings=len(locations),
                            raw_credential_pattern_matches=raw_match_count, reviewed_pattern_matches=len(reviewed),
                            locations_by_kind=dict(sorted(Counter(row["kind"] for row in locations).items())),
                            locations_by_area=dict(sorted(Counter(row["area"] for row in locations).items())),
                            skipped_files=len(skipped), errors=len(errors)),
                scope="Pattern audit of selected text files; reviewed generic fixture/reference assignments are separately bound to exact file bytes. Deployment locations are inventory findings and do not fail the credential check.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="release directory to inspect")
    parser.add_argument("--summary", action="store_true", help="omit the potentially long deployment-location and skipped-file inventories")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(json.dumps(dict(schema="jev-release-audit-v1", status="SCAN_ERROR", errors=[dict(kind="root-not-directory")])))
        return 2
    report = audit(root)
    if args.summary:
        report.pop("location_findings")
        report.pop("skipped_files")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 2 if report["errors"] else 1 if report["secret_findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
