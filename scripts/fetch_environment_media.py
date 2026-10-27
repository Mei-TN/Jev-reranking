"""Fetch the recorded environment artifacts without installing or importing them.

Downloads remain under the explicitly selected media directory. Completed files
are immutable and hash-checked on reuse. An interrupted transfer retains its
partial bytes; a later invocation can resume when the server supports ranges.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPOSITORY / "requirements/training-media.json"
CHUNK = 1024 * 1024


def default_spec():
    """Use the public external-media inventory without importing project code."""
    return DEFAULT_SPEC


def public_url(value):
    if not isinstance(value, str):
        raise ValueError("artifact URL must be a string")
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.fragment):
        raise ValueError("artifact URL requires HTTPS without embedded credentials or a fragment")
    _ = parsed.port
    return value


def validate_record(record):
    name = record.get("filename")
    if not isinstance(name, str):
        raise ValueError("artifact filename must be a string")
    path = PurePosixPath(name)
    device = name.split(".", 1)[0].upper()
    reserved = {"CON", "PRN", "AUX", "NUL"} | {prefix + str(number) for prefix in ("COM", "LPT") for number in range(1, 10)}
    if (not name or path.is_absolute() or len(path.parts) != 1 or name in {".", "..", "MEDIA_MANIFEST.json"}
            or name.startswith(".") or "\\" in name or ":" in name or name.endswith((" ", ".")) or device in reserved):
        raise ValueError("artifact filename must be a safe basename: " + repr(name))
    if type(record.get("bytes")) is not int or record["bytes"] < 0:
        raise ValueError("artifact byte count must be a nonnegative integer: " + name)
    if not isinstance(record.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
        raise ValueError("artifact SHA-256 must be a lowercase hexadecimal digest: " + name)
    public_url(record.get("public_url"))
    return {key: record[key] for key in ("filename", "bytes", "sha256", "public_url")}


def load_media_spec(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(spec, list):
        records = spec
    else:
        records = spec["wheels"] + [spec["python_archive"], spec["pillow_overlay"]] + spec["isolated_system_libraries"]
    result, names = [], set()
    for record in records:
        pin = validate_record(record)
        if pin["filename"].casefold() in names:
            raise ValueError("duplicate or case-colliding artifact filename: " + pin["filename"])
        names.add(pin["filename"].casefold())
        result.append(pin)
    if not result:
        raise ValueError("environment media specification is empty")
    return result


def file_pin(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_immutable(path, data):
    if path.is_symlink():
        raise ValueError("refusing symlink destination: " + str(path))
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError("refusing changed existing metadata: " + str(path))
        return
    with path.open("xb") as stream:
        stream.write(data)


def media_root(destination):
    root = Path(destination).expanduser()
    if root.is_symlink():
        raise ValueError("media root must not be a symlink")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("media root must be a directory")
    return root


class PublicArtifactRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        public_url(newurl)
        return super().redirect_request(request, response, code, message, headers, newurl)


def fetch_one(record, root, timeout=60, opener=None):
    pin = validate_record(record)
    root = media_root(root)
    target = root / pin["filename"]
    partial = root / (pin["filename"] + ".part")
    metadata = root / (pin["filename"] + ".part.json")
    expected = {key: pin[key] for key in ("bytes", "sha256")}
    for path in (target, partial, metadata):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("artifact path must remain a regular file inside the media root")
    if target.exists():
        if not target.is_file() or file_pin(target) != expected:
            raise ValueError("existing complete artifact differs from its pin: " + pin["filename"])
        return {**pin, "status": "VERIFIED_EXISTING", "resumed_from": 0}
    if partial.exists() and not partial.is_file():
        raise ValueError("partial artifact must be a regular file")
    if partial.exists() and not metadata.exists():
        raise ValueError("partial artifact has no matching source metadata: " + pin["filename"])
    if metadata.exists():
        if not metadata.is_file() or json.loads(metadata.read_text(encoding="utf-8")) != pin:
            raise ValueError("partial artifact source metadata differs: " + pin["filename"])
    else:
        write_immutable(metadata, json_bytes(pin))
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > pin["bytes"]:
        raise ValueError("partial artifact exceeds its declared size: " + pin["filename"])
    if offset < pin["bytes"]:
        headers = {"Accept-Encoding": "identity", "User-Agent": "jev-environment-media/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(pin["public_url"], headers=headers)
        opener = opener or build_opener(PublicArtifactRedirect())
        with opener.open(request, timeout=timeout) as response:
            public_url(response.geturl())
            status = response.status
            if offset:
                expected_range = f"bytes {offset}-{pin['bytes'] - 1}/{pin['bytes']}"
                if status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise ValueError("server did not honor the exact resume range; partial preserved: " + pin["filename"])
            elif status != 200:
                raise ValueError("expected HTTP 200 for a fresh artifact: " + pin["filename"])
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != pin["bytes"] - offset:
                raise ValueError("server content length differs from the recorded artifact: " + pin["filename"])
            count = offset
            with partial.open("ab" if partial.exists() else "xb") as stream:
                while chunk := response.read(CHUNK):
                    if count + len(chunk) > pin["bytes"]:
                        raise ValueError("response exceeds the recorded artifact size: " + pin["filename"])
                    stream.write(chunk)
                    count += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    elif not partial.exists():
        partial.touch(exist_ok=False)
    actual = file_pin(partial)
    if actual != expected:
        raise ValueError("downloaded artifact size or SHA-256 differs; partial preserved: " + pin["filename"])
    # A hard link publishes the verified bytes atomically without replacing an
    # existing complete file. Both names are inside this same media directory.
    os.link(partial, target)
    partial.unlink()
    metadata.unlink()
    return {**pin, "status": "DOWNLOADED", "resumed_from": offset}


def fetch_media(spec_path, destination, workers=4, timeout=60):
    if type(workers) is not int or not 1 <= workers <= 16 or timeout <= 0:
        raise ValueError("workers must be 1..16 and timeout must be positive")
    spec_path = Path(spec_path).resolve()
    records = load_media_spec(spec_path)
    spec_sha256 = file_pin(spec_path)["sha256"]
    root = media_root(destination)
    reports = root / ".fetch-reports"
    if reports.is_symlink():
        raise ValueError("report directory must not be a symlink")
    reports.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc)
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_one, record, root, timeout): record for record in records}
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {**record, "status": "FAILED", "error_type": type(error).__name__, "error": str(error)}
            results[record["filename"]] = result
            print(result["status"] + " " + record["filename"], flush=True)
    succeeded = all(value["status"] != "FAILED" for value in results.values())
    report = {"schema": "jev.environment-media-fetch.v1", "status": "PASS" if succeeded else "FAILED",
              "source_spec_sha256": spec_sha256, "destination": str(root),
              "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
              "expected_files": len(records), "expected_bytes": sum(record["bytes"] for record in records),
              "files": [results[record["filename"]] for record in records],
              "scope": "Artifact download and size/SHA-256 verification only; no package installation or execution."}
    if succeeded:
        manifest = {"schema": "jev.environment-media.v1", "source_spec_sha256": spec_sha256,
                    "files": {record["filename"]: record for record in records}}
        try:
            write_immutable(root / "MEDIA_MANIFEST.json", json_bytes(manifest))
        except Exception as error:
            report.update(status="FAILED", manifest_error={"type": type(error).__name__, "message": str(error)})
    report_name = started.strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}.json"
    report_path = reports / report_name
    write_immutable(report_path, json_bytes(report))
    report["report_path"] = str(report_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, help="media specification; defaults to requirements/training-media.json")
    parser.add_argument("--output", type=Path, required=True, help="task-owned artifact directory; supports verified reuse and range resume")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    try:
        specification = args.spec if args.spec is not None else default_spec()
        report = fetch_media(specification, args.output, args.workers, args.timeout)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAILED", "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({key: report[key] for key in ("status", "expected_files", "expected_bytes", "report_path")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
