"""Pinned environment media transfer checks; all HTTP responses are in memory."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
from jev.config import root

SPECIFICATION = root() / "requirements/training-media.json"


def load_downloader():
    name = "jev_environment_media_for_tests"
    source = REPOSITORY / "scripts/fetch_environment_media.py"
    specification = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class MemoryResponse:
    """A urllib-compatible response, optionally failing after one saved chunk."""

    def __init__(self, payload, *, status=200, headers=None, error_after_chunk=False):
        self.body = io.BytesIO(payload)
        self.status = status
        self.headers = dict(headers or {})
        self.error_after_chunk = error_after_chunk
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        self.body.close()

    def geturl(self):
        return "https://media.example.invalid/package.whl"

    def read(self, size=-1):
        if self.error_after_chunk and self.reads:
            raise OSError("simulated interrupted connection")
        self.reads += 1
        return self.body.read(size)


class MemoryOpener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        if not self.responses:
            raise AssertionError("unexpected network request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class EnvironmentMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.downloader = load_downloader()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.payload = b"a pinned environment wheel\x00with binary bytes\xff\n"
        self.record = {
            "filename": "package.whl",
            "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "public_url": "https://media.example.invalid/package.whl",
        }

    def partial(self):
        return self.folder / (self.record["filename"] + ".part")

    def sidecar(self):
        return self.folder / (self.record["filename"] + ".part.json")

    def seed_partial(self, payload):
        self.partial().write_bytes(payload)
        self.sidecar().write_text(json.dumps(self.record), encoding="utf-8")

    def write_spec(self, specification):
        path = self.folder / "SPEC.json"
        path.write_text(json.dumps(specification), encoding="utf-8")
        return path

    def test_default_spec_uses_adjacent_public_requirements_without_fetching(self):
        with patch.object(self.downloader, "fetch_media") as transfer:
            path = self.downloader.default_spec()
        self.assertEqual(path, REPOSITORY / "requirements/training-media.json")
        transfer.assert_not_called()

    def test_explicit_spec_never_resolves_implicit_resources(self):
        specification = self.folder / "custom-media.json"
        result = dict(status="PASS", expected_files=1, expected_bytes=17, report_path="receipt.json")
        with patch.object(self.downloader.sys, "argv", ["fetch_environment_media.py", "--spec", str(specification), "--output", str(self.folder / "media")]), \
                patch.object(self.downloader, "default_spec", side_effect=AssertionError("implicit resources must remain lazy")), \
                patch.object(self.downloader, "fetch_media", return_value=result) as transfer, \
                patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(self.downloader.main(), 0)
        self.assertEqual(transfer.call_args.args[0], specification)

    def test_exact_download_verifies_before_publishing(self):
        opener = MemoryOpener(MemoryResponse(self.payload))
        result = self.downloader.fetch_one(self.record, self.folder, timeout=17, opener=opener)
        self.assertEqual(result["status"], "DOWNLOADED")
        self.assertEqual(result["bytes"], len(self.payload))
        self.assertEqual(result["sha256"], self.record["sha256"])
        self.assertEqual(result["resumed_from"], 0)
        self.assertEqual((self.folder / self.record["filename"]).read_bytes(), self.payload)
        self.assertFalse(self.partial().exists())
        self.assertEqual(len(opener.requests), 1)
        request, timeout = opener.requests[0]
        self.assertEqual(request.full_url, self.record["public_url"])
        self.assertIsNone(request.get_header("Range"))
        self.assertEqual(timeout, 17)

    def test_verified_existing_payload_never_opens_network(self):
        complete = self.folder / self.record["filename"]
        complete.write_bytes(self.payload)
        result = self.downloader.fetch_one(self.record, self.folder, opener=MemoryOpener())
        self.assertEqual(result["status"], "VERIFIED_EXISTING")
        self.assertEqual(complete.read_bytes(), self.payload)
        self.assertFalse(self.partial().exists())

    def test_changed_complete_payload_is_preserved_and_refused(self):
        complete = self.folder / self.record["filename"]
        changed = bytes([self.payload[0] ^ 1]) + self.payload[1:]
        complete.write_bytes(changed)
        with self.assertRaises(ValueError):
            self.downloader.fetch_one(self.record, self.folder, opener=MemoryOpener())
        self.assertEqual(complete.read_bytes(), changed)
        self.assertFalse(self.partial().exists())

    def test_hash_mismatch_keeps_partial_without_publishing(self):
        changed = bytes([self.payload[0] ^ 1]) + self.payload[1:]
        opener = MemoryOpener(MemoryResponse(changed))
        with self.assertRaises(ValueError):
            self.downloader.fetch_one(self.record, self.folder, opener=opener)
        self.assertFalse((self.folder / self.record["filename"]).exists())
        self.assertEqual(self.partial().read_bytes(), changed)
        self.assertEqual(json.loads(self.sidecar().read_text(encoding="utf-8")), self.record)

    def test_interrupted_transfer_resumes_only_the_missing_suffix(self):
        offset = 13
        prefix = self.payload[:offset]
        interrupted = MemoryOpener(MemoryResponse(prefix, error_after_chunk=True))
        with self.assertRaises(OSError):
            self.downloader.fetch_one(self.record, self.folder, opener=interrupted)
        self.assertEqual(self.partial().read_bytes(), prefix)
        self.assertFalse((self.folder / self.record["filename"]).exists())
        self.assertEqual(json.loads(self.sidecar().read_text(encoding="utf-8")), self.record)

        resumed = MemoryOpener(MemoryResponse(
            self.payload[offset:], status=206,
            headers={"Content-Range": f"bytes {offset}-{len(self.payload) - 1}/{len(self.payload)}"},
        ))
        result = self.downloader.fetch_one(self.record, self.folder, opener=resumed)
        self.assertEqual(result["status"], "DOWNLOADED")
        self.assertEqual(result["resumed_from"], offset)
        self.assertEqual(resumed.requests[0][0].get_header("Range"), f"bytes={offset}-")
        self.assertEqual((self.folder / self.record["filename"]).read_bytes(), self.payload)
        self.assertFalse(self.partial().exists())

    def test_server_ignoring_range_cannot_replace_or_append_to_partial(self):
        prefix = self.payload[:11]
        self.seed_partial(prefix)
        opener = MemoryOpener(MemoryResponse(self.payload, status=200))
        with self.assertRaises(ValueError):
            self.downloader.fetch_one(self.record, self.folder, opener=opener)
        self.assertEqual(self.partial().read_bytes(), prefix)
        self.assertFalse((self.folder / self.record["filename"]).exists())

    def test_resume_rejects_wrong_content_range_without_appending(self):
        offset = 11
        prefix = self.payload[:offset]
        for content_range in (
            None,
            f"bytes 0-{len(self.payload) - 1}/{len(self.payload)}",
            f"bytes {offset}-{len(self.payload) - 1}/{len(self.payload) + 1}",
            "malformed",
        ):
            with self.subTest(content_range=content_range):
                self.seed_partial(prefix)
                headers = {} if content_range is None else {"Content-Range": content_range}
                opener = MemoryOpener(MemoryResponse(self.payload[offset:], status=206, headers=headers))
                with self.assertRaises(ValueError):
                    self.downloader.fetch_one(self.record, self.folder, opener=opener)
                self.assertEqual(self.partial().read_bytes(), prefix)
                self.assertFalse((self.folder / self.record["filename"]).exists())

    def test_partial_for_different_identity_is_preserved_and_refused(self):
        prefix = self.payload[:11]
        self.seed_partial(prefix)
        original = copy.deepcopy(self.record)
        original["public_url"] = "https://different.example.invalid/package.whl"
        self.sidecar().write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.downloader.fetch_one(self.record, self.folder, opener=MemoryOpener())
        self.assertEqual(self.partial().read_bytes(), prefix)
        self.assertEqual(json.loads(self.sidecar().read_text(encoding="utf-8")), original)

    def test_download_name_must_be_a_plain_basename(self):
        for filename in ("../escape.whl", "..\\escape.whl", "/escape.whl", "subdir/package.whl",
                         "C:\\escape.whl", ".", ""):
            with self.subTest(filename=filename):
                record = dict(self.record, filename=filename)
                with self.assertRaises(ValueError):
                    self.downloader.fetch_one(record, self.folder, opener=MemoryOpener())
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_public_url_must_be_absolute_https_without_credentials(self):
        for public_url in ("http://media.example.invalid/file", "ftp://media.example.invalid/file",
                           "https:///file", "/relative/file", "https://user:secret@media.example.invalid/file",
                           "https://user@media.example.invalid/file", ""):
            with self.subTest(public_url=public_url):
                record = dict(self.record, public_url=public_url)
                with self.assertRaises(ValueError):
                    self.downloader.fetch_one(record, self.folder, opener=MemoryOpener())
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_recorded_template_covers_all_86_media_payloads(self):
        records = self.downloader.load_media_spec(SPECIFICATION)
        specification = json.loads(SPECIFICATION.read_text(encoding="utf-8"))
        expected = specification
        self.assertEqual(len(records), 86)
        self.assertEqual(len({record["filename"] for record in records}), 86)
        fields = ("filename", "bytes", "sha256", "public_url")
        self.assertEqual(
            {record["filename"]: {key: record[key] for key in fields} for record in records},
            {record["filename"]: {key: record[key] for key in fields} for record in expected},
        )

    def test_invalid_manifest_is_rejected_before_any_download(self):
        original = json.loads(SPECIFICATION.read_text(encoding="utf-8"))
        for key, value in (("filename", "../escape.whl"), ("public_url", "https://u:dummy_password@example.invalid/file"),
                           ("sha256", "not-a-digest"), ("bytes", -1)):
            with self.subTest(key=key):
                specification = copy.deepcopy(original)
                specification[0][key] = value
                with self.assertRaises(ValueError):
                    self.downloader.load_media_spec(self.write_spec(specification))

    def test_fetch_media_reports_a_failed_file_and_keeps_successful_results(self):
        other = dict(self.record, filename="second.whl")
        spec_path = self.write_spec({"testing": "manifest loading is patched below"})

        def transfer(record, root, timeout=60):
            if record["filename"] == "second.whl":
                raise OSError("simulated network failure")
            return dict(record, status="VERIFIED_EXISTING", resumed_from=0)

        with patch.object(self.downloader, "load_media_spec", return_value=[self.record, other]), \
             patch.object(self.downloader, "fetch_one", side_effect=transfer):
            report = self.downloader.fetch_media(spec_path, self.folder / "media", workers=2)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(len(report["files"]), 2)
        self.assertEqual({record["filename"] for record in report["files"]}, {"package.whl", "second.whl"})
        self.assertEqual({record["status"] for record in report["files"]}, {"VERIFIED_EXISTING", "FAILED"})
        reports = list((self.folder / "media" / ".fetch-reports").glob("*.json"))
        self.assertEqual(len(reports), 1)
        persisted = json.loads(reports[0].read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "FAILED")
        self.assertEqual(persisted["files"], report["files"])


if __name__ == "__main__":
    unittest.main()
