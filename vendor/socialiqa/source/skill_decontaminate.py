#!/usr/bin/env python3
"""Deterministic train-eval decontamination for canonical SFT releases.

The implementation deliberately keeps large-scale matching statistical.  It
never calls an LLM and never edits a source release or evaluation asset.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import math
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


TOOL_VERSION = "dataset-decontamination/1.4.0"
REGISTRY_SCHEMA = "decontamination-eval-registry/v1"
SCAN_SCHEMA = "decontamination-scan/v1"
RELEASE_SCHEMA = "decontaminated-canonical-release/v1"
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\uff02": '"',
        "\uff07": "'",
    }
)
TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==|!=|<=|>=|->|=>|::|"
    r"&&|\|\||\+\+|--|<<|>>|[\u3400-\u4dbf\u4e00-\u9fff]|[^\s]",
    re.UNICODE,
)
ALLOWED_EVAL_FIELDS = ("question", "answer", "rationale", "test", "choices")
ALLOWED_REVIEW_DECISIONS = {"remove", "keep", "uncertain"}


class DecontaminationError(RuntimeError):
    """Expected fail-closed validation error."""


_FORK_INDEX: "LexicalIndex | None" = None
_FORK_RED_THRESHOLD = 0.80
_FORK_ORANGE_THRESHOLD = 0.20
_FORK_FILE_DESCRIPTORS: dict[str, int] = {}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_facts(path: Path, *, count_lines: bool = True) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    lines = 0
    last = b""
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
            size += len(chunk)
            if count_lines:
                lines += chunk.count(b"\n")
                last = chunk[-1:]
    if count_lines and size and last != b"\n":
        lines += 1
    result: dict[str, Any] = {"bytes": size, "sha256": digest.hexdigest()}
    if count_lines:
        result["rows"] = lines
    return result


def normalize_conservative(value: str) -> str:
    value = html.unescape(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(ZERO_WIDTH).translate(QUOTE_TRANSLATION).casefold()
    return " ".join(value.split())


def tokenize(value: str) -> list[str]:
    return TOKEN_RE.findall(normalize_conservative(value))


def shingle_digest(tokens: Sequence[str]) -> bytes:
    payload = "\x1f".join(tokens).encode("utf-8", errors="strict")
    return hashlib.blake2b(payload, digest_size=16, person=b"decontam-v1").digest()


def shingle_set(tokens: Sequence[str], width: int) -> set[bytes]:
    if len(tokens) < width:
        return set()
    return {shingle_digest(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def json_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return canonical_json_bytes(value).decode("utf-8")
    return str(value)


def dotted_get(record: Mapping[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def require_keys(record: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in record]
    if missing:
        raise DecontaminationError(f"{context}: missing keys {missing}")


def require_new_directory_target(path: Path) -> Path:
    path = path.resolve()
    if path.parent == path:
        raise DecontaminationError(f"output must not be a filesystem root: {path}")
    if path.exists():
        raise DecontaminationError(f"output already exists: {path}")
    incomplete = path.with_name(path.name + ".incomplete")
    if incomplete.exists():
        raise DecontaminationError(f"incomplete output already exists: {incomplete}")
    return incomplete


def safe_new_directory(path: Path) -> Path:
    incomplete = require_new_directory_target(path)
    incomplete.mkdir(parents=True)
    return incomplete


def finish_directory(incomplete: Path, final: Path) -> None:
    os.replace(incomplete, final.resolve())


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any], bytes]]:
    with path.open("rb") as handle:
        for line_number, raw_with_newline in enumerate(handle, 1):
            raw = raw_with_newline.rstrip(b"\r\n")
            if not raw:
                raise DecontaminationError(f"{path}:{line_number}: blank JSONL row")
            value = parse_json_object(raw, f"{path}:{line_number}")
            yield line_number, value, raw


def parse_json_object(raw: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DecontaminationError(
                    f"{context}: non-standard JSON constant {token!r}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecontaminationError(f"{context}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise DecontaminationError(f"{context}: row is not an object")
    return value


def iter_eval_records(path: Path, format_name: str) -> Iterator[dict[str, Any]]:
    if format_name == "jsonl":
        for _, record, _ in iter_jsonl(path):
            yield record
        return
    if format_name != "parquet":
        raise DecontaminationError(f"unsupported eval format: {format_name}")
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as error:
        raise DecontaminationError("Parquet input requires pyarrow") from error
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1024):
        for record in batch.to_pylist():
            if not isinstance(record, dict):
                raise DecontaminationError(f"{path}: Parquet row is not an object")
            yield record


def resolve_config_path(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve()


def command_build_registry(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DecontaminationError(f"cannot read eval config {config_path}: {error}") from error
    if not isinstance(config, dict) or not isinstance(config.get("assets"), list) or not config["assets"]:
        raise DecontaminationError("eval config must contain a non-empty assets list")

    output = Path(args.out).resolve()
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    manifest_assets: list[dict[str, Any]] = []
    try:
        for asset_index, asset in enumerate(config["assets"]):
            context = f"assets[{asset_index}]"
            if not isinstance(asset, dict):
                raise DecontaminationError(f"{context}: asset is not an object")
            require_keys(
                asset,
                ("benchmark", "split", "revision", "path", "sha256", "format", "id_field", "fields"),
                context,
            )
            if not isinstance(asset["fields"], dict) or "question" not in asset["fields"]:
                raise DecontaminationError(f"{context}: fields must contain question")
            unknown_fields = set(asset["fields"]) - set(ALLOWED_EVAL_FIELDS)
            if unknown_fields:
                raise DecontaminationError(f"{context}: unsupported canonical fields {sorted(unknown_fields)}")
            source_path = resolve_config_path(config_path, str(asset["path"]))
            if not source_path.is_file():
                raise DecontaminationError(f"{context}: source is not a file: {source_path}")
            actual_sha = sha256_file(source_path)
            expected_sha = str(asset["sha256"]).lower()
            if actual_sha != expected_sha:
                raise DecontaminationError(
                    f"{context}: SHA-256 mismatch for {source_path}: expected {expected_sha}, got {actual_sha}"
                )
            count = 0
            field_coverage: collections.Counter[str] = collections.Counter()
            for row_number, source_record in enumerate(
                iter_eval_records(source_path, str(asset["format"])), 1
            ):
                raw_id = dotted_get(source_record, str(asset["id_field"]))
                if raw_id is None or not str(raw_id).strip():
                    raise DecontaminationError(f"{context} row {row_number}: empty eval item ID")
                item_id = str(raw_id)
                identity = (str(asset["benchmark"]), str(asset["split"]), item_id)
                if identity in seen:
                    raise DecontaminationError(f"duplicate eval identity: {identity}")
                seen.add(identity)
                fields: dict[str, str] = {}
                for canonical_field in ALLOWED_EVAL_FIELDS:
                    source_fields = asset["fields"].get(canonical_field, [])
                    if isinstance(source_fields, str):
                        source_fields = [source_fields]
                    if not isinstance(source_fields, list):
                        raise DecontaminationError(
                            f"{context}: field mapping for {canonical_field} must be a string or list"
                        )
                    parts = [
                        json_value_to_text(dotted_get(source_record, str(field_path)))
                        for field_path in source_fields
                    ]
                    fields[canonical_field] = normalize_conservative("\n".join(p for p in parts if p))
                    if fields[canonical_field]:
                        field_coverage[canonical_field] += 1
                if not fields["question"]:
                    raise DecontaminationError(
                        f"{context} row {row_number}: required question field resolved to empty text"
                    )
                items.append(
                    {
                        "benchmark": str(asset["benchmark"]),
                        "split": str(asset["split"]),
                        "revision": str(asset["revision"]),
                        "item_id": item_id,
                        "fields": fields,
                    }
                )
                count += 1
            if count == 0:
                raise DecontaminationError(f"{context}: eval asset contains zero items")
            manifest_assets.append(
                {
                    "benchmark": str(asset["benchmark"]),
                    "split": str(asset["split"]),
                    "revision": str(asset["revision"]),
                    "source_name": source_path.name,
                    "source_sha256": actual_sha,
                    "format": str(asset["format"]),
                    "id_field": str(asset["id_field"]),
                    "fields": asset["fields"],
                    "license": str(asset.get("license", "unknown")),
                    "protocol": str(asset.get("protocol", "unspecified")),
                    "items": count,
                    "field_coverage": {
                        field: int(field_coverage[field]) for field in ALLOWED_EVAL_FIELDS
                    },
                }
            )

        if not items:
            raise DecontaminationError("eval registry contains zero items")
        incomplete = safe_new_directory(output)
        items.sort(key=lambda row: (row["benchmark"], row["split"], row["item_id"]))
        items_path = incomplete / "items.jsonl"
        with items_path.open("wb") as handle:
            for item in items:
                handle.write(canonical_json_bytes(item) + b"\n")
        manifest = {
            "schema": REGISTRY_SCHEMA,
            "tool_version": TOOL_VERSION,
            "assets": sorted(
                manifest_assets,
                key=lambda row: (row["benchmark"], row["split"], row["revision"]),
            ),
            "items": len(items),
            "items_file": file_facts(items_path),
        }
        write_json(incomplete / "MANIFEST.json", manifest)
        finish_directory(incomplete, output)
    except Exception:
        # Preserve the incomplete directory for forensic inspection; never erase it silently.
        raise
    print(json.dumps({"registry": str(output), "items": len(items)}, sort_keys=True))


def load_registry(
    path: Path, shingle_width: int, cjk_shingle_width: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = path / "MANIFEST.json"
    items_path = path / "items.jsonl"
    if not manifest_path.is_file() or not items_path.is_file():
        raise DecontaminationError(f"incomplete registry: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != REGISTRY_SCHEMA:
        raise DecontaminationError(f"unsupported registry schema: {manifest.get('schema')}")
    facts = file_facts(items_path)
    if facts != manifest.get("items_file"):
        raise DecontaminationError("registry items.jsonl does not match its manifest")
    items: list[dict[str, Any]] = []
    for _, item, _ in iter_jsonl(items_path):
        fields = item.get("fields")
        if not isinstance(fields, dict):
            raise DecontaminationError("registry item has invalid fields")
        question_text = "\n".join(
            str(fields.get(field, "")) for field in ("question", "choices") if fields.get(field)
        )
        answer_text = "\n".join(
            str(fields.get(field, "")) for field in ("answer", "rationale") if fields.get(field)
        )
        # Registry fields were normalized exactly once by build-registry.  Do
        # not call tokenize(), whose normalizer is intentionally non-idempotent
        # for nested HTML entities.
        item["_q_tokens"] = TOKEN_RE.findall(question_text)
        item["_a_tokens"] = TOKEN_RE.findall(answer_text)
        item["_test_tokens"] = TOKEN_RE.findall(str(fields.get("test", "")))
        item["_qa_tokens"] = item["_q_tokens"] + item["_a_tokens"]
        cjk_count = sum(1 for char in question_text if "\u3400" <= char <= "\u9fff")
        item["_is_cjk"] = (
            cjk_count >= cjk_shingle_width and cjk_count * 2 >= max(1, len(item["_q_tokens"]))
        )
        base_width = cjk_shingle_width if item["_is_cjk"] else shingle_width

        # Only questions get an exact short-field fallback. Short answer/test
        # values such as "A" or "72" are far too collision-prone even for an
        # orange candidate, and q+a must not inherit the short question width.
        item["_q_shingle_width"] = (
            min(base_width, len(item["_q_tokens"]))
            if item["_q_tokens"]
            else base_width
        )
        item["_a_shingle_width"] = base_width
        item["_test_shingle_width"] = base_width
        item["_qa_shingle_width"] = base_width
        item["_short_field_fallback"] = item["_q_shingle_width"] < base_width
        item["_long_text_threshold"] = 48 if item["_is_cjk"] else 32
        eval_question_cjk_chars = sum(
            1 for char in str(fields.get("question", "")) if "\u3400" <= char <= "\u9fff"
        )
        item["_weak_key_long_enough"] = (
            len(item["_q_tokens"]) >= 16 or eval_question_cjk_chars >= 24
        )
        items.append(item)
    if len(items) != manifest.get("items"):
        raise DecontaminationError("registry item count does not match its manifest")
    if not items:
        raise DecontaminationError("registry contains zero items")
    return items, manifest


def estimate_index_postings(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    def window_count(tokens: Sequence[str], width: int) -> int:
        return max(0, len(tokens) - width + 1)

    # This deliberately counts windows rather than unique per-item digests, so it
    # is a safe pre-allocation upper bound and can reject before shingle sets exist.
    question = sum(
        window_count(item["_q_tokens"], int(item["_q_shingle_width"])) for item in items
    )
    answer = sum(
        window_count(item["_a_tokens"], int(item["_a_shingle_width"])) for item in items
    )
    test = sum(
        window_count(item["_test_tokens"], int(item["_test_shingle_width"])) for item in items
    )
    question_answer = sum(
        window_count(item["_qa_tokens"], int(item["_qa_shingle_width"])) for item in items
    )
    long_text = sum(
        max(0, len(item["_q_tokens"]) - int(item["_long_text_threshold"]) + 1)
        for item in items
    )
    long_code = sum(
        max(0, len(item["_a_tokens"]) - 50 + 1)
        + max(0, len(item["_test_tokens"]) - 50 + 1)
        for item in items
    )
    total = question + answer + test + question_answer + long_text + long_code
    return {
        "question": question,
        "answer": answer,
        "test": test,
        "question_answer": question_answer,
        "long_text": long_text,
        "long_code": long_code,
        "total": total,
        "estimated_python_bytes_upper_bound": total * 192,
    }


class LexicalIndex:
    def __init__(self, items: list[dict[str, Any]], width: int, cjk_width: int):
        self.items = items
        self.width = width
        self.cjk_width = cjk_width
        width_fields = (
            "_q_shingle_width",
            "_a_shingle_width",
            "_test_shingle_width",
            "_qa_shingle_width",
        )
        self.widths = sorted({int(item[field]) for item in items for field in width_fields})
        self.q_item_widths = [int(item["_q_shingle_width"]) for item in items]
        self.a_item_widths = [int(item["_a_shingle_width"]) for item in items]
        self.test_item_widths = [int(item["_test_shingle_width"]) for item in items]
        self.qa_item_widths = [int(item["_qa_shingle_width"]) for item in items]
        self.q_widths = sorted(set(self.q_item_widths))
        self.a_widths = sorted(set(self.a_item_widths))
        self.test_widths = sorted(set(self.test_item_widths))
        self.qa_widths = sorted(set(self.qa_item_widths))
        self.high_frequency_cutoff = max(100, int(1e-5 * len(items)))
        self.q_inv: dict[tuple[int, bytes], list[int]] = collections.defaultdict(list)
        self.a_inv: dict[tuple[int, bytes], list[int]] = collections.defaultdict(list)
        self.test_inv: dict[tuple[int, bytes], list[int]] = collections.defaultdict(list)
        self.qa_inv: dict[tuple[int, bytes], list[int]] = collections.defaultdict(list)
        self.q_weights: list[dict[bytes, float]] = []
        self.a_weights: list[dict[bytes, float]] = []
        self.test_weights: list[dict[bytes, float]] = []
        self.qa_weights: list[dict[bytes, float]] = []
        self.q_denominators: list[float] = []
        self.a_denominators: list[float] = []
        self.test_denominators: list[float] = []
        self.qa_denominators: list[float] = []
        self.ids: dict[str, list[int]] = collections.defaultdict(list)
        self.long_text: dict[tuple[int, bytes], list[tuple[int, int]]] = collections.defaultdict(list)
        self.long_code: dict[bytes, list[tuple[int, str, str, int]]] = collections.defaultdict(list)

        for item in items:
            item["_q_shingles"] = shingle_set(
                item["_q_tokens"], int(item["_q_shingle_width"])
            )
            item["_a_shingles"] = shingle_set(
                item["_a_tokens"], int(item["_a_shingle_width"])
            )
            item["_test_shingles"] = shingle_set(
                item["_test_tokens"], int(item["_test_shingle_width"])
            )
            item["_qa_shingles"] = shingle_set(
                item["_qa_tokens"], int(item["_qa_shingle_width"])
            )

        q_df: collections.Counter[bytes] = collections.Counter()
        a_df: collections.Counter[bytes] = collections.Counter()
        test_df: collections.Counter[bytes] = collections.Counter()
        qa_df: collections.Counter[bytes] = collections.Counter()
        for item in items:
            q_df.update(item["_q_shingles"])
            a_df.update(item["_a_shingles"])
            test_df.update(item["_test_shingles"])
            qa_df.update(item["_qa_shingles"])
        for index, item in enumerate(items):
            self.ids[str(item["item_id"])].append(index)
            q_weight = self._weights(item["_q_shingles"], q_df)
            a_weight = self._weights(item["_a_shingles"], a_df)
            test_weight = self._weights(item["_test_shingles"], test_df)
            qa_weight = self._weights(item["_qa_shingles"], qa_df)
            self.q_weights.append(q_weight)
            self.a_weights.append(a_weight)
            self.test_weights.append(test_weight)
            self.qa_weights.append(qa_weight)
            self.q_denominators.append(math.fsum(q_weight[digest] for digest in sorted(q_weight)))
            self.a_denominators.append(math.fsum(a_weight[digest] for digest in sorted(a_weight)))
            self.test_denominators.append(
                math.fsum(test_weight[digest] for digest in sorted(test_weight))
            )
            self.qa_denominators.append(math.fsum(qa_weight[digest] for digest in sorted(qa_weight)))
            for digest in q_weight:
                self.q_inv[(self.q_item_widths[index], digest)].append(index)
            for digest in a_weight:
                self.a_inv[(self.a_item_widths[index], digest)].append(index)
            for digest in test_weight:
                self.test_inv[(self.test_item_widths[index], digest)].append(index)
            for digest in qa_weight:
                self.qa_inv[(self.qa_item_widths[index], digest)].append(index)
            q_tokens = item["_q_tokens"]
            long_text_threshold = int(item["_long_text_threshold"])
            if len(q_tokens) >= long_text_threshold:
                for position in range(len(q_tokens) - long_text_threshold + 1):
                    digest = shingle_digest(q_tokens[position : position + long_text_threshold])
                    self.long_text[(long_text_threshold, digest)].append((index, position))
            for label, attr, tokens in (
                ("answer", "_a_tokens", item["_a_tokens"]),
                ("test", "_test_tokens", item["_test_tokens"]),
            ):
                if len(tokens) >= 50:
                    for position in range(len(tokens) - 50 + 1):
                        digest = shingle_digest(tokens[position : position + 50])
                        self.long_code[digest].append((index, label, attr, position))

    def _weights(
        self, shingles: set[bytes], document_frequency: Mapping[bytes, int]
    ) -> dict[bytes, float]:
        result: dict[bytes, float] = {}
        item_count = len(self.items)
        for digest in sorted(shingles):
            df = int(document_frequency[digest])
            if df > self.high_frequency_cutoff:
                continue
            result[digest] = math.log((item_count + 1) / (df + 1)) + 1.0
        return result

    @staticmethod
    def _candidate_scores(
        train_shingles_by_width: Mapping[int, set[bytes]],
        inverse: Mapping[tuple[int, bytes], Sequence[int]],
        weights: Sequence[Mapping[bytes, float]],
        denominators: Sequence[float],
        item_widths: Sequence[int],
    ) -> dict[int, tuple[float, int]]:
        candidates: set[int] = set()
        for width, train_shingles in train_shingles_by_width.items():
            for digest in train_shingles:
                candidates.update(inverse.get((width, digest), ()))
        result: dict[int, tuple[float, int]] = {}
        for index in candidates:
            train_shingles = train_shingles_by_width[item_widths[index]]
            shared = train_shingles.intersection(weights[index])
            denominator = denominators[index]
            score = (
                math.fsum(weights[index][digest] for digest in sorted(shared)) / denominator
                if denominator
                else 0.0
            )
            result[index] = (score, len(shared))
        return result

    @staticmethod
    def sets_by_width(
        tokens: Sequence[str], widths: Iterable[int]
    ) -> dict[int, set[bytes]]:
        return {width: shingle_set(tokens, width) for width in widths}

    def q_scores(
        self, train_sets: Mapping[int, set[bytes]]
    ) -> dict[int, tuple[float, int]]:
        return self._candidate_scores(
            train_sets,
            self.q_inv,
            self.q_weights,
            self.q_denominators,
            self.q_item_widths,
        )

    def qa_scores(
        self, train_sets: Mapping[int, set[bytes]]
    ) -> dict[int, tuple[float, int]]:
        return self._candidate_scores(
            train_sets,
            self.qa_inv,
            self.qa_weights,
            self.qa_denominators,
            self.qa_item_widths,
        )

    def answer_scores(
        self, train_sets: Mapping[int, set[bytes]]
    ) -> dict[int, tuple[float, int]]:
        return self._candidate_scores(
            train_sets,
            self.a_inv,
            self.a_weights,
            self.a_denominators,
            self.a_item_widths,
        )

    def test_scores(
        self, train_sets: Mapping[int, set[bytes]]
    ) -> dict[int, tuple[float, int]]:
        return self._candidate_scores(
            train_sets,
            self.test_inv,
            self.test_weights,
            self.test_denominators,
            self.test_item_widths,
        )

    def long_matches(
        self,
        question_tokens: Sequence[str],
        answer_tokens: Sequence[str],
        observation_tokens: Sequence[str],
    ) -> dict[int, tuple[int, str]]:
        result: dict[int, tuple[int, str]] = {}
        field_priority = {
            "question": 5,
            "answer": 4,
            "test": 3,
            "observation_answer": 2,
            "observation_test": 1,
        }

        def record(item_index: int, threshold: int, field: str) -> None:
            current = result.get(item_index)
            if current is None or (field_priority[field], threshold) > (
                field_priority[current[1]],
                current[0],
            ):
                result[item_index] = (threshold, field)

        for threshold in sorted({key[0] for key in self.long_text}):
            if len(question_tokens) < threshold:
                continue
            for position in range(len(question_tokens) - threshold + 1):
                window = question_tokens[position : position + threshold]
                digest = shingle_digest(window)
                for item_index, eval_position in self.long_text.get((threshold, digest), ()):
                    expected = self.items[item_index]["_q_tokens"][eval_position : eval_position + threshold]
                    if list(window) == expected:
                        record(item_index, threshold, "question")
        for channel, tokens in (("", answer_tokens), ("observation_", observation_tokens)):
            if len(tokens) < 50:
                continue
            for position in range(len(tokens) - 50 + 1):
                window = tokens[position : position + 50]
                digest = shingle_digest(window)
                for item_index, label, attr, eval_position in self.long_code.get(digest, ()):
                    expected = self.items[item_index][attr][eval_position : eval_position + 50]
                    if list(window) == expected:
                        record(item_index, 50, channel + label)
        return result

    def stats(self) -> dict[str, int]:
        return {
            "question_postings": sum(len(value) for value in self.q_inv.values()),
            "answer_postings": sum(len(value) for value in self.a_inv.values()),
            "test_postings": sum(len(value) for value in self.test_inv.values()),
            "question_answer_postings": sum(len(value) for value in self.qa_inv.values()),
            "long_text_postings": sum(len(value) for value in self.long_text.values()),
            "long_code_postings": sum(len(value) for value in self.long_code.values()),
        }


def load_provenance(release: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], collections.deque[dict[str, Any]]]]:
    ledger_path = release / "ledgers" / "provenance.jsonl"
    if not ledger_path.is_file():
        raise DecontaminationError(f"missing provenance ledger: {ledger_path}")
    records: list[dict[str, Any]] = []
    mapping: dict[tuple[str, str], collections.deque[dict[str, Any]]] = collections.defaultdict(collections.deque)
    for line_number, record, _ in iter_jsonl(ledger_path):
        require_keys(record, ("stable_key", "split", "canonical_row_sha256"), f"{ledger_path}:{line_number}")
        stable_key = str(record["stable_key"])
        split = str(record["split"])
        row_sha = str(record["canonical_row_sha256"])
        records.append(record)
        mapping[(split, row_sha)].append(record)
    return records, mapping


def source_splits(release: Path) -> list[tuple[str, Path]]:
    messages_dir = release / "messages"
    if not messages_dir.is_dir():
        raise DecontaminationError(f"missing messages directory: {messages_dir}")
    splits = [(path.stem, path) for path in sorted(messages_dir.glob("*.jsonl"))]
    if not splits:
        raise DecontaminationError(f"no message split JSONL files in {messages_dir}")
    return splits


def message_text_fields(row: Mapping[str, Any], context: str) -> tuple[str, str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise DecontaminationError(f"{context}: messages is not a list")
    question: list[str] = []
    answer: list[str] = []
    auxiliary: list[str] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise DecontaminationError(f"{context}: message {message_index} is not an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise DecontaminationError(f"{context}: message {message_index} lacks string role/content")
        if role == "user":
            question.append(content)
        elif role in {"assistant", "tool_call"}:
            answer.append(content)
        elif role in {"system", "developer", "tool", "tool_response", "observation"}:
            auxiliary.append(content)
        else:
            raise DecontaminationError(
                f"{context}: message {message_index} has unsupported role {role!r}"
            )
    return "\n".join(question), "\n".join(answer), "\n".join(auxiliary)


def provenance_item_ids(record: Mapping[str, Any]) -> set[str]:
    candidates: set[str] = set()
    for key in ("item_id", "instance_id"):
        value = record.get(key)
        if not isinstance(value, str) or not value:
            continue
        candidates.add(value)
    lineage = record.get("lineage_group_id")
    if isinstance(lineage, str) and "\x00" in lineage:
        suffix = lineage.rsplit("\x00", 1)[-1]
        if suffix:
            candidates.add(suffix)
    metadata = record.get("source_metadata")
    if isinstance(metadata, dict):
        for key in ("item_id", "instance_id", "task_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                candidates.add(value)
    # Bare numeric/very short IDs are unsafe across benchmark namespaces.
    return {value for value in candidates if len(value) >= 8 and not value.isdecimal()}


def make_match(
    metadata: Mapping[str, Any],
    item: Mapping[str, Any],
    zone: str,
    rule: str,
    q_score: float,
    qa_score: float,
    answer_score: float,
    test_score: float,
    auxiliary_qa_score: float,
    context_score: float,
    shared: int,
    long_span: int,
    long_field: str,
    normalized_question: str,
    auxiliary_answer_score: float,
    auxiliary_test_score: float,
) -> dict[str, Any]:
    return {
        "stable_key": str(metadata["stable_key"]),
        "split": str(metadata["split"]),
        "canonical_row_sha256": str(metadata["canonical_row_sha256"]),
        "benchmark": str(item["benchmark"]),
        "eval_split": str(item["split"]),
        "item_id": str(item["item_id"]),
        "zone": zone,
        "rule": rule,
        "scores": {
            "question_containment": round(q_score, 8),
            "question_answer_containment": round(qa_score, 8),
            "answer_containment": round(answer_score, 8),
            "test_containment": round(test_score, 8),
            "auxiliary_question_answer_containment": round(auxiliary_qa_score, 8),
            "auxiliary_answer_containment": round(auxiliary_answer_score, 8),
            "auxiliary_test_containment": round(auxiliary_test_score, 8),
            "context_question_containment": round(context_score, 8),
            "shared_nonfrequent_shingles": shared,
            "long_exact_tokens": long_span,
            "long_exact_field": long_field,
        },
        "train_excerpt": normalized_question[:1200],
        "eval_excerpt": str(item["fields"].get("question", ""))[:1200],
    }


def evaluate_row(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
    index: LexicalIndex,
    red_threshold: float,
    orange_threshold: float,
) -> list[dict[str, Any]]:
    question, answer, auxiliary = message_text_fields(row, str(metadata["stable_key"]))
    question_tokens = tokenize(question)
    answer_tokens = tokenize(answer)
    auxiliary_tokens = tokenize(auxiliary)
    normalized_question = normalize_conservative(question)
    qa_tokens = question_tokens + answer_tokens
    question_sets = index.sets_by_width(question_tokens, index.q_widths)
    answer_sets = index.sets_by_width(
        answer_tokens, sorted(set(index.a_widths) | set(index.test_widths))
    )
    auxiliary_sets = index.sets_by_width(auxiliary_tokens, index.widths)
    qa_sets = index.sets_by_width(qa_tokens, index.qa_widths)
    q_scores = index.q_scores(question_sets)
    qa_scores = index.qa_scores(qa_sets)
    answer_scores = index.answer_scores(answer_sets)
    test_scores = index.test_scores(answer_sets)
    auxiliary_qa_scores = index.qa_scores(auxiliary_sets)
    auxiliary_answer_scores = index.answer_scores(auxiliary_sets)
    auxiliary_test_scores = index.test_scores(auxiliary_sets)
    context_scores = index.q_scores(auxiliary_sets)
    long_matches = index.long_matches(question_tokens, answer_tokens, auxiliary_tokens)
    identity_indexes: set[int] = set()
    for candidate_id in provenance_item_ids(metadata):
        resolved = index.ids.get(candidate_id, ())
        identity_indexes.update(resolved)
    candidate_indexes = (
        set(q_scores)
        | set(qa_scores)
        | set(answer_scores)
        | set(test_scores)
        | set(auxiliary_qa_scores)
        | set(auxiliary_answer_scores)
        | set(auxiliary_test_scores)
        | set(context_scores)
        | set(long_matches)
        | identity_indexes
    )
    matches: list[dict[str, Any]] = []
    for item_index in sorted(candidate_indexes):
        item = index.items[item_index]
        q_score, q_shared = q_scores.get(item_index, (0.0, 0))
        qa_score, qa_shared = qa_scores.get(item_index, (0.0, 0))
        answer_score, answer_shared = answer_scores.get(item_index, (0.0, 0))
        test_score, test_shared = test_scores.get(item_index, (0.0, 0))
        auxiliary_qa_score, auxiliary_qa_shared = auxiliary_qa_scores.get(
            item_index, (0.0, 0)
        )
        auxiliary_answer_score, auxiliary_answer_shared = auxiliary_answer_scores.get(
            item_index, (0.0, 0)
        )
        auxiliary_test_score, auxiliary_test_shared = auxiliary_test_scores.get(
            item_index, (0.0, 0)
        )
        context_score, context_shared = context_scores.get(item_index, (0.0, 0))
        long_span, long_field = long_matches.get(item_index, (0, ""))
        zone = ""
        rule = ""
        nonauxiliary_score = max(q_score, qa_score, answer_score, test_score)
        auxiliary_score = max(
            context_score,
            auxiliary_qa_score,
            auxiliary_answer_score,
            auxiliary_test_score,
        )
        weak_key_long_enough = bool(item["_weak_key_long_enough"])
        if item_index in identity_indexes:
            zone, rule = "red", "exact_lineage_item_id"
        elif q_score >= red_threshold and q_shared >= 2 and weak_key_long_enough:
            zone, rule = "red", "question_containment"
        elif (
            qa_score >= red_threshold
            and qa_shared >= 2
            and weak_key_long_enough
        ):
            zone, rule = "red", "question_answer_containment"
        elif long_span:
            if nonauxiliary_score >= orange_threshold:
                zone, rule = (
                    "orange",
                    "answer_or_test_candidate"
                    if max(answer_score, test_score) >= orange_threshold
                    else "lexical_candidate",
                )
            else:
                zone, rule = "yellow", f"isolated_long_exact_{long_field}"
        elif nonauxiliary_score >= orange_threshold:
            zone, rule = (
                "orange",
                "answer_or_test_candidate"
                if max(answer_score, test_score) >= orange_threshold
                else "lexical_candidate",
            )
        elif auxiliary_score >= orange_threshold:
            zone, rule = "yellow", "context_only"
        if zone:
            if rule == "question_containment":
                shared = q_shared
            elif rule == "question_answer_containment":
                shared = qa_shared
            elif rule == "answer_or_test_candidate":
                shared = max(answer_shared, test_shared)
            elif rule == "lexical_candidate":
                shared = max(q_shared, qa_shared)
            elif rule == "context_only":
                shared = max(
                    context_shared,
                    auxiliary_qa_shared,
                    auxiliary_answer_shared,
                    auxiliary_test_shared,
                )
            elif long_field == "question":
                shared = q_shared
            elif long_field == "answer":
                shared = answer_shared
            elif long_field == "test":
                shared = test_shared
            elif long_field == "observation_answer":
                shared = auxiliary_answer_shared
            elif long_field == "observation_test":
                shared = auxiliary_test_shared
            else:
                shared = 0
            matches.append(
                make_match(
                    metadata,
                    item,
                    zone,
                    rule,
                    q_score,
                    qa_score,
                    answer_score,
                    test_score,
                    auxiliary_qa_score,
                    context_score,
                    shared,
                    long_span,
                    long_field,
                    normalized_question,
                    auxiliary_answer_score,
                    auxiliary_test_score,
                )
            )
    return matches


def initialize_fork_worker(red_threshold: float, orange_threshold: float) -> None:
    import gc

    global _FORK_RED_THRESHOLD, _FORK_ORANGE_THRESHOLD, _FORK_FILE_DESCRIPTORS
    if _FORK_INDEX is None:
        raise DecontaminationError("fork worker did not inherit the lexical index")
    _FORK_RED_THRESHOLD = red_threshold
    _FORK_ORANGE_THRESHOLD = orange_threshold
    _FORK_FILE_DESCRIPTORS = {}
    gc.enable()


def evaluate_offset_task(
    task: tuple[int, str, int, int, int, dict[str, Any]]
) -> tuple[int, str, list[dict[str, Any]]]:
    sequence, path_text, offset, length, line_number, metadata = task
    if _FORK_INDEX is None:
        raise DecontaminationError("parallel scan worker has no lexical index")
    descriptor = _FORK_FILE_DESCRIPTORS.get(path_text)
    if descriptor is None:
        descriptor = os.open(path_text, os.O_RDONLY)
        _FORK_FILE_DESCRIPTORS[path_text] = descriptor
    raw_with_newline = os.pread(descriptor, length, offset)
    if len(raw_with_newline) != length:
        raise DecontaminationError(
            f"{path_text}:{line_number}: short pread: expected {length}, got {len(raw_with_newline)}"
        )
    raw = raw_with_newline.rstrip(b"\r\n")
    if not raw:
        raise DecontaminationError(f"{path_text}:{line_number}: blank JSONL row")
    row = parse_json_object(raw, f"{path_text}:{line_number}")
    matches = evaluate_row(
        row,
        metadata,
        _FORK_INDEX,
        red_threshold=_FORK_RED_THRESHOLD,
        orange_threshold=_FORK_ORANGE_THRESHOLD,
    )
    return sequence, str(metadata["split"]), matches


def source_inventory(release: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for split, path in source_splits(release):
        files[f"messages/{split}.jsonl"] = file_facts(path)
    provenance = release / "ledgers" / "provenance.jsonl"
    files["ledgers/provenance.jsonl"] = file_facts(provenance)
    manifest = release / "MANIFEST.json"
    return {
        "release_name": release.name,
        "manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
        "files": files,
    }


def command_scan(args: argparse.Namespace) -> None:
    global _FORK_INDEX
    release = Path(args.release).resolve()
    registry = Path(args.registry).resolve()
    output = Path(args.out).resolve()
    require_new_directory_target(output)
    if not release.is_dir():
        raise DecontaminationError(f"release is not a directory: {release}")
    if args.workers > 1:
        import multiprocessing

        if "fork" not in multiprocessing.get_all_start_methods() or not hasattr(os, "pread"):
            raise DecontaminationError(
                "--workers > 1 requires a POSIX platform with fork and os.pread"
            )
    items, registry_manifest = load_registry(
        registry, args.shingle_width, args.cjk_shingle_width
    )
    estimated_index = estimate_index_postings(items)
    print(
        "index_postings_estimate="
        + json.dumps(estimated_index, ensure_ascii=True, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )
    if estimated_index["total"] > args.max_index_postings:
        raise DecontaminationError(
            f"estimated index postings {estimated_index['total']} exceed --max-index-postings "
            f"{args.max_index_postings}; shard the eval registry or raise the audited cap"
        )
    index = LexicalIndex(items, args.shingle_width, args.cjk_shingle_width)
    unindexable_eval_items = sum(
        1
        for item_index in range(len(items))
        if not index.q_denominators[item_index] and not index.qa_denominators[item_index]
    )
    if unindexable_eval_items:
        raise DecontaminationError(
            f"registry contains {unindexable_eval_items} eval items with no weighted "
            "question or question+answer shingles after frequency filtering"
        )
    provenance_records, provenance_map = load_provenance(release)
    matches: list[dict[str, Any]] = []
    match_positions: dict[tuple[str, str, str, str], int] = {}
    zone_rank = {"yellow": 1, "orange": 2, "red": 3}

    def upsert_match(match: dict[str, Any]) -> None:
        key = (
            str(match["stable_key"]),
            str(match["benchmark"]),
            str(match["eval_split"]),
            str(match["item_id"]),
        )
        position = match_positions.get(key)
        if position is None:
            if len(matches) >= args.max_match_records:
                raise DecontaminationError(
                    f"match records exceed --max-match-records {args.max_match_records}; "
                    "recalibrate or shard before continuing"
                )
            match_positions[key] = len(matches)
            matches.append(match)
        elif zone_rank[str(match["zone"])] > zone_rank[str(matches[position]["zone"])]:
            matches[position] = match
    scanned = 0
    split_counts: collections.Counter[str] = collections.Counter()
    if args.workers == 1:
        for split, messages_path in source_splits(release):
            for line_number, row, raw in iter_jsonl(messages_path):
                row_sha = sha256_bytes(raw)
                queue = provenance_map.get((split, row_sha))
                if not queue:
                    raise DecontaminationError(
                        f"{messages_path}:{line_number}: no provenance row for canonical SHA {row_sha}"
                    )
                metadata = queue.popleft()
                for match in evaluate_row(
                    row,
                    metadata,
                    index,
                    red_threshold=args.red_threshold,
                    orange_threshold=args.orange_threshold,
                ):
                    upsert_match(match)
                scanned += 1
                split_counts[split] += 1
                if args.progress_every and scanned % args.progress_every == 0:
                    print(
                        f"scanned_rows={scanned} matches={len(matches)} split={split}",
                        file=sys.stderr,
                        flush=True,
                    )
    else:
        import concurrent.futures
        import gc
        import multiprocessing

        def tasks() -> Iterator[tuple[int, str, int, int, int, dict[str, Any]]]:
            sequence = 0
            for split, messages_path in source_splits(release):
                offset = 0
                with messages_path.open("rb") as handle:
                    for line_number, raw_with_newline in enumerate(handle, 1):
                        length = len(raw_with_newline)
                        raw = raw_with_newline.rstrip(b"\r\n")
                        if not raw:
                            raise DecontaminationError(
                                f"{messages_path}:{line_number}: blank JSONL row"
                            )
                        row_sha = sha256_bytes(raw)
                        queue = provenance_map.get((split, row_sha))
                        if not queue:
                            raise DecontaminationError(
                                f"{messages_path}:{line_number}: no provenance row for canonical SHA {row_sha}"
                            )
                        metadata = queue.popleft()
                        yield (
                            sequence,
                            str(messages_path),
                            offset,
                            length,
                            line_number,
                            metadata,
                        )
                        sequence += 1
                        offset += length

        _FORK_INDEX = index
        context = multiprocessing.get_context("fork")
        executor: concurrent.futures.ProcessPoolExecutor | None = None
        parent_gc_enabled = gc.isenabled()
        gc.disable()
        gc.freeze()
        try:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=context,
                initializer=initialize_fork_worker,
                initargs=(args.red_threshold, args.orange_threshold),
            )
            task_iterator = iter(tasks())
            pending: collections.deque[concurrent.futures.Future[Any]] = collections.deque()
            for _ in range(args.workers * 2):
                try:
                    pending.append(executor.submit(evaluate_offset_task, next(task_iterator)))
                except StopIteration:
                    break
            gc.unfreeze()
            if parent_gc_enabled:
                gc.enable()
            while pending:
                sequence, split, row_matches = pending.popleft().result()
                if sequence != scanned:
                    raise DecontaminationError(
                        f"parallel scan result order mismatch: expected {scanned}, got {sequence}"
                    )
                for match in row_matches:
                    upsert_match(match)
                scanned += 1
                split_counts[split] += 1
                try:
                    pending.append(executor.submit(evaluate_offset_task, next(task_iterator)))
                except StopIteration:
                    pass
                if args.progress_every and scanned % args.progress_every == 0:
                    print(
                        f"scanned_rows={scanned} matches={len(matches)} split={split}",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            if gc.get_freeze_count():
                gc.unfreeze()
            if parent_gc_enabled:
                gc.enable()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            _FORK_INDEX = None
    leftovers = sum(len(queue) for queue in provenance_map.values())
    if leftovers:
        raise DecontaminationError(f"{leftovers} provenance rows were not matched to message rows")
    if scanned != len(provenance_records):
        raise DecontaminationError(
            f"message/provenance count mismatch: messages={scanned}, provenance={len(provenance_records)}"
        )

    def propagate_cluster(
        groups: Mapping[str, Sequence[dict[str, Any]]], rule: str
    ) -> None:
        red_by_stable_key = {
            str(match["stable_key"]): match for match in matches if match["zone"] == "red"
        }
        for records in groups.values():
            template = next(
                (red_by_stable_key[str(record["stable_key"])] for record in records if str(record["stable_key"]) in red_by_stable_key),
                None,
            )
            if template is None:
                continue
            if len(records) > args.max_lineage_group:
                raise DecontaminationError(
                    f"red {rule} group has {len(records)} rows, exceeding the "
                    f"--max-lineage-group cluster cap of {args.max_lineage_group}"
                )
            for record in records:
                propagated = dict(template)
                propagated.update(
                    {
                        "stable_key": str(record["stable_key"]),
                        "split": str(record["split"]),
                        "canonical_row_sha256": str(record["canonical_row_sha256"]),
                        "rule": rule,
                        "train_excerpt": "",
                        "scores": {
                            "question_containment": 0.0,
                            "question_answer_containment": 0.0,
                            "answer_containment": 0.0,
                            "test_containment": 0.0,
                            "auxiliary_question_answer_containment": 0.0,
                            "auxiliary_answer_containment": 0.0,
                            "auxiliary_test_containment": 0.0,
                            "context_question_containment": 0.0,
                            "shared_nonfrequent_shingles": 0,
                            "long_exact_tokens": 0,
                            "long_exact_field": "",
                        },
                        "inherited_from_stable_key": str(template["stable_key"]),
                    }
                )
                upsert_match(propagated)

    exact_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in provenance_records:
        exact_groups[str(record["canonical_row_sha256"])].append(record)
    lineage_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in provenance_records:
        lineage = record.get("lineage_group_id")
        if isinstance(lineage, str) and lineage:
            lineage_groups[lineage].append(record)
    while True:
        red_before = {
            str(match["stable_key"]) for match in matches if match["zone"] == "red"
        }
        propagate_cluster(exact_groups, "exact_duplicate_cluster")
        propagate_cluster(lineage_groups, "lineage_cluster")
        red_after = {
            str(match["stable_key"]) for match in matches if match["zone"] == "red"
        }
        if red_after == red_before:
            break

    zone_counts = collections.Counter(match["zone"] for match in matches)
    rule_counts = collections.Counter(match["rule"] for match in matches)
    unique_zone_rows: dict[str, set[str]] = collections.defaultdict(set)
    for match in matches:
        unique_zone_rows[match["zone"]].add(match["stable_key"])
    source_stable_key_counts = collections.Counter(
        str(record["stable_key"]) for record in provenance_records
    )
    matches.sort(
        key=lambda row: (
            row["stable_key"],
            row["benchmark"],
            row["eval_split"],
            row["item_id"],
            row["rule"],
        )
    )
    incomplete = safe_new_directory(output)
    try:
        matches_path = incomplete / "matches.jsonl"
        with matches_path.open("wb") as handle:
            for match in matches:
                handle.write(canonical_json_bytes(match) + b"\n")
        llm_review_budget = min(
            100000, max(args.min_review_budget, int(1e-4 * scanned))
        )
        review_orange_rows = unique_zone_rows.get("orange", set()) - unique_zone_rows.get(
            "red", set()
        )
        unique_orange_rows = len(review_orange_rows)
        summary = {
            "schema": SCAN_SCHEMA,
            "tool_version": TOOL_VERSION,
            "source": source_inventory(release),
            "registry_manifest_sha256": sha256_file(registry / "MANIFEST.json"),
            "registry_items_sha256": str(registry_manifest["items_file"]["sha256"]),
            "parameters": {
                "shingle_width": args.shingle_width,
                "cjk_shingle_width": args.cjk_shingle_width,
                "red_threshold": args.red_threshold,
                "orange_threshold": args.orange_threshold,
                "high_frequency_cutoff": index.high_frequency_cutoff,
                "index_entries": index.stats(),
                "index_postings_estimate": estimated_index,
                "max_index_postings": args.max_index_postings,
                "max_match_records": args.max_match_records,
                "max_lineage_group": args.max_lineage_group,
                "min_review_budget": args.min_review_budget,
                "indexed_field_widths": {
                    "question": index.q_widths,
                    "answer": index.a_widths,
                    "test": index.test_widths,
                    "question_answer": index.qa_widths,
                },
                "short_field_fallback_lengths": dict(
                    sorted(
                        collections.Counter(
                            int(item["_q_shingle_width"])
                            for item in items
                            if item["_short_field_fallback"]
                        ).items()
                    )
                ),
                "unindexable_eval_items": unindexable_eval_items,
                "registry_field_coverage": [
                    asset.get("field_coverage", {}) for asset in registry_manifest["assets"]
                ],
            },
            "counts": {
                "rows_scanned": scanned,
                "unique_stable_keys_scanned": len(source_stable_key_counts),
                "duplicate_stable_key_rows": sum(
                    count - 1 for count in source_stable_key_counts.values() if count > 1
                ),
                "duplicate_stable_key_groups": sum(
                    1 for count in source_stable_key_counts.values() if count > 1
                ),
                "by_split": dict(sorted(split_counts.items())),
                "match_records_by_zone": dict(sorted(zone_counts.items())),
                "unique_rows_by_zone": {
                    zone: len(keys) for zone, keys in sorted(unique_zone_rows.items())
                },
                "by_rule": dict(sorted(rule_counts.items())),
                "llm_review_budget": llm_review_budget,
                "orange_budget_exceeded": unique_orange_rows > llm_review_budget,
            },
            "runtime": {"workers": args.workers},
            "matches_file": file_facts(matches_path),
            "residual_risks": [
                "unregistered evaluation assets",
                "deep paraphrase and cross-language pure-text leakage",
                "approximate near-duplicate channel (MinHash-LSH) not run",
                "structure keys not run (code AST/test constants, math canonicalization, option permutation)",
                "semantic-only similarity (semantic channel not run)",
                "adversarial poisoning",
            ],
        }
        summary_path = incomplete / "summary.json"
        write_json(summary_path, summary)
        write_json(
            incomplete / "SCAN_COMPLETE",
            {
                "matches_sha256": sha256_file(matches_path),
                "summary_sha256": sha256_file(summary_path),
            },
        )
        finish_directory(incomplete, output)
    except Exception:
        raise
    print(json.dumps(summary["counts"], ensure_ascii=True, sort_keys=True))


def load_complete_scan(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matches_path = path / "matches.jsonl"
    summary_path = path / "summary.json"
    complete_path = path / "SCAN_COMPLETE"
    if not all(candidate.is_file() for candidate in (matches_path, summary_path, complete_path)):
        raise DecontaminationError(f"scan is incomplete: {path}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("matches_sha256") != sha256_file(matches_path):
        raise DecontaminationError("scan matches hash does not match SCAN_COMPLETE")
    if complete.get("summary_sha256") != sha256_file(summary_path):
        raise DecontaminationError("scan summary hash does not match SCAN_COMPLETE")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != SCAN_SCHEMA:
        raise DecontaminationError(f"unsupported scan schema: {summary.get('schema')}")
    matches = [record for _, record, _ in iter_jsonl(matches_path)]
    return matches, summary


def load_reviews(path: Path | None) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if path is None:
        return {}
    reviews: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for line_number, review, _ in iter_jsonl(path):
        require_keys(
            review,
            (
                "stable_key",
                "benchmark",
                "eval_split",
                "item_id",
                "decision",
                "reviewer",
                "reason",
            ),
            f"{path}:{line_number}",
        )
        decision = str(review["decision"])
        if decision not in ALLOWED_REVIEW_DECISIONS:
            raise DecontaminationError(f"{path}:{line_number}: invalid review decision {decision!r}")
        key = (
            str(review["stable_key"]),
            str(review["benchmark"]),
            str(review["eval_split"]),
            str(review["item_id"]),
        )
        if key in reviews:
            raise DecontaminationError(f"duplicate review key: {key}")
        reviews[key] = review
    return reviews


def decide_removals(
    matches: Sequence[dict[str, Any]],
    reviews: Mapping[tuple[str, str, str, str], dict[str, Any]],
    orange_policy: str,
) -> tuple[set[str], dict[str, list[dict[str, Any]]]]:
    orange_keys = {
        (
            str(match["stable_key"]),
            str(match["benchmark"]),
            str(match["eval_split"]),
            str(match["item_id"]),
        )
        for match in matches
        if match["zone"] == "orange"
    }
    unknown_reviews = set(reviews) - orange_keys
    if unknown_reviews:
        raise DecontaminationError(f"reviews refer to non-orange candidates: {sorted(unknown_reviews)[:3]}")
    red_stable_keys = {
        str(match["stable_key"]) for match in matches if match["zone"] == "red"
    }
    removals: set[str] = set()
    evidence: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    unresolved = 0
    for match in matches:
        stable_key = str(match["stable_key"])
        if match["zone"] == "red":
            removals.add(stable_key)
            evidence[stable_key].append(match)
        elif match["zone"] == "orange":
            if stable_key in red_stable_keys:
                continue
            key = (
                stable_key,
                str(match["benchmark"]),
                str(match["eval_split"]),
                str(match["item_id"]),
            )
            review = reviews.get(key)
            decision = str(review["decision"]) if review else None
            if decision == "remove" or (decision in {None, "uncertain"} and orange_policy == "remove"):
                removals.add(stable_key)
                enriched = dict(match)
                enriched["review"] = review or {
                    "decision": "remove",
                    "reviewer": "policy",
                    "reason": "explicit --orange-policy remove",
                }
                evidence[stable_key].append(enriched)
            elif decision in {None, "uncertain"}:
                unresolved += 1
    if unresolved:
        raise DecontaminationError(
            f"{unresolved} orange candidates are unresolved; provide reviews or --orange-policy remove"
        )
    return removals, evidence


def write_filtered_messages(
    source: Path,
    destination: Path,
    split: str,
    provenance_map: dict[tuple[str, str], collections.deque[dict[str, Any]]],
    removals: set[str],
) -> tuple[int, int]:
    kept = 0
    removed = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        for line_number, _, raw in iter_jsonl(source):
            row_sha = sha256_bytes(raw)
            queue = provenance_map.get((split, row_sha))
            if not queue:
                raise DecontaminationError(
                    f"{source}:{line_number}: no provenance row for canonical SHA {row_sha}"
                )
            metadata = queue.popleft()
            if str(metadata["stable_key"]) in removals:
                removed += 1
            else:
                output.write(raw + b"\n")
                kept += 1
    return kept, removed


def command_apply(args: argparse.Namespace) -> None:
    release = Path(args.release).resolve()
    scan_path = Path(args.scan).resolve()
    output = Path(args.out).resolve()
    matches, scan_summary = load_complete_scan(scan_path)
    if scan_summary.get("counts", {}).get("orange_budget_exceeded") and not args.acknowledge_orange_budget:
        raise DecontaminationError(
            "orange candidate set exceeds the recorded LLM review budget; recalibrate the scan "
            "or pass --acknowledge-orange-budget after documenting approval"
        )
    current_inventory = source_inventory(release)
    if current_inventory != scan_summary.get("source"):
        raise DecontaminationError("source release changed after the scan; refusing to apply stale decisions")
    reviews_path = Path(args.reviews).resolve() if args.reviews else None
    reviews = load_reviews(reviews_path)
    removals, evidence = decide_removals(matches, reviews, args.orange_policy)
    provenance_records, provenance_map = load_provenance(release)
    stable_key_to_record = {str(record["stable_key"]): record for record in provenance_records}
    missing_keys = removals - set(stable_key_to_record)
    if missing_keys:
        raise DecontaminationError(f"removal keys missing from provenance: {sorted(missing_keys)[:3]}")
    incomplete = safe_new_directory(output)
    try:
        kept_by_split: dict[str, int] = {}
        removed_by_split: dict[str, int] = {}
        for split, source_path in source_splits(release):
            kept, removed = write_filtered_messages(
                source_path,
                incomplete / "messages" / source_path.name,
                split,
                provenance_map,
                removals,
            )
            kept_by_split[split] = kept
            removed_by_split[split] = removed
        leftovers = sum(len(queue) for queue in provenance_map.values())
        if leftovers:
            raise DecontaminationError(f"{leftovers} provenance rows were not matched while applying")

        (incomplete / "ledgers").mkdir(parents=True, exist_ok=True)
        provenance_out = incomplete / "ledgers" / "provenance.jsonl"
        removed_out = incomplete / "ledgers" / "decontaminated.jsonl"
        with provenance_out.open("wb") as kept_handle, removed_out.open("wb") as removed_handle:
            for record in provenance_records:
                stable_key = str(record["stable_key"])
                if stable_key in removals:
                    release_evidence = []
                    for evidence_record in evidence[stable_key]:
                        sanitized = {
                            key: value
                            for key, value in evidence_record.items()
                            if key not in {"train_excerpt", "eval_excerpt", "review"}
                        }
                        review = evidence_record.get("review")
                        if isinstance(review, dict):
                            sanitized["review"] = {
                                "decision": str(review.get("decision", "")),
                                "reviewer": str(review.get("reviewer", "")),
                                "reason_sha256": sha256_bytes(
                                    str(review.get("reason", "")).encode("utf-8")
                                ),
                            }
                        release_evidence.append(sanitized)
                    removal_record = {
                        "stable_key": stable_key,
                        "split": str(record["split"]),
                        "canonical_row_sha256": str(record["canonical_row_sha256"]),
                        "action": "remove_whole_sft_row",
                        "evidence": release_evidence,
                    }
                    removed_handle.write(canonical_json_bytes(removal_record) + b"\n")
                else:
                    kept_handle.write(canonical_json_bytes(record) + b"\n")

        for relative in (
            Path("ledgers/filtered.jsonl"),
            Path("quarantine/quarantine.jsonl"),
        ):
            source_file = release / relative
            if source_file.is_file():
                destination = incomplete / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, destination)
        source_manifest = release / "MANIFEST.json"
        if source_manifest.is_file():
            shutil.copyfile(source_manifest, incomplete / "SOURCE_MANIFEST.json")

        output_files: dict[str, Any] = {}
        for path in sorted(incomplete.rglob("*.jsonl")):
            output_files[path.relative_to(incomplete).as_posix()] = file_facts(path)
        if (incomplete / "SOURCE_MANIFEST.json").is_file():
            output_files["SOURCE_MANIFEST.json"] = file_facts(
                incomplete / "SOURCE_MANIFEST.json", count_lines=False
            )
        rules = collections.Counter(
            evidence_record["rule"]
            for records in evidence.values()
            for evidence_record in records
        )
        orange_review_records: collections.Counter[str] = collections.Counter()
        for review in reviews.values():
            orange_review_records[str(review["decision"])] += 1
        red_stable_keys = {
            str(match["stable_key"]) for match in matches if match["zone"] == "red"
        }
        explicitly_removed_stable_keys = {
            str(match["stable_key"])
            for match in matches
            if match["zone"] == "orange"
            and str(
                reviews.get(
                    (
                        str(match["stable_key"]),
                        str(match["benchmark"]),
                        str(match["eval_split"]),
                        str(match["item_id"]),
                    ),
                    {},
                ).get("decision", "")
            )
            == "remove"
        }
        policy_removed_stable_keys = {
            str(match["stable_key"])
            for match in matches
            if match["zone"] == "orange"
            and str(match["stable_key"]) not in red_stable_keys
            and str(match["stable_key"]) not in explicitly_removed_stable_keys
            and args.orange_policy == "remove"
            and str(
                reviews.get(
                    (
                        str(match["stable_key"]),
                        str(match["benchmark"]),
                        str(match["eval_split"]),
                        str(match["item_id"]),
                    ),
                    {},
                ).get("decision", "")
            )
            in {"", "uncertain"}
        }
        orange_decisions = {
            "review_records": dict(sorted(orange_review_records.items())),
            "policy_removed_stable_keys": len(policy_removed_stable_keys),
        }
        manifest = {
            "schema": RELEASE_SCHEMA,
            "tool_version": TOOL_VERSION,
            "status": "ok",
            "source_release_name": release.name,
            "source_manifest_sha256": current_inventory["manifest_sha256"],
            "scan_complete_sha256": sha256_file(scan_path / "SCAN_COMPLETE"),
            "registry_manifest_sha256": scan_summary["registry_manifest_sha256"],
            "scan_parameters": scan_summary["parameters"],
            "policy": {"orange_policy": args.orange_policy},
            "reviews_file_sha256": sha256_file(reviews_path) if reviews_path else None,
            "orange_decisions": orange_decisions,
            "counts": {
                "selected_before": len(provenance_records),
                "unique_stable_keys_before": len(
                    {str(record["stable_key"]) for record in provenance_records}
                ),
                "removed_stable_key_clusters": len(removals),
                "removed": sum(removed_by_split.values()),
                "selected_after": len(provenance_records) - sum(removed_by_split.values()),
                "kept_by_split": dict(sorted(kept_by_split.items())),
                "removed_by_split": dict(sorted(removed_by_split.items())),
                "removal_evidence_by_rule": dict(sorted(rules.items())),
                "accounting_closed": sum(kept_by_split.values()) + sum(removed_by_split.values())
                == len(provenance_records),
            },
            "outputs": output_files,
            "residual_risks": scan_summary.get("residual_risks", []),
        }
        if not manifest["counts"]["accounting_closed"]:
            raise DecontaminationError("derived release accounting did not close")
        for stable_key in removals:
            if not evidence.get(stable_key):
                raise DecontaminationError(f"removal has no audit evidence: {stable_key!r}")
        write_json(incomplete / "MANIFEST.json", manifest)
        verify_release(incomplete)
        finish_directory(incomplete, output)
    except Exception:
        raise
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))


def verify_release(release: Path) -> dict[str, Any]:
    manifest_path = release / "MANIFEST.json"
    if not manifest_path.is_file():
        raise DecontaminationError(f"missing derived MANIFEST.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != RELEASE_SCHEMA or manifest.get("status") != "ok":
        raise DecontaminationError("not an ok derived release manifest")
    failures: list[str] = []
    for relative, expected in manifest.get("outputs", {}).items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            failures.append(f"unsafe output path in manifest: {relative!r}")
            continue
        path = (release / relative_path).resolve()
        try:
            path.relative_to(release.resolve())
        except ValueError:
            failures.append(f"output path escapes release: {relative!r}")
            continue
        if not path.is_file():
            failures.append(f"missing output {relative}")
            continue
        facts = file_facts(path, count_lines=relative.endswith(".jsonl"))
        if facts != expected:
            failures.append(f"facts mismatch for {relative}: expected {expected}, got {facts}")

    expected_file_set = set(manifest.get("outputs", {})) | {"MANIFEST.json"}
    actual_file_set = {
        path.relative_to(release).as_posix() for path in release.rglob("*") if path.is_file()
    }
    if actual_file_set != expected_file_set:
        failures.append(
            "release file set mismatch: "
            f"missing={sorted(expected_file_set - actual_file_set)[:3]}, "
            f"unexpected={sorted(actual_file_set - expected_file_set)[:3]}"
        )

    message_hashes: collections.Counter[tuple[str, str]] = collections.Counter()
    message_rows = 0
    for split, path in source_splits(release):
        for _, _, raw in iter_jsonl(path):
            message_hashes[(split, sha256_bytes(raw))] += 1
            message_rows += 1
    provenance_hashes: collections.Counter[tuple[str, str]] = collections.Counter()
    stable_keys: set[str] = set()
    provenance_rows = 0
    provenance_path = release / "ledgers" / "provenance.jsonl"
    for line_number, record, _ in iter_jsonl(provenance_path):
        require_keys(record, ("stable_key", "split", "canonical_row_sha256"), f"{provenance_path}:{line_number}")
        stable_key = str(record["stable_key"])
        stable_keys.add(stable_key)
        provenance_rows += 1
        provenance_hashes[(str(record["split"]), str(record["canonical_row_sha256"]))] += 1
    if message_hashes != provenance_hashes:
        failures.append("message/provenance canonical SHA-256 multisets differ")

    removed_keys: set[str] = set()
    removed_rows = 0
    removed_path = release / "ledgers" / "decontaminated.jsonl"
    if removed_path.is_file():
        for line_number, record, _ in iter_jsonl(removed_path):
            require_keys(record, ("stable_key", "action", "evidence"), f"{removed_path}:{line_number}")
            stable_key = str(record["stable_key"])
            removed_keys.add(stable_key)
            removed_rows += 1
    overlap = stable_keys.intersection(removed_keys)
    if overlap:
        failures.append(f"stable keys are both selected and removed: {sorted(overlap)[:3]}")
    counts = manifest.get("counts", {})
    if message_rows != counts.get("selected_after"):
        failures.append(
            f"selected_after mismatch: manifest={counts.get('selected_after')}, messages={message_rows}"
        )
    if removed_rows != counts.get("removed"):
        failures.append(f"removed mismatch: manifest={counts.get('removed')}, ledger={removed_rows}")
    if provenance_rows != counts.get("selected_after"):
        failures.append(
            f"provenance selected_after mismatch: manifest={counts.get('selected_after')}, ledger={provenance_rows}"
        )
    if failures:
        raise DecontaminationError("verification failed:\n- " + "\n- ".join(failures))
    return {
        "schema": "decontamination-verification/v1",
        "release_manifest_sha256": sha256_file(manifest_path),
        "selected_rows": message_rows,
        "removed_rows": removed_rows,
        "failures": [],
    }


def command_verify(args: argparse.Namespace) -> None:
    release = Path(args.release).resolve()
    result = verify_release(release)
    if args.post_scan:
        _, post_summary = load_complete_scan(Path(args.post_scan).resolve())
        manifest = json.loads((release / "MANIFEST.json").read_text(encoding="utf-8"))
        if post_summary.get("registry_manifest_sha256") != manifest.get("registry_manifest_sha256"):
            raise DecontaminationError("post-scan registry does not match the derived release")
        if post_summary.get("parameters") != manifest.get("scan_parameters"):
            raise DecontaminationError("post-scan parameters do not match the pinned pre-scan parameters")
        if post_summary.get("source") != source_inventory(release):
            raise DecontaminationError("post-scan was not run against this derived release")
        red_rows = post_summary.get("counts", {}).get("unique_rows_by_zone", {}).get("red", 0)
        if red_rows:
            raise DecontaminationError(f"post-scan contains {red_rows} red rows")
        result["post_scan_complete_sha256"] = sha256_file(
            Path(args.post_scan).resolve() / "SCAN_COMPLETE"
        )
    if args.report:
        report = Path(args.report).resolve()
        if report.exists():
            raise DecontaminationError(f"verification report already exists: {report}")
        write_json(report, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    registry_parser = subparsers.add_parser("build-registry", help="freeze eval assets")
    registry_parser.add_argument("--config", required=True)
    registry_parser.add_argument("--out", required=True)
    registry_parser.set_defaults(func=command_build_registry)

    scan_parser = subparsers.add_parser("scan", help="scan a canonical release")
    scan_parser.add_argument("--release", required=True)
    scan_parser.add_argument("--registry", required=True)
    scan_parser.add_argument("--out", required=True)
    scan_parser.add_argument("--shingle-width", type=int, default=13)
    scan_parser.add_argument("--cjk-shingle-width", type=int, default=15)
    scan_parser.add_argument("--red-threshold", type=float, default=0.80)
    scan_parser.add_argument("--orange-threshold", type=float, default=0.20)
    scan_parser.add_argument("--progress-every", type=int, default=1000)
    scan_parser.add_argument("--max-index-postings", type=int, default=10000000)
    scan_parser.add_argument("--max-match-records", type=int, default=1000000)
    scan_parser.add_argument("--max-lineage-group", type=int, default=1000)
    scan_parser.add_argument("--min-review-budget", type=int, default=1)
    scan_parser.add_argument("--workers", type=int, default=1)
    scan_parser.set_defaults(func=command_scan)

    apply_parser = subparsers.add_parser("apply", help="create a derived release")
    apply_parser.add_argument("--release", required=True)
    apply_parser.add_argument("--scan", required=True)
    apply_parser.add_argument("--out", required=True)
    apply_parser.add_argument("--reviews")
    apply_parser.add_argument("--orange-policy", choices=("require-review", "remove"), default="require-review")
    apply_parser.add_argument("--acknowledge-orange-budget", action="store_true")
    apply_parser.set_defaults(func=command_apply)

    verify_parser = subparsers.add_parser("verify", help="verify a derived release")
    verify_parser.add_argument("--release", required=True)
    verify_parser.add_argument("--report")
    verify_parser.add_argument("--post-scan")
    verify_parser.set_defaults(func=command_verify)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "shingle_width", 13) < 2:
        raise DecontaminationError("--shingle-width must be at least 2")
    if getattr(args, "cjk_shingle_width", 15) < 2:
        raise DecontaminationError("--cjk-shingle-width must be at least 2")
    if getattr(args, "progress_every", 1000) < 0:
        raise DecontaminationError("--progress-every must be non-negative")
    if getattr(args, "max_index_postings", 10000000) < 1:
        raise DecontaminationError("--max-index-postings must be positive")
    if getattr(args, "max_match_records", 1000000) < 1:
        raise DecontaminationError("--max-match-records must be positive")
    if getattr(args, "max_lineage_group", 1000) < 1:
        raise DecontaminationError("--max-lineage-group must be positive")
    min_review_budget = getattr(args, "min_review_budget", 1)
    if not (1 <= min_review_budget <= 100000):
        raise DecontaminationError("--min-review-budget must be in [1, 100000]")
    if getattr(args, "workers", 1) < 1:
        raise DecontaminationError("--workers must be positive")
    worker_cap = min(os.cpu_count() or 1, 32)
    if getattr(args, "workers", 1) > worker_cap:
        raise DecontaminationError(
            f"--workers exceeds the audited host cap of {worker_cap}"
        )
    red = getattr(args, "red_threshold", 0.80)
    orange = getattr(args, "orange_threshold", 0.20)
    if not (0.0 <= orange < red <= 1.0):
        raise DecontaminationError("thresholds must satisfy 0 <= orange < red <= 1")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        args.func(args)
    except DecontaminationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
