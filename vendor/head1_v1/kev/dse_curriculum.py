"""C18 ordering only: stratified easy-to-hard prefix, baseline-order suffix.

This module has no torch/numpy dependency, RNG calls, model loading, file I/O or
training integration. Pass the already shuffled *clean source records* and an
index measured before augmentation, e.g.::

    index = {record_id: {"packed_tokens": 412, "max_candidates": 4}, ...}
    ordered, report = reorder_epoch(
        shuffled_records, index, fraction=.2, seed=17, epoch=0,
        length_source="clean strict encode: len(ids), tokenizer/revision ...",
        candidate_source="max len(options) over clean materialized questions")

Each source record must have a unique ``_meta.id``, a nonempty ``_meta.source``
and raw ``questions`` keyed by question ID with type choice/noul/score. The
returned records are the original objects; neither labels nor text are edited.
Across epochs, retain a separate persistent baseline-shuffle sequence and consume
the returned order only as that epoch's view. Replacing the baseline sequence
with this view would change the starting order of the next baseline shuffle.
The clean index may contain additional records, but only this epoch's records
participate in ranks, allocation and diagnostics.

The prefix size is floor(decimal fraction * N). Its source x question-type-count
stratum quotas use largest remainder, without a minimum quota. Within a stratum,
SHA-256(seed, epoch, stratum, record ID) supplies the independent sampling order.
The selected prefix is sorted by the mean of length and max-K percentile ranks.
Each percentile is the average 1-based tied rank divided by N (pandas rank/pct
convention). Thus all ties have equal difficulty. Quota and difficulty ties use
domain-separated, seed-independent SHA-256 keys. The suffix preserves the exact
relative order of the supplied baseline shuffle.

At fraction=0 the original sequence object is returned immediately, without
reading the index/records, hashing, validating seed/epoch or computing ranks.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from numbers import Real
from statistics import median


SCHEMA_VERSION = 1
_TYPES = frozenset(("choice", "noul", "score"))
_DOMAIN = "jev-c18-v1"


def _sha(kind, *values):
    payload = json.dumps([_DOMAIN, kind, *values], ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stratum(source, composition):
    return json.dumps([source, list(composition)], ensure_ascii=False, separators=(",", ":"))


def _average_percentile_ranks(values):
    """Exact Fractions for average 1-based rank / N, equal values averaged."""
    n = len(values)
    ordered = sorted(values, key=lambda rid: (values[rid], rid))
    result = {}
    lo = 0
    while lo < n:
        end = lo + 1
        while end < n and values[ordered[end]] == values[ordered[lo]]:
            end += 1
        percentile = Fraction(lo + 1 + end, 2 * n)
        for rid in ordered[lo:end]:
            result[rid] = percentile
        lo = end
    return result


def _quotas(counts, prefix_size, n):
    ideals = {key: Fraction(count * prefix_size, n) for key, count in counts.items()}
    quotas = {key: ideal.numerator // ideal.denominator for key, ideal in ideals.items()}
    unassigned = prefix_size - sum(quotas.values())
    by_remainder = sorted(counts, key=lambda key: (-(ideals[key] - quotas[key]), _sha("quota-tie", key), key))
    for key in by_remainder[:unassigned]:
        quotas[key] += 1
    return quotas


def _stats(values):
    if not values:
        return {"count": 0, "sum": 0, "min": None, "max": None, "mean": None, "median": None}
    return {"count": len(values), "sum": sum(values), "min": min(values), "max": max(values),
            "mean": sum(values) / len(values), "median": median(values)}


def _diagnostics(ids, info, difficulty):
    sources, types, strata = Counter(), Counter(), Counter()
    for rid in ids:
        item = info[rid]
        sources[item["source"]] += 1
        types.update(dict(item["composition"]))
        strata[item["stratum"]] += 1
    return {"records": len(ids), "source_record_counts": dict(sorted(sources.items())),
            "question_type_counts": dict(sorted(types.items())),
            "stratum_record_counts": dict(sorted(strata.items())),
            "clean_packed_tokens": _stats([info[rid]["packed_tokens"] for rid in ids]),
            "clean_max_candidates": _stats([info[rid]["max_candidates"] for rid in ids]),
            "difficulty": _stats([float(difficulty[rid]) for rid in ids])}


def reorder_epoch(shuffled_records, clean_index=None, *, fraction=0.0, seed=0, epoch=0,
                  length_source=None, candidate_source=None):
    """Return ``(ordered_records, JSON-serializable_metadata)``; see module contract.

    ``length_source`` and ``candidate_source`` are required nonempty provenance
    descriptions when enabled. The caller owns measurement correctness; this
    function validates positive integer metrics but never fabricates/recomputes
    clean lengths or candidate counts from augmented views.
    """
    if isinstance(fraction, bool) or not isinstance(fraction, Real) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be finite and in [0, 1]")
    if fraction == 0:
        return shuffled_records, {"schema_version": SCHEMA_VERSION, "enabled": False, "fraction": 0.0,
                                  "prefix_records": 0, "suffix_records": len(shuffled_records),
                                  "returned_input_identity": True, "stage_diagnostics": None,
                                  "diagnostics_reason": "fraction=0: exact identity bypass, index and records not inspected"}
    if isinstance(seed, bool) or not isinstance(seed, int) or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("seed must be an integer; epoch must be a nonnegative integer")
    if any(not isinstance(value, str) or not value.strip() for value in (length_source, candidate_source)):
        raise ValueError("enabled curriculum requires length_source and candidate_source descriptions")
    if not isinstance(clean_index, dict):
        raise ValueError("clean_index must map record IDs to clean measurements")

    info, by_id, strata = {}, {}, defaultdict(list)
    ids = []
    for record in shuffled_records:
        if not isinstance(record, dict) or not isinstance(record.get("_meta"), dict):
            raise ValueError("each source record requires _meta.id and _meta.source")
        rid, source = record["_meta"].get("id"), record["_meta"].get("source")
        if any(not isinstance(value, str) or not value for value in (rid, source)):
            raise ValueError("record ID and source must be nonempty strings")
        if rid in by_id:
            raise ValueError(f"duplicate source record ID: {rid}")
        questions = record.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError(f"record {rid} needs a nonempty raw questions mapping")
        types = []
        for question in questions.values():
            if not isinstance(question, dict) or question.get("type") not in _TYPES:
                raise ValueError(f"record {rid} has an unsupported question type")
            types.append(question["type"])
        composition = tuple(sorted(Counter(types).items()))
        key = _stratum(source, composition)
        measurement = clean_index.get(rid)
        if not isinstance(measurement, dict):
            raise ValueError(f"missing clean measurements for {rid}")
        length, max_k = measurement.get("packed_tokens"), measurement.get("max_candidates")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (length, max_k)):
            raise ValueError(f"clean length and max_candidates must be positive integers: {rid}")
        by_id[rid] = record
        ids.append(rid)
        info[rid] = {"source": source, "composition": composition, "stratum": key,
                     "packed_tokens": length, "max_candidates": max_k}
        strata[key].append(rid)

    n = len(ids)
    prefix_size = (Fraction(str(fraction)) * n).__floor__()
    lengths = _average_percentile_ranks({rid: item["packed_tokens"] for rid, item in info.items()})
    candidates = _average_percentile_ranks({rid: item["max_candidates"] for rid, item in info.items()})
    difficulty = {rid: (lengths[rid] + candidates[rid]) / 2 for rid in ids}
    counts = {key: len(members) for key, members in strata.items()}
    quotas = _quotas(counts, prefix_size, n) if n else {}
    selected = []
    for key in sorted(strata):
        sampled = sorted(strata[key], key=lambda rid: (_sha("sample", seed, epoch, key, rid), rid))
        selected.extend(sampled[:quotas[key]])
    prefix = sorted(selected, key=lambda rid: (difficulty[rid], _sha("difficulty-tie", rid), rid))
    selected_ids = set(prefix)
    suffix = [rid for rid in ids if rid not in selected_ids]
    ordered_ids = prefix + suffix
    ordered = [by_id[rid] for rid in ordered_ids]
    metadata = {"schema_version": SCHEMA_VERSION, "enabled": True, "fraction": float(fraction),
                "seed": seed, "epoch": epoch, "total_records": n, "prefix_records": prefix_size,
                "suffix_records": n - prefix_size, "returned_input_identity": False,
                "prefix_rounding": "floor(Fraction(str(fraction)) * N)",
                "stratification": "source x exact question-type counts per clean source record",
                "rank_population": "only source records present in this epoch, before augmentation",
                "percentile_definition": "average 1-based tied rank / N; difficulty is mean(length percentile, max-K percentile)",
                "sampling": "ascending SHA-256(jev-c18-v1, sample, seed, epoch, stratum, record ID)",
                "quota_tie": "descending fractional remainder, then fixed SHA-256(jev-c18-v1, quota-tie, stratum)",
                "difficulty_tie": "fixed SHA-256(jev-c18-v1, difficulty-tie, record ID)",
                "length_source": length_source, "candidate_source": candidate_source,
                "input_order_sha256": _sha("order", ids), "output_order_sha256": _sha("order", ordered_ids),
                "clean_measurements_used_sha256": _sha("clean-measurements", [[rid, info[rid]] for rid in sorted(info)]),
                "stratum_prefix_quotas": dict(sorted(quotas.items())),
                "stage_diagnostics": {"all": _diagnostics(ids, info, difficulty),
                                      "prefix": _diagnostics(prefix, info, difficulty),
                                      "suffix": _diagnostics(suffix, info, difficulty)}}
    return ordered, metadata
