"""Bind surface augmentations to the trainer with an independent RNG and coverage counters."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .data import source_seed
from . import dse_surface


def runtime_binding():
    path = Path(dse_surface.__file__)
    body = path.read_bytes()
    tree = ast.parse(body)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        tree.body.pop(0)  # Same convention as standalone disclosure-v2 AST witness.
    return {"module_sha256": hashlib.sha256(body).hexdigest(),
            "runtime_ast_sha256": hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
            "whitelist_version": dse_surface.WHITELIST_VERSION,
            "candidate_rendering": "a: Desc -> [a] Desc / a:\\nDesc / [a]\\nDesc; description preserved",
            "C14_operator_shortfall": "sample min(requested, available) without replacement or retries; eligible pool-one yields one actual operation",
            "launch_authorization": "engineering binding only; C14 scientific coverage HOLD remains a separate registry gate"}


def private_rng(seed, epoch, source, source_id, sibling_index):
    # Deliberately omit op_count/probability so activation is paired across arms;
    # each sibling gets a fresh stream, so different sample() draws cannot drift.
    value = source_seed(seed, json.dumps(["surface", epoch, source, source_id, sibling_index], separators=(",", ":")))
    return random.Random(value)


def new_coverage():
    return {"records": 0, "questions": 0, "protected_records": 0,
            "status": Counter(), "eligible_ops": Counter(), "sampled_ops": Counter(), "applied_ops": Counter(),
            "eligible_pool_sizes": Counter(), "operator_shortfall": Counter(),
            "changed_instruction_questions": 0, "changed_candidate_questions": 0,
            "rejection": Counter(), "truncated_records": 0,
            "strict_encoding_attempts": 0, "strict_encodings_accepted": 0,
            "by_source": {}, "by_question_type": {}}


def collect_coverage(summary, original, view, source, *, encoding_attempts, accepted_encodings):
    c = view.coverage
    summary["records"] += 1
    summary["questions"] += len(original["questions"])
    summary["protected_records"] += int(c["preserve_candidates"])
    summary["status"][c["status"]] += 1
    for field in ("eligible_ops", "sampled_ops", "applied_ops"):
        summary[field].update(c[field])
    summary["eligible_pool_sizes"][str(len(c["eligible_ops"]))] += 1
    summary["operator_shortfall"][str(c["operator_shortfall"])] += 1
    for field in ("changed_instruction_questions", "changed_candidate_questions"):
        summary[field] += c[field]
    if c["rejection"] is not None:
        summary["rejection"][c["rejection"]] += 1
    summary["truncated_records"] += int(c["truncated"])
    summary["strict_encoding_attempts"] += encoding_attempts
    summary["strict_encodings_accepted"] += accepted_encodings
    local = summary["by_source"].setdefault(source, {"records": 0, "questions": 0, "status": Counter(), "eligible_pool_sizes": Counter(), "operator_shortfall": Counter(), "eligible_ops": Counter(), "sampled_ops": Counter(), "applied_ops": Counter(), "changed_instruction_questions": 0, "changed_candidate_questions": 0, "rejection": Counter()})
    local["records"] += 1; local["questions"] += len(original["questions"])
    local["status"][c["status"]] += 1
    local["eligible_pool_sizes"][str(len(c["eligible_ops"]))] += 1
    local["operator_shortfall"][str(c["operator_shortfall"])] += 1
    for field in ("eligible_ops", "sampled_ops", "applied_ops"):
        local[field].update(c[field])
    for field in ("changed_instruction_questions", "changed_candidate_questions"):
        local[field] += c[field]
    if c["rejection"] is not None:
        local["rejection"][c["rejection"]] += 1
    kinds = set()
    for before, after in zip(original["questions"], view.record["questions"]):
        kind = before["qtype"]; kinds.add(kind)
        item = summary["by_question_type"].setdefault(kind, {"questions": 0, "records_containing_type": 0, "record_status": Counter(), "changed_instruction_questions": 0, "changed_candidate_questions": 0})
        item["questions"] += 1
        item["changed_instruction_questions"] += int(before["instr"] != after["instr"])
        item["changed_candidate_questions"] += int(before["options"] != after["options"])
    for kind in kinds:
        summary["by_question_type"][kind]["records_containing_type"] += 1
        summary["by_question_type"][kind]["record_status"][c["status"]] += 1
