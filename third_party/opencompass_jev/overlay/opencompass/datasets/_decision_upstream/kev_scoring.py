"""Unmodified pure scoring functions extracted from pinned Kev sources."""

import math

import numpy as np

from .kev_metrics import EPSILON, grouped_metrics, metrics, unknowable_report

def question_keys(qtype: str, criteria) -> list[str]:
    """The keys a question's probabilities are reported under, in option order: the criteria names (choice),
    ["false", "true"] (noul), the level indices as strings (score). Labels, targets and anchors use the same keys."""
    if qtype == "choice": return list(criteria)
    if qtype == "noul": return ["false", "true"]
    return [str(i) for i in range(len(criteria))]

def paired_flip(rows):
    """Pair-level metrics from benchmark rows carrying pair_id/sibling. A model that ignores the state cannot flip."""
    by_pair = {}
    for r in rows:
        if r.get("pair_id"):
            key = (r["pair_id"], r.get("question", "decision"))
            pair = by_pair.setdefault(key, {})
            if r["sibling"] in pair:
                raise ValueError("duplicate contrastive sibling")
            pair[r["sibling"]] = r
    if not by_pair:
        return None
    if any(set(p) != {"a", "b"} for p in by_pair.values()):
        raise ValueError("incomplete contrastive pair")
    prediction = lambda r: r["keys"][max(range(len(r["p"])), key=r["p"].__getitem__)]
    truth = lambda r: r["keys"][r["label"]]
    relevant = [p for p in by_pair.values() if truth(p["a"]) != truth(p["b"])]
    invariant = [p for p in by_pair.values() if truth(p["a"]) == truth(p["b"])]
    both = lambda ps: sum(all(prediction(r) == truth(r) for r in p.values()) for p in ps) / len(ps) if ps else None
    result = {"pairs": len(relevant),
              "flip_rate": sum(prediction(p["a"]) != prediction(p["b"]) for p in relevant) / len(relevant) if relevant else None,
              "both_correct_rate": both(relevant)}
    if invariant:
        result.update(invariant_pairs=len(invariant), invariance_rate=sum(prediction(p["a"]) == prediction(p["b"]) for p in invariant) / len(invariant),
                      invariant_both_correct_rate=both(invariant))
    return result

def labels(q):
    """(option keys, label index) of a labelled request question."""
    keys = question_keys(q["type"], q.get("criteria"))
    return keys, keys.index(q["label"]) if q["type"] == "choice" else int(q["label"])

def validate_distribution(raw, keys):
    if set(raw) != set(keys):
        raise ValueError("probability keys do not match requested options")
    p = np.array([raw[k] for k in keys], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("non-finite or out-of-range probabilities")
    total = float(p.sum())
    if total <= 0 or abs(total - 1) > max(1e-5, len(keys) * 0.005 + 1e-8):
        raise ValueError(f"invalid probability sum: {total}")
    return p / total, total

def prediction_rows(record, prediction):
    if set(prediction["probabilities"]) != set(record["questions"]):
        raise ValueError("answer IDs differ from request IDs")
    meta = record["_meta"]
    rows = []
    for qid, q in record["questions"].items():
        keys, y = labels(q)
        p, total = validate_distribution(prediction["probabilities"][qid], keys)
        row = {"id": meta["id"], "group": meta["group_id"], "question": qid,
               "source": meta["source"], "task": q["src"], "type": q["type"],
               "variant": meta["variant"], "keys": keys, "label": y, "control_id": meta.get("control_id"),
               "pair_id": meta.get("pair_id"), "sibling": meta.get("sibling"), # suites frozen before parent_id existed stored the parent's id in group_id for variants
               "parent": meta.get("parent_id") or (meta["id"] if meta["variant"] == "clean" else meta["group_id"]),
               "p": p.tolist(), "raw_probability_sum": total, "zero_count": int((p == 0).sum())}
        if "logits" in prediction:
            raw_logits = prediction["logits"][qid]
            if set(raw_logits) != set(keys) or not all(math.isfinite(raw_logits[k]) for k in keys):
                raise ValueError("logit keys or values do not match the requested options")
            row["logits"] = [float(raw_logits[k]) for k in keys]
            row["inference_temperature"] = prediction["inference_temperature"]
        rows.append(row)
    return rows

def summarize(rows, temperature=1.0, heldout_sources=()):
    """Report over benchmark rows. heldout_sources: sources the scored model never trained on; their tasks are also
    reported as a separate block."""
    clean = [r for r in rows if r["variant"] == "clean"]
    tasks = grouped_metrics(clean, "task")
    variants = grouped_metrics(rows, "variant")
    lookup = {(r["id"], r["question"]): r for r in clean}
    diffs, flips = [], []
    for row in rows:
        if row["variant"] == "permuted" and row["type"] == "choice":
            original = lookup[(row["parent"], row["question"])]
            aligned = [row["p"][row["keys"].index(k)] for k in original["keys"]]
            diffs.append(float(np.max(np.abs(np.array(aligned) - original["p"]))))
            flips.append(int(np.argmax(aligned) != np.argmax(original["p"])))
    knowable = [r for r in clean if r["source"] != "unknowable"]     # unknowable records are scored on confidence, never on accuracy
    return {"objective": -float(np.mean([v["nll"] for k, v in tasks.items() if not k.startswith("unknowable_") or k.startswith("unknowable_control")])),
            "paired_flip": paired_flip(clean), "unknowable": unknowable_report(clean),
            "clean": metrics(knowable), "tasks": tasks, "variants": variants,
            "heldout_tasks": grouped_metrics([r for r in clean if r["source"] in heldout_sources], "task") if any(r["source"] in heldout_sources for r in clean) else {},
            "permutation": {"n": len(diffs), "mean_max_delta": float(np.mean(diffs)) if diffs else None,
                            "flip_rate": float(np.mean(flips)) if flips else None},
            "temperature": temperature, "calibrated_clean": metrics(knowable, temperature),
            "metric_policy": {"version": 2, "selective_ties": "whole_confidence_groups",
                              "coverage_at_error": "in-sample maximum over confidence thresholds; not a deployed error guarantee",
                              "aurc": "right-step integral over whole confidence groups",
                              "confident_error_rate": "high-confidence errors divided by all questions",
                              "error_rate_at_0_9": "errors divided by questions accepted at p_max >= 0.9",
                              "nll": "exact from logits when recorded; otherwise from floored probabilities",
                              "nll_floor": EPSILON, "renormalize_returned_probabilities": True,
                              "raw_sums_outside_1e_5": sum(abs(r["raw_probability_sum"] - 1) > 1e-5 for r in rows),
                              "returned_zeros": sum(r["zero_count"] for r in rows)}}
