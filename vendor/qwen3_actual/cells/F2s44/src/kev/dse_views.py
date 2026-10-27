"""Supervised option-order views with semantic-aligned Jensen--Shannon divergence.

Permute Choice options while preserving Noul and Score order. Pass an
independent random.Random stream to make_views, encode both records with the
same strict limits, and call record_loss. An optional validator can reject
the secondary view before encoding.
"""
from collections import Counter
import copy
from dataclasses import dataclass
import math
import numbers

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DecisionViews:
    primary: dict
    secondary: dict | None
    changed_questions: tuple[int, ...]
    coverage: dict


@dataclass(frozen=True)
class ViewLoss:
    total: torch.Tensor
    ce: torch.Tensor
    js: torch.Tensor
    views_used: int


def _question(q):
    if not isinstance(q, dict):
        raise ValueError("questions must be materialized dictionaries")
    keys, options = q.get("keys"), q.get("options")
    if (not isinstance(keys, (list, tuple)) or not keys
            or any(not isinstance(k, str) or not k for k in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError("views require nonempty, unique candidate semantic IDs")
    if (not isinstance(options, (list, tuple)) or len(options) != len(keys)
            or any(not isinstance(x, str) for x in options)):
        raise ValueError("options must be text aligned with semantic IDs")
    label = q.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(keys):
        raise ValueError("label must index a real candidate")
    kind = q.get("qtype")
    if kind not in ("choice", "noul", "score"):
        raise ValueError("unsupported question type")
    if kind == "noul" and list(keys) != ["false", "true"]:
        raise ValueError("Noul order must remain false,true")
    if kind == "score" and (len(keys) < 2 or list(keys) != [str(i) for i in range(len(keys))]):
        raise ValueError("Score level order must remain canonical")
    mask = q.get("valid_mask")
    if mask is not None and (not isinstance(mask, (list, tuple)) or len(mask) != len(keys)
                             or any(type(x) is not bool or not x for x in mask)):
        raise ValueError("C16 v1 accepts ragged real candidates, not padded/invalid candidates")
    if q.get("target") is not None:
        target = q["target"]
        if (not isinstance(target, (list, tuple)) or len(target) != len(keys)
                or any(isinstance(x, bool) or not isinstance(x, numbers.Real)
                       or not math.isfinite(float(x)) or x < 0 for x in target)
                or not math.isclose(math.fsum(target), 1.0, rel_tol=1e-6, abs_tol=1e-6)):
            raise ValueError("native target must be finite, nonnegative and sum to one")
    return len(keys)


def _validate_record(record):
    if (not isinstance(record, dict) or "state" not in record
            or not isinstance(record.get("questions"), list) or not record["questions"]):
        raise ValueError("views require a materialized record with a nonempty questions list")
    for q in record["questions"]:
        _question(q)


def make_views(record, rng, validate_record=None):
    """Copy a materialized record and permute only nontrivial Choice questions.

    Gold indices, semantic IDs, option text, targets, and optional all-valid masks
    move together. A failed secondary context check produces explicit single_view
    coverage; an invalid PRIMARY record raises. Validators receive private copies.
    A mixed record retains its unchanged Noul/Score questions in the second view;
    they receive two CE terms if forwarded twice, but no consistency penalty.
    """
    _validate_record(record)
    primary = copy.deepcopy(record)
    if validate_record is not None and validate_record(copy.deepcopy(primary)) is False:
        raise ValueError("primary view failed context validation")
    secondary = copy.deepcopy(primary)
    changed, reasons = [], []
    for i, q in enumerate(primary["questions"]):
        k = len(q["keys"])
        if q["qtype"] != "choice":
            reasons.append("ordered_type_not_augmented")
            continue
        if k < 2:
            reasons.append("choice_single_candidate")
            continue
        if len(set(q["options"])) == 1:
            reasons.append("choice_rendering_identical")
            continue
        order = list(range(k))
        rng.shuffle(order)
        if [q["options"][j] for j in order] == list(q["options"]):
            # A one-position rotation changes every nonconstant option sequence.
            order = list(range(1, k)) + [0]
        out = secondary["questions"][i]
        for field in ("keys", "options", "target", "valid_mask"):
            if q.get(field) is not None:
                out[field] = [q[field][j] for j in order]
        out["label"] = order.index(q["label"])
        changed.append(i)
        reasons.append("choice_permuted")
    attempted = len(changed)
    rejection = None
    if not changed:
        secondary = None
        rejection = "no_nontrivial_choice_permutation"
    elif validate_record is not None:
        try:
            if validate_record(copy.deepcopy(secondary)) is False:
                raise ValueError("secondary view failed context validation")
        except ValueError as error:
            rejection = str(error)
            secondary = None
            changed = []
            reasons = ["secondary_context_rejected" if reason == "choice_permuted" else reason
                       for reason in reasons]
    coverage = {
        "status": "two_views" if secondary is not None else "single_view",
        "record_count": 1, "views_used": 2 if secondary is not None else 1,
        "questions": len(primary["questions"]),
        "questions_by_type": dict(Counter(q["qtype"] for q in primary["questions"])),
        "attempted_permuted_questions": attempted, "applied_permuted_questions": len(changed),
        "question_reasons": reasons, "second_view_rejection": rejection,
        "context_validator_used": validate_record is not None,
        "augmentation_scope": "Choice candidate order only; Noul/Score are unchanged",
        "truncated": False,
    }
    return DecisionViews(primary, secondary, tuple(changed), coverage)


def _log_probs(z, force_double=False):
    if (not isinstance(z, torch.Tensor) or z.ndim != 1 or not z.numel()
            or not z.is_floating_point() or not bool(torch.isfinite(z).all())):
        raise ValueError("logits must be a finite nonempty floating vector")
    dtype = torch.float64 if force_double or z.dtype == torch.float64 else torch.float32
    if dtype == torch.float32 and bool((z.abs() > torch.finfo(torch.float32).max / 4).any()):
        dtype = torch.float64
    lp = F.log_softmax(z.to(dtype), dim=-1)
    if not bool(torch.isfinite(lp).all()):
        raise ValueError("logit dynamic range exceeds finite float64 log-probability representation")
    return lp


def _target(q, lp):
    _question(q)
    if len(q["keys"]) != len(lp):
        raise ValueError("logit and candidate counts differ")
    if q.get("target") is None:
        return F.one_hot(torch.tensor(q["label"], device=lp.device), len(lp)).to(lp.dtype)
    # Do not renormalize, clamp, smooth or harden native supervision.
    return torch.tensor(q["target"], dtype=lp.dtype, device=lp.device)


def native_ce(z, q):
    lp = _log_probs(z)
    target = _target(q, lp)
    value = -(target * lp).sum()
    if not bool(torch.isfinite(value)):
        raise ValueError("CE exceeds finite arithmetic range")
    return value


def align_second(primary_q, secondary_q, primary_logits, secondary_logits):
    """Validate semantic/supervision invariants and align the second logits."""
    _question(primary_q)
    _question(secondary_q)
    a, b = primary_q["keys"], secondary_q["keys"]
    if set(a) != set(b):
        raise ValueError("candidate semantic sets differ between views")
    if primary_q["qtype"] != secondary_q["qtype"]:
        raise ValueError("question type changed between views")
    if primary_q["qtype"] != "choice" and list(a) != list(b):
        raise ValueError("ordered Noul/Score candidate order changed")
    for field in ("instr", "qid", "src"):
        if primary_q.get(field) != secondary_q.get(field):
            raise ValueError(f"question {field} changed between views")
    order = [b.index(k) for k in a]
    if any(primary_q["options"][i] != secondary_q["options"][j] for i, j in enumerate(order)):
        raise ValueError("candidate text/semantic correspondence changed")
    if a[primary_q["label"]] != b[secondary_q["label"]]:
        raise ValueError("gold semantic ID changed between views")
    ta, tb = primary_q.get("target"), secondary_q.get("target")
    if (ta is None) != (tb is None) or ta is not None and list(ta) != [tb[j] for j in order]:
        raise ValueError("native target changed between views")
    _log_probs(primary_logits)
    _log_probs(secondary_logits)
    if len(primary_logits) != len(a) or len(secondary_logits) != len(b):
        raise ValueError("logit and candidate counts differ")
    if primary_logits.device != secondary_logits.device:
        raise ValueError("view logits must be on the same device")
    return secondary_logits[torch.tensor(order, dtype=torch.long, device=secondary_logits.device)]


def jensen_shannon(first, second):
    """JS in nats, differentiable in BOTH logits, using double log-domain math.

    Finite log probabilities guarantee an underflowed probability is multiplied
    by a finite log-ratio, never 0 * inf. Target distributions are not modified.
    A negative divergence within 64 double eps is roundoff and returns exact zero;
    a larger negative value raises. This guard acts on the scalar JS, not targets.
    """
    if (not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor)
            or first.shape != second.shape or first.device != second.device):
        raise ValueError("JS requires aligned vectors on the same device")
    lp, lq = _log_probs(first, True), _log_probs(second, True)
    lm = torch.logsumexp(torch.stack((lp, lq)), dim=0) - math.log(2.0)
    value = 0.5 * ((lp.exp() * (lp - lm)).sum() + (lq.exp() * (lq - lm)).sum())
    if not bool(torch.isfinite(value)):
        raise ValueError("nonfinite Jensen-Shannon divergence")
    if bool(value < 0):
        if bool(value < -64 * torch.finfo(torch.float64).eps):
            raise ValueError("JS below numerical roundoff tolerance")
        value = value * 0.0
    return value


def _weight(weight):
    if isinstance(weight, bool) or not isinstance(weight, numbers.Real) or not math.isfinite(weight) or weight < 0:
        raise ValueError("JS weight must be finite and nonnegative")


def two_view_question_loss(first, primary_q, second, secondary_q, js_weight):
    """Always mean the TWO supervised CEs, including at js_weight=0."""
    _weight(js_weight)
    aligned = align_second(primary_q, secondary_q, first, second)
    ce = 0.5 * (native_ce(first, primary_q) + native_ce(second, secondary_q))
    js = jensen_shannon(first, aligned)
    return ViewLoss(ce + js_weight * js, ce, js, 2)


def record_loss(views, first_logits, second_logits=None, js_weight=0.0):
    """Question mean within one source record; caller handles sibling/accum weights.

    For mixed two-view records, all questions receive mean(two CE), while only
    actually permuted Choice questions contribute JS. JS is averaged over ALL
    questions (zero for unchanged ones), so eligibility does not amplify weight.
    A single_view record uses exactly one CE per question and no JS.
    """
    _weight(js_weight)
    _validate_record(views.primary)
    qs = views.primary["questions"]
    if len(first_logits) != len(qs):
        raise ValueError("primary logits must cover every question")
    if views.secondary is None:
        if second_logits is not None or views.changed_questions:
            raise ValueError("single_view must not carry second logits or changed questions")
        ce = torch.stack([native_ce(z, q) for z, q in zip(first_logits, qs)]).mean()
        return ViewLoss(ce, ce, ce * 0.0, 1)
    _validate_record(views.secondary)
    if views.primary["state"] != views.secondary["state"]:
        raise ValueError("state changed between candidate-order views")
    if len(views.secondary["questions"]) != len(qs) or second_logits is None or len(second_logits) != len(qs):
        raise ValueError("second view/logits must cover the same questions")
    changed = set(views.changed_questions)
    if (not changed or len(changed) != len(views.changed_questions)
            or any(type(i) is not int or i < 0 or i >= len(qs) for i in changed)):
        raise ValueError("invalid changed-question coverage")
    ce_terms, js_terms = [], []
    for i, (a, qa, b, qb) in enumerate(zip(first_logits, qs, second_logits, views.secondary["questions"])):
        actual_change = qa["options"] != qb["options"]
        if actual_change != (i in changed) or actual_change and qa["qtype"] != "choice":
            raise ValueError("changed-question coverage disagrees with actual Choice permutation")
        parts = two_view_question_loss(a, qa, b, qb, js_weight)
        ce_terms.append(parts.ce)
        js_terms.append(parts.js if i in changed else parts.js * 0.0)
    ce, js = torch.stack(ce_terms).mean(), torch.stack(js_terms).mean()
    return ViewLoss(ce + js_weight * js, ce, js, 2)
