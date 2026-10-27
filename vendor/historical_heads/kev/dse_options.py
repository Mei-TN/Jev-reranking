"""Optional JEV training objectives and adaptation controls.

The disabled path calls the original loss.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DSEOptions:
    target_mode: str = "native"
    label_smoothing: float = 0.0
    smoothing_scope: str = "hard_only"
    smoothing_score: bool = False
    brier_weight: float = 0.0
    brier_reduction: str = "sum"
    noul_objective: str = "ce"
    head_dropout: float = 0.0
    lora_dropout: float = 0.05
    head_only: bool = False
    freeze_fraction: float = 0.0
    adapter_scheduler: str = "global_full_horizon_no_restart"
    stop_after_steps: int = 0
    supervised_views: int = 1
    js_weight: float = 0.0
    curriculum_prefix: float = 0.0
    ema_mode: str = "off"
    ema_decay: float = 0.99
    ema_consistency_weight: float = 0.0
    ema_ramp_fraction: float = 0.2
    original_augmentation: int = 1
    surface_ops_per_view: int = 1
    surface_probability: float = 0.0

    def validate(self):
        for name in ("label_smoothing", "brier_weight", "head_dropout", "lora_dropout", "freeze_fraction", "js_weight", "curriculum_prefix", "ema_decay", "ema_consistency_weight", "ema_ramp_fraction", "surface_probability"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.target_mode not in ("native", "harden"):
            raise ValueError("target_mode must be native or harden")
        if self.smoothing_scope not in ("hard_only", "all"):
            raise ValueError("smoothing_scope must be hard_only or all")
        if self.brier_reduction not in ("sum", "half_sum"):
            raise ValueError("brier_reduction must be sum or half_sum")
        if self.noul_objective not in ("ce", "difference_bce"):
            raise ValueError("noul_objective must be ce or difference_bce")
        if not 0 <= self.label_smoothing < 1 or not 0 <= self.head_dropout < 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("smoothing and dropout probabilities must be in [0, 1)")
        if self.brier_weight < 0 or not 0 <= self.freeze_fraction <= 1:
            raise ValueError("brier_weight must be nonnegative; freeze_fraction must be in [0, 1]")
        if isinstance(self.stop_after_steps, bool) or not isinstance(self.stop_after_steps, int) or self.stop_after_steps < 0:
            raise ValueError("stop_after_steps must be a nonnegative integer")
        if self.head_only and self.freeze_fraction:
            raise ValueError("head_only and freeze_fraction are separate experiments")
        if type(self.supervised_views) is not int or self.supervised_views not in (1, 2):
            raise ValueError("supervised_views must be 1 or 2")
        if self.js_weight < 0 or self.js_weight > 0 and self.supervised_views != 2:
            raise ValueError("nonnegative js_weight requires supervised_views=2 when nonzero")
        if not 0 <= self.curriculum_prefix <= 1:
            raise ValueError("curriculum_prefix must be in [0, 1]")
        if self.supervised_views == 2 and (not self.native_loss or self.curriculum_prefix > 0):
            raise ValueError("C16 currently requires native loss and no C18 composition; combinations need separate review")
        if self.ema_mode not in ("off", "mean_teacher", "eval_only"):
            raise ValueError("ema_mode must be off, mean_teacher or eval_only")
        if not 0 <= self.ema_decay < 1 or self.ema_consistency_weight < 0 or not 0 <= self.ema_ramp_fraction <= 1:
            raise ValueError("invalid EMA decay, consistency weight or ramp fraction")
        if self.ema_mode != "mean_teacher" and self.ema_consistency_weight != 0:
            raise ValueError("EMA consistency requires explicit mean_teacher mode")
        if self.ema_mode == "off" and (self.ema_decay != .99 or self.ema_ramp_fraction != .2):
            raise ValueError("EMA hyperparameters require an explicit enabled ema_mode")
        if self.ema_mode != "off" and (not self.native_loss or self.supervised_views != 1 or self.curriculum_prefix > 0 or self.head_only or self.freeze_fraction > 0):
            raise ValueError("EMA single-factor stage requires native loss, one student view, no curriculum or freezing composition")
        if self.adapter_scheduler not in ("global_full_horizon_no_restart", "restart_adapter_group_only"):
            raise ValueError("invalid adapter_scheduler")
        if self.adapter_scheduler == "restart_adapter_group_only":
            if self.head_only or not 0 < self.freeze_fraction < 1:
                raise ValueError("adapter restart requires staged freezing with an actual later thaw")
            if (not self.native_loss or self.supervised_views != 1 or self.curriculum_prefix != 0
                    or self.ema_mode != "off" or self.head_dropout != 0 or self.lora_dropout != .05):
                raise ValueError("adapter restart is a single-factor E3 diagnostic; incompatible loss/view/EMA/curriculum/dropout combinations")
        if type(self.original_augmentation) is not int or self.original_augmentation not in (0, 1):
            raise ValueError("original_augmentation must be 0 or 1")
        if type(self.surface_ops_per_view) is not int or self.surface_ops_per_view not in (1, 2):
            raise ValueError("surface_ops_per_view must be 1 or 2; disable via probability=0")
        if not 0 <= self.surface_probability <= 1:
            raise ValueError("surface_probability must be in [0,1]")
        if self.surface_probability > 0 or not self.original_augmentation:
            if (not self.native_loss or self.supervised_views != 1 or self.curriculum_prefix != 0 or self.ema_mode != "off"
                    or self.head_only or self.freeze_fraction != 0 or self.adapter_scheduler != "global_full_horizon_no_restart"
                    or self.head_dropout != 0 or self.lora_dropout != .05):
                raise ValueError("surface/original-augmentation controls currently require the isolated native-loss adaptation recipe")
        return self

    @property
    def native_loss(self):
        return (self.target_mode == "native" and self.label_smoothing == 0
                and self.brier_weight == 0 and self.noul_objective == "ce")

    def metadata(self):
        return {**asdict(self),
                "brier_definition": "sum_k (p_k - t_k)^2" if self.brier_reduction == "sum" else "0.5 * sum_k (p_k - t_k)^2",
                "brier_target": "same effective target as CE, after optional hardening and smoothing",
                "loss_curve_definition": ("C16: logged loss/run.ce is record question-mean of mean(two native CE) + js_weight * JS on changed Choice questions; single_view uses one native CE"
                                          if self.supervised_views == 2 else "logged loss/run.ce is the supervised objective: CE (or equivalent Noul BCE), plus enabled Brier and applicable Score RPS; anchor and permutation KL are logged separately"),
                "score_rps_definition": "with ord_w > 0, native hard Score targets get original RPS and native supplied targets get none; hardening enables RPS for supplied Score targets, and enabled Score smoothing changes the RPS reference to the effective smoothed CDF",
                "loss_curve_comparable_to_native": self.native_loss and self.supervised_views == 1,
                "c16_enabled": self.supervised_views == 2,
                "c16_rng": "independent random.Random(source_seed(seed, JSON([c16, epoch, source, source_id, sibling_index])))",
                "c16_weighting": "record question mean; two views averaged even at JS weight zero; unchanged original sibling/accum denominator",
                "c18_enabled": self.curriculum_prefix > 0,
                "surface_enabled": self.surface_probability > 0,
                "original_augmentation_definition": "0 bypasses both original augment and none_pair; 1 preserves their exact per-record RNG/calls",
                "surface_rng": "independent source_seed(seed, JSON([surface, epoch, source, source_id, sibling_index])); shared seed across requested-op/probability arms",
                "surface_weighting": "one forward per existing materialized sibling; original source/sibling/accum weighting retained",
                "surface_coverage_scope": "actual post-original-augmentation records; p0 bypasses surface API and leaves coverage unobserved",
                "ema_enabled": self.ema_mode != "off",
                "ema_logging": "run.ce and displayed loss remain native supervised CE; EMA consistency and full objective are separately reported under metrics.ema",
                "ema_objective": "native supervised CE + ramp_weight(completed optimizer updates) * question-mean KL(detached EMA teacher || student), T=1; no confidence gating or label replacement",
                "ema_teacher_view": "independent candidate-order RNG per seed/epoch/source/source-ID/sibling; unchanged ordered questions; same record when no legal permutation",
                "ema_snapshot": "ema_shadow.pt is separate LoRA/head shadow state; ordinary saved adapter/head remain student; no full training resume support",
                "c18_index": "clean strict 2048/2048/4096 packed len(ids) and maximum candidate count, before augmentation; built only when enabled",
                "smoothing_hard_definition": "one-hot effective target, including supplied one-hot distributions",
                "harden_tie_break": "lexicographically smallest candidate semantic key among equal maxima",
                "harden_is_gold": False,
                "noul_bce_definition": "BCEWithLogits(z_true - z_false, t_true); equivalent to two-class softmax CE",
                "head_only_adapter": "frozen PEFT shell with zero-initialized B (zero adapter output); A retains original initialization RNG" if self.head_only else None,
                "freeze_scheduler": ("head retains global OneCycle; only adapter LR/beta1 restart at thaw for remaining full horizon, AdamW state retained"
                                     if self.adapter_scheduler == "restart_adapter_group_only" else "original full-horizon OneCycle; existing empty adapter group populated on thaw"),
                "freeze_update_rounding": "ceil(Fraction(str(freeze_fraction)) * planned_optimizer_steps)",
                "sanity_horizon": "stop_after_steps never changes dataset, epochs, or OneCycle total_steps"}


def add_arguments(parser):
    parser.add_argument("--target_mode", choices=["native", "harden"], default="native")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--smoothing_scope", choices=["hard_only", "all"], default="hard_only")
    parser.add_argument("--smoothing_score", action="store_true", help="also allow smoothing Score targets")
    parser.add_argument("--brier_weight", type=float, default=0.0)
    parser.add_argument("--brier_reduction", choices=["sum", "half_sum"], default="sum")
    parser.add_argument("--noul_objective", choices=["ce", "difference_bce"], default="ce")
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--head_only", action="store_true")
    parser.add_argument("--freeze_fraction", type=float, default=0.0, help="fraction of full-horizon optimizer updates training only the head")
    parser.add_argument("--adapter_scheduler", choices=["global_full_horizon_no_restart", "restart_adapter_group_only"], default="global_full_horizon_no_restart")
    parser.add_argument("--stop_after_steps", type=int, default=0, help="sanity-only early stop; retains full dataset and scheduler horizon")
    parser.add_argument("--supervised_views", type=int, choices=[1, 2], default=1, help="C16: 2 enables two supervised views even when js_weight=0")
    parser.add_argument("--js_weight", type=float, default=0.0)
    parser.add_argument("--curriculum_prefix", type=float, default=0.0)
    parser.add_argument("--ema_mode", choices=["off", "mean_teacher", "eval_only"], default="off")
    parser.add_argument("--ema_decay", type=float, default=.99)
    parser.add_argument("--ema_consistency_weight", type=float, default=0.0)
    parser.add_argument("--ema_ramp_fraction", type=float, default=.2)
    parser.add_argument("--original_augmentation", type=int, choices=[0, 1], default=1)
    parser.add_argument("--surface_ops_per_view", type=int, choices=[1, 2], default=1)
    parser.add_argument("--surface_probability", type=float, default=0.0)


def from_arguments(args):
    options = DSEOptions(**{name: getattr(args, name) for name in DSEOptions.__dataclass_fields__}).validate()
    if args.full_finetune and (options != DSEOptions() or args.lora_targets in ("no_attention", "no_deltanet")):
        raise ValueError("DSE options require the adapter recipe; do not combine with full_finetune")
    if options.head_only and args.init_from:
        raise ValueError("head_only requires fresh zero-output adapter initialization, not init_from")
    if options.supervised_views == 2 and (args.perm_kl != 0 or args.anchor or args.anchor_w != 0 or args.ord_w != 0):
        raise ValueError("C16 cannot mix with legacy perm_kl, anchor or ordinal objectives in the single-factor stage")
    if options.ema_mode != "off" and (args.perm_kl != 0 or args.anchor or args.anchor_w != 0 or args.ord_w != 0 or args.special_embeddings):
        raise ValueError("EMA currently rejects legacy perm_kl, anchors, ordinal objectives and special embeddings")
    if options.adapter_scheduler == "restart_adapter_group_only" and (args.perm_kl != 0 or args.anchor or args.anchor_w != 0 or args.ord_w != 0
            or args.special_embeddings or args.init_from or args.option_isolation or args.lora_targets != "all"):
        raise ValueError("adapter restart single-factor E3 requires the original objective, fresh all-target adapters and original representation settings")
    if (options.surface_probability > 0 or not options.original_augmentation) and (args.perm_kl != 0 or args.anchor or args.anchor_w != 0
            or args.ord_w != 0 or args.special_embeddings or args.init_from or args.option_isolation or args.lora_targets != "all"):
        raise ValueError("surface/original-augmentation controls currently reject auxiliary objectives, warm starts and representation/adapter-target changes")
    return options


def _checked_target(q, k, device, dtype):
    """Strict finite probability target; no silent renormalization or invention of gold."""
    if q.get("target") is None:
        y = q.get("label")
        if isinstance(y, bool) or not isinstance(y, int) or not 0 <= y < k:
            raise ValueError("label must be an integer indexing a valid candidate")
        return F.one_hot(torch.tensor(y, device=device), k).to(dtype)
    t = torch.as_tensor(q["target"], device=device, dtype=dtype)
    if t.shape != (k,) or not bool(torch.isfinite(t).all()) or bool((t < 0).any()):
        raise ValueError("target must be a finite nonnegative vector matching candidates")
    if not bool(torch.isclose(t.sum(), t.new_tensor(1.0), atol=1e-6, rtol=1e-6)):
        raise ValueError("target probabilities must sum to one")
    return t


def effective_target(z, q, options):
    """Return valid logits, their effective target and candidate indices.

    Native encodings already return exactly K real candidate logits. A caller that
    supplies padded/invalid candidates must provide valid_mask. Invalid logits
    are removed before either normalization; their target mass must be zero.
    """
    if z.ndim != 1 or not len(z):
        raise ValueError("question logits must be a nonempty one-dimensional tensor")
    mask = q.get("valid_mask")
    if mask is None:
        mask = torch.ones(len(z), dtype=torch.bool, device=z.device)
    else:
        mask = torch.as_tensor(mask, device=z.device)
        if mask.dtype != torch.bool or mask.shape != z.shape or not bool(mask.any()):
            raise ValueError("valid_mask must be a boolean vector with at least one real candidate")
    if not bool(torch.isfinite(z[mask]).all()):
        raise ValueError("valid candidate logits must be finite")
    t = _checked_target(q, len(z), z.device, z.dtype)
    if bool((t[~mask] != 0).any()):
        raise ValueError("target puts mass on an invalid candidate")
    indices = mask.nonzero(as_tuple=True)[0]
    t = t[mask]
    if options.target_mode == "harden" and q.get("target") is not None:
        keys = q.get("keys")
        if (not isinstance(keys, (list, tuple)) or len(keys) != len(z)
                or any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys)):
            raise ValueError("hardening requires unique candidate semantic keys")
        ties = (t == t.max()).nonzero(as_tuple=True)[0].tolist()
        original_indices = indices.tolist()
        winner = min(ties, key=lambda i: keys[original_indices[i]])
        t = F.one_hot(torch.tensor(winner, device=z.device), len(t)).to(z.dtype)
    hard = bool(((t == 0) | (t == 1)).all())
    if (options.label_smoothing > 0 and (hard or options.smoothing_scope == "all")
            and (q["qtype"] != "score" or options.smoothing_score)):
        t = (1 - options.label_smoothing) * t + options.label_smoothing / len(t)
    return z[mask], t, indices


def loss_wrapper(z, q, dev, ord_w, options, original_loss):
    # No reconstruction of targets, tensor ops, validation, or RNG on this path.
    if options.native_loss and q.get("valid_mask") is None:
        return original_loss(z, q, dev, ord_w)
    v, t, indices = effective_target(z, q, options)
    if options.noul_objective == "difference_bce" and q["qtype"] == "noul":
        keys = q.get("keys")
        valid_keys = [keys[i] for i in indices.tolist()] if keys is not None else []
        if len(v) != 2 or set(valid_keys) != {"false", "true"}:
            raise ValueError("difference_bce requires exactly false/true Noul candidates")
        f, y = valid_keys.index("false"), valid_keys.index("true")
        loss = F.binary_cross_entropy_with_logits(v[y] - v[f], t[y])
    else:
        loss = -(t * F.log_softmax(v, -1)).sum()
    if q["qtype"] == "score" and ord_w > 0 and (q.get("target") is None or options.target_mode == "harden"):
        # Original native soft Score targets have no RPS term. Transformed hard
        # Score targets use their full effective CDF, including smoothing.
        if len(v) < 2 or len(v) != len(z):
            raise ValueError("ordinal loss requires all ordered levels and at least two levels")
        loss = loss + ord_w * (F.softmax(v, -1).cumsum(-1)[:-1] - t.cumsum(-1)[:-1]).square().mean()
    if options.brier_weight > 0:
        brier = (F.softmax(v, -1) - t).square().sum()
        loss = loss + options.brier_weight * (0.5 if options.brier_reduction == "half_sum" else 1.0) * brier
    return loss


def freeze_updates(options, total_steps):
    """Decimal CLI fraction, rounded upward without binary-float overshoot."""
    return total_steps if options.head_only else math.ceil(Fraction(str(options.freeze_fraction)) * total_steps)


class FreezeController:
    """Freeze only already-trainable backbone parameters, exposing none to AdamW.

    Construct only for head_only/freeze_fraction > 0; the baseline has no controller.
    The optimizer's original group 0 is retained (empty while frozen), so the
    scheduler always has its original number of groups and full training horizon.
    """
    def __init__(self, model, options, total_steps):
        self.model = model
        if options.head_only:
            # Real PEFT shells must have zero B; plain test/backbone modules with
            # no LoRA tensors are still valid frozen-feature probes.
            for name, parameter in model.lm.named_parameters():
                if "lora_B" in name.split(".") and not bool((parameter.detach() == 0).all()):
                    raise ValueError(f"head_only requires zero-output LoRA initialization; nonzero or nonfinite B tensor: {name}")
        self.parameters = [p for p in model.lm.parameters() if p.requires_grad]
        self.freeze_steps = freeze_updates(options, total_steps)
        self.permanent = options.head_only or self.freeze_steps >= total_steps
        self.frozen = self.freeze_steps > 0
        if self.frozen:
            for p in self.parameters:
                p.requires_grad_(False)

    def apply_modes(self):
        if self.frozen:
            self.model.lm.eval()
            self.model.head.train()

    def after_update(self, completed_steps, optimizer):
        if not self.frozen or self.permanent or completed_steps < self.freeze_steps:
            return False
        if optimizer.param_groups[0]["params"]:
            raise RuntimeError("adapter optimizer group must be empty before thaw")
        for p in self.parameters:
            p.requires_grad_(True)
        optimizer.param_groups[0]["params"] = self.parameters
        self.frozen = False
        self.model.lm.train()
        self.model.head.train()
        return True


def processed_row_tokens(enc, hybrid):
    """Unpadded actual input tokens; hybrid state repeats once per question."""
    packed = len(enc["ids"])
    return packed + (len(enc["decide_idx"]) - 1) * enc["seg"].count(0) if hybrid else packed


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py")) if not p.name.startswith("._")}


def target_inventory(records):
    """Source-side audit only; never change the native materialization/data stream."""
    from collections import Counter
    counts, sources, provenance = Counter(), Counter(), Counter()
    for r in records:
        for q in r["questions"].values():
            source = q.get("src", r.get("_meta", {}).get("source", "unspecified"))
            if q.get("target") is None:
                kind = "hard_label"
            else:
                values = list(q["target"].values())
                # Input requests use keyed weights; materialize normalizes them.
                # Audit that existing interface without renormalizing its output.
                if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values) or sum(values) <= 0:
                    raise ValueError("input target weights must be finite, nonnegative and have positive sum")
                kind = "supplied_onehot" if sum(x > 0 for x in values) == 1 else "supplied_soft"
                target_source = q.get("target_source", r.get("_meta", {}).get("target_source", "unspecified_in_record"))
                provenance[str(target_source)] += 1
            counts[kind] += 1
            sources[f"{source}:{kind}"] += 1
    return {"counts": dict(counts), "by_source": dict(sources), "target_provenance": dict(provenance),
            "supplied_target_interface": "existing keyed nonnegative weights, normalized by materialize",
            "unspecified_provenance_is_teacher": False}
