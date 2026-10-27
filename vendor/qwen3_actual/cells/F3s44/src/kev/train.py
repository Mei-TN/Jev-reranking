import argparse, contextlib, json, math, os, random, resource, sys, time
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from .data import EVAL_ONLY, build, augment, load_records, materialize, none_pair, source_seed
from .suite import digest, load_split, write_json
from .model import MAX_BRANCH, MAX_STATE, DecisionModel, load_tokenizer, encode
from .heads import normalize_head_config
from .dse_options import (add_arguments, from_arguments, loss_wrapper, FreezeController,
                          freeze_updates, processed_row_tokens, source_hashes, target_inventory)


def permuted_copy(rec, rng):
    """Re-shuffle options of every Choice question with K>=3; return (record, perms) with perms[q] = new->old index or None."""
    out, perms = {"state": rec["state"], "questions": []}, []
    for q in rec["questions"]:
        if q["qtype"] == "choice" and len(q["options"]) >= 3:
            perm = list(range(len(q["options"]))); rng.shuffle(perm)
            q2 = {**q, "options": [q["options"][j] for j in perm], "label": perm.index(q["label"])}
            for field in ("keys", "target", "valid_mask"):
                if q.get(field) is not None: q2[field] = [q[field][j] for j in perm]
            out["questions"].append(q2); perms.append(perm)
        else:
            out["questions"].append(q); perms.append(None)
    return out, perms


def question_loss(z, q, dev, ord_w):
    """Cross-entropy (or cross-entropy against a soft target when the question carries one), optionally plus the
    normalized ranked probability score for ordered levels."""
    if q.get("target") is not None:
        t = torch.tensor(q["target"], device=dev, dtype=z.dtype)
        return -(t * F.log_softmax(z, -1)).sum()
    y = torch.tensor([q["label"]], device=dev)
    loss = F.cross_entropy(z[None], y)
    if q["qtype"] == "score" and ord_w > 0:
        p = F.softmax(z, -1)
        observed_cdf = (torch.arange(len(p) - 1, device=dev) >= q["label"]).to(p.dtype)
        loss = loss + ord_w * (p.cumsum(-1)[:-1] - observed_cdf).square().mean()
    return loss


def anchor_loss(z, q, target, dev):
    """KL(teacher || student) for one question, teacher = frozen base zero-shot distribution keyed by option key.
    Skips (returns None) when the current option set is not exactly the teacher's (e.g. a none-option was inserted)."""
    if target is None or set(target) != set(q["keys"]): return None
    t = torch.tensor([target[k] for k in q["keys"]], device=dev, dtype=torch.float32).clamp_min(1e-6); t = t / t.sum()
    return F.kl_div(F.log_softmax(z, -1), t, reduction="sum")


def fits_context(tok, req):
    """True when the clean record encodes within the training limits (the rule frozen suites are filtered by)."""
    try:
        return len(encode(tok, materialize(req), max_state=2048, max_branch=2048, strict=True)["ids"]) <= 4096
    except ValueError:
        return False


def accumulation_records(n, batch, accum, microbatch):
    start = (microbatch // accum) * accum * batch
    return min(accum * batch, n - start)


def strict_training_encode(model, tok, rec):
    enc = model.encode(tok, rec, max_state=2048, max_branch=2048, strict=True)
    if len(enc["ids"]) > 4096:
        raise ValueError("training request exceeds 4096 packed tokens")
    return enc


def collect_view_coverage(summary, views, source):
    coverage = views.coverage
    summary["records"] += 1
    summary[coverage["status"] + "_records"] += 1
    summary["views_forwarded"] += coverage["views_used"]
    for name in ("questions", "attempted_permuted_questions", "applied_permuted_questions"):
        summary[name] += coverage[name]
    summary["questions_by_type"].update(coverage["questions_by_type"])
    summary["question_reasons"].update(coverage["question_reasons"])
    if coverage["second_view_rejection"]:
        summary["second_view_rejections"][coverage["second_view_rejection"]] += 1
    summary["by_source"].setdefault(source, Counter())[coverage["status"]] += 1


def ema_record_kl(student_record, teacher_record, student_logits, teacher_logits):
    """Question-mean KL(detached teacher || student), semantic aligned, T=1."""
    from .dse_views import align_second
    sq, tq = student_record["questions"], teacher_record["questions"]
    if not sq or len(sq) != len(tq) or len(student_logits) != len(sq) or len(teacher_logits) != len(tq):
        raise ValueError("EMA view question/logit counts differ")
    terms = []
    for q, t, z, tz in zip(sq, tq, student_logits, teacher_logits):
        aligned = align_second(q, t, z, tz.detach())
        lp, lt = F.log_softmax(z, -1), F.log_softmax(aligned, -1)
        terms.append(F.kl_div(lp, lt, log_target=True, reduction="sum"))
    value = sum(terms) / len(terms)
    if not bool(torch.isfinite(value)):
        raise ValueError("nonfinite EMA consistency loss")
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--n_per_source", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head_lr", type=float, default=0.0, help="separate learning rate for the pointer head (0 = same as --lr); the head trains from scratch")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay on LoRA and head parameters")
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--full_finetune", action="store_true", help="train the complete backbone and pointer head, without adapters")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--holdout", default="", help="comma-separated sources excluded from training (evaluated as out-of-source)")
    ap.add_argument("--perm_kl", type=float, default=0.0, help="weight of symmetric KL between predictions under two option orders")
    ap.add_argument("--perm_frac", type=float, default=0.3, help="fraction of records that get the second permuted forward pass")
    ap.add_argument("--ord_w", type=float, default=0.0, help="weight of ranked probability score for Score questions")
    ap.add_argument("--suite", help="frozen suite directory; train only on its training partition")
    ap.add_argument("--train_sources", default="", help="comma-separated subset of the suite's trainable sources (ablations); default all")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--batch", type=int, default=1, help="records per forward pass (padded batch); optimizer step every --accum micro-batches")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32", help="bf16 = autocast forward with fp32 master weights (CUDA only)")
    ap.add_argument("--weights_dtype", choices=["fp32", "bf16"], default="fp32", help="dtype of the frozen backbone weights. bf16 halves memory and is required by the fused MoE experts "
                                                                                        "(torch._grouped_mm wants bf16); LoRA and head stay fp32 (peft upcasts adapters). Evaluation of such a run must also load bf16 (KEV_DTYPE=bf16).")
    ap.add_argument("--checkpointing", type=int, choices=[0, 1], default=0)
    ap.add_argument("--option_isolation", type=int, choices=[0, 1], default=0, help="option spans are isolated sub-branches with shared positions (exact permutation invariance)")
    ap.add_argument("--special_embeddings", type=int, choices=[0, 1], default=0, help="also train the embeddings of the 5 delimiter tokens")
    ap.add_argument("--head_dim", type=int, default=256, help="pointer head dimension")
    ap.add_argument("--head_config", default="", help="versioned head configuration as a JSON object; omitted = legacy pointer/end")
    ap.add_argument("--supplement_diagnostics", type=int, choices=[0, 1], default=0,
                    help="fixed paper supplement observational state/step/same-forward diagnostics")
    ap.add_argument("--lora_targets", choices=["all", "dense", "attn", "qv", "no_attention", "no_deltanet"], default="all", help="LoRA module set; no_attention removes standard attention, no_deltanet removes DeltaNet; both retain MLP. Original all/dense/attn/qv definitions unchanged")
    ap.add_argument("--base_revision", default="", help="pin the base commit when the suite manifest does not pin this base")
    ap.add_argument("--p_none", type=float, default=0.1)
    ap.add_argument("--p_none_distract", type=float, default=0.12)
    ap.add_argument("--p_distract", type=float, default=0.15)
    ap.add_argument("--p_none_pair", type=float, default=0.0, help="fraction of Choice records that additionally emit a none-present/none-absent minimal pair")
    ap.add_argument("--synthetic_repeat", type=int, default=1, help="oversample synthetic policy sources (legacy_policy, compositional, contrastive) this many times per epoch")
    ap.add_argument("--public_frac", type=float, default=1.0, help="deterministic subsample of public-source training records (mix ablations)")
    ap.add_argument("--anchor", default="", help="JSON of frozen-base zero-shot distributions {record_id: {qid: {key: p}}} (kev.anchors); enables the anchoring loss")
    ap.add_argument("--anchor_w", type=float, default=0.0, help="weight of KL(base || model) toward the frozen base model's zero-shot distribution, per anchored question")
    ap.add_argument("--anchor_sources", default="", help="comma-separated sources to anchor (default: every record with a target)")
    ap.add_argument("--out", default="runs/kev")
    ap.add_argument("--data", default="", help="your own labelled requests, one JSON object per line (see kev.data.load_records); an alternative to --suite for fine-tuning, or combined with --suite and --replay")
    ap.add_argument("--replay", type=int, default=0, help="with --data and --suite: mix in this many records sampled (by --seed) from the suite's training partition, so a delta fine-tune does not forget the released recipe")
    ap.add_argument("--init_from", default="", help="delta mode: warm-start LoRA and the pointer head from an existing run "
                                                   "(local directory or hub id) instead of starting from the base model; keeps the "
                                                   "released model's in-domain skill while adapting to a new domain")
    ap.add_argument("--seed", type=int, default=0)
    add_arguments(ap)
    a = ap.parse_args()
    try:
        head_config = normalize_head_config(json.loads(a.head_config) if a.head_config else None, head_dim=a.head_dim)
        a.head_dim = head_config["dim"]
        dse = from_arguments(a)
    except ValueError as exc:
        ap.error(str(exc))

    if min(a.epochs, a.accum, a.n_per_source, a.lora, a.batch, a.synthetic_repeat) < 1 or not 0 < a.public_frac <= 1:
        ap.error("epochs, accum, n_per_source, lora, batch and synthetic_repeat must be positive; 0 < public_frac <= 1")
    if a.dtype == "bf16" and a.device != "cuda":
        ap.error("--dtype bf16 requires --device cuda")
    if a.full_finetune and (a.init_from or a.special_embeddings or a.weights_dtype != 'fp32'):
        ap.error('full_finetune starts from the base with fp32 master weights; no adapter warm-start or special-token-only mode')
    if a.lr <= 0 or a.head_lr < 0 or a.weight_decay < 0 or min(a.ord_w, a.perm_kl, a.anchor_w) < 0 or not 0 <= a.perm_frac <= 1:
        ap.error("invalid learning rate or loss weights")
    if bool(a.anchor) != (a.anchor_w > 0):
        ap.error("--anchor and --anchor_w > 0 go together")
    anchors = json.loads(Path(a.anchor).read_text()).get("targets", {}) if a.anchor else {}
    if a.anchor: print(f"anchor targets: {len(anchors)} records from {a.anchor}", flush=True)
    anchor_sources = set(a.anchor_sources.split(",")) if a.anchor_sources else None
    out_dir = Path(a.out)
    if out_dir.exists():
        ap.error("refusing to overwrite an existing run")
    out_dir.mkdir(parents=True)
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if a.dtype == "bf16" else contextlib.nullcontext()
    manifest = json.loads((Path(a.suite) / "manifest.json").read_text()) if a.suite else None
    revision = manifest["base_revisions"].get(a.base) if manifest else None
    if a.base_revision:
        if revision and revision != a.base_revision: raise ValueError("--base_revision conflicts with the suite's pinned revision")
        revision = a.base_revision
    if manifest and not revision:
        raise ValueError("base not pinned by the suite; pass --base_revision")
    tok = load_tokenizer(a.base, revision=revision)
    model = DecisionModel(a.base, tok, dev, lora=None if a.full_finetune else a.lora, revision=revision, head_dim=a.head_dim, lora_targets=a.lora_targets,
                          option_isolation=bool(a.option_isolation), special_embeddings=bool(a.special_embeddings),
                          dtype=torch.bfloat16 if a.weights_dtype == "bf16" else torch.float32,
                          head_dropout=dse.head_dropout, lora_dropout=dse.lora_dropout, head_config=head_config)
    # Candidate-only paired step-zero binding, before any optimizer construction.
    from .paired_candidate_init import bind_initial
    bind_initial(model, out_dir, a.seed, revision)
    if a.full_finetune:
        assert all(p.requires_grad for p in model.parameters()), 'full fine-tuning must not leave frozen parameters'
    if a.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    init_source = None
    if a.init_from:
        # delta mode (PR #9, Radexito): start from an already trained adapter + pointer head instead of the base model, so a
        # fine-tune on new data keeps what the released checkpoint knows. Compatibility is checked field by field BEFORE
        # loading, because peft loads matching keys silently and a half-loaded adapter still trains and still reports a loss.
        from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
        from .evaluate import resolve_run
        src = resolve_run(a.init_from)
        meta = torch.load(f"{src}/head.pt", map_location="cpu")
        if meta.get("stopped_early"):
            raise ValueError(f"--init_from {src}: refusing a stopped_early sanity checkpoint")
        previous_head = normalize_head_config(meta.get("head_config"), head_dim=meta.get("head_dim", 256))
        if previous_head != model.head_config:
            raise ValueError(f"--init_from {src}: head_config is {previous_head!r} there and {model.head_config!r} here")
        checks = [("base", meta.get("base"), a.base), ("base_revision", meta.get("base_revision"), revision), ("lora", int(meta.get("lora", 0)), a.lora),
                  ("head_dim", int(meta.get("head_dim", 256)), a.head_dim), ("option_isolation", bool(meta.get("option_isolation", False)), bool(a.option_isolation)),
                  ("special_embeddings", bool(meta.get("special_embeddings", False)), bool(a.special_embeddings))]
        for field, theirs, ours in checks:
            if theirs != ours and not (field == "base_revision" and (theirs is None or ours is None)):
                raise ValueError(f"--init_from {src}: {field} is {theirs!r} there and {ours!r} here")
        weights = load_peft_weights(src, device="cpu")
        mine = set(get_peft_model_state_dict(model.lm))
        unexpected, missing = sorted(set(weights) - mine), sorted(mine - set(weights))
        if unexpected:
            raise ValueError(f"--init_from {src} carries {len(unexpected)} adapter tensors this model does not have (e.g. {unexpected[:2]}); check --lora_targets / --lora against its adapter_config.json")
        if missing:
            raise ValueError(f"--init_from {src} does not cover {len(missing)} of this model's adapter tensors (e.g. {missing[:2]}); check --lora_targets")
        set_peft_model_state_dict(model.lm, weights)
        model.head.load_state_dict(meta["head"], strict=True)
        init_source = {"init_from": a.init_from, "resolved": str(src), "adapter_sha256": digest(Path(src) / "adapter_model.safetensors"), "head_sha256": digest(Path(src) / "head.pt")}
        print(f"delta: warm start from {src}: {len(weights)} adapter tensors and the pointer head loaded", flush=True)
    print(f"device={dev} trainable params={sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M", flush=True)

    holdout = manifest["holdout_sources"] if manifest else [s for s in a.holdout.split(",") if s]
    if a.replay and not (a.data and a.suite): ap.error("--replay needs both --data and --suite")
    if a.data:
        reqs = load_records(a.data)
        if a.replay:
            pool = load_split(a.suite, "train"); replay = random.Random(f"replay:{a.seed}").sample(pool, min(a.replay, len(pool)))
            print(f"replay: {len(replay)} of {len(pool)} suite training records mixed with {len(reqs)} from {a.data}", flush=True); reqs = reqs + replay
    else:
        reqs = load_split(a.suite, "train") if manifest else build(a.n_per_source, "train", a.seed, exclude=holdout)
    if not manifest or a.data:
        # frozen suites are filtered to the training context when they are frozen (kev.suite.select_unique); records built
        # on the fly here are not, so apply the same rule instead of letting the strict encoder abort the run (issue #5)
        kept = [r for r in reqs if fits_context(tok, r)]
        if len(kept) < len(reqs):
            raise ValueError(f"preflight disagreement: {len(reqs)-len(kept)} records exceed 2048 state / 2048 branch / 4096 packed")
        reqs = kept
    if not reqs:
        raise ValueError("empty training set")
    forbidden = {r["_meta"]["source"] for r in reqs} & set(EVAL_ONLY)
    if forbidden:
        raise ValueError(f"training partition contains eval-only sources: {sorted(forbidden)}")
    if a.train_sources:
        wanted = set(a.train_sources.split(","))
        unknown = wanted - {r["_meta"]["source"] for r in reqs}
        if unknown: raise ValueError(f"--train_sources not in the training partition: {sorted(unknown)}")
        reqs = [r for r in reqs if r["_meta"]["source"] in wanted]
        print(f"ablation: training on {sorted(wanted)} -> {len(reqs)} records", flush=True)
    if manifest:
        from .study_v3 import validate_training
        # --data records are the user's own (validated by load_records; never an eval-only source by construction of their
        # names); the suite's rules apply to the replay sample and to suite-only runs
        validate_training([r for r in reqs if not a.data or not r["_meta"]["source"].startswith(("custom", "night2_"))], manifest)
        if a.data and any(r["_meta"]["source"] in set(EVAL_ONLY) | set(manifest.get("eval_only_sources", [])) for r in reqs):
            raise ValueError("--data contains an eval-only source")
    SYNTHETIC = ("legacy_policy", "compositional", "contrastive")
    if a.public_frac < 1:
        mix_rng = random.Random(source_seed(a.seed, "public_frac"))
        public = [r for r in reqs if r["_meta"]["source"] not in SYNTHETIC]; synth = [r for r in reqs if r["_meta"]["source"] in SYNTHETIC]
        keep = sorted(mix_rng.sample(range(len(public)), int(round(a.public_frac * len(public)))))
        reqs = [public[i] for i in keep] + synth
        print(f"mix: public_frac {a.public_frac} -> {len(keep)} public + {len(synth)} synthetic records", flush=True)
    if a.synthetic_repeat > 1:
        extra = [r for r in reqs if r["_meta"]["source"] in SYNTHETIC] * (a.synthetic_repeat - 1)
        reqs = reqs + extra
        print(f"mix: synthetic_repeat {a.synthetic_repeat} -> +{len(extra)} records", flush=True)
    clean_index = None
    curriculum_reports = []
    curriculum_precompute = {"records": 0, "packed_tokens_encoded": 0, "wall_seconds": 0.0}
    if dse.curriculum_prefix > 0:
        from .dse_curriculum import reorder_epoch
        clean_index = {}
        index_started = time.time()
        for req in reqs:
            rid = req["_meta"]["id"]
            if rid in clean_index:
                raise ValueError("C18 requires each source record ID exactly once; repeated inputs are unsupported")
            clean = materialize(req)
            encoded = strict_training_encode(model, tok, clean)
            clean_index[rid] = {"packed_tokens": len(encoded["ids"]), "max_candidates": max(len(q["options"]) for q in clean["questions"])}
            curriculum_precompute["records"] += 1
            curriculum_precompute["packed_tokens_encoded"] += len(encoded["ids"])
        curriculum_precompute["wall_seconds"] = time.time() - index_started
    if dse.supervised_views == 2 or dse.ema_mode == "mean_teacher":
        from .dse_views import make_views, record_loss
    surface_binding = None
    if dse.surface_probability > 0:
        from .dse_surface import apply_surface
        from .dse_surface_training import runtime_binding, private_rng as surface_rng, new_coverage, collect_coverage
        surface_binding = runtime_binding()
    suite_hash = digest(Path(a.suite) / "manifest.json") if manifest else None
    micro_per_epoch = math.ceil(len(reqs) / a.batch)
    steps = a.epochs * math.ceil(micro_per_epoch / a.accum)
    if dse.adapter_scheduler == "restart_adapter_group_only":
        from .dse_restart import AdapterGroupRestart, validate_restart_horizon
        validate_restart_horizon(steps, freeze_updates(dse, steps))
    dse_metadata = {**dse.metadata(), "source_sha256": source_hashes(), "target_inventory": target_inventory(reqs),
                    "planned_optimizer_steps": steps, "requested_source_records": a.epochs * len(reqs),
                    "lora_target_modules": getattr(model, "lora_target_modules", None),
                    "freeze_optimizer_updates": freeze_updates(dse, steps),
                    "parameter_counts": {"total": sum(p.numel() for p in model.parameters()),
                                         "head": sum(p.numel() for p in model.head.parameters()),
                                         "adapter_and_special_embeddings": sum(p.numel() for p in model.lm.parameters() if p.requires_grad),
                                         "initial_trainable": sum(p.numel() for p in model.head.parameters()) if dse.head_only or dse.freeze_fraction > 0 else sum(p.numel() for p in model.trainable_parameters())},
                    "processed_row_tokens_definition": "unpadded model input tokens including per-question state repetition on hybrid models",
                    "extra_view_tokens_definition": "additional perm_kl, C16 or EMA teacher forward passes; original none-pair siblings remain primary records",
                    "supervised_processed_row_tokens_definition": "primary passes plus C16 supervised second views; excludes legacy permutation-KL and EMA teacher views",
                    "curriculum_precompute": curriculum_precompute, "surface_binding": surface_binding}
    write_json(out_dir / "training_config.json", {"args": vars(a), "suite_sha256": suite_hash, "base_revision": revision, "init_source": init_source,
                                                "head_config": model.head_config,
                                                "ordinal_objective": "ranked_probability_score", "holdout": holdout, "dse": dse_metadata})
    print(f"{len(reqs)} training requests (holdout={holdout}), questions by type "
          f"{dict(Counter(q['qtype'] for r in reqs for q in materialize(r)['questions']))}")

    diagnostics = None
    if a.supplement_diagnostics:
        from .supplement_diagnostics import TrainingDiagnostics
        if not a.data or a.init_from or a.full_finetune:
            raise ValueError("supplement diagnostics require exact fresh LoRA training data")
        diagnostics = TrainingDiagnostics(model, out_dir, reqs, materialize, tok, autocast, digest(Path(a.data)))
    freeze = FreezeController(model, dse, steps) if dse.head_only or dse.freeze_fraction > 0 else None
    if freeze:
        print(f"frozen backbone for {freeze.freeze_steps}/{steps} updates; initial trainable parameters={sum(p.numel() for p in model.trainable_parameters())}", flush=True)
    head_params = list(model.head.parameters()); head_ids = {id(p) for p in head_params}
    groups = [{"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": a.lr},
              {"params": head_params, "lr": a.head_lr or a.lr}]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay,
                            foreach=False if a.full_finetune else None)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr or a.lr], total_steps=max(steps, 1), pct_start=0.1)
    adapter_restart = (AdapterGroupRestart(opt, total_steps=steps, freeze_steps=freeze.freeze_steps, max_lr=a.lr)
                       if dse.adapter_scheduler == "restart_adapter_group_only" else None)
    ema = teacher = None
    if dse.ema_mode != "off":
        from .dse_ema import AdapterHeadEMA
        ema = AdapterHeadEMA(model, dse.ema_decay)
        if dse.ema_mode == "mean_teacher":
            teacher = model.make_ema_teacher(ema.names)
            ema.copy_to(teacher)
    model.train(); t0 = time.time(); run = Counter(); step = 0; seen = 0
    if freeze: freeze.apply_modes()
    tokens_seen = peak_mem = 0
    row_tokens = extra_view_tokens = extra_view_row_tokens = source_records_seen = 0
    c16_tokens = c16_row_tokens = 0
    c16_summary = {"enabled": dse.supervised_views == 2, "records": 0, "two_views_records": 0, "single_view_records": 0,
                   "views_forwarded": 0, "questions": 0, "attempted_permuted_questions": 0, "applied_permuted_questions": 0,
                   "questions_by_type": Counter(), "question_reasons": Counter(), "second_view_rejections": Counter(), "by_source": {},
                   "native_ce_sum": 0.0, "js_sum": 0.0, "objective_sum": 0.0,
                   "loss_sums_definition": "sum of per-primary-record question means before original sibling/accum normalization"}
    ema_tokens = ema_row_tokens = 0
    ema_summary = {"mode": dse.ema_mode, "teacher_records_forwarded": 0, "permuted_teacher_records": 0,
                   "same_record_teacher_records": 0, "fallback_reasons": Counter(), "questions_by_type": Counter(),
                   "by_source": {}, "kl_sum": 0.0, "weighted_kl_sum": 0.0,
                   "native_ce_sum": 0.0, "objective_sum": 0.0, "weights_by_optimizer_update": [],
                   "loss_sums_definition": "sum of per-primary-record question means before original sibling/accum normalization",
                   "ramp_counter": "completed EMA updates equal successful optimizer steps before the next forward"}
    surface_summary = new_coverage() if dse.surface_probability > 0 else None
    stop_requested = False
    for ep in range(a.epochs):
        rng.shuffle(reqs)
        epoch_order = reqs  # reqs remains the persistent baseline-shuffle sequence.
        if clean_index is not None:
            epoch_order, curriculum_report = reorder_epoch(
                reqs, clean_index, fraction=dse.curriculum_prefix, seed=a.seed, epoch=ep,
                length_source=f"clean model.encode len(ids), base={a.base}, revision={revision}, strict max_state=2048 max_branch=2048 packed<=4096",
                candidate_source="maximum len(options) across clean materialize(req) questions, before augmentation")
            curriculum_report["diagnostics_scope"] = "planned full epoch order; progress below is actual processed source records"
            curriculum_report["progress"] = {"source_records_seen": 0, "prefix_records_seen": 0, "suffix_records_seen": 0,
                                             "by_source": Counter(), "epoch_completed": False}
            curriculum_reports.append(curriculum_report)
        for mb in range(micro_per_epoch):
            chunk = epoch_order[mb * a.batch : (mb + 1) * a.batch]
            recs, encs, perm_jobs, rec_ids, rec_sources = [], [], [], [], []
            c16_views, c16_jobs = [], []
            ema_records, ema_encs = [], []
            ema_weight = ema.ramp_weight(steps, maximum_weight=dse.ema_consistency_weight, ramp_fraction=dse.ema_ramp_fraction) if teacher is not None else 0.0
            if teacher is not None and mb % a.accum == 0:
                ema_summary["weights_by_optimizer_update"].append(ema_weight)
            for req in chunk:
                item_rng = random.Random(source_seed(a.seed, f"{ep}:{req['_meta']['id']}"))
                preserve = req['_meta'].get('r9_preserve_candidates', False)
                variants = ([augment(req, item_rng, p_none=0 if preserve else a.p_none, p_none_distract=0 if preserve else a.p_none_distract, p_distract=0 if preserve else a.p_distract)]
                            if dse.original_augmentation else [req])
                if dse.original_augmentation and not preserve and a.p_none_pair > 0 and item_rng.random() < a.p_none_pair:
                    variants += none_pair(req, item_rng)
                for sibling_index, v in enumerate(variants):
                    rec = materialize(v)
                    surface_encoding = None
                    if dse.surface_probability > 0:
                        original_surface_record = rec
                        surface_encodings = []
                        surface_encoding_attempts = 0
                        def validate_surface_record(view):
                            nonlocal surface_encoding_attempts
                            surface_encoding_attempts += 1
                            surface_encodings.append(strict_training_encode(model, tok, view))
                            return True
                        surface_view = apply_surface(rec, surface_rng(a.seed, ep, req["_meta"]["source"], req["_meta"]["id"], sibling_index),
                                                     probability=dse.surface_probability, op_count=dse.surface_ops_per_view,
                                                     preserve_candidates=preserve, validate_record=validate_surface_record)
                        rec = surface_view.record
                        surface_encoding = surface_encodings[-1] if surface_view.coverage["applied_ops"] else surface_encodings[0]
                        collect_coverage(surface_summary, original_surface_record, surface_view, req["_meta"]["source"],
                                         encoding_attempts=surface_encoding_attempts, accepted_encodings=len(surface_encodings))
                    if dse.supervised_views == 2:
                        view_encodings = []
                        def validate_view(view):
                            view_encodings.append(strict_training_encode(model, tok, view))
                            return True
                        view_seed = source_seed(a.seed, json.dumps(["c16", ep, req["_meta"]["source"], req["_meta"]["id"], sibling_index], separators=(",", ":")))
                        views = make_views(rec, random.Random(view_seed), validate_record=validate_view)
                        rec, enc = views.primary, view_encodings[0]
                        c16_views.append(views)
                        if views.secondary is not None:
                            c16_jobs.append((len(recs), view_encodings[1]))
                        collect_view_coverage(c16_summary, views, req["_meta"]["source"])
                    else:
                        enc = surface_encoding if surface_encoding is not None else model.encode(tok, rec, max_state=2048, max_branch=2048, strict=True)
                        if len(enc["ids"]) > 4096:
                            raise ValueError("training request exceeds 4096 packed tokens")
                    if teacher is not None:
                        # Validate/reuse the already encoded primary. The secondary has the same strict limits.
                        teacher_encodings = []
                        def validate_teacher_view(view):
                            teacher_encodings.append(enc if not teacher_encodings else strict_training_encode(model, tok, view))
                            return True
                        teacher_seed = source_seed(a.seed, json.dumps(["e9", ep, req["_meta"]["source"], req["_meta"]["id"], sibling_index], separators=(",", ":")))
                        teacher_views = make_views(rec, random.Random(teacher_seed), validate_record=validate_teacher_view)
                        permuted = teacher_views.secondary is not None
                        ema_records.append(teacher_views.secondary if permuted else teacher_views.primary)
                        ema_encs.append(teacher_encodings[1] if permuted else enc)
                        status = "permuted" if permuted else "same_record"
                        ema_summary[status + "_teacher_records"] += 1
                        ema_summary["by_source"].setdefault(req["_meta"]["source"], Counter())[status] += 1
                        ema_summary["questions_by_type"].update(teacher_views.coverage["questions_by_type"])
                        if not permuted:
                            ema_summary["fallback_reasons"][teacher_views.coverage["second_view_rejection"]] += 1
                    recs.append(rec); encs.append(enc); tokens_seen += len(enc["ids"])
                    row_tokens += processed_row_tokens(enc, model.hybrid)
                    rec_ids.append(req["_meta"]["id"]); rec_sources.append(req["_meta"]["source"])
                if a.perm_kl > 0 and item_rng.random() < a.perm_frac and any(q["qtype"] == "choice" and len(q["options"]) >= 3 for q in rec["questions"]):
                    rec2, perms = permuted_copy(rec, item_rng)
                    enc2 = model.encode(tok, rec2, max_state=2048, max_branch=2048, strict=True)
                    if len(enc2["ids"]) > 4096: raise ValueError("permuted training request exceeds 4096 packed tokens")
                    perm_jobs.append((len(recs) - 1, enc2, perms))
            with autocast:
                logits_b = model.forward_batch(encs)
                logits2_b = model.forward_batch([e for _, e, _ in perm_jobs]) if perm_jobs else []
                c16_logits2_b = model.forward_batch([e for _, e in c16_jobs]) if c16_jobs else []
                if teacher is not None:
                    with ema.teacher_prediction(teacher):
                        teacher_logits_b = teacher.forward_batch(ema_encs)
            for extra_enc in ema_encs:
                nt, nr = len(extra_enc["ids"]), processed_row_tokens(extra_enc, model.hybrid)
                tokens_seen += nt; row_tokens += nr
                extra_view_tokens += nt; extra_view_row_tokens += nr
                ema_tokens += nt; ema_row_tokens += nr
                ema_summary["teacher_records_forwarded"] += 1
            c16_second = {ri: logits for (ri, _), logits in zip(c16_jobs, c16_logits2_b)}
            for _, extra_enc in c16_jobs:
                nt, nr = len(extra_enc["ids"]), processed_row_tokens(extra_enc, model.hybrid)
                tokens_seen += nt; row_tokens += nr
                extra_view_tokens += nt; extra_view_row_tokens += nr
                c16_tokens += nt; c16_row_tokens += nr
            loss = 0.0
            for ri, (logits, rec, rid, src) in enumerate(zip(logits_b, recs, rec_ids, rec_sources)):
                if dse.supervised_views == 2:
                    secondary = c16_second.get(ri)
                    parts = record_loss(c16_views[ri], [z.float() for z in logits],
                                        [z.float() for z in secondary] if secondary is not None else None, js_weight=dse.js_weight)
                    ce = parts.total
                    c16_summary["native_ce_sum"] += parts.ce.item()
                    c16_summary["js_sum"] += parts.js.item()
                    c16_summary["objective_sum"] += parts.total.item()
                else:
                    ce = sum(loss_wrapper(z.float(), q, dev, a.ord_w, dse, question_loss) for z, q in zip(logits, rec["questions"])) / len(logits)
                run["ce"] += ce.item(); loss = loss + ce
                if teacher is not None:
                    consistency = ema_record_kl(rec, ema_records[ri], [z.float() for z in logits],
                                                [z.float() for z in teacher_logits_b[ri]])
                    if ema_weight > 0:
                        loss = loss + ema_weight * consistency
                    ema_summary["native_ce_sum"] += ce.item()
                    ema_summary["kl_sum"] += consistency.item()
                    ema_summary["weighted_kl_sum"] += ema_weight * consistency.item()
                    ema_summary["objective_sum"] += ce.item() + ema_weight * consistency.item()
                if anchors and rid in anchors and (anchor_sources is None or src in anchor_sources):
                    terms = [t for t in (anchor_loss(z.float(), q, anchors[rid].get(q["qid"]), dev) for z, q in zip(logits, rec["questions"])) if t is not None]
                    if terms:
                        kl_a = sum(terms) / len(terms); loss = loss + a.anchor_w * kl_a; run["anchor"] += kl_a.item(); run["anchor_n"] += 1
            for (ri, enc2, perms), logits2 in zip(perm_jobs, logits2_b):
                kl, n = 0.0, 0; tokens_seen += len(enc2["ids"])
                extra_view_tokens += len(enc2["ids"])
                extra_view_row_tokens += processed_row_tokens(enc2, model.hybrid)
                row_tokens += processed_row_tokens(enc2, model.hybrid)
                for z1, z2, perm in zip(logits_b[ri], logits2, perms):
                    if perm is None: continue
                    lp1 = F.log_softmax(z1.float(), -1); lp2 = F.log_softmax(z2.float(), -1)[torch.tensor([perm.index(j) for j in range(len(perm))], device=dev)]
                    kl = kl + 0.5 * (F.kl_div(lp2, lp1, log_target=True, reduction="sum") + F.kl_div(lp1, lp2, log_target=True, reduction="sum")); n += 1
                kl = kl / n; loss = loss + a.perm_kl * kl; run["kl"] += kl.item(); run["kl_n"] += 1
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            # weight by source records in the accumulation group so none-pair siblings do not inflate a record's share
            group_records = accumulation_records(len(reqs), a.batch, a.accum, mb) * (len(recs) / len(chunk))
            (loss / group_records).backward(); run["n"] += len(recs); seen += len(recs)
            source_records_seen += len(chunk)
            if clean_index is not None:
                progress = curriculum_report["progress"]
                start = progress["source_records_seen"]
                prefix_now = max(0, min(len(chunk), curriculum_report["prefix_records"] - start))
                progress["prefix_records_seen"] += prefix_now
                progress["suffix_records_seen"] += len(chunk) - prefix_now
                progress["source_records_seen"] += len(chunk)
                progress["by_source"].update(req["_meta"]["source"] for req in chunk)
                progress["epoch_completed"] = progress["source_records_seen"] == len(reqs)
            if dev == "mps": peak_mem = max(peak_mem, torch.mps.current_allocated_memory())
            elif dev == "cuda": peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())
            if (mb + 1) % a.accum == 0 or mb + 1 == micro_per_epoch:
                used_lrs = [group["lr"] for group in opt.param_groups]
                used_beta1 = [group["betas"][0] for group in opt.param_groups]
                adapter_count = sum(p.numel() for p in opt.param_groups[0]["params"])
                preclip_norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step()
                if diagnostics is not None:
                    diagnostics.record_update(step + 1, preclip_norm, used_lrs, used_beta1, adapter_count)
                if ema is not None:
                    # This trainer has no GradScaler/skipped-step path. Exceptions from step abort before EMA.
                    ema.update(step + 1)
                    if teacher is not None:
                        ema.copy_to(teacher)
                sched.step(); opt.zero_grad(); step += 1
                thawed = bool(freeze and step < steps and freeze.after_update(step, opt))
                if adapter_restart:
                    adapter_restart.after_update(step, thawed=thawed)
                if thawed:
                    schedule_message = (f"adapter-only restart horizon={adapter_restart.remaining}; head global horizon={steps}"
                                        if adapter_restart else f"retained OneCycle horizon={steps}")
                    print(f"thawed backbone adapters after {step} optimizer updates; {schedule_message}", flush=True)
                if dev == 'cuda':
                    peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())
                if step == 1:
                    print(f'first optimizer update complete; peak_cuda_gib={peak_mem / 2**30:.2f}', flush=True)
                if dev == "mps": torch.mps.empty_cache()
                if step % 10 == 0:
                    print(f"ep{ep} step {step}/{steps} loss {run['ce']/run['n']:.3f} kl {run['kl']/max(run['kl_n'],1):.3f} anchor {run['anchor']/max(run['anchor_n'],1):.3f} {(time.time()-t0)/seen:.3f}s/rec", flush=True)
                    run = Counter()
                if dse.stop_after_steps and step >= dse.stop_after_steps:
                    stop_requested = True
                    break
        if stop_requested: break
    stopped_early = step < steps
    if diagnostics is not None:
        diagnostics.finish(step)
    os.makedirs(a.out, exist_ok=True)
    model.lm.save_pretrained(str(out_dir / 'backbone') if a.full_finetune else a.out)
    torch.save({"head": model.head.state_dict(), "base": a.base, "base_revision": revision, "lora": 0 if a.full_finetune else a.lora, "head_dim": a.head_dim,
                "head_config": model.head_config,
                "full_finetune": a.full_finetune,
                "option_isolation": bool(a.option_isolation), "special_embeddings": bool(a.special_embeddings), "weights_dtype": a.weights_dtype,
                "holdout": holdout, "args": vars(a), "suite_sha256": suite_hash, "init_source": init_source,
                "dse": dse_metadata, "optimizer_steps": step, "stopped_early": stopped_early}, f"{a.out}/head.pt")
    if ema is not None:
        shadow_state = ema.state_dict()
        shadow_state["shadow"] = {name: value.cpu() for name, value in shadow_state["shadow"].items()}
        student_artifacts = ("head.pt", "adapter_config.json", "adapter_model.safetensors")
        if not all((out_dir / name).is_file() for name in student_artifacts):
            raise ValueError("EMA checkpoint binding requires saved head.pt, adapter_config.json and adapter_model.safetensors")
        student_hashes = {name: digest(out_dir / name) for name in student_artifacts}
        torch.save({"ema": shadow_state, "metadata": {**ema.metadata(), "mode": dse.ema_mode,
                    "checkpoint_schema_version": 1, "args": vars(a), "student_artifacts_sha256": student_hashes,
                    "base": a.base, "base_revision": revision, "suite_sha256": suite_hash,
                    "optimizer_steps": step, "planned_optimizer_steps": steps, "stopped_early": stopped_early,
                    "dse": dse_metadata}}, out_dir / "ema_shadow.pt")
        ema_summary["state"] = ema.metadata()
    tok.save_pretrained(a.out)
    write_json(out_dir / "training_metrics.json", {"wall_seconds": time.time() - t0, "records_seen": seen,
               "requested_records": a.epochs * len(reqs), "truncated_records": 0, "rejected_records": 0,
               "optimizer_steps": step, "forward_tokens": tokens_seen,
               "planned_optimizer_steps": steps, "stopped_early": stopped_early,
               "status": "stopped_early_sanity" if stopped_early else "complete",
               "source_records_seen": source_records_seen, "processed_row_tokens": row_tokens,
               "extra_view_tokens": extra_view_tokens, "extra_view_processed_row_tokens": extra_view_row_tokens,
               "primary_processed_row_tokens": row_tokens - extra_view_row_tokens,
               "supervised_processed_row_tokens": row_tokens - extra_view_row_tokens + c16_row_tokens,
               "c16_extra_view_tokens": c16_tokens, "c16_extra_view_processed_row_tokens": c16_row_tokens,
               "c16": c16_summary, "curriculum_epochs": curriculum_reports,
               "ema_teacher_forward_tokens": ema_tokens, "ema_teacher_processed_row_tokens": ema_row_tokens,
               "ema": ema_summary,
               "adapter_restart": None if adapter_restart is None else adapter_restart.metadata(),
               "surface": surface_summary,
               "dse": dse_metadata,
               "peak_device_bytes": peak_mem, "device": dev, "dtype": a.dtype, "batch": a.batch,
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)})
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
