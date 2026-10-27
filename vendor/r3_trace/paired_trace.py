"""One bounded real-model pointer/frozen-zero trajectory diagnostic.

The frozen protocol is PROTOCOL.md. This deliberately does not invoke kev.train
or write a production checkpoint; it reproduces its first four update groups.
"""
import argparse
import hashlib
import json
import os
import random
import socket
import time
import traceback
from pathlib import Path

import torch

from kev.data import augment, load_records, materialize, none_pair, source_seed
from kev.dse_options import DSEOptions, loss_wrapper
from kev.model import DecisionModel, load_tokenizer
from kev.train import question_loss


SEED = 44
STEPS = 4
ACCUM = 8
FULL_HORIZON = 1932
PEAK_LR = 2.5e-5
BASE_REVISION = "56f0584bb40578a2c33b1b40a08ccd17243ad710"
POINTER = {"version": 1, "kind": "pointer", "dim": 256, "readout": "end", "rng_anchor_dim": 256}
RESIDUAL = {"version": 1, "kind": "residual_pair_mlp", "dim": 256, "readout": "end", "rng_anchor_dim": 256, "mlp_hidden": 128}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    blob = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        os.unlink(path)
        raise
    dfd = os.open(str(Path(path).parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def bytes_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def compare_tensor(a, b):
    if a is None or b is None:
        return {"both_absent": a is None and b is None, "max_abs": None, "exact": a is None and b is None}
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"both_absent": False, "shape_dtype_equal": False, "max_abs": None, "exact": False}
    with torch.no_grad():
        af, bf = a.detach().float(), b.detach().float()
        delta = (af - bf).abs()
        maximum = float(delta.max().item()) if delta.numel() else 0.0
        finite = bool(torch.isfinite(af).all().item() and torch.isfinite(bf).all().item())
        return {"both_absent": False, "shape_dtype_equal": True, "max_abs": maximum,
                "exact": bool(torch.equal(a.detach(), b.detach())), "finite": finite}


def summarize(comparisons):
    values = list(comparisons.values())
    return {"count": len(values), "exact": all(v["exact"] for v in values),
            "max_abs": max((v["max_abs"] or 0.0 for v in values), default=0.0),
            "nonfinite": [k for k, v in comparisons.items() if v.get("finite") is False],
            "mismatched": [k for k, v in comparisons.items() if not v["exact"]][:20]}


def compare_named(a, b):
    assert set(a) == set(b), (sorted(set(a)-set(b))[:5], sorted(set(b)-set(a))[:5])
    return summarize({k: compare_tensor(a[k], b[k]) for k in sorted(a)})


def trainable(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def make_optimizer(model):
    head = [p for p in model.head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head}
    groups = [{"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": PEAK_LR},
              {"params": head, "lr": PEAK_LR}]
    opt = torch.optim.AdamW(groups, lr=PEAK_LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[PEAK_LR, PEAK_LR],
                                                 total_steps=FULL_HORIZON, pct_start=0.1)
    return opt, sched


def optimizer_state(opt, named):
    out = {}
    for name, param in named.items():
        state = opt.state.get(param, {})
        out[name] = {k: state.get(k) for k in ("step", "exp_avg", "exp_avg_sq")}
    return out


def compare_optimizer(opt_a, na, opt_b, nb):
    sa, sb = optimizer_state(opt_a, na), optimizer_state(opt_b, nb)
    assert set(sa) == set(sb)
    flattened = {}
    for name in sa:
        for key in ("step", "exp_avg", "exp_avg_sq"):
            flattened[name + "." + key] = compare_tensor(sa[name][key], sb[name][key])
    return summarize(flattened)


def logits_comparison(aa, bb):
    if len(aa) != len(bb) or any(len(a) != len(b) for a, b in zip(aa, bb)):
        raise ValueError("question/logit count mismatch")
    return summarize({f"{i}.{j}": compare_tensor(a, b)
                      for i, (ra, rb) in enumerate(zip(aa, bb))
                      for j, (a, b) in enumerate(zip(ra, rb))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--expected-data-sha256", required=True)
    ap.add_argument("--expected-base-config-sha256", required=True)
    args = ap.parse_args()
    output = Path(args.output)
    output.mkdir(mode=0o755, parents=False, exist_ok=False)
    started = time.time()
    try:
        if sha(args.data) != args.expected_data_sha256:
            raise ValueError("F2 data SHA mismatch")
        if sha(Path(args.base) / "config.json") != args.expected_base_config_sha256:
            raise ValueError("base config SHA mismatch")
        if torch.cuda.device_count() != 1:
            raise ValueError("exactly one H20 must be visible")
        if "H20" not in torch.cuda.get_device_name(0):
            raise ValueError("visible GPU is not an H20")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        tok = load_tokenizer(args.base, revision=BASE_REVISION)
        reqs = load_records(args.data)
        if len(reqs) != 15456:
            raise ValueError(f"expected 15456 F2 records, got {len(reqs)}")
        random.Random(SEED).shuffle(reqs)
        fixed = reqs[:STEPS * ACCUM]
        ids = [r["_meta"]["id"] for r in fixed]
        if len(set(ids)) != len(ids):
            raise ValueError("selected source IDs are not unique")
        publish(output / "FIXED_BATCHES.json", {"seed": SEED, "records": len(reqs), "updates": STEPS,
                                                "accum": ACCUM, "ids": ids, "data_sha256": sha(args.data)})

        def construct(config):
            torch.manual_seed(SEED)
            model = DecisionModel(args.base, tok, "cuda", lora=64, revision=BASE_REVISION,
                                  head_dim=256, lora_targets="all", dtype=torch.float32,
                                  head_dropout=0.0, lora_dropout=0.05, head_config=config)
            model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.lm.config.use_cache = False
            model.train()
            return model, torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()

        pointer, ptr_cpu, ptr_cuda = construct(POINTER)
        sham, sham_cpu, sham_cuda = construct(RESIDUAL)
        frozen = {n: p for n, p in sham.head.pair_mlp.named_parameters()}
        if set(frozen) != {"0.weight", "0.bias", "2.weight", "2.bias"}:
            raise ValueError("unexpected residual tensor set")
        frozen_start = {n: bytes_sha(p) for n, p in frozen.items()}
        if any(bool(p.detach().count_nonzero().item()) for n, p in frozen.items() if n.startswith("2.")):
            raise ValueError("residual last layer is not exactly zero")
        for p in frozen.values():
            p.requires_grad_(False)
        shared_pointer, shared_sham = trainable(pointer), trainable(sham)
        init = compare_named(shared_pointer, shared_sham)
        if not init["exact"]:
            raise ValueError("shared trainable initialization differs")
        if not torch.equal(ptr_cpu, sham_cpu) or not torch.equal(ptr_cuda, sham_cuda):
            raise ValueError("post-construction RNG states differ")
        opt_ptr, sched_ptr = make_optimizer(pointer)
        opt_sham, sched_sham = make_optimizer(sham)
        if set(shared_pointer) != set(shared_sham):
            raise ValueError("shared trainable names differ")
        initial_optimizer = compare_optimizer(opt_ptr, shared_pointer, opt_sham, shared_sham)
        if not initial_optimizer["exact"]:
            raise ValueError("initial optimizer states differ")
        publish(output / "INITIAL.json", {"host": socket.gethostname(),
                                            "boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                                            "gpu": torch.cuda.get_device_name(0),
                                            "torch": torch.__version__, "base": args.base,
                                            "base_config_sha256": args.expected_base_config_sha256,
                                            "initial_shared": init, "initial_optimizer": initial_optimizer,
                                            "frozen_branch_sha256": frozen_start,
                                            "post_construction_rng_equal": True,
                                            "full_scheduler_horizon": FULL_HORIZON,
                                            "peak_lr": PEAK_LR, "elapsed_seconds": time.time()-started})
        options = DSEOptions().validate()
        first_divergence = None
        for update in range(STEPS):
            micro = []
            for j in range(ACCUM):
                req = fixed[update * ACCUM + j]
                item_rng = random.Random(source_seed(SEED, f"0:{req['_meta']['id']}"))
                preserve = req["_meta"].get("r9_preserve_candidates", False)
                variants = [augment(req, item_rng, p_none=0 if preserve else 0.1,
                                    p_none_distract=0 if preserve else 0.12,
                                    p_distract=0 if preserve else 0.15)]
                if not preserve and item_rng.random() < 0.25:
                    variants += none_pair(req, item_rng)
                recs = [materialize(v) for v in variants]
                encs = [pointer.encode(tok, rec, max_state=2048, max_branch=2048, strict=True) for rec in recs]
                if any(len(enc["ids"]) > 4096 for enc in encs):
                    raise ValueError("fixed record exceeds production packed-token limit")
                before_cpu = torch.get_rng_state().clone()
                before_cuda = torch.cuda.get_rng_state().clone()

                def arm(model):
                    torch.set_rng_state(before_cpu)
                    torch.cuda.set_rng_state(before_cuda)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model.forward_batch(encs)
                    loss = sum(sum(loss_wrapper(z.float(), q, "cuda", 0.0, options, question_loss)
                                   for z, q in zip(row, rec["questions"])) / len(row)
                               for row, rec in zip(logits, recs))
                    if not bool(torch.isfinite(loss).item()):
                        raise ValueError("nonfinite loss")
                    (loss / (ACCUM * len(recs))).backward()
                    return logits, loss, torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()

                lp, loss_p, cpu_p, cuda_p = arm(pointer)
                ls, loss_s, cpu_s, cuda_s = arm(sham)
                torch.set_rng_state(cpu_p)
                torch.cuda.set_rng_state(cuda_p)
                entry = {"id": req["_meta"]["id"], "siblings": len(recs),
                         "packed_tokens": [len(e["ids"]) for e in encs],
                         "logits": logits_comparison(lp, ls),
                         "loss": compare_tensor(loss_p, loss_s),
                         "accumulated_shared_grad": compare_named(
                             {n: p.grad for n, p in shared_pointer.items()},
                             {n: p.grad for n, p in shared_sham.items()}),
                         "post_forward_backward_cpu_rng_equal": bool(torch.equal(cpu_p, cpu_s)),
                         "post_forward_backward_cuda_rng_equal": bool(torch.equal(cuda_p, cuda_s))}
                if first_divergence is None and (not entry["logits"]["exact"] or not entry["loss"]["exact"]
                                                 or not entry["accumulated_shared_grad"]["exact"]
                                                 or not entry["post_forward_backward_cpu_rng_equal"]
                                                 or not entry["post_forward_backward_cuda_rng_equal"]):
                    first_divergence = {"update": update+1, "microbatch": j+1, "stage": "forward_backward"}
                micro.append(entry)
                del lp, ls, loss_p, loss_s
            grad_before = compare_named({n: p.grad for n, p in shared_pointer.items()},
                                        {n: p.grad for n, p in shared_sham.items()})
            norm_ptr = torch.nn.utils.clip_grad_norm_(pointer.trainable_parameters(), 1.0)
            norm_sham = torch.nn.utils.clip_grad_norm_(sham.trainable_parameters(), 1.0)
            grad_after = compare_named({n: p.grad for n, p in shared_pointer.items()},
                                       {n: p.grad for n, p in shared_sham.items()})
            pre_ptr = {n: p.detach().clone() for n, p in shared_pointer.items()}
            pre_sham = {n: p.detach().clone() for n, p in shared_sham.items()}
            lr_before = [[g["lr"] for g in opt.param_groups] for opt in (opt_ptr, opt_sham)]
            opt_ptr.step(); opt_sham.step()
            moments = compare_optimizer(opt_ptr, shared_pointer, opt_sham, shared_sham)
            updates = summarize({n: compare_tensor(shared_pointer[n].detach() - pre_ptr[n],
                                                    shared_sham[n].detach() - pre_sham[n])
                                 for n in shared_pointer})
            parameters = compare_named(shared_pointer, shared_sham)
            sched_ptr.step(); sched_sham.step()
            lr_after = [[g["lr"] for g in opt.param_groups] for opt in (opt_ptr, opt_sham)]
            opt_ptr.zero_grad(); opt_sham.zero_grad()
            frozen_after = {n: bytes_sha(p) for n, p in frozen.items()}
            if any(p.grad is not None for p in frozen.values()) or frozen_after != frozen_start:
                raise ValueError("frozen branch changed or received a gradient")
            if any(report["nonfinite"] for report in (grad_before, grad_after, moments, updates, parameters)):
                raise ValueError("nonfinite shared trajectory value")
            if first_divergence is None and (not grad_before["exact"] or not grad_after["exact"] or
                                             not moments["exact"] or not updates["exact"] or not parameters["exact"] or
                                             lr_before[0] != lr_before[1] or lr_after[0] != lr_after[1]):
                first_divergence = {"update": update+1, "stage": "gradient_or_update"}
            receipt = {"update": update+1, "microbatches": micro, "grad_preclip": grad_before,
                       "grad_postclip": grad_after, "clip_pre_norm": [float(norm_ptr.item()), float(norm_sham.item())],
                       "moments": moments, "updates": updates, "parameters": parameters,
                       "lr_before": lr_before, "lr_after": lr_after,
                       "frozen_branch_unchanged": True, "elapsed_seconds": time.time()-started,
                       "first_divergence_so_far": first_divergence}
            publish(output / f"UPDATE_{update+1}.json", receipt)
            del pre_ptr, pre_sham
        publish(output / "RESULT.json", {"status": "FOUR_UPDATES_COMPLETE", "updates": STEPS,
                                         "first_divergence": first_divergence,
                                         "frozen_branch_sha256": frozen_start,
                                         "elapsed_seconds": time.time()-started})
    except BaseException as exc:
        publish(output / "FAILURE.json", {"status": "FAILED", "type": type(exc).__name__,
                                           "message": str(exc), "traceback": traceback.format_exc(),
                                           "elapsed_seconds": time.time()-started})
        raise


if __name__ == "__main__":
    main()
