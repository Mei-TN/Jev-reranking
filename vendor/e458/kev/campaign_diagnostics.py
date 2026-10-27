"""Observe training scores, gradients, and parameter updates.

Hooks read tensors from the existing training step and return None, preserving
head inputs and gradients. All arms use the same hooks; residual-branch fields
are recorded only for models with that branch. Observations use training data.
"""
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import random
import socket
import sys

import torch

from .supplement_diagnostics import (HeadObserver, TrainingDiagnostics, canonical, capture_sample,
                                     digest, rng_equal, rng_state, save_json, tensor_stats)


def tensor_sha256(value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def group_digest(tensors, keep):
    """One digest over the sorted name:sha256 lines of a tensor group."""
    selected = [name for name in sorted(tensors) if keep(name)]
    if not selected:
        return None
    lines = "".join(f"{name}:{tensors[name]}\n" for name in selected)
    return {"tensors": len(selected), "sha256": hashlib.sha256(lines.encode()).hexdigest()}


def torch_rng_digests():
    """Seeded generators only: the CPU generator and every visible, already initialized CUDA generator."""
    cpu = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    cuda = None
    if torch.cuda.is_initialized():
        cuda = hashlib.sha256(b"".join(s.numpy().tobytes() for s in torch.cuda.get_rng_state_all())).hexdigest()
    return cpu, cuda


def unseeded_global_digest():
    """Python's and NumPy's global generators are never seeded by the trainer; an unchanged digest shows they were not drawn from."""
    import numpy
    state = numpy.random.get_state()
    h = hashlib.sha256(repr(random.getstate()).encode())
    h.update(state[1].tobytes())
    h.update(repr(state[2:]).encode())
    return h.hexdigest()


def nested(flat):
    out = {}
    for name, value in flat.items():
        group, _, key = name.partition(".")
        out.setdefault(group, {})[key] = value
    return out


class CampaignHeadObserver(HeadObserver):
    """One observer for every head kind.

    Sample mode is the unchanged supplement behavior. In training mode the base
    path of every arm gets the score statistics that only the residual head had
    before, and detached copies of the gradients reaching the head projections
    are kept until the optimizer update, for a derived split of the base and
    residual gradient paths without a second backward pass.
    """
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.residual = hasattr(head, "pair_mlp")
        self.inputs = self.features = None
        self.pending = []
        self.last_summary = {}
        self.scores = self._zero_scores()
        head.register_forward_pre_hook(self._head_inputs)
        if self.residual:
            head.pair_mlp.register_forward_pre_hook(self._pair_inputs)

    # Pre-hooks only remember the arguments of the call in progress.
    def _head_inputs(self, module, args):
        self.inputs = args

    def _pair_inputs(self, module, args):
        self.features = args[0]

    def _zero_scores(self):
        scores = dict(questions=0, options=0, base_centered_sum_sq=0.0)
        if self.residual:
            scores.update(residual_centered_sum_sq=0.0, base_residual_centered_dot=0.0,
                          full_base_argmax_flips=0, full_base_kl_sum=0.0, full_base_kl_max=0.0,
                          full_base_tv_sum=0.0, full_base_tv_max=0.0)
        return scores

    def __call__(self, *, q, k, logits, base_logits=None, residual_logits=None):
        inputs, features, self.inputs, self.features = self.inputs, self.features, None, None
        if self.sample_rows is not None:
            return super().__call__(q=q, k=k, logits=logits, base_logits=base_logits,
                                    residual_logits=residual_logits)
        base = logits if residual_logits is None else base_logits
        rows = {"base_logits": base}
        if residual_logits is not None:
            rows["residual_logits"] = residual_logits
        with torch.no_grad():
            for name, value in rows.items():
                row = tensor_stats(value)
                aggregate = self.training_totals.setdefault(name, dict(
                    numel=0, sum_sq=0.0, sum_abs=0.0, max_abs=0.0))
                for key in ("numel", "sum_sq", "sum_abs"):
                    aggregate[key] += row[key]
                aggregate["max_abs"] = max(aggregate["max_abs"], row["max_abs"])
            self._score_shape(base, residual_logits, logits)
        if logits.requires_grad:
            self._watch(q, k, logits, inputs, features)

    def _score_shape(self, base, residual, logits):
        """A softmax ignores a per-question constant, so the spread around the question mean is what can move a prediction."""
        scores = self.scores
        b = base.detach().to(device="cpu", dtype=torch.float64)
        centered = b - b.mean()
        scores["questions"] += 1
        scores["options"] += b.numel()
        scores["base_centered_sum_sq"] += float(centered.square().sum())
        if residual is None:
            return
        r = residual.detach().to(device="cpu", dtype=torch.float64)
        full = logits.detach().to(device="cpu", dtype=torch.float64)
        r_centered = r - r.mean()
        scores["residual_centered_sum_sq"] += float(r_centered.square().sum())
        scores["base_residual_centered_dot"] += float((centered * r_centered).sum())
        log_full, log_base = torch.log_softmax(full, -1), torch.log_softmax(b, -1)
        p_full = log_full.exp()
        kl = float((p_full * (log_full - log_base)).sum())
        tv = float(0.5 * (p_full - log_base.exp()).abs().sum())
        scores["full_base_argmax_flips"] += int(full.argmax() != b.argmax())
        scores["full_base_kl_sum"] += kl
        scores["full_base_kl_max"] = max(scores["full_base_kl_max"], kl)
        scores["full_base_tv_sum"] += tv
        scores["full_base_tv_max"] = max(scores["full_base_tv_max"], tv)

    def _watch(self, q, k, logits, inputs, features):
        if inputs is None or (self.residual and features is None):
            raise ValueError("head inputs of the observed forward were not seen")
        entry = {"q": q.detach(), "k": k.detach(),
                 "h_decide": inputs[0].detach(), "h_opts": inputs[1].detach()}

        def keep(name):
            def hook(grad):
                entry[name] = grad.detach().clone()
            return hook

        q.register_hook(keep("grad_q"))
        k.register_hook(keep("grad_k"))
        logits.register_hook(keep("grad_logits"))
        if self.residual and features.requires_grad:
            entry["features"] = features.detach()
            features.register_hook(keep("grad_features"))
        self.pending.append(entry)

    def update_summary(self):
        """The supplement's per-update score magnitudes, now also defined for a head without a residual."""
        rows, self.training_totals = self.training_totals, {}
        result = {name: {**row, "rms": math.sqrt(row["sum_sq"] / row["numel"]),
                         "mean_abs": row["sum_abs"] / row["numel"]} for name, row in rows.items()}
        if "residual_logits" in result:
            denominator = result["base_logits"]["rms"]
            result["residual_base_rms_ratio"] = (
                result["residual_logits"]["rms"] / denominator if denominator else None)
        self.last_summary = result
        return result

    def score_shape_summary(self):
        scores, self.scores = self.scores, self._zero_scores()
        return scores

    def gradient_paths(self):
        """Derived local split of this accumulation group's pre-clip gradients at the head projections.

        A double-precision recomputation from the gradients and features the run
        actually observed, not an exact per-operator decomposition of the
        mixed-precision backward: base = the bilinear score path from the
        observed dL/dlogits; residual = what arrived through the pair_mlp input,
        from the observed dL/dfeatures; closure = observed total - base -
        residual, kept so the error of the split is visible. hidden_state maps
        both parts through the FP32 q/k weights converted to double (not the
        autocast-rounded weights the backward used), and qk_parameters applies
        the same split to the accumulated q/k projection gradients. Nothing here
        decomposes the backbone or adapter parameter gradients.
        """
        pending, self.pending = self.pending, []
        if not pending:
            return {"questions": 0}
        head = self.head
        dp, scale = head.q.out_features, head.scale
        wq, wk = head.q.weight.detach().double(), head.k.weight.detach().double()
        total = {}

        def add(name, value):
            total[name] = total[name] + value if name in total else value

        parts = [torch.zeros_like(wq), torch.zeros(dp, dtype=wq.dtype, device=wq.device),
                 torch.zeros_like(wk), torch.zeros(dp, dtype=wk.dtype, device=wk.device)] if self.residual else None
        for entry in pending:
            wanted = ["grad_q", "grad_k", "grad_logits"] + (["grad_features"] if "features" in entry else [])
            missing = [name for name in wanted if name not in entry]
            if missing:
                raise ValueError("an observed head forward never received its gradient: " + ",".join(missing))
            q, k = entry["q"].double(), entry["k"].double()
            grad_q, grad_k, delta = entry["grad_q"].double(), entry["grad_k"].double(), entry["grad_logits"].double()
            h_decide, h_opts = entry["h_decide"].double(), entry["h_opts"].double()
            base_k = delta[:, None] * q[None, :] * scale
            base_q = (delta @ k) * scale
            residual_k = residual_q = None
            if "features" in entry:
                features, grad_features = entry["features"].double(), entry["grad_features"].double()
                queries, keys = features[:, :dp], features[:, dp:2 * dp]
                sign = torch.sign(queries - keys)
                residual_k = (grad_features[:, dp:2 * dp] + grad_features[:, 2 * dp:3 * dp] * queries
                              - grad_features[:, 3 * dp:] * sign)
                residual_q = (grad_features[:, :dp] + grad_features[:, 2 * dp:3 * dp] * keys
                              + grad_features[:, 3 * dp:] * sign).sum(0)
            for tag, grad, base, residual in (("option_projection", grad_k, base_k, residual_k),
                                              ("decision_projection", grad_q, base_q, residual_q)):
                add(tag + ".total_sq", grad.square().sum())
                add(tag + ".base_sq", base.square().sum())
                if residual is None:
                    add(tag + ".closure_sq", (grad - base).square().sum())
                else:
                    add(tag + ".residual_sq", residual.square().sum())
                    add(tag + ".base_dot_residual", (base * residual).sum())
                    add(tag + ".closure_sq", (grad - base - residual).square().sum())
            hidden_base_decide, hidden_base_opts = base_q @ wq, base_k @ wk
            add("hidden_state.base_sq", hidden_base_decide.square().sum() + hidden_base_opts.square().sum())
            if residual_k is not None:
                hidden_residual_decide, hidden_residual_opts = residual_q @ wq, residual_k @ wk
                add("hidden_state.residual_sq",
                    hidden_residual_decide.square().sum() + hidden_residual_opts.square().sum())
                add("hidden_state.base_dot_residual", (hidden_base_decide * hidden_residual_decide).sum()
                    + (hidden_base_opts * hidden_residual_opts).sum())
                parts[0] += torch.outer(residual_q, h_decide)
                parts[1] += residual_q
                parts[2] += residual_k.T @ h_opts
                parts[3] += residual_k.sum(0)
        grads = [head.q.weight.grad, head.q.bias.grad, head.k.weight.grad, head.k.bias.grad]
        if parts is not None and all(g is not None for g in grads):
            add("qk_parameters.residual_sq", sum(part.square().sum() for part in parts))
            add("qk_parameters.total_dot_residual",
                sum((g.detach().double() * part).sum() for g, part in zip(grads, parts)))
        names = list(total)
        values = torch.stack([total[name] for name in names]).tolist()
        return {"questions": len(pending), **nested(dict(zip(names, values)))}


class CampaignDiagnostics(TrainingDiagnostics):
    """The supplement diagnostics plus campaign receipts, per-update records and intermediate checkpoints."""
    def __init__(self, model, out_dir, requests, materialize, tokenizer, autocast, data_sha256):
        super().__init__(model, out_dir, requests, materialize, tokenizer, autocast, data_sha256)
        self.campaign_directory = Path(out_dir) / "campaign"
        self.campaign_directory.mkdir(exist_ok=False)
        self.checkpoint_root = Path(out_dir) / "checkpoints"
        self.data_sha256 = data_sha256
        self.names = {id(p): name for name, p in model.named_parameters()}
        self.observer = CampaignHeadObserver(model.head)
        model.head._diagnostic_observer = self.observer
        self.campaign_log = None

    def begin(self, optimizer, data_rng, planned_steps, checkpoint_steps, timeout_seconds, save_checkpoint):
        """Step-0 receipt: actual initial tensors, optimizer membership and seeded RNG state, for cross-arm comparison."""
        before = rng_state()
        tensors = {name: tensor_sha256(value) for name, value in self.initial.items()}
        in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
        groups = []
        for role, group in zip(("adapter", "head"), optimizer.param_groups):
            members = [self.names[id(p)] for p in group["params"]]
            row = {"role": role, "tensors": len(members), "elements": sum(p.numel() for p in group["params"]),
                   "names_sha256": hashlib.sha256("\n".join(members).encode()).hexdigest(),
                   "initial_lr": group["lr"], "max_lr": group.get("max_lr"), "weight_decay": group["weight_decay"]}
            if role == "head":
                row["names"] = members
            groups.append(row)
        cpu, cuda = torch_rng_digests()
        self.unseeded_at_entry = unseeded_global_digest()
        packages = {}
        for name in ("peft", "transformers", "safetensors", "tokenizers", "numpy"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        on_cuda = torch.cuda.is_initialized()
        save_json(self.campaign_directory / "initial_receipt.json", {
            "schema": "jev-e1e2-campaign-initial-receipt-v1", "optimizer_steps": 0,
            "nominal_seed_reconstruction": False, "head_config": self.model.head_config,
            "training_data_sha256": self.data_sha256, "planned_optimizer_steps": planned_steps,
            "checkpoint_steps": list(checkpoint_steps), "timeout_seconds": timeout_seconds,
            "tensor_sha256": tensors,
            "tensor_groups": {
                "adapter": group_digest(tensors, lambda name: not name.startswith("head.")),
                "head_qk": group_digest(tensors, lambda name: name.startswith(("head.q.", "head.k."))),
                "pair_mlp": group_digest(tensors, lambda name: name.startswith("head.pair_mlp."))},
            "head_parameters": [{"name": name, "requires_grad": p.requires_grad, "in_optimizer": id(p) in in_optimizer}
                                for name, p in self.model.head.named_parameters()],
            "optimizer": {"class": type(optimizer).__name__, "groups": groups},
            "rng_at_loop_entry": {
                "torch_cpu_sha256": cpu, "torch_cuda_sha256": cuda,
                "data_order_rng_sha256": hashlib.sha256(repr(data_rng.getstate()).encode()).hexdigest()},
            "environment": {
                "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
                "packages": packages, "host": socket.gethostname(), "device": str(self.model.device),
                "gpu_name": torch.cuda.get_device_name(0) if on_cuda else None,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "threads": torch.get_num_threads(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}})
        self.planned_steps = planned_steps
        self.checkpoint_steps = set(checkpoint_steps)
        self.save_checkpoint = save_checkpoint
        self.checkpoints = []
        self.input_hash = hashlib.sha256()
        self.rng_trace = hashlib.sha256()
        self.clipped_updates = self.campaign_updates = 0
        self.group = dict(objective_sum=0.0, records=0, source_records=0, microbatches=0)
        self.clip = self.paths = None
        self.campaign_log = (self.campaign_directory / "updates.jsonl").open("xb")
        if not rng_equal(before, rng_state()):
            raise ValueError("campaign receipt changed RNG")

    def after_microbatch(self, record_ids, records, encodings, source_records, loss):
        """Hash what the model was actually given, after augmentation, in the order it was given."""
        for record_id, record, encoding in zip(record_ids, records, encodings):
            self.input_hash.update(canonical([record_id, encoding["ids"],
                                              [[q["label"], q.get("target")] for q in record["questions"]]]))
        group = self.group
        group["objective_sum"] += float(loss.detach())
        group["records"] += len(records)
        group["source_records"] += source_records
        group["microbatches"] += 1

    def before_clip(self, optimizer):
        """Read the accumulated gradients before clip_grad_norm_ rescales them in place."""
        norms = {"adapter": [], "head_qk": [], "pair_mlp": []}
        for role, group in zip(("adapter", "head"), optimizer.param_groups):
            for p in group["params"]:
                if p.grad is None:
                    continue
                key = role if role == "adapter" else (
                    "pair_mlp" if self.names[id(p)].startswith("head.pair_mlp.") else "head_qk")
                norms[key].append(torch.linalg.vector_norm(p.grad.detach()))
        present = [key for key, values in norms.items() if values]
        squares = dict(zip(present, torch.stack(
            [torch.stack(norms[key]).double().square().sum() for key in present]).tolist())) if present else {}
        shared = squares.get("adapter", 0.0) + squares.get("head_qk", 0.0)
        self.clip = {
            "group_l2": {key: math.sqrt(squares[key]) if key in squares else None for key in norms},
            "recomputed_global_l2": math.sqrt(shared + squares.get("pair_mlp", 0.0)),
            "shared_only_l2": math.sqrt(shared)}
        self.paths = self.observer.gradient_paths()

    def record_update(self, step, preclip_norm, used_lrs, used_beta1, adapter_count):
        super().record_update(step, preclip_norm, used_lrs, used_beta1, adapter_count)
        if self.clip is None or step != self.campaign_updates + 1:
            raise ValueError("campaign update record order differs")
        norm = preclip_norm.detach()
        value = float(norm)
        # The expression clip_grad_norm_ applied to every gradient, evaluated on the norm it returned.
        coefficient = float(torch.clamp(1.0 / (norm + 1e-6), max=1.0))
        # The ratio compares this update's global clip scaling of the shared gradients with the scaling a shared-only
        # norm would give. It says nothing about the AdamW update or step size. Both sides use the group norms read
        # before clipping, so it is exactly 1 without a branch gradient. The original global clipping is unchanged.
        recomputed_coefficient = min(1.0, 1.0 / (self.clip["recomputed_global_l2"] + 1e-6))
        shared_coefficient = min(1.0, 1.0 / (self.clip["shared_only_l2"] + 1e-6))
        cpu, cuda = torch_rng_digests()
        rng_digest = hashlib.sha256((cpu + ":" + (cuda or "")).encode()).hexdigest()
        self.rng_trace.update(bytes.fromhex(rng_digest))
        row = {"optimizer_update": step,
               "clip": {"threshold": 1.0, "preclip_global_l2": value, "coefficient": coefficient,
                        "postclip_global_l2_derived": value * coefficient, **self.clip,
                        "shared_only_coefficient": shared_coefficient,
                        "coefficient_over_shared_only": recomputed_coefficient / shared_coefficient},
               "used_group_lrs": used_lrs, "used_group_beta1": used_beta1,
               "objective": self.group, "score_magnitude": self.observer.last_summary,
               "score_shape": self.observer.score_shape_summary(), "gradient_path": self.paths,
               "torch_rng_sha256": rng_digest[:16],
               "input_stream_sha256": self.input_hash.copy().hexdigest()[:16]}
        raw = canonical(row)
        if self.campaign_log.write(raw) != len(raw):
            raise IOError("short campaign update write")
        self.campaign_log.flush()
        self.clipped_updates += coefficient < 1.0
        self.campaign_updates = step
        self.group = dict(objective_sum=0.0, records=0, source_records=0, microbatches=0)
        self.clip = self.paths = None

    def after_update(self, step):
        if step not in self.checkpoint_steps:
            return
        before = rng_state()
        modes = [module.training for module in self.model.modules()]
        trainable = [p.requires_grad for p in self.model.parameters()]
        name = f"step_{step:06d}"
        directory = self.checkpoint_root / name
        directory.mkdir(parents=True, exist_ok=False)
        self.save_checkpoint(directory, step)
        sample = self.campaign_directory / f"fixed_sample_{name}.json"
        save_json(sample, capture_sample(self.model, self.tokenizer, self.sample, self.observer, self.autocast))
        if (not rng_equal(before, rng_state()) or modes != [module.training for module in self.model.modules()]
                or trainable != [p.requires_grad for p in self.model.parameters()]):
            raise ValueError("intermediate checkpoint changed RNG, module modes or trainable flags")
        self.checkpoints.append({
            "optimizer_update": step, "directory": "checkpoints/" + name, "fixed_sample": sample.name,
            "files_sha256": {member: digest(directory / member)
                             for member in ("head.pt", "adapter_model.safetensors", "adapter_config.json")}})
        self.campaign_log.flush()
        os.fsync(self.campaign_log.fileno())

    def finish(self, optimizer_steps):
        super().finish(optimizer_steps)
        if optimizer_steps != self.campaign_updates:
            raise ValueError("campaign and actual optimizer counts differ")
        self.campaign_log.flush()
        os.fsync(self.campaign_log.fileno())
        self.campaign_log.close()
        save_json(self.campaign_directory / "completion.json", {
            "status": "complete", "optimizer_updates": optimizer_steps,
            "planned_optimizer_steps": self.planned_steps, "clipped_updates": self.clipped_updates,
            "checkpoints": self.checkpoints,
            "input_stream_sha256": self.input_hash.hexdigest(),
            "torch_rng_trace_sha256": self.rng_trace.hexdigest(),
            "unseeded_python_numpy_rng_unchanged": unseeded_global_digest() == self.unseeded_at_entry,
            "files_sha256": {p.name: digest(p) for p in sorted(self.campaign_directory.iterdir()) if p.is_file()},
            "definitions": {
                "scope": "training-mode forwards of the training stream; pre-clip gradients including the "
                         "accumulation normalization; not held-out evaluation",
                "clip.coefficient": "min(1, 1/(preclip_global_l2+1e-6)), the factor applied to every gradient",
                "clip.shared_only_coefficient": "the same factor if only adapter and q/k gradients entered the norm. "
                                                "coefficient_over_shared_only compares the two scalings of the "
                                                "current shared gradients; it is not an AdamW update or step-size "
                                                "ratio, and the run's global clipping is the original one",
                "score_magnitude": "the supplement's raw per-update base/residual score statistics",
                "score_shape": "per-question mean-centered sums; full/base compare the returned logits with the "
                               "pre-addition base scores of the same forward",
                "gradient_path": "derived local diagnostic in double from the observed gradients and features, not "
                                 "an exact mixed-precision per-operator decomposition: sums over the update's "
                                 "question forwards of squared L2 norms at the option and decision projections; "
                                 "closure_sq is observed total minus base minus residual. hidden_state uses the "
                                 "FP32 q/k weights converted to double; qk_parameters splits the accumulated q/k "
                                 "projection gradient. Backbone and adapter parameter gradients are not decomposed",
                "checkpoints": "optional saved endpoints with a fixed training-sample observation; not held-out "
                               "performance until separately evaluated",
                "torch_rng_sha256": "seeded CPU and CUDA generator state after each update's forward/backward",
                "input_stream_sha256": "running digest of record id, token ids, labels and targets per microbatch"}})
        descriptor = os.open(self.campaign_directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
