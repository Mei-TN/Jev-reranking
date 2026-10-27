"""Observe head scores and gradients during training.

Detach observations from the existing forward pass and preserve RNG and
optimizer state. The fixed diagnostic sample comes from training data.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random

import torch


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(",", ":")) + "\n").encode()


def save_json(path, value):
    with Path(path).open("xb") as stream:
        raw = canonical(value)
        if stream.write(raw) != len(raw):
            raise IOError("short diagnostics write")
        stream.flush()
        os.fsync(stream.fileno())


def rng_state():
    return (random.getstate(), torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else [])


def rng_equal(before, after):
    return (before[0] == after[0] and torch.equal(before[1], after[1]) and
            len(before[2]) == len(after[2]) and
            all(torch.equal(a, b) for a, b in zip(before[2], after[2])))


def tensor_stats(value):
    value = value.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite observed head tensor")
    return {"numel": value.numel(), "sum_sq": float(value.square().sum()),
            "sum_abs": float(value.abs().sum()), "max_abs": float(value.abs().max())}


class HeadObserver:
    """Receives the actual intermediates; never recalculates a head forward."""
    def __init__(self):
        self.training_totals = {}
        self.sample_rows = None

    def __call__(self, *, q, k, logits, base_logits=None, residual_logits=None):
        if self.sample_rows is None and residual_logits is None:
            return
        values = {"q": q, "k": k, "logits": logits}
        if residual_logits is not None:
            values.update(base_logits=base_logits, residual_logits=residual_logits)
        with torch.no_grad():
            if self.sample_rows is not None:
                self.sample_rows.append({
                    "same_forward": True,
                    "head_input_projection_dtype": str(q.dtype),
                    **{name: value.detach().float().cpu().tolist()
                       for name, value in values.items()}})
            elif residual_logits is not None:
                for name in ("base_logits", "residual_logits"):
                    row = tensor_stats(values[name])
                    aggregate = self.training_totals.setdefault(name, dict(
                        numel=0, sum_sq=0.0, sum_abs=0.0, max_abs=0.0))
                    for key in ("numel", "sum_sq", "sum_abs"):
                        aggregate[key] += row[key]
                    aggregate["max_abs"] = max(aggregate["max_abs"], row["max_abs"])

    def update_summary(self):
        rows = self.training_totals
        self.training_totals = {}
        result = {}
        for name, row in rows.items():
            result[name] = {**row, "rms": math.sqrt(row["sum_sq"] / row["numel"]),
                            "mean_abs": row["sum_abs"] / row["numel"]}
        if rows:
            denominator = result["base_logits"]["rms"]
            result["residual_base_rms_ratio"] = (
                result["residual_logits"]["rms"] / denominator if denominator else None)
        return result


def select_sample(requests, materialize, each_qtype=32):
    """First 32 questions of each type, in exact training-file order."""
    counts = dict(choice=0, noul=0, score=0)
    records = []
    for raw in requests:
        rec = materialize(raw)
        selected = []
        for question in rec["questions"]:
            typ = question["qtype"]
            if typ not in counts:
                raise ValueError("unsupported diagnostic qtype")
            if counts[typ] < each_qtype:
                selected.append(question)
                counts[typ] += 1
        if selected:
            records.append({"record_id": raw["_meta"]["id"],
                            "record": {**rec, "questions": selected}})
        if all(n == each_qtype for n in counts.values()):
            break
    if not records:
        raise ValueError("empty fixed diagnostic sample")
    return {"schema": "jev-head-supplement-fixed-training-diagnostic-sample-v1",
            "selection": "first32_each_qtype_in_original_training_file_order",
            "permutation_policy": "one_reverse_order_for_each_selected_choice_K_ge_2",
            "held_out": False, "counts": counts, "records": records}


def capture_sample(model, tokenizer, sample, observer, autocast):
    before = rng_state()
    flags = [(module, module.training) for module in model.modules()]
    rows = []
    try:
        model.eval()
        with torch.no_grad():
            for entry in sample["records"]:
                rec = entry["record"]
                variants = [("original", rec, [None]*len(rec["questions"]))]
                choices, mappings = [], []
                for question in rec["questions"]:
                    if question["qtype"] != "choice" or len(question["options"]) < 2:
                        continue
                    mapping = list(reversed(range(len(question["options"]))))
                    permuted = {**question, "options": [question["options"][i] for i in mapping],
                                "label": mapping.index(question["label"])}
                    for field in ("keys", "target", "valid_mask"):
                        if question.get(field) is not None:
                            permuted[field] = [question[field][i] for i in mapping]
                    choices.append(permuted)
                    mappings.append(mapping)
                if choices:
                    variants.append(("fixed_reverse_choice", {**rec, "questions": choices}, mappings))
                for variant, observed_rec, permutations in variants:
                    enc = model.encode(tokenizer, observed_rec, max_state=2048, max_branch=2048, strict=True)
                    if len(enc["ids"]) > 4096:
                        raise ValueError("fixed diagnostic sample exceeds the original context")
                    observer.sample_rows = []
                    with autocast:
                        logits = model.forward(enc)
                    if len(logits) != len(observed_rec["questions"]) or len(observer.sample_rows) != len(logits):
                        raise ValueError("same-forward diagnostic question coverage differs")
                    for i, (question, output, observed) in enumerate(zip(
                            observed_rec["questions"], logits, observer.sample_rows)):
                        if observed["logits"] != output.detach().float().cpu().tolist():
                            raise ValueError("observer logits differ from actual returned logits")
                        if "opt_spans" in enc:
                            lengths = [end-start for start, end in enc["opt_spans"][i]]
                        else:
                            lengths = [sum(seg == i+1 and opt == j for seg, opt in zip(enc["seg"], enc["opt"]))-2
                                       for j in range(len(question["options"]))]
                        if len(lengths) != len(question["options"]) or min(lengths) < 0:
                            raise ValueError("actual option-content token coverage differs")
                        rows.append({"record_id": entry["record_id"], "question": question,
                                     "sample_variant": variant, "new_to_original_option": permutations[i],
                                     "option_content_token_lengths": lengths,
                                     "packed_token_count": len(enc["ids"]), **observed})
    finally:
        observer.sample_rows = None
        for module, flag in flags:
            module.training = flag
    if not rng_equal(before, rng_state()):
        raise ValueError("diagnostic forward changed RNG state; no restoration hides it")
    return rows


def parameter_snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()
            if name.startswith("head.") or "lora_" in name}


def update_norms(initial, final, scales):
    if initial.keys() != final.keys():
        raise ValueError("initial/final trainable state names differ")
    per_tensor = {}
    for name, before in initial.items():
        after = final[name]
        if before.shape != after.shape:
            raise ValueError("initial/final trainable state shapes differ")
        a, b = before.double(), after.double()
        n0, n1, delta = float(a.norm()), float(b.norm()), float((b-a).norm())
        per_tensor[name] = dict(initial_l2=n0, final_l2=n1, update_l2=delta,
                               relative_update=delta/n0 if n0 else None)
    effective = {}
    for name, factor in scales.items():
        an, bn = name + ".lora_A.default.weight", name + ".lora_B.default.weight"
        ai, bi, af, bf = (initial[an].double(), initial[bn].double(),
                          final[an].double(), final[bn].double())
        # Frobenius norm of scale*(Bf@Af-Bi@Ai), without a full dense allocation.
        squared = float(((af@af.T)*(bf.T@bf).T).sum() +
                        ((ai@ai.T)*(bi.T@bi).T).sum() -
                        2*((af@ai.T)*(bi.T@bf).T).sum())
        effective[name] = abs(factor)*math.sqrt(max(0.0, squared))
    return {"per_tensor": per_tensor, "effective_adapter_update_l2": effective,
            "head_update_l2": math.sqrt(sum(row["update_l2"]**2 for name, row in per_tensor.items()
                                           if name.startswith("head."))),
            "adapter_effective_update_l2": math.sqrt(sum(x*x for x in effective.values())),
            "definition": "initial/final difference, not optimization path length"}


class TrainingDiagnostics:
    def __init__(self, model, out_dir, requests, materialize, tokenizer, autocast, data_sha256):
        self.directory = Path(out_dir) / "diagnostics"
        self.directory.mkdir(exist_ok=False)
        self.model = model
        self.tokenizer = tokenizer
        self.autocast = autocast
        self.observer = HeadObserver()
        model.head._diagnostic_observer = self.observer
        before = rng_state()
        self.initial = parameter_snapshot(model)
        self.scales = {name: float(module.scaling["default"])
                       for name, module in model.named_modules()
                       if hasattr(module, "lora_A") and "default" in module.lora_A}
        self.sample = select_sample(requests, materialize)
        save_json(self.directory / "fixed_sample.json", self.sample)
        self._save_state("initial", self.initial)
        save_json(self.directory / "initial_capture.json", {
            "optimizer_steps": 0, "genuine_training_capture": True,
            "training_data_sha256": data_sha256, "head_config": model.head_config,
            "initial_head_sha256": digest(self.directory / "head_initial.safetensors"),
            "initial_adapter_sha256": digest(self.directory / "adapter_initial.safetensors"),
            "sample_sha256": digest(self.directory / "fixed_sample.json"),
            "adapter_scaling": self.scales, "nominal_seed_reconstruction": False})
        save_json(self.directory / "fixed_sample_initial.json", capture_sample(
            model, tokenizer, self.sample, self.observer, autocast))
        if not rng_equal(before, rng_state()):
            raise ValueError("diagnostic initialization changed RNG")
        self.log = (self.directory / "optimizer_updates.jsonl").open("xb")
        self.updates = 0

    def _save_state(self, label, state):
        from safetensors.torch import save_file
        for group in ("head", "adapter"):
            selected = {name.removeprefix("head.") if group == "head" else name: value.contiguous()
                        for name, value in state.items()
                        if name.startswith("head.") == (group == "head")}
            if not selected:
                raise ValueError("empty diagnostic parameter snapshot")
            path = self.directory / (group + "_" + label + ".safetensors")
            if path.exists():
                raise ValueError("diagnostic snapshot already exists")
            save_file(selected, str(path))
            with path.open("rb") as stream:
                os.fsync(stream.fileno())

    def record_update(self, step, preclip_norm, used_lrs, used_beta1, adapter_count, *,
                      adapter_active_count=None, adapter_non_none_grad_count=None, reverse_phase=None):
        if step != self.updates + 1:
            raise ValueError("diagnostic optimizer update order differs")
        value = float(preclip_norm.detach().cpu())
        if not math.isfinite(value):
            raise ValueError("nonfinite measured preclip norm")
        row = {"optimizer_update": step, "preclip_global_l2": value,
               "clip_threshold": 1.0, "clip_event": value > 1.0,
               "used_group_lrs": used_lrs, "used_group_beta1": used_beta1,
               "adapter_group_parameter_count": adapter_count,
               "active_adapter_parameter_count": adapter_count if adapter_active_count is None else adapter_active_count,
               "adapter_requires_grad_parameter_count": adapter_active_count,
               "adapter_non_none_grad_parameter_count": adapter_non_none_grad_count,
               "reverse_phase_for_this_update": reverse_phase,
               "h4_actual_same_forward": self.observer.update_summary()}
        raw = canonical(row)
        if self.log.write(raw) != len(raw):
            raise IOError("short optimizer diagnostic write")
        self.log.flush()
        os.fsync(self.log.fileno())
        self.updates = step

    def record_reverse_boundary(self, event):
        before = rng_state()
        save_json(self.directory / "reverse_freeze_boundary.json", event)
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if not rng_equal(before, rng_state()):
            raise ValueError("reverse boundary logging changed RNG")

    def finish(self, optimizer_steps):
        if optimizer_steps != self.updates:
            raise ValueError("diagnostic and actual optimizer counts differ")
        before = rng_state()
        final = parameter_snapshot(self.model)
        self._save_state("final", final)
        save_json(self.directory / "fixed_sample_final.json", capture_sample(
            self.model, self.tokenizer, self.sample, self.observer, self.autocast))
        save_json(self.directory / "update_norms.json", update_norms(self.initial, final, self.scales))
        self.log.close()
        if not rng_equal(before, rng_state()):
            raise ValueError("diagnostic finalization changed RNG")
        save_json(self.directory / "completion.json", {
            "status": "complete", "optimizer_steps": optimizer_steps,
            "training_population_only": True, "sample_counts": self.sample["counts"],
            "RNG_preserved": True, "optimizer_state_not_modified": True,
            "files_sha256": {p.name: digest(p) for p in sorted(self.directory.iterdir()) if p.is_file()},
            "claims_excluded": ["held_out_generalization", "causal_undertraining", "optimization_path_length"]})
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
