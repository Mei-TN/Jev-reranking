"""Initialize a readout or adapter-scope comparison from its matched pointer run.

Copy head tensors and LoRA tensors with matching names and shapes from saved
step-zero records, then verify the copied values. Parameters with different
shapes retain their own initialization. Restore the recorded torch CPU and CUDA
generator states while verifying the separate data-order and augmentation
generators. Missing or inconsistent records stop initialization.

After training begins, each adapter configuration consumes its own dropout
stream; configurations with different modules need not have identical masks.
"""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

import torch

from . import e458_arms
from .campaign_diagnostics import group_digest, tensor_sha256
from .dse_options import source_hashes
from .heads import normalize_head_config
from .supplement_diagnostics import digest, rng_equal, rng_state, save_json

SCHEMA = "jev-e458-reference-initialization-v1"
LORA_SUFFIXES = (".lora_A.default.weight", ".lora_B.default.weight")
HEAD_TENSORS = ["k.bias", "k.weight", "q.bias", "q.weight"]
# Step-zero records of the reference run, relative to its output directory. All exist before its first update.
REFERENCE_FILES = {
    "head_initial": "diagnostics/head_initial.safetensors",
    "adapter_initial": "diagnostics/adapter_initial.safetensors",
    "initial_capture": "diagnostics/initial_capture.json",
    "binary_map": "r3_control/initial_complete_trainable_tensor_binary_map.json",
    "generator_record": "r3_control/model_to_device_complete.RNG.json",
    "initial_receipt": "campaign/initial_receipt.json",
    "training_config": "training_config.json"}


class ReferenceUnusable(ValueError):
    """Required reference records are missing or inconsistent."""


def read_json(path):
    return json.loads(Path(path).read_bytes())


def same_bytes(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def leaf_of(name):
    """Target module leaf of a LoRA tensor name ...<leaf>.lora_A.default.weight."""
    for suffix in LORA_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)].rsplit(".", 1)[-1]
    raise ValueError("not a plain LoRA tensor name: " + name)


def plain(value):
    """A PEFT configuration value as JSON."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(plain(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(getattr(value, "value", value))


def environment(device):
    """The part of the campaign receipt's environment that the tensor bytes and generator layouts depend on."""
    packages = {}
    for name in ("peft", "transformers", "safetensors", "tokenizers", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "packages": packages, "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_initialized() else None}


class ReferenceInitialization:
    def __init__(self, out_dir, args, model):
        self.out_dir, self.args, self.model = Path(out_dir), args, model
        self.directory = self.out_dir / "e458_reference"
        self.directory.mkdir(exist_ok=False)
        self.arm = e458_arms.arm(args.e458_arm)
        self.reference_dir = Path(args.e458_reference_run)
        entered = rng_state()
        self.receipt = {"schema": SCHEMA, "optimizer_steps": 0, "arm": self.arm,
                        "reference_run": str(self.reference_dir), "nominal_seed_reconstruction": False}
        self.receipt["structure"] = self.check_structure()
        self.receipt["reference"] = self.check_reference()
        self.receipt["tensors"] = self.install_tensors()
        if not rng_equal(entered, rng_state()):
            raise ValueError("the reference checks or tensor copies moved a generator")
        self.receipt["generators"] = self.restore_generators()

    # --- the arm itself: actual structure and element counts of the constructed model -----------------------------

    def check_structure(self):
        args, arm, model, problems = self.args, self.arm, self.model, []

        def need(ok, message):
            if not ok:
                problems.append(message)
        need(args.lora == arm["rank"] and args.lora_targets == arm["targets"],
             f"--lora {args.lora} / --lora_targets {args.lora_targets} are not the arm's "
             f"{arm['rank']} / {arm['targets']}")
        need(model.head_config == arm["head_config"], f"head configuration {model.head_config} is not the arm's")
        need(list(getattr(model, "lora_target_modules", None) or []) == arm["target_modules"],
             "LoRA target modules are not the arm's (the attn scope needs the hybrid backbone's DeltaNet leaves)")
        config = model.lm.peft_config["default"]
        # No rsLoRA, no DoRA, no per-module rank or alpha override: alpha / rank is the only scaling.
        wanted = {"r": arm["rank"], "lora_alpha": arm["alpha"], "lora_dropout": args.lora_dropout,
                  "use_rslora": False, "use_dora": False, "rank_pattern": {}, "alpha_pattern": {}, "bias": "none",
                  "lora_bias": False, "init_lora_weights": True}
        actual = {key: plain(getattr(config, key, value)) for key, value in wanted.items()}
        need(actual == wanted, f"LoRA configuration {actual} is not the plain {wanted}")
        modules = []
        for name, module in model.lm.named_modules():
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                a, b = module.lora_A["default"].weight, module.lora_B["default"].weight
                modules.append({"module": name, "leaf": name.rsplit(".", 1)[-1], "in_features": a.shape[1],
                                "out_features": b.shape[0], "ranks": [a.shape[0], b.shape[1], module.r["default"]],
                                "alpha": module.lora_alpha["default"], "scaling": module.scaling["default"],
                                "dropout": getattr(module.lora_dropout["default"], "p", 0.0),
                                "elements": a.numel() + b.numel()})
        need(bool(modules) and all(
            m["ranks"] == [arm["rank"]] * 3 and m["alpha"] == arm["alpha"] and m["scaling"] == arm["alpha"] / arm["rank"]
            and m["scaling"] == 2.0 and m["dropout"] == args.lora_dropout for m in modules),
            "a LoRA module's rank, alpha, alpha/rank scaling or dropout is not the arm's")
        need({m["leaf"] for m in modules} == set(arm["target_modules"]), "adapted leaves are not the arm's target modules")
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        head = {name: p for name, p in trainable.items() if name.startswith("head.")}
        adapter = {name: p for name, p in trainable.items() if not name.startswith("head.")}
        need(sorted(head) == ["head." + name for name in HEAD_TENSORS], "head tensors are not the pointer q/k")
        need(all(name.endswith(LORA_SUFFIXES) for name in adapter) and len(adapter) == 2 * len(modules)
             and sorted(adapter) == sorted(name for name, _ in model.named_parameters() if "lora_" in name),
             "trainable backbone tensors are not exactly the plain LoRA A/B weights")
        numel = {"head": sum(p.numel() for p in head.values()), "adapter": sum(p.numel() for p in adapter.values())}
        numel["trainable"] = numel["head"] + numel["adapter"]
        planned = {"head": args.e458_expected_head_numel, "adapter": args.e458_expected_adapter_numel}
        planned["trainable"] = planned["head"] + planned["adapter"]
        need(numel == planned, f"actual trainable elements {numel} are not the planned {planned}")
        need(numel["adapter"] == sum(m["elements"] for m in modules), "LoRA module elements do not add up")
        if problems:
            raise ValueError(f"{arm['name']} is not the declared arm: " + "; ".join(problems))
        leaves = {}
        for m in modules:
            row = leaves.setdefault(m["leaf"], {"modules": 0, "elements": 0, "in_out": []})
            row["modules"] += 1
            row["elements"] += m["elements"]
            if [m["in_features"], m["out_features"]] not in row["in_out"]:
                row["in_out"].append([m["in_features"], m["out_features"]])
        return {"actual_numel": numel, "planned_numel": planned, "lora_modules": len(modules),
                "lora_tensors": len(adapter), "by_leaf": leaves, "lora_configuration": actual,
                "scaling_values": sorted({m["scaling"] for m in modules}),
                "peft_config": plain({key: value for key, value in vars(config).items()}),
                "scaling_rule": "lora_alpha / r, with lora_alpha = 2 * r as for the reference; no rsLoRA, DoRA, "
                                "rank_pattern or alpha_pattern"}

    # --- the reference run: its own step-zero records must agree before anything is taken from it ----------------

    def check_reference(self):
        args, arm, ref = self.args, self.arm, self.reference_dir
        paths = {key: ref / name for key, name in REFERENCE_FILES.items()}
        absent = [name for key, name in REFERENCE_FILES.items()
                  if not paths[key].is_file() or paths[key].stat().st_size == 0]
        if absent:
            raise ReferenceUnusable(f"reference run {ref} has no step-zero record {absent}")
        files = {REFERENCE_FILES[key]: digest(path) for key, path in paths.items()}
        problems = []

        def need(ok, message):
            if not ok:
                problems.append(message)
        try:
            capture, binary_map = read_json(paths["initial_capture"]), read_json(paths["binary_map"])
            receipt, config = read_json(paths["initial_receipt"]), read_json(paths["training_config"])
            for group in ("head", "adapter"):
                binary = files[REFERENCE_FILES[group + "_initial"]]
                need(capture[f"initial_{group}_sha256"] == binary == binary_map[f"full_{group}_binary_sha256"],
                     f"{group} step-zero binary is not the file its own receipts hashed")
            need(capture["optimizer_steps"] == 0 and capture["genuine_training_capture"] is True
                 and capture["nominal_seed_reconstruction"] is False and receipt["optimizer_steps"] == 0
                 and receipt["nominal_seed_reconstruction"] is False, "step-zero records are not a genuine capture")
            reference_head = e458_arms.head_config(e458_arms.REFERENCE["readout"])
            need(receipt["head_config"] == reference_head == config["head_config"],
                 "reference head is not the pointer with the end readout")
            need(config["args"]["lora"] == e458_arms.REFERENCE["rank"]
                 and config["args"]["lora_targets"] == e458_arms.REFERENCE["targets"]
                 and config["dse"]["lora_target_modules"] == list(e458_arms.TARGET_MODULES["all"]),
                 "reference adapter is not rank 64 on every target module")
            data = digest(Path(args.data))
            need(receipt["training_data_sha256"] == data == capture["training_data_sha256"],
                 "reference trained on different data bytes")
            need(set(capture["adapter_scaling"].values()) == {2.0}, "reference alpha / rank scaling is not 2")
            arguments = self.compare_arguments(config["args"], need)
            mine, theirs = environment(self.model.device), receipt["environment"]
            need(all(theirs.get(key) == value for key, value in mine.items()),
                 f"environment differs: here {mine}, reference { {key: theirs.get(key) for key in mine} }")
            here, there = source_hashes(), config["dse"]["source_sha256"]
            changed = sorted(name for name in set(here) | set(there) if here.get(name) != there.get(name))
            need(set(changed) <= set(e458_arms.SOURCE_FILES_CHANGED),
                 f"training source differs from the reference's outside the declared files: {changed}")
            facts = {"files_sha256": files, "training_data_sha256": data, "head_config": reference_head,
                     "planned_optimizer_steps": receipt["planned_optimizer_steps"], "arguments": arguments,
                     "environment": mine, "reference_host": theirs.get("host"),
                     "reference_CUDA_VISIBLE_DEVICES": theirs.get("CUDA_VISIBLE_DEVICES"),
                     "source_files_differing_from_reference": changed}
            if not isinstance(receipt["tensor_sha256"], dict) or not isinstance(receipt["rng_at_loop_entry"], dict):
                raise TypeError("tensor or generator digests are not mappings")
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            raise ReferenceUnusable(f"reference run {ref} has an unreadable step-zero record: "
                                    f"{type(exc).__name__}: {exc}") from exc
        if problems:
            raise ReferenceUnusable(f"reference run {ref} cannot be used for {arm['name']}: " + "; ".join(problems))
        self.paths, self.reference_receipt = paths, receipt
        return facts

    def compare_arguments(self, theirs, need):
        """Recorded trainer arguments: equal except where the arm is declared to differ."""
        mine = json.loads(json.dumps(vars(self.args)))

        def value(row, key):
            if key == "head_config":
                return normalize_head_config(json.loads(row[key]) if row[key] else None, head_dim=row.get("head_dim", 256))
            return row[key]
        here, there = set(mine) - set(e458_arms.ARGUMENTS), set(theirs) - set(e458_arms.ARGUMENTS)
        compared = sorted((here & there) - set(e458_arms.ARGUMENTS_NOT_COMPARED))
        different = [key for key in compared if value(mine, key) != value(theirs, key)]
        declared = sorted(self.arm["argument_changes"])
        undeclared = [key for key in different if key not in declared]
        need(here == there, f"argument names differ from the reference's: {sorted(here ^ there)}")
        need(not any(theirs.get(key) for key in e458_arms.ARGUMENTS), "the reference run is itself an E4a/E5/E8 arm")
        need(not undeclared, f"arguments {undeclared} differ from the reference; the arm only changes {declared}")
        need(all(key in different for key in declared), f"declared changes {declared} are not all changes: {different}")
        return {"compared": len(compared), "different": different, "declared_changes": self.arm["argument_changes"],
                "not_compared": {key: [mine.get(key), theirs.get(key)] for key in e458_arms.ARGUMENTS_NOT_COMPARED}}

    # --- action 1: copy the actual common step-zero tensors and compare what the model now holds -----------------

    def install_tensors(self):
        from safetensors import safe_open
        arm, model = self.arm, self.model
        recorded = self.reference_receipt["tensor_sha256"]
        head = dict(model.head.named_parameters())
        adapter = {name: p for name, p in model.named_parameters() if "lora_" in name}
        rows, actual = [], {}

        def install(key, name, target, source):
            value = source.get_tensor(key)
            sha = tensor_sha256(value)
            if recorded.get(name) != sha:
                raise ReferenceUnusable(f"reference binary tensor {name} is not the tensor its step-zero receipt hashed")
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ReferenceUnusable(f"reference tensor {name} is {list(value.shape)} {value.dtype}, "
                                        f"here {list(target.shape)} {target.dtype}")
            natural = same_bytes(target.detach().cpu(), value)     # what this arm's own construction had produced
            with torch.no_grad():
                target.copy_(value.to(target.device))
            if not same_bytes(target.detach().cpu(), value):
                raise ValueError(f"tensor {name} read back from the model differs from the reference value")
            rows.append([name, list(value.shape), sha, natural])
            actual[name] = sha

        with safe_open(str(self.paths["head_initial"]), framework="pt", device="cpu") as source:
            if sorted(source.keys()) != HEAD_TENSORS or sorted(head) != HEAD_TENSORS:
                raise ReferenceUnusable("head tensor names differ from the reference's pointer q/k")
            for key in HEAD_TENSORS:
                install(key, "head." + key, head[key], source)
        with safe_open(str(self.paths["adapter_initial"]), framework="pt", device="cpu") as source:
            names = set(source.keys())
            arm_only, reference_only = sorted(set(adapter) - names), sorted(names - set(adapter))
            excluded = set(e458_arms.TARGET_MODULES["all"]) - set(arm["target_modules"])
            if arm_only or {leaf_of(name) for name in reference_only} != excluded or any(
                    leaf_of(name) in excluded for name in adapter):
                raise ReferenceUnusable(
                    f"LoRA tensor names do not follow the arm's scope: {len(arm_only)} only here, "
                    f"{len(reference_only)} only in the reference on leaves "
                    f"{sorted({leaf_of(name) for name in reference_only})}, expected exactly {sorted(excluded)}")
            if arm["adapter"] == "reference":
                for name in sorted(adapter):
                    install(name, name, adapter[name], source)
            else:
                # A different rank: same modules, no tensor of the reference's shape. Nothing is copied or matched.
                for name in sorted(adapter):
                    there, here = source.get_slice(name).get_shape(), list(adapter[name].shape)
                    axis = 0 if ".lora_A." in name else 1
                    if (there[axis], here[axis], there[1 - axis]) != (e458_arms.REFERENCE["rank"], arm["rank"], here[1 - axis]):
                        raise ReferenceUnusable(f"{name}: {here} here and {there} in the reference do not differ in the rank alone")
                    actual[name] = tensor_sha256(adapter[name])
        # Properties of the actual step-zero LoRA values, whichever way they got there (PEFT's default rule:
        # kaiming_uniform_(a=sqrt(5)) on A, i.e. uniform within 1/sqrt(in_features), and zeros on B).
        spread = {}
        for name, p in adapter.items():
            value = p.detach()
            if ".lora_B." in name:
                if torch.count_nonzero(value).item():
                    raise ValueError(f"step-zero LoRA B is not zero: {name}")
            else:
                bound = 1 / math.sqrt(value.shape[1])
                if not bool(torch.isfinite(value).all()) or float(value.abs().max()) > bound * (1 + 1e-6):
                    raise ValueError(f"step-zero LoRA A is outside the default uniform bound: {name}")
                row = spread.setdefault(leaf_of(name), [math.inf, -math.inf])
                scaled = float(value.double().std()) / bound       # sqrt(1/3) for a uniform draw
                row[0], row[1] = min(row[0], scaled), max(row[1], scaled)
        copied = [row for row in rows if not row[0].startswith("head.")]
        return {
            "head": {"copied_and_read_back_equal": 4, "naturally_equal_before_copy": sum(row[3] for row in rows[:4])},
            "adapter": {
                "mode": arm["adapter"], "arm_tensors": len(adapter), "reference_tensors": len(names),
                "copied_and_read_back_equal": len(copied),
                "naturally_equal_before_copy": sum(row[3] for row in copied) if copied else None,
                "only_in_reference": len(reference_only),
                "only_in_reference_by_leaf": {leaf: sum(leaf_of(name) == leaf for name in reference_only)
                                              for leaf in sorted(excluded)},
                "only_here": 0,
                "matched_to_reference": arm["adapter"] == "reference",
                "initialization": ("reference run's actual step-zero values" if arm["adapter"] == "reference" else
                                   "ordinary: PEFT default at this rank (kaiming_uniform_(a=sqrt(5)) on A, zeros on B), "
                                   "drawn by this run's own construction from the trainer-seeded CPU generator; "
                                   "no tensor has the reference's shape")},
            "lora_B_all_zero": True, "lora_A_within_default_uniform_bound": True,
            "lora_A_std_over_bound_by_leaf": spread,
            "group_sha256": {"adapter": group_digest(actual, lambda name: not name.startswith("head.")),
                             "head_qk": group_digest(actual, lambda name: name.startswith("head."))},
            "copied": {"columns": ["tensor", "shape", "sha256", "naturally_equal_before_copy"], "rows": rows}}

    # --- action 2: training starts from the reference run's recorded seeded torch generator state ----------------

    def restore_generators(self):
        ref, entry = self.reference_dir / "r3_control", self.reference_receipt["rng_at_loop_entry"]
        raw = {}
        try:
            record = read_json(self.paths["generator_record"])
            for row in record["full_bytes"]:
                data = (ref / row["file"]).read_bytes()
                if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ReferenceUnusable(f"reference generator bytes {row['file']} are not what their receipt hashed")
                raw[row["role"]] = data
            cuda_recorded = bool(record["cuda_was_initialized"])
        except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            raise ReferenceUnusable(f"reference generator record is unreadable: {type(exc).__name__}: {exc}") from exc
        cuda_roles = [f"cuda_visible_{i}" for i in range(len(raw) - 1)]
        if sorted(raw) != sorted(["cpu"] + cuda_roles) or cuda_recorded != bool(cuda_roles):
            raise ReferenceUnusable(f"reference generator record has unexpected roles {sorted(raw)}")
        digests = {"torch_cpu_sha256": hashlib.sha256(raw["cpu"]).hexdigest(),
                   "torch_cuda_sha256": (hashlib.sha256(b"".join(raw[role] for role in cuda_roles)).hexdigest()
                                         if cuda_roles else None)}
        # The raw bytes were saved at the end of the reference's model construction. Its training-entry state
        # exists only as digests, so the raw bytes are that state only if those digests say nothing moved between.
        if any(digests[key] != entry.get(key) for key in digests):
            raise ReferenceUnusable(
                "the reference's raw generator bytes (end of model construction) are not its training-entry state: "
                f"their digests {digests} differ from its loop-entry digests "
                f"{ {key: entry.get(key) for key in digests} }. No raw bytes of the entry state were recorded, so "
                "it cannot be restored; it is not reconstructed from the seed")
        initialized = torch.cuda.is_initialized()
        natural = [torch.get_rng_state().clone()] + ([s.clone() for s in torch.cuda.get_rng_state_all()] if initialized else [])
        target = [torch.frombuffer(bytearray(raw[role]), dtype=torch.uint8).clone() for role in ["cpu"] + cuda_roles]
        if len(natural) != len(target) or any(a.numel() != b.numel() for a, b in zip(natural, target)):
            raise ReferenceUnusable(
                f"generator layout differs: reference has CPU and {len(cuda_roles)} CUDA states of "
                f"{[t.numel() for t in target]} bytes, here {len(natural) - 1} CUDA states of {[t.numel() for t in natural]}")
        # Bind the saved construction state to this initialization.
        own = read_json(self.out_dir / "r3_control/model_to_device_complete.RNG.json")
        natural_sha = [hashlib.sha256(state.numpy().tobytes()).hexdigest() for state in natural]
        if natural_sha != [row["sha256"] for row in own["full_bytes"]]:
            raise ValueError("generator state moved after this run's own end-of-construction capture")
        torch.set_rng_state(target[0])
        for index, state in enumerate(target[1:]):
            torch.cuda.set_rng_state(state, device=index)
        now = [torch.get_rng_state()] + (list(torch.cuda.get_rng_state_all()) if initialized else [])
        if len(now) != len(target) or not all(torch.equal(a, b) for a, b in zip(now, target)):
            raise ValueError("generator state read back after the restore differs from the reference bytes")
        self.restored = target
        return {"restored_from": {role: f"r3_control/model_to_device_complete.{role}.bin" for role in ["cpu"] + cuda_roles},
                "restored_digests_equal_reference_loop_entry": digests,
                "read_back_equal": True,
                "natural_sha256_in_own_r3_capture": natural_sha,
                "natural_equal_to_reference": [bool(torch.equal(a, b)) for a, b in zip(natural, target)],
                "cuda_states": len(cuda_roles),
                "not_restored": "unseeded Python and NumPy global generators (the trainer never draws from them); "
                                "the dedicated data-order and augmentation generators (compared, never set)"}

    def at_loop_entry(self, data_rng, planned_steps):
        """Last check before the first update, then the receipt. Nothing may have drawn since the restore."""
        now = [torch.get_rng_state()] + (list(torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else [])
        if len(now) != len(self.restored) or not all(torch.equal(a, b) for a, b in zip(now, self.restored)):
            raise ValueError("a seeded torch generator moved between the restore and training entry")
        entry = self.reference_receipt["rng_at_loop_entry"]
        data = hashlib.sha256(repr(data_rng.getstate()).encode()).hexdigest()
        if data != entry["data_order_rng_sha256"]:
            raise ValueError("the data-order generator at training entry differs from the reference run's")
        if planned_steps != self.reference_receipt["planned_optimizer_steps"]:
            raise ValueError(f"{planned_steps} planned updates here, {self.reference_receipt['planned_optimizer_steps']} "
                             "in the reference")
        self.receipt["loop_entry"] = {
            "torch_generators_equal_restored_state": True, "data_order_rng_sha256": data,
            "data_order_rng_equal_reference": True, "planned_optimizer_steps": planned_steps}
        self.receipt["after_training_entry"] = {
            "dropout_draws": ("same LoRA modules and input shapes as the reference" if self.arm["same_structure_as_reference"]
                              else "this arm's own LoRA modules; the seeded generator sequence may leave the reference's"),
            "not_forced": "whole-run CUDA generator identity", "not_claimed": "identical dropout masks"}
        save_json(self.directory / "receipt.json", self.receipt)
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
