"""Initialize two-epoch runs from their matched one-epoch references.

Pointer runs use LL_A; mean-span runs use POOL_ALL. Reuse kev.e458_reference
to validate structure, copy head and adapter tensors, and restore torch RNG
states. Record the OneCycleLR configuration and cumulative updates, source
records, and augmented-input digest after each epoch.

Data-order and augmentation generators continue across epoch boundaries.
The learning-rate schedule depends on the full training horizon.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch

from . import e11_arms, e458_arms
from .campaign_diagnostics import torch_rng_digests
from .dse_options import source_hashes
from .e458_reference import (REFERENCE_FILES, ReferenceInitialization, ReferenceUnusable, environment, plain,
                             read_json)
from .e458_reference import SCHEMA as E458_SCHEMA
from .heads import normalize_head_config
from .supplement_diagnostics import canonical, digest, rng_equal, rng_state, save_json

SCHEMA = "jev-e11-reference-initialization-v1"
EPOCH_SCHEMA = "jev-e11-epoch-record-v1"
E458_RECEIPT = "e458_reference/receipt.json"


def schedule_record(optimizer, scheduler, args, planned_steps):
    """The schedule as the objects hold it at training entry. Read only: no step, no value set, nothing recomputed.

    `source_call` is the trainer's own call; `constructor_defaults` are the installed constructor's; `resolved` is
    what the object holds. The three must agree, or the record would describe another schedule than the one in use.
    """
    signature = inspect.signature(type(scheduler).__init__)
    defaults = {name: parameter.default for name, parameter in signature.parameters.items()
                if parameter.default is not inspect.Parameter.empty}
    call = {"max_lr": [args.lr, args.head_lr or args.lr], "total_steps": max(planned_steps, 1),
            "pct_start": e11_arms.PCT_START}
    used = {**defaults, **call}
    phases = [dict(phase) for phase in scheduler._schedule_phases]
    groups = [{"role": role, **{key: group.get(key) for key in ("initial_lr", "max_lr", "min_lr", "max_momentum", "base_momentum")},
               "lr": group["lr"], "betas": list(group["betas"]), "eps": group["eps"], "weight_decay": group["weight_decay"]}
              for role, group in zip(("adapter", "head"), optimizer.param_groups)]
    resolved = {"total_steps": scheduler.total_steps, "anneal_strategy": getattr(scheduler, "_anneal_func_type", None),
                "cycle_momentum": scheduler.cycle_momentum, "cycles_beta1": getattr(scheduler, "use_beta1", None),
                "three_phase": len(phases) == 3, "phases": phases, "groups": groups}
    problems = []

    def need(ok, message):
        if not ok:
            problems.append(message)
    need(type(scheduler).__name__ == "OneCycleLR" and len(groups) == len(optimizer.param_groups) == 2,
         "the scheduler is not OneCycleLR over the adapter and head groups")
    need(resolved["total_steps"] == call["total_steps"] == planned_steps, "the horizon is not the planned optimizer updates")
    need(len(phases) == (3 if used["three_phase"] else 2) and not used["three_phase"]
         and phases[0]["end_step"] == float(used["pct_start"] * resolved["total_steps"]) - 1
         and phases[-1]["end_step"] == resolved["total_steps"] - 1, "the phases are not pct_start of the horizon, then its end")
    need((resolved["anneal_strategy"], resolved["cycle_momentum"], resolved["cycles_beta1"])
         == (used["anneal_strategy"], used["cycle_momentum"], True), "annealing or momentum cycling is not the constructor default")
    for group, peak in zip(groups, call["max_lr"]):
        need(group["max_lr"] == peak and group["initial_lr"] == peak / used["div_factor"]
             and group["min_lr"] == group["initial_lr"] / used["final_div_factor"]
             and (group["max_momentum"], group["base_momentum"]) == (used["max_momentum"], used["base_momentum"]),
             f"{group['role']} group bounds do not follow from the peak rate and the constructor defaults")
    if problems:
        raise ValueError("the schedule in use is not the one the trainer line and the installed defaults give: "
                         + "; ".join(problems))
    return {"class": f"{type(scheduler).__module__}.{type(scheduler).__name__}", "torch": torch.__version__,
            "source_call": call, "constructor_defaults": plain(defaults),
            "resolved": {**resolved, "div_factor": used["div_factor"], "final_div_factor": used["final_div_factor"],
                         "pct_start": used["pct_start"], "base_momentum": used["base_momentum"],
                         "max_momentum": used["max_momentum"]},
            "optimizer": {"class": type(optimizer).__name__, "defaults": plain(dict(optimizer.defaults))},
            "definitions": {
                "groups.lr, groups.betas": "the values in the optimizer at training entry: what the first update uses",
                "used_sequence": "campaign/updates.jsonl used_group_lrs and used_group_beta1, read before every update",
                "horizon": "optimizer updates of both epochs; each epoch ends with a flush of its last accumulation group"}}


class TwoEpochInitialization(ReferenceInitialization):
    def __init__(self, out_dir, args, model):
        self.out_dir, self.run_args, self.model = Path(out_dir), args, model
        # The inherited structure check reads the planned element counts under its own block's argument names.
        self.args = argparse.Namespace(**{**vars(args), "e458_expected_head_numel": args.e11_expected_head_numel,
                                          "e458_expected_adapter_numel": args.e11_expected_adapter_numel})
        self.directory = self.out_dir / "e11_reference"
        self.directory.mkdir(exist_ok=False)
        self.arm = e11_arms.arm(args.e11_arm)
        self.reference_dir = Path(args.e11_reference_run)
        if args.epochs != self.arm["epochs"]:
            raise ValueError(f"{self.arm['name']} trains {self.arm['epochs']} epochs, not --epochs {args.epochs}")
        entered = rng_state()
        self.receipt = {"schema": SCHEMA, "optimizer_steps": 0, "arm": self.arm,
                        "reference_run": str(self.reference_dir), "nominal_seed_reconstruction": False}
        self.receipt["structure"] = self.check_structure()
        self.receipt["reference"] = self.check_reference()
        self.receipt["tensors"] = self.install_tensors()
        if not rng_equal(entered, rng_state()):
            raise ValueError("the reference checks or tensor copies moved a generator")
        self.receipt["generators"] = self.restore_generators()
        self.log = None

    # --- the one-epoch run: its own step-zero records must agree before anything is taken from it ----------------

    def check_reference(self):
        args, arm, ref = self.run_args, self.arm, self.reference_dir
        paths = {key: ref / name for key, name in REFERENCE_FILES.items()}
        if arm["reference"]["e458_arm"]:         # an E4a/E5/E8 arm recorded how it was itself initialized
            paths["own_initialization"] = ref / E458_RECEIPT
        names = {**REFERENCE_FILES, "own_initialization": E458_RECEIPT}
        absent = [names[key] for key, path in paths.items() if not path.is_file() or path.stat().st_size == 0]
        if absent:
            raise ReferenceUnusable(f"reference run {ref} has no step-zero record {absent}")
        files = {names[key]: digest(path) for key, path in paths.items()}
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
            need(receipt["head_config"] == arm["head_config"] == config["head_config"],
                 f"reference head is not the pointer with the {arm['readout']} readout")
            need(config["args"]["lora"] == arm["rank"] and config["args"]["lora_targets"] == arm["targets"]
                 and config["dse"]["lora_target_modules"] == arm["target_modules"],
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
            need(set(changed) <= set(arm["source_files_changed"]),
                 f"training source differs from the reference's outside the declared files: {changed}")
            facts = {"files_sha256": files, "training_data_sha256": data, "head_config": arm["head_config"],
                     "planned_optimizer_steps": receipt["planned_optimizer_steps"], "arguments": arguments,
                     "environment": mine, "reference_host": theirs.get("host"),
                     "reference_CUDA_VISIBLE_DEVICES": theirs.get("CUDA_VISIBLE_DEVICES"),
                     "source_files_differing_from_reference": changed}
            if "own_initialization" in paths:
                own = read_json(paths["own_initialization"])
                need(own["schema"] == E458_SCHEMA and own["arm"]["name"] == arm["reference"]["e458_arm"]
                     and own["tensors"]["adapter"]["matched_to_reference"] is True,
                     f"the reference's own receipt is not that of a reference-initialized {arm['reference']['e458_arm']}")
                facts["reference_was_initialized_from"] = own["reference_run"]
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
        """Recorded trainer arguments: equal except the two a second epoch changes. The reference is the declared
        one-epoch arm: a plain E1 run for the pointer, the E4a/E5/E8 mean-span arm on these counts for the pooling."""
        mine, arm = json.loads(json.dumps(vars(self.run_args))), self.arm
        own, block = set(e11_arms.ARGUMENTS), set(e458_arms.ARGUMENTS)

        def value(row, key):
            if key == "head_config":
                return normalize_head_config(json.loads(row[key]) if row[key] else None, head_dim=row.get("head_dim", 256))
            return row[key]
        here, there = set(mine) - own - block, set(theirs) - own - block
        compared = sorted((here & there) - set(e458_arms.ARGUMENTS_NOT_COMPARED))
        different = [key for key in compared if value(mine, key) != value(theirs, key)]
        declared = sorted(arm["argument_changes"])
        undeclared = [key for key in different if key not in declared]
        wanted = arm["reference"]["e458_arm"]
        need(here == there, f"argument names differ from the reference's: {sorted(here ^ there)}")
        need(not any(theirs.get(key) for key in own), "the reference run is itself a two-epoch arm")
        need(theirs.get("e458_arm", "") == wanted,
             f"the reference run is {theirs.get('e458_arm') or 'not an E4a/E5/E8 arm'}, not the {arm['reference']['arm']} "
             f"of {arm['reference']['block']}")
        if wanted:
            need((theirs.get("e458_expected_head_numel"), theirs.get("e458_expected_adapter_numel"))
                 == (mine["e11_expected_head_numel"], mine["e11_expected_adapter_numel"]),
                 "the reference arm was planned with other element counts")
        need((mine.get("epochs"), theirs.get("epochs")) == (arm["epochs"], e11_arms.REFERENCE_EPOCHS),
             f"epochs here and in the reference are {mine.get('epochs')} and {theirs.get('epochs')}, "
             f"not {arm['epochs']} and {e11_arms.REFERENCE_EPOCHS}")
        need(not undeclared, f"arguments {undeclared} differ from the reference; the arm only changes {declared}")
        need(all(key in different for key in declared), f"declared changes {declared} are not all changes: {different}")
        return {"compared": len(compared), "different": different, "declared_changes": declared,
                "reference_e458_arm": theirs.get("e458_arm", ""),
                "not_compared": {key: [mine.get(key), theirs.get(key)] for key in e458_arms.ARGUMENTS_NOT_COMPARED}}

    # --- training entry, then one row per complete epoch ---------------------------------------------------------

    def at_loop_entry(self, data_rng, planned_steps, optimizer, scheduler, records, microbatches_per_epoch):
        """Last check before the first update, then the receipt. Nothing may have drawn since the restore."""
        now = [torch.get_rng_state()] + (list(torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else [])
        if len(now) != len(self.restored) or not all(torch.equal(a, b) for a, b in zip(now, self.restored)):
            raise ValueError("a seeded torch generator moved between the restore and training entry")
        entry = self.reference_receipt["rng_at_loop_entry"]
        data = hashlib.sha256(repr(data_rng.getstate()).encode()).hexdigest()
        if data != entry["data_order_rng_sha256"]:
            raise ValueError("the data-order generator at training entry differs from the reference run's")
        args, arm = self.run_args, self.arm
        per_epoch = -(-microbatches_per_epoch // args.accum)
        reference_steps = self.reference_receipt["planned_optimizer_steps"]
        if not (planned_steps == arm["epochs"] * per_epoch == args.e11_expected_optimizer_steps
                and reference_steps == e11_arms.REFERENCE_EPOCHS * per_epoch):
            raise ValueError(
                f"{planned_steps} planned updates here, {arm['epochs']} epochs of {per_epoch}; the plan expects "
                f"{args.e11_expected_optimizer_steps} and the one-epoch reference planned {reference_steps}")
        self.receipt["loop_entry"] = {
            "torch_generators_equal_restored_state": True, "data_order_rng_sha256": data,
            "data_order_rng_equal_reference": True, "planned_optimizer_steps": planned_steps,
            "reference_planned_optimizer_steps": reference_steps, "epochs": arm["epochs"],
            "source_records_per_epoch": records, "microbatches_per_epoch": microbatches_per_epoch,
            "optimizer_steps_per_epoch": per_epoch,
            "source_records_in_the_last_update_of_an_epoch": e11_arms.epoch_updates(records, args.batch, args.accum)[-1]}
        self.receipt["schedule"] = schedule_record(optimizer, scheduler, args, planned_steps)
        self.receipt["after_training_entry"] = {
            "first_epoch": "order from the data-order generator checked above and each record's augmentation from the "
                           "trainer's seed/epoch/record generator, as in the reference",
            "second_epoch": "continues the same data-order generator; no reseed",
            "schedule": "over the two-epoch horizon, so not the reference's; the values each update used are recorded "
                        "per update and are the only statement of where the two differ",
            "not_claimed": ["the reference's parameter trajectory", "learning rates or beta1 equal to, or different "
                            "from, the reference's at any particular update"]}
        save_json(self.directory / "receipt.json", self.receipt)
        self.log = (self.directory / "epochs.jsonl").open("xb")
        self.sync_directory()

    def after_epoch(self, epoch, optimizer_updates, source_records_seen, input_stream, data_rng):
        """A complete epoch: counts so far and the running input digest. Reads only; `input_stream` is not advanced."""
        cpu, cuda = torch_rng_digests()
        raw = canonical({"schema": EPOCH_SCHEMA, "epoch": epoch, "optimizer_updates": optimizer_updates,
                         "source_records_seen": source_records_seen,
                         "input_stream_sha256": input_stream.copy().hexdigest(),
                         "data_order_rng_sha256": hashlib.sha256(repr(data_rng.getstate()).encode()).hexdigest(),
                         "torch_cpu_sha256": cpu, "torch_cuda_sha256": cuda})
        if self.log.write(raw) != len(raw):
            raise IOError("short epoch record write")
        self.log.flush()
        os.fsync(self.log.fileno())
        if epoch + 1 == self.arm["epochs"]:
            self.log.close()
            self.sync_directory()

    def sync_directory(self):
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
