"""CPU tests of the E11 two-epoch path: epoch boundaries, the first epoch against the one-epoch run, the schedule.

The real kev.train loop, the real DecisionModel and the real PEFT LoRA run on a tiny hybrid backbone
(tests/tiny_hybrid.py). One-epoch runs of this tree stand in for the references: a plain pointer run for the E1
low/low pointer, and a mean-span arm trained through the unchanged E4a/E5/E8 path for POOL_ALL. 46 records with
accumulation 4 give 12 updates per epoch, the last on 2 records, so two epochs are 24 updates and not
ceil(92 / 4) = 23, and both horizons start in the schedule's first phase as the real ones do.

The second class feeds these real records to the controller's standard-library readers (../../run_e11.py).
Run in the training environment with CUDA_VISIBLE_DEVICES='' and PYTHONPATH at this source tree. The CUDA
generator branch and the device-side copy are the inherited ones and cannot run here.
"""
import contextlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import load_file

from kev import e11_arms, e458_arms
from kev import model as model_module
from kev.dse_options import source_hashes
from kev.e11_reference import schedule_record
from kev.e458_reference import ReferenceUnusable
from kev.supplement_diagnostics import digest
from kev.train import accumulation_records

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_hybrid import HIDDEN, FakeAuto, FakeTokenizer, lora_modules, run_trainer, write_training_file  # noqa: E402

CAMPAIGN = Path(__file__).resolve().parents[2]
RECORDS, ACCUM, PER_EPOCH, STEPS, LAST = 46, 4, 12, 24, 2
RATE = 1e-3
COMMON = ["--base", "synthetic", "--lr", "1e-3", "--head_lr", "1e-3", "--weight_decay", "0.01", "--lora", "64",
          "--lora_targets", "all", "--accum", str(ACCUM), "--device", "cpu", "--batch", "1", "--dtype", "fp32",
          "--weights_dtype", "fp32", "--checkpointing", "0", "--p_none_pair", "0.25", "--seed", "44",
          "--lora_dropout", "0.05", "--supplement_diagnostics", "1", "--r3_control_capture", "1",
          "--r3_sham_residual", "0", "--campaign_diagnostics", "1"]
ONE = ["--epochs", "1", "--campaign_checkpoint_steps", "2,6"]
TWO = ["--epochs", "2", "--campaign_checkpoint_steps", "3,12,18"]          # 12 is the end of the first epoch
NUMEL = {"head": e458_arms.head_numel(HIDDEN), "adapter": e458_arms.adapter_numel(lora_modules(), 64, "all")}
ORDER = "r3_control/epoch_{}_actual_original_data_order.json"
UPDATES = "campaign/updates.jsonl"


def head(readout):
    return ["--head_config", json.dumps(e458_arms.head_config(readout), sort_keys=True, separators=(",", ":"))]


def e11(name, reference, **changes):
    values = {"head": NUMEL["head"], "adapter": NUMEL["adapter"], "steps": STEPS, **changes}
    return head(e11_arms.ARMS[name]["readout"]) + [
        "--e11_arm", name, "--e11_reference_run", str(reference), "--e11_expected_head_numel", str(values["head"]),
        "--e11_expected_adapter_numel", str(values["adapter"]), "--e11_expected_optimizer_steps", str(values["steps"])]


def e458(name, reference):
    return head(e458_arms.ARMS[name]["readout"]) + [
        "--e458_arm", name, "--e458_reference_run", str(reference), "--e458_expected_head_numel", str(NUMEL["head"]),
        "--e458_expected_adapter_numel", str(NUMEL["adapter"])]


def changed(flags, **values):
    flags = list(flags)
    for key, value in values.items():
        flags[flags.index("--" + key) + 1] = value
    return flags


def train(out, data, *flags):
    run_trainer([*[item for group in flags for item in group], "--data", str(data), "--out", str(out)])


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines()]


def state(out, label):
    tensors = load_file(str(out / f"diagnostics/head_{label}.safetensors"))
    return {**{"head." + name: value for name, value in tensors.items()},
            **load_file(str(out / f"diagnostics/adapter_{label}.safetensors"))}


def same(left, right):
    return left.shape == right.shape and torch.equal(left.view(torch.uint8), right.view(torch.uint8))


def replay(total_steps, rate=RATE):
    """The trainer's own scheduler call on a fresh two-group AdamW: what every update uses."""
    groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": rate} for _ in range(2)]
    optimizer = torch.optim.AdamW(groups, lr=rate, weight_decay=0.01, foreach=None)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[rate, rate], total_steps=max(total_steps, 1), pct_start=0.1)
    lrs, beta1 = [], []
    for _ in range(total_steps):
        lrs.append([group["lr"] for group in optimizer.param_groups])
        beta1.append([group["betas"][0] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    return lrs, beta1


RUNS = {}


def setUpModule():
    torch.set_num_threads(1)
    root = RUNS["root"] = Path(tempfile.mkdtemp(prefix="jev-e11-test-"))
    data = RUNS["data"] = root / "train.jsonl"
    write_training_file(data, records=RECORDS)
    out = RUNS["out"] = {name: root / name for name in ("A", "POOL_ALL", "PLAIN_POOL", "POINTER_2EP", "POOL_2EP", "PLAIN_2EP")}
    train(out["A"], data, COMMON, ONE, head("end"))                                  # stands in for E1 LL_A
    train(out["POOL_ALL"], data, COMMON, ONE, e458("POOL_ALL", out["A"]))            # The one-epoch reference initialization.
    train(out["PLAIN_POOL"], data, COMMON, ONE, head("mean_span"))                   # mean-span, but not that arm
    train(out["POINTER_2EP"], data, COMMON, TWO, e11("POINTER_2EP", out["A"]))
    train(out["POOL_2EP"], data, COMMON, TWO, e11("POOL_2EP", out["POOL_ALL"]))
    train(out["PLAIN_2EP"], data, COMMON, TWO, head("end"))                          # two epochs without this path
    RUNS["pairs"] = {"POINTER_2EP": "A", "POOL_2EP": "POOL_ALL"}
    RUNS["updates"] = {name: lines(path / UPDATES) for name, path in out.items()}
    RUNS["done"] = {name: read(path / "campaign/completion.json") for name, path in out.items()}
    RUNS["entry"] = {name: read(path / "campaign/initial_receipt.json") for name, path in out.items()}
    RUNS["receipt"] = {name: read(out[name] / "e11_reference/receipt.json") for name in RUNS["pairs"]}
    RUNS["epochs"] = {name: lines(out[name] / "e11_reference/epochs.jsonl") for name in RUNS["pairs"]}


def tearDownModule():
    shutil.rmtree(RUNS["root"], ignore_errors=True)


class TwoEpochTrainingTest(unittest.TestCase):
    def setUp(self):
        self.root, self.data, self.out, self.pairs = RUNS["root"], RUNS["data"], RUNS["out"], RUNS["pairs"]

    # --- epoch boundaries --------------------------------------------------------------------------------------

    def test_each_epoch_ends_with_a_flush_so_two_epochs_are_twice_one_epoch(self):
        self.assertEqual((e11_arms.optimizer_steps(RECORDS, 1, ACCUM), math.ceil(2 * RECORDS / ACCUM)), (STEPS, STEPS - 1))
        self.assertEqual(e11_arms.epoch_updates(RECORDS, 1, ACCUM), [ACCUM] * (PER_EPOCH - 1) + [LAST])
        # The real folds: F3's last group of each epoch holds 2 records, so 3862 and not ceil(30884 / 8) = 3861.
        self.assertEqual([e11_arms.optimizer_steps(records, 1, 8) for records in (15456, 15442)], [3864, 3862])
        self.assertEqual((math.ceil(2 * 15442 / 8), e11_arms.epoch_updates(15442, 1, 8)[-1], e11_arms.epoch_updates(15456, 1, 8)[-1]),
                         (3861, 2, 8))
        # The trainer's own normalization of the group a microbatch belongs to, by its index inside the epoch.
        self.assertEqual([accumulation_records(RECORDS, 1, ACCUM, mb) for mb in (0, 43, 44, 45)], [ACCUM, ACCUM, LAST, LAST])
        for name in self.pairs:
            with self.subTest(arm=name):
                out, rows = self.out[name], RUNS["updates"][name]
                metrics, config = read(out / "training_metrics.json"), read(out / "training_config.json")
                self.assertEqual((metrics["optimizer_steps"], metrics["planned_optimizer_steps"], metrics["status"],
                                  metrics["stopped_early"], metrics["source_records_seen"], metrics["requested_records"]),
                                 (STEPS, STEPS, "complete", False, 2 * RECORDS, 2 * RECORDS))
                self.assertEqual((config["args"]["epochs"], config["dse"]["planned_optimizer_steps"],
                                  config["dse"]["requested_source_records"]), (2, STEPS, 2 * RECORDS))
                self.assertEqual([row["optimizer_update"] for row in rows], list(range(1, STEPS + 1)))
                self.assertEqual([row["objective"]["source_records"] for row in rows], ([ACCUM] * (PER_EPOCH - 1) + [LAST]) * 2)
                self.assertEqual([row["objective"]["microbatches"] for row in rows], ([ACCUM] * (PER_EPOCH - 1) + [LAST]) * 2)
                epochs = RUNS["epochs"][name]
                self.assertEqual([(row["epoch"], row["optimizer_updates"], row["source_records_seen"]) for row in epochs],
                                 [(0, PER_EPOCH, RECORDS), (1, STEPS, 2 * RECORDS)])
                loop = RUNS["receipt"][name]["loop_entry"]
                self.assertEqual((loop["epochs"], loop["planned_optimizer_steps"], loop["reference_planned_optimizer_steps"],
                                  loop["optimizer_steps_per_epoch"], loop["source_records_per_epoch"],
                                  loop["source_records_in_the_last_update_of_an_epoch"]), (2, STEPS, PER_EPOCH, PER_EPOCH, RECORDS, LAST))
                # Every recorder covered both epochs; the saved endpoints are the planned ones.
                for record, key in (("diagnostics", "optimizer_steps"), ("r3_control", "actual_optimizer_updates"),
                                    ("campaign", "optimizer_updates")):
                    self.assertEqual(read(out / record / "completion.json")[key], STEPS)
                self.assertEqual([row["optimizer_update"] for row in RUNS["done"][name]["checkpoints"]], [3, 12, 18])

    def test_epoch_rows_are_the_unchanged_recorders_digests_at_the_boundaries(self):
        for name in self.pairs:
            epochs, rows, done = RUNS["epochs"][name], RUNS["updates"][name], RUNS["done"][name]
            for row in epochs:
                self.assertEqual(row["input_stream_sha256"][:16], rows[row["optimizer_updates"] - 1]["input_stream_sha256"])
                self.assertEqual(len(row["input_stream_sha256"]), 64)
            self.assertEqual(epochs[-1]["input_stream_sha256"], done["input_stream_sha256"])
            self.assertNotEqual(epochs[0]["input_stream_sha256"], epochs[1]["input_stream_sha256"])

    def test_the_second_epoch_continues_the_data_order_generator_without_a_reseed(self):
        for name in self.pairs:
            first, second = (read(self.out[name] / ORDER.format(epoch)) for epoch in (0, 1))
            self.assertEqual(second["original_data_rng_state_before"], first["original_data_rng_state_after"])
            self.assertNotEqual(first["original_data_rng_state_before"], first["original_data_rng_state_after"])
            self.assertEqual(sorted(first["record_ids"]), sorted(second["record_ids"]))
            self.assertNotEqual(first["record_ids"], second["record_ids"])
            self.assertEqual(len(first["record_ids"]), RECORDS)
            self.assertFalse((self.out[name] / ORDER.format(2)).exists())
            self.assertEqual([path.name for path in sorted((self.out[name] / "r3_control").glob("first_update_micro_*.json"))],
                             [f"first_update_micro_{i}.json" for i in range(ACCUM)])      # the first update of the run only

    # --- the first epoch against the one-epoch run ---------------------------------------------------------------

    def test_the_first_epoch_input_stream_is_the_one_epoch_runs_whole_stream(self):
        for name, reference in self.pairs.items():
            with self.subTest(arm=name):
                epochs, done, theirs = RUNS["epochs"][name], RUNS["done"][name], RUNS["done"][reference]
                self.assertEqual(epochs[0]["input_stream_sha256"], theirs["input_stream_sha256"])
                self.assertEqual((self.out[name] / ORDER.format(0)).read_bytes(), (self.out[reference] / ORDER.format(0)).read_bytes())
                mine, other = RUNS["updates"][name], RUNS["updates"][reference]
                self.assertEqual(len(other), PER_EPOCH)
                self.assertEqual([row["input_stream_sha256"] for row in mine[:PER_EPOCH]], [row["input_stream_sha256"] for row in other])
                self.assertEqual([row["objective"]["source_records"] for row in mine[:PER_EPOCH]],
                                 [row["objective"]["source_records"] for row in other])
                # The digests of the whole runs are different things and are not what is compared.
                self.assertNotEqual(done["input_stream_sha256"], theirs["input_stream_sha256"])
                self.assertNotEqual(done["torch_rng_trace_sha256"], theirs["torch_rng_trace_sha256"])
        # The two readouts consume the same stream, order and seeded generator sequence over both epochs.
        pointer, pooling = RUNS["done"]["POINTER_2EP"], RUNS["done"]["POOL_2EP"]
        self.assertEqual(pointer["input_stream_sha256"], pooling["input_stream_sha256"])
        self.assertEqual(pointer["torch_rng_trace_sha256"], pooling["torch_rng_trace_sha256"])
        for epoch in (0, 1):
            self.assertEqual((self.out["POINTER_2EP"] / ORDER.format(epoch)).read_bytes(),
                             (self.out["POOL_2EP"] / ORDER.format(epoch)).read_bytes())

    def test_every_initial_tensor_and_the_entry_generators_are_the_one_epoch_runs(self):
        for name, reference in self.pairs.items():
            with self.subTest(arm=name):
                mine, theirs = state(self.out[name], "initial"), state(self.out[reference], "initial")
                self.assertEqual(set(mine), set(theirs))
                self.assertEqual(len(mine), 4 + 2 * len(lora_modules()))
                for tensor, value in mine.items():
                    self.assertTrue(same(value, theirs[tensor]), tensor)
                entry, other, receipt = RUNS["entry"][name], RUNS["entry"][reference], RUNS["receipt"][name]
                self.assertEqual(entry["tensor_sha256"], other["tensor_sha256"])           # Check the tensor recorder independently.
                self.assertEqual(entry["tensor_groups"], other["tensor_groups"])
                self.assertEqual(entry["rng_at_loop_entry"], other["rng_at_loop_entry"])
                summary = receipt["tensors"]
                self.assertEqual((summary["head"]["copied_and_read_back_equal"], summary["adapter"]["copied_and_read_back_equal"],
                                  summary["adapter"]["only_in_reference"], summary["adapter"]["matched_to_reference"]),
                                 (4, len(mine) - 4, 0, True))
                self.assertEqual(summary["group_sha256"]["adapter"], entry["tensor_groups"]["adapter"])
                generators = receipt["generators"]
                self.assertTrue(generators["read_back_equal"])
                self.assertEqual(generators["restored_digests_equal_reference_loop_entry"],
                                 {key: other["rng_at_loop_entry"][key] for key in ("torch_cpu_sha256", "torch_cuda_sha256")})
                self.assertEqual(receipt["reference_run"], str(self.out[reference]))
                self.assertEqual(receipt["reference"]["arguments"]["different"], ["campaign_checkpoint_steps", "epochs"])
                self.assertEqual(receipt["reference"]["source_files_differing_from_reference"], [])   # trained by this tree here
                # Not a continuation of the one-epoch endpoint: it starts at step zero and ends elsewhere.
                final, reference_final = state(self.out[name], "final"), state(self.out[reference], "final")
                self.assertFalse(all(same(value, reference_final[tensor]) for tensor, value in final.items()))
        # The pooling reference is itself a reference-initialized arm; its own receipt was read and is named.
        receipt = RUNS["receipt"]["POOL_2EP"]["reference"]
        self.assertEqual(receipt["reference_was_initialized_from"], str(self.out["A"]))
        self.assertEqual(receipt["files_sha256"]["e458_reference/receipt.json"], digest(self.out["POOL_ALL"] / "e458_reference/receipt.json"))
        self.assertEqual(receipt["arguments"]["reference_e458_arm"], "POOL_ALL")
        self.assertNotIn("e458_reference/receipt.json", RUNS["receipt"]["POINTER_2EP"]["reference"]["files_sha256"])
        self.assertEqual(RUNS["receipt"]["POINTER_2EP"]["reference"]["arguments"]["reference_e458_arm"], "")

    # --- the schedule --------------------------------------------------------------------------------------------

    def test_the_schedule_record_is_what_the_scheduler_object_held(self):
        defaults = {"anneal_strategy": "cos", "cycle_momentum": True, "base_momentum": 0.85, "max_momentum": 0.95,
                    "div_factor": 25.0, "final_div_factor": 10000.0, "three_phase": False, "last_epoch": -1, "pct_start": 0.3}
        for name in self.pairs:
            record = RUNS["receipt"][name]["schedule"]
            resolved = record["resolved"]
            self.assertEqual((record["class"], record["torch"]), ("torch.optim.lr_scheduler.OneCycleLR", torch.__version__))
            self.assertEqual(record["source_call"], {"max_lr": [RATE, RATE], "total_steps": STEPS, "pct_start": 0.1})
            self.assertEqual({key: record["constructor_defaults"][key] for key in defaults}, defaults)   # of the installed torch
            self.assertEqual((resolved["total_steps"], resolved["anneal_strategy"], resolved["cycle_momentum"], resolved["cycles_beta1"],
                              resolved["three_phase"], resolved["pct_start"], resolved["div_factor"], resolved["final_div_factor"]),
                             (STEPS, "cos", True, True, False, 0.1, 25.0, 10000.0))
            self.assertEqual([phase["end_step"] for phase in resolved["phases"]], [float(0.1 * STEPS) - 1, STEPS - 1])
            self.assertEqual([(phase["start_lr"], phase["end_lr"], phase["start_momentum"], phase["end_momentum"]) for phase in resolved["phases"]],
                             [("initial_lr", "max_lr", "max_momentum", "base_momentum"), ("max_lr", "min_lr", "base_momentum", "max_momentum")])
            for role, group in zip(("adapter", "head"), resolved["groups"]):
                self.assertEqual((group["role"], group["max_lr"], group["initial_lr"], group["min_lr"], group["max_momentum"],
                                  group["base_momentum"], group["weight_decay"], group["betas"][1], group["eps"]),
                                 (role, RATE, RATE / 25.0, RATE / 25.0 / 10000.0, 0.95, 0.85, 0.01, 0.999, 1e-8))
            self.assertEqual((record["optimizer"]["class"], record["optimizer"]["defaults"]["weight_decay"],
                              record["optimizer"]["defaults"]["betas"]), ("AdamW", 0.01, [0.9, 0.999]))

    def test_the_schedule_record_at_the_real_rates_and_horizons_and_its_refusal_of_any_other_schedule(self):
        args = types.SimpleNamespace(lr=1.25e-5, head_lr=1.25e-5)

        def built(steps, **changes):
            groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1.25e-5} for _ in range(2)]
            optimizer = torch.optim.AdamW(groups, lr=1.25e-5, weight_decay=0.01, foreach=None)
            call = {"max_lr": [1.25e-5, 1.25e-5], "total_steps": steps, "pct_start": 0.1, **changes}
            return optimizer, torch.optim.lr_scheduler.OneCycleLR(optimizer, **call)
        for steps, boundary in ((3864, 385.40000000000003), (3862, 385.20000000000005)):
            record = schedule_record(*built(steps), args, steps)
            resolved = record["resolved"]
            self.assertEqual([phase["end_step"] for phase in resolved["phases"]], [boundary, steps - 1])
            # What the optimizer holds at entry is the annealing expression at 0, not the nominal max_lr / 25.
            self.assertEqual([(group["initial_lr"], group["max_lr"], group["min_lr"], group["lr"], group["betas"]) for group in resolved["groups"]],
                             [(5e-07, 1.25e-05, 4.9999999999999995e-11, 5.000000000000003e-07, [0.95, 0.999])] * 2)
            self.assertEqual(record["source_call"], {"max_lr": [1.25e-05, 1.25e-05], "total_steps": steps, "pct_start": 0.1})
            json.dumps(record, allow_nan=False)
        for changes in ({"pct_start": 0.3}, {"div_factor": 10.0}, {"final_div_factor": 100.0}, {"anneal_strategy": "linear"},
                        {"three_phase": True}, {"cycle_momentum": False}, {"base_momentum": 0.8}, {"max_lr": [1.25e-5, 2.5e-5]}):
            with self.subTest(changes=changes), self.assertRaisesRegex(
                    ValueError, "is not the one the trainer line and the installed defaults give"):
                schedule_record(*built(3864, **changes), args, 3864)
        with self.assertRaisesRegex(ValueError, "the horizon is not the planned optimizer updates"):
            schedule_record(*built(1932), args, 3864)

    def test_every_update_used_the_two_epoch_schedule_and_the_reference_the_one_epoch_one(self):
        two, one = replay(STEPS), replay(PER_EPOCH)
        for name, reference in self.pairs.items():
            with self.subTest(arm=name):
                mine, theirs = RUNS["updates"][name], RUNS["updates"][reference]
                self.assertEqual(([row["used_group_lrs"] for row in mine], [row["used_group_beta1"] for row in mine]), two)
                self.assertEqual(([row["used_group_lrs"] for row in theirs], [row["used_group_beta1"] for row in theirs]), one)
                resolved = RUNS["receipt"][name]["schedule"]["resolved"]
                self.assertEqual(([group["lr"] for group in resolved["groups"]], [group["betas"][0] for group in resolved["groups"]]),
                                 (mine[0]["used_group_lrs"], mine[0]["used_group_beta1"]))
        # The same in the diagnostics' own per-update log.
        rows = lines(self.out["POINTER_2EP"] / "diagnostics/optimizer_updates.jsonl")
        self.assertEqual([row["used_group_lrs"] for row in rows], two[0])

    def test_where_the_two_schedules_coincide_is_measured_not_assumed(self):
        """What this build gives, read from the records: the first update uses the schedule's initial value, which
        does not depend on the horizon, so it is the reference's update exactly; the second already is not."""
        for name, reference in self.pairs.items():
            with self.subTest(arm=name):
                mine, theirs = RUNS["updates"][name], RUNS["updates"][reference]
                equal = [a["used_group_lrs"] == b["used_group_lrs"] and a["used_group_beta1"] == b["used_group_beta1"]
                         for a, b in zip(mine, theirs)]
                self.assertEqual(equal, [True] + [False] * (PER_EPOCH - 1))
                self.assertEqual(mine[0], theirs[0])                              # the whole first update record
                self.assertEqual({key for key in mine[1] if mine[1][key] != theirs[1][key]}, {"used_group_lrs", "used_group_beta1"})
                self.assertNotEqual(mine[2]["objective"], theirs[2]["objective"])  # other parameters from the third forward on
                # Dropout draws do not depend on the parameters: the seeded generator sequence of the first epoch is the reference's.
                self.assertEqual([row["torch_rng_sha256"] for row in mine[:PER_EPOCH]], [row["torch_rng_sha256"] for row in theirs])
        # The initial value is not the nominal max_lr / 25: it is what the annealing expression returns at 0.
        first = RUNS["updates"]["POINTER_2EP"][0]["used_group_lrs"][0]
        self.assertEqual(first, RATE + (RATE / 25.0 - RATE) / 2.0 * (math.cos(math.pi * 0.0) + 1))

    # --- nothing but reads ---------------------------------------------------------------------------------------

    def test_the_two_epoch_path_changes_nothing_a_plain_two_epoch_run_does(self):
        mine, plain = self.out["POINTER_2EP"], self.out["PLAIN_2EP"]
        self.assertEqual((mine / UPDATES).read_bytes(), (plain / UPDATES).read_bytes())
        for label in ("initial", "final"):
            ours, theirs = state(mine, label), state(plain, label)
            for tensor, value in ours.items():
                self.assertTrue(same(value, theirs[tensor]), (label, tensor))
        self.assertEqual(RUNS["done"]["POINTER_2EP"]["input_stream_sha256"], RUNS["done"]["PLAIN_2EP"]["input_stream_sha256"])
        self.assertEqual(RUNS["done"]["POINTER_2EP"]["torch_rng_trace_sha256"], RUNS["done"]["PLAIN_2EP"]["torch_rng_trace_sha256"])
        self.assertFalse((plain / "e11_reference").exists())
        self.assertFalse(any(read(plain / "training_config.json")["args"][key] for key in e11_arms.ARGUMENTS))
        self.assertEqual(RUNS["receipt"]["POINTER_2EP"]["generators"]["natural_equal_to_reference"], [True])
        self.assertTrue(RUNS["done"]["POINTER_2EP"]["unseeded_python_numpy_rng_unchanged"])

    def test_saved_head_metadata_is_the_existing_contract_and_reloads(self):
        import kev.evaluate as evaluate
        from runtime import bundle_contract
        for name, reference in self.pairs.items():
            with self.subTest(arm=name):
                out, readout = self.out[name], e11_arms.ARMS[name]["readout"]
                meta = torch.load(out / "head.pt", map_location="cpu")
                theirs = torch.load(self.out[reference] / "head.pt", map_location="cpu")
                self.assertEqual(set(meta), set(theirs))
                self.assertEqual(meta["head_config"], theirs["head_config"])
                self.assertEqual(bundle_contract.config_identity(meta["head_config"]), e458_arms.head_config(readout))
                self.assertEqual((meta["optimizer_steps"], meta["stopped_early"], meta["args"]["epochs"], meta["args"]["e11_arm"],
                                  meta["dse"]["planned_optimizer_steps"]), (STEPS, False, 2, name, STEPS))
                boundary = torch.load(out / "checkpoints/step_000012/head.pt", map_location="cpu")
                self.assertEqual((boundary["optimizer_steps"], boundary["stopped_early"], boundary["intermediate_checkpoint"]),
                                 (PER_EPOCH, True, True))
                with mock.patch.object(model_module, "AutoModelForCausalLM", FakeAuto), \
                        mock.patch.object(evaluate, "load_tokenizer", lambda base, revision=None: FakeTokenizer()):
                    tok, model = evaluate.load(str(out), "cpu")
                self.assertEqual(model.head_config["readout"], readout)
                sample = read(out / "diagnostics/fixed_sample.json")
                recorded = [row for row in read(out / "diagnostics/fixed_sample_final.json") if row["sample_variant"] == "original"]
                position = 0
                with torch.no_grad():
                    for entry in sample["records"]:
                        enc = model.encode(tok, entry["record"], max_state=2048, max_branch=2048, strict=True)
                        for logits in model.forward(enc):
                            self.assertTrue(torch.allclose(logits, torch.tensor(recorded[position]["logits"]), atol=1e-5), (name, position))
                            position += 1
                self.assertEqual(position, len(recorded))

    # --- refusals: each stops before the first update ------------------------------------------------------------

    def refuse(self, label, flags, error, text, shared=(COMMON, TWO), entered=False):
        out = self.root / label
        with self.assertRaises(error) as caught:
            train(out, self.data, *shared, flags)
        self.assertIn(text, str(caught.exception))
        members = ["e11_reference/receipt.json", "e11_reference/epochs.jsonl", "campaign/initial_receipt.json", UPDATES, "head.pt"]
        for member in members + ([] if entered else ["diagnostics", "campaign"]):
            self.assertFalse((out / member).exists(), member)
        return str(caught.exception)

    def test_another_epoch_count_or_update_count_stops_the_run(self):
        a = self.out["A"]
        for epochs in ("1", "3"):
            self.refuse("epochs_" + epochs, e11("POINTER_2EP", a), ValueError, "trains 2 epochs, not --epochs " + epochs,
                        shared=(COMMON, changed(TWO, epochs=epochs, campaign_checkpoint_steps="3")))
        # ceil(two epochs' records / accumulation) is one update short; the trainer's own count is what runs.
        self.refuse("ceil_of_the_total", e11("POINTER_2EP", a, steps=STEPS - 1), ValueError,
                    f"{STEPS} planned updates here, 2 epochs of {PER_EPOCH}; the plan expects {STEPS - 1}", entered=True)
        self.refuse("wrong_count", e11("POINTER_2EP", a, adapter=NUMEL["adapter"] + 64), ValueError, "are not the planned")
        self.refuse("unknown_arm", e11("POINTER_2EP", a)[:2] + ["--e11_arm", "POINTER_3EP"] + e11("POINTER_2EP", a)[4:],
                    ValueError, "unknown E11 arm")

    def test_a_reference_that_is_not_the_paired_one_epoch_run_stops_the_run(self):
        out = self.out
        # A mean-span arm of the earlier block offered as the pointer's reference is named for what it is.
        text = self.refuse("pointer_on_pooling", e11("POINTER_2EP", out["POOL_ALL"]), ReferenceUnusable,
                           "reference head is not the pointer with the end readout")
        self.assertIn("the reference run is POOL_ALL, not the LL_A of E1/E2", text)
        self.refuse("pooling_on_pointer", e11("POOL_2EP", out["A"]), ReferenceUnusable, "has no step-zero record ['e458_reference/receipt.json']")
        self.refuse("pooling_on_plain_pooling", e11("POOL_2EP", out["PLAIN_POOL"]), ReferenceUnusable,
                    "has no step-zero record ['e458_reference/receipt.json']")
        self.refuse("two_epoch_reference", e11("POINTER_2EP", out["POINTER_2EP"]), ReferenceUnusable, "is itself a two-epoch arm",
                    shared=(COMMON, changed(TWO, campaign_checkpoint_steps="3,12")))
        self.refuse("absent_reference", e11("POINTER_2EP", self.root / "nowhere"), ReferenceUnusable, "has no step-zero record")

    def test_a_protocol_that_differs_outside_the_two_declared_arguments_stops_the_run(self):
        a = self.out["A"]
        self.refuse("other_rate", e11("POINTER_2EP", a), ReferenceUnusable, "arguments ['lr'] differ from the reference",
                    shared=(changed(COMMON, lr="2e-3"), TWO))
        self.refuse("other_seed", e11("POINTER_2EP", a), ReferenceUnusable, "arguments ['seed'] differ from the reference",
                    shared=(changed(COMMON, seed="45"), TWO))
        self.refuse("other_accumulation", e11("POINTER_2EP", a, steps=46), ReferenceUnusable, "arguments ['accum'] differ from the reference",
                    shared=(changed(COMMON, accum="2"), TWO))
        self.refuse("same_saved_steps", e11("POINTER_2EP", a), ReferenceUnusable, "are not all changes",
                    shared=(COMMON, changed(TWO, campaign_checkpoint_steps="2,6")))
        other = self.root / "other.jsonl"
        write_training_file(other, records=RECORDS + 4)
        with self.assertRaises(ReferenceUnusable) as caught:
            train(self.root / "other_data", other, COMMON, TWO, e11("POINTER_2EP", a, steps=2 * math.ceil((RECORDS + 4) / ACCUM)))
        self.assertIn("reference trained on different data bytes", str(caught.exception))

    def test_arm_flags_go_together_and_exclude_the_earlier_blocks_arm(self):
        a = self.out["A"]
        for flags in (["--e11_arm", "POINTER_2EP"], ["--e11_reference_run", str(a)], ["--e11_expected_optimizer_steps", "24"],
                      e11("POINTER_2EP", a) + ["--e458_arm", "POOL_ALL", "--e458_reference_run", str(a)]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                train(self.root / "rejected", self.data, COMMON, TWO, head("end") if "--head_config" not in flags else [], flags)
        self.assertFalse((self.root / "rejected").exists())

    def test_the_arm_table_is_the_two_frozen_arms(self):
        self.assertEqual(list(e11_arms.ARMS), ["POINTER_2EP", "POOL_2EP"])
        rows = {name: e11_arms.arm(name) for name in e11_arms.ARMS}
        self.assertEqual({name: (row["epochs"], row["rank"], row["alpha"], row["targets"], row["readout"], row["adapter"], row["structure"])
                          for name, row in rows.items()},
                         {"POINTER_2EP": (2, 64, 128, "all", "end", "reference", "all/r64"),
                          "POOL_2EP": (2, 64, 128, "all", "mean_span", "reference", "all/r64")})
        self.assertEqual({name: (row["reference"]["arm"], row["reference"]["block"]) for name, row in rows.items()},
                         {"POINTER_2EP": ("LL_A", "E1/E2"), "POOL_2EP": ("POOL_ALL", "E4a/E5/E8")})
        self.assertEqual({name: row["argument_changes"] for name, row in rows.items()},
                         dict.fromkeys(rows, ["campaign_checkpoint_steps", "epochs"]))
        self.assertEqual(rows["POOL_2EP"]["source_files_changed"], ["e11_arms.py", "e11_reference.py", "train.py"])
        self.assertEqual(rows["POINTER_2EP"]["source_files_changed"],
                         ["e11_arms.py", "e11_reference.py", "e458_arms.py", "e458_reference.py", "heads.py", "model.py", "train.py"])


@unittest.skipUnless((CAMPAIGN / "run_e11.py").is_file(), "the controller is read from ../../run_e11.py")
class ControllerReadersOnRealRecordsTest(unittest.TestCase):
    """The controller's own readers on what the real trainer wrote: the interface, without a stand-in in between."""
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(CAMPAIGN))
        import run_e11
        cls.controller, cls.e458, cls.basis = run_e11, run_e11.e458, run_e11.basis
        out, data = RUNS["out"], RUNS["data"]
        by_leaf = {}
        for module in lora_modules():
            by_leaf[module["leaf"]] = by_leaf.get(module["leaf"], 0) + 1
        cls.structure = {"numel": {**NUMEL, "trainable": NUMEL["head"] + NUMEL["adapter"]}, "lora_tensors": 2 * len(lora_modules()),
                         "lora_modules_by_leaf": by_leaf, "copied_adapter_tensors": 2 * len(lora_modules())}
        pinned = {"kev/" + name: value for name, value in source_hashes().items()}
        lora = {"rank": 64, "alpha": 128, "targets": "all", "target_modules": list(e458_arms.TARGET_MODULES["all"]), "dropout": 0.05}

        def entry(name, arm, readout, steps, records, checkpoints, **more):
            directory = out[name]
            return {"run_id": name, "cell": "F9s44", "fold": "F9", "seed": 44, "arm": arm, "head_config": e458_arms.head_config(readout),
                    "head_peak_lr": RATE, "adapter_peak_lr": RATE, "frozen_zero_residual": False, "lora": lora, "data_sha256": digest(data),
                    "checkpoint_steps": checkpoints,
                    "expected": {"optimizer_steps": steps, "planned_optimizer_steps": steps, "source_records": records, "status": "complete",
                                 "head_optimizer_tensors": list(cls.e458.QK), **cls.structure},
                    "paths": {"run_dir": str(directory), "output_dir": str(directory), "training_config": str(directory / "training_config.json"),
                              "per_update_log": str(directory / UPDATES),
                              "reference_initialization": str(directory / "e11_reference/receipt.json"),
                              "epoch_records": str(directory / "e11_reference/epochs.jsonl")},
                    "argv": ["python", "-P", "-m", "kev.train", "--device", "cpu", "--data", str(data)], **more}
        cls.references = {"POINTER_2EP": entry("A", "LL_A", "end", PER_EPOCH, RECORDS, [2, 6]),
                          "POOL_2EP": entry("POOL_ALL", "POOL_ALL", "mean_span", PER_EPOCH, RECORDS, [2, 6])}
        cls.runs = {}
        for name, reference in cls.references.items():
            run = entry(name, name, e11_arms.ARMS[name]["readout"], STEPS, 2 * RECORDS, [3, 12, 18], epochs=2,
                        argument_changes=list(e11_arms.ARGUMENT_CHANGES),
                        reference={"run_id": reference["run_id"], "output_dir": reference["paths"]["output_dir"],
                                   "optimizer_steps": PER_EPOCH})
            run["expected"].update(epochs=2, source_records_per_epoch=RECORDS, optimizer_steps_per_epoch=PER_EPOCH,
                                   source_records_in_the_last_update_of_an_epoch=LAST)
            cls.runs[name] = run
        blocks = {name: e11_arms.ARMS[name]["reference"]["block"] for name in cls.runs}
        cls.reference_plan = {"source_sha256": pinned}                      # the tree that trained the stand-in references
        cls.plan = {
            "references": {name: {"block": blocks[name], "arm": cls.references[name]["arm"], "campaign": "tiny",
                                  "expected": {key: cls.structure[key] for key in ("numel", "lora_tensors", "lora_modules_by_leaf")}}
                           for name in cls.runs},
            "schedule": {"class": "torch.optim.lr_scheduler.OneCycleLR",
                         "source_call": {"max_lr": [RATE, RATE], "pct_start": 0.1, "total_steps": {"F9": STEPS}},
                         "constructor_defaults": {"anneal_strategy": "cos", "cycle_momentum": True, "base_momentum": 0.85, "max_momentum": 0.95,
                                                  "div_factor": 25.0, "final_div_factor": 10000.0, "three_phase": False, "last_epoch": -1},
                         "one_epoch_total_steps": {"F9": PER_EPOCH}},
            "recipe": {"peak_lr": {"head": RATE, "adapter": RATE}, "weight_decay": 0.01},
            "source_sha256": pinned, "source_basis": {block: {"kev_files_differing_from_reference": []} for block in blocks.values()},
            "trainer_arguments": {"e11": list(e11_arms.ARGUMENTS), "e458": list(e458_arms.ARGUMENTS),
                                  "not_compared": list(e458_arms.ARGUMENTS_NOT_COMPARED)}}

    def facts(self, name):
        problems, facts = self.e458.reference_record({"reference_campaign": self.plan["references"][name]}, self.reference_plan,
                                                     self.references[name])
        self.assertEqual(problems, [])
        return facts

    def test_every_check_reads_the_real_records_and_finds_them_as_planned(self):
        controller, basis, table, cache = self.controller, self.basis, {}, {}
        for name, run in self.runs.items():
            with self.subTest(arm=name):
                reference = self.references[name]
                self.assertEqual(basis.check_outputs(run), [])                     # Check completion for two planned epochs.
                mine, theirs = basis.step_zero(run), basis.step_zero(reference)
                self.assertEqual(controller.step_zero(run, reference, mine, theirs), [])
                self.assertIn(("F9s44", "realized epoch order"), mine[0])
                problems, observed = controller.initialization(self.plan, run, reference, self.facts(name))
                self.assertEqual(problems, [])
                self.assertEqual((observed["natural_generator_state_equal_reference"], observed["head_tensors_naturally_equal_before_copy"],
                                  observed["actual_numel"]), ([True], 4, self.structure["numel"]))
                self.assertEqual(controller.recorded_arguments(self.plan, run, reference), [])
                problems, observed = controller.two_epochs(run, cache)
                self.assertEqual(problems, [])
                self.assertEqual(observed, {"updates_per_epoch": [PER_EPOCH, PER_EPOCH], "second_epoch_order_equals_first": False,
                                            "source_records_in_the_last_update_of_each_epoch": [LAST, LAST]})
                problems, observed = controller.first_epoch(run, reference, cache)
                self.assertEqual(problems, [])
                self.assertEqual(observed, {
                    "first_epoch_updates": PER_EPOCH, "first_update_used_the_reference_lr_and_beta1": True,
                    "first_epoch_updates_that_used_the_reference_lr_and_beta1": 1, "first_update_with_another_lr_or_beta1": 2,
                    "first_epoch_updates_with_the_reference_generator_state": PER_EPOCH,
                    "first_epoch_updates_with_the_reference_input_digest": PER_EPOCH})
                problems, observed = controller.schedule(self.plan, run, reference, cache)
                self.assertEqual(problems, [])
                self.assertEqual((observed["largest_relative_deviation_from_the_schedule_two_epoch"],
                                  observed["largest_relative_deviation_from_the_schedule_one_epoch_reference"]), (0.0, 0.0))
                self.assertEqual(observed["first_update_lr_and_beta1"], observed["entry_lr_and_beta1_in_receipt"])
                shared = controller.whole_run(run, cache)
                self.assertEqual(len(shared), 4)
                self.assertEqual(basis.compare(table, name, shared), [])           # the second arm against the first
        self.assertEqual({holder for _, holder in table.values()}, {"POINTER_2EP"})

    def test_one_epoch_of_output_is_not_a_complete_two_epoch_run(self):
        run = self.repointed(self.runs["POINTER_2EP"], RUNS["out"]["POINTER_2EP"], RUNS["out"]["A"])   # the plan entry, one epoch of output
        problems = self.basis.check_outputs(run)
        self.assertIn(f"optimizer steps {PER_EPOCH}/{PER_EPOCH}, expected {STEPS}/{STEPS}", problems)
        self.assertIn("training status or source record count differs from the plan", problems)
        self.assertIn("campaign record does not cover every optimizer update", problems)
        self.assertIsNone(self.controller.two_epochs(run, {}))                     # and it has no epoch record at all
        # A run that stopped inside its second epoch would have one epoch row; nothing is concluded from that.
        partial = self.copy("partial", "POINTER_2EP")
        path = Path(partial["paths"]["epoch_records"])
        path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])
        problems, _ = self.controller.two_epochs(partial, {})
        self.assertIn("the epoch record does not hold 2 complete epochs of 12 updates and 46 records", problems)

    @staticmethod
    def repointed(run, old, new):
        run = json.loads(json.dumps(run))
        run["paths"] = {key: value.replace(str(old), str(new)) for key, value in run["paths"].items()}
        return run

    def copy(self, label, name):
        directory = RUNS["root"] / ("copy_" + label)
        shutil.copytree(RUNS["out"][name], directory)
        run = self.runs.get(name) or next(row for row in self.references.values() if row["run_id"] == name)
        return {**self.repointed(run, RUNS["out"][name], directory), "run_id": label}

    def rewrite_updates(self, run, change):
        path = Path(run["paths"]["per_update_log"])
        rows = lines(path)
        change(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_what_differs_from_the_plan_or_the_reference_is_found_in_real_records(self):
        controller, reference = self.controller, self.references["POINTER_2EP"]
        # an update that did not flush at the epoch end
        run = self.copy("no_flush", "POINTER_2EP")
        self.rewrite_updates(run, lambda rows: rows[PER_EPOCH - 1]["objective"].update(source_records=ACCUM))
        self.assertIn("first difference at update 12", controller.two_epochs(run, {})[0][0])
        # a first epoch that is not the reference's stream
        other = self.copy("other_stream_reference", "A")
        path = Path(other["paths"]["output_dir"]) / "campaign/completion.json"
        path.write_text(json.dumps({**read(path), "input_stream_sha256": "0" * 64}))
        problems, _ = controller.first_epoch(self.runs["POINTER_2EP"], other, {})
        self.assertEqual(problems, ["augmented input stream of the first epoch differs from other_stream_reference's whole run"])
        # used values that are not the recorded schedule, for the run and for the reference
        run = self.copy("other_rate", "POINTER_2EP")
        self.rewrite_updates(run, lambda rows: rows[4].update(used_group_lrs=[value * 1.01 for value in rows[4]["used_group_lrs"]]))
        problems, _ = controller.schedule(self.plan, run, reference, {})
        self.assertEqual(len(problems), 1)
        self.assertIn("this run used are not this schedule over 24 updates", problems[0])
        other = self.copy("other_rate_reference", "A")
        self.rewrite_updates(other, lambda rows: rows[4].update(used_group_beta1=[0.9, 0.9]))
        problems, _ = controller.schedule(self.plan, self.runs["POINTER_2EP"], other, {})
        self.assertIn("the reference other_rate_reference used are not this schedule over 12 updates", problems[0])
        # a plan that expects other constructor defaults than the installed ones
        plan = json.loads(json.dumps(self.plan))
        plan["schedule"]["constructor_defaults"]["div_factor"] = 10.0
        problems, _ = controller.schedule(plan, self.runs["POINTER_2EP"], reference, {})
        self.assertIn("the installed constructor defaults are not the planned ones: {'div_factor': 25.0}", problems)
        # the other arm's reference is not this arm's
        problems, _ = controller.initialization(self.plan, self.runs["POINTER_2EP"], self.references["POOL_2EP"], self.facts("POOL_2EP"))
        self.assertIn("the reference files the run read are not the reference's files now", problems)

    def test_the_recomputed_schedule_is_the_installed_schedulers_at_the_real_horizons(self):
        """The controller's standard-library schedule against torch's own object, value for value, for the two real
        folds at one and two epochs and the real peak rate, and for the short horizons of these tests."""
        for steps, rate in ((1932, 1.25e-5), (3864, 1.25e-5), (1931, 1.25e-5), (3862, 1.25e-5), (PER_EPOCH, RATE), (STEPS, RATE), (7, RATE)):
            with self.subTest(steps=steps):
                bounds = {"initial_lr": rate / 25.0, "max_lr": rate, "min_lr": rate / 25.0 / 10000.0, "max_momentum": 0.95, "base_momentum": 0.85}
                resolved = {"pct_start": 0.1, "groups": [bounds, bounds], "phases": [
                    {"start_lr": "initial_lr", "end_lr": "max_lr", "start_momentum": "max_momentum", "end_momentum": "base_momentum"},
                    {"start_lr": "max_lr", "end_lr": "min_lr", "start_momentum": "base_momentum", "end_momentum": "max_momentum"}]}
                self.assertEqual(self.controller.one_cycle(resolved, steps), replay(steps, rate))
        one, two = replay(1932, 1.25e-5), replay(3864, 1.25e-5)
        self.assertEqual((one[0][0], one[1][0]), (two[0][0], two[1][0]))           # the first update: the same values at both horizons
        self.assertNotEqual(one[0][1], two[0][1])                                  # the second: not
        self.assertEqual(sum(a == b for a, b in zip(one[0], two[0])), 1)
        self.assertEqual(sum(a == b for a, b in zip(*(replay(steps, 1.25e-5)[0] for steps in (1931, 3862)))), 1)


if __name__ == "__main__":
    unittest.main()
