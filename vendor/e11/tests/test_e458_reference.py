"""CPU tests of the E4a/E5/E8 step-zero path.

The real kev.train loop, the real DecisionModel and the real PEFT LoRA run on a tiny hybrid backbone
(tests/tiny_hybrid.py). A plain rank-64 all-target pointer run of this tree stands in for the E1 low/low
reference; the five fixed arms, and plain runs without the reference path, are compared with its actual files.
Run in the training environment with CUDA_VISIBLE_DEVICES='' and PYTHONPATH at this source tree. The CUDA
generator branch and the device-side copy cannot run here.
"""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import load_file, save_file

from kev import e458_arms
from kev import model as model_module
from kev.e458_reference import ReferenceUnusable, leaf_of
from kev.heads import normalize_head_config
from kev.supplement_diagnostics import digest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_hybrid import HIDDEN, FakeAuto, FakeTokenizer, lora_modules, run_trainer, write_training_file  # noqa: E402

COMMON = ["--base", "synthetic", "--epochs", "1", "--lr", "1e-3", "--head_lr", "1e-3", "--weight_decay", "0.01",
          "--accum", "4", "--device", "cpu", "--batch", "1", "--dtype", "fp32", "--weights_dtype", "fp32",
          "--checkpointing", "0", "--p_none_pair", "0.25", "--seed", "44", "--lora_dropout", "0.05",
          "--supplement_diagnostics", "1", "--r3_control_capture", "1", "--r3_sham_residual", "0",
          "--campaign_diagnostics", "1", "--campaign_checkpoint_steps", "2"]
RANK64 = ("POINTER_NOFFN", "POOL_ALL", "POOL_NOFFN", "POINTER_LAST_CONTENT")
SAME_STRUCTURE = ("POOL_ALL", "POINTER_LAST_CONTENT")
ORDER = "r3_control/epoch_0_actual_original_data_order.json"


def structure_flags(rank, targets, readout):
    return ["--lora", str(rank), "--lora_targets", targets,
            "--head_config", json.dumps(e458_arms.head_config(readout), sort_keys=True, separators=(",", ":"))]


def planned(name):
    arm = e458_arms.arm(name)
    return {"head": e458_arms.head_numel(HIDDEN),
            "adapter": e458_arms.adapter_numel(lora_modules(), arm["rank"], arm["targets"])}


def arm_flags(name, reference, **numel):
    arm, numel = e458_arms.arm(name), {**planned(name), **numel}
    return structure_flags(arm["rank"], arm["targets"], arm["readout"]) + [
        "--e458_arm", name, "--e458_reference_run", str(reference),
        "--e458_expected_head_numel", str(numel["head"]), "--e458_expected_adapter_numel", str(numel["adapter"])]


def common(**changes):
    flags = list(COMMON)
    for key, value in changes.items():
        flags[flags.index("--" + key) + 1] = value
    return flags


def train(out, data, flags, shared=COMMON):
    run_trainer([*shared, "--data", str(data), "--out", str(out), *flags])


def read(path):
    return json.loads(Path(path).read_text())


def state(out, label):
    head = load_file(str(out / f"diagnostics/head_{label}.safetensors"))
    return {**{"head." + name: value for name, value in head.items()},
            **load_file(str(out / f"diagnostics/adapter_{label}.safetensors"))}


def same(left, right):
    return left.shape == right.shape and torch.equal(left.view(torch.uint8), right.view(torch.uint8))


class ReferenceInitializationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path(tempfile.mkdtemp(prefix="jev-e458-test-"))
        cls.data = cls.root / "train.jsonl"
        write_training_file(cls.data)
        cls.out = {}
        # Plain runs of this tree, without the reference path: the reference, its repeat, and what a scope or
        # rank change gives by the same seed alone.
        for name, setting in (("A", (64, "all", "end")), ("A_REPEAT", (64, "all", "end")),
                              ("PLAIN_NOFFN", (64, "attn", "end")), ("PLAIN_R28", (28, "all", "end"))):
            cls.out[name] = cls.root / name
            train(cls.out[name], cls.data, structure_flags(*setting))
        for name in e458_arms.ARMS:
            cls.out[name] = cls.root / name
            train(cls.out[name], cls.data, arm_flags(name, cls.out["A"]))
        # Not an arm of the block: the reference's own structure through the copy and the restore.
        unchanged = {"SAME_AS_REFERENCE": {"rank": 64, "targets": "all", "readout": "end", "adapter": "reference", "roles": []}}
        with mock.patch.dict(e458_arms.ARMS, unchanged):
            cls.out["SAME_AS_REFERENCE"] = cls.root / "SAME_AS_REFERENCE"
            train(cls.out["SAME_AS_REFERENCE"], cls.data, arm_flags("SAME_AS_REFERENCE", cls.out["A"]))
        cls.initial = {name: state(out, "initial") for name, out in cls.out.items()}
        cls.receipt = {name: read(out / "e458_reference/receipt.json") for name, out in cls.out.items()
                       if (out / "e458_reference").exists()}
        cls.entry = {name: read(out / "campaign/initial_receipt.json") for name, out in cls.out.items()}
        cls.done = {name: read(out / "campaign/completion.json") for name, out in cls.out.items()}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_rank64_arms_start_from_the_reference_tensors(self):
        reference = self.initial["A"]
        mlp = {name for name in reference if not name.startswith("head.") and leaf_of(name) in e458_arms.MLP}
        self.assertEqual(len(mlp), 2 * 3 * 4)                           # A and B of three leaves in four layers
        for name in RANK64:
            with self.subTest(arm=name):
                arm, tensors, summary = e458_arms.arm(name), self.initial[name], self.receipt[name]["tensors"]
                self.assertEqual(set(reference) - set(tensors), mlp if arm["targets"] == "attn" else set())
                self.assertFalse(set(tensors) - set(reference))
                for tensor, value in tensors.items():
                    self.assertTrue(same(value, reference[tensor]), tensor)
                self.assertEqual(summary["head"]["copied_and_read_back_equal"], 4)
                self.assertEqual(summary["adapter"]["copied_and_read_back_equal"], len(tensors) - 4)
                self.assertEqual(summary["adapter"]["only_in_reference"], len(mlp) if arm["targets"] == "attn" else 0)
                self.assertEqual(set(summary["adapter"]["only_in_reference_by_leaf"]),
                                 set(e458_arms.MLP) if arm["targets"] == "attn" else set())
                self.assertTrue(summary["adapter"]["matched_to_reference"] and summary["lora_B_all_zero"])
                self.assertEqual(len(summary["copied"]["rows"]), len(tensors))
                # Check the recorded step-zero tensor hashes.
                hashed = self.entry[name]["tensor_sha256"]
                self.assertEqual(set(hashed), set(tensors))
                self.assertEqual({tensor: self.entry["A"]["tensor_sha256"][tensor] for tensor in hashed}, hashed)
                self.assertEqual(self.entry[name]["tensor_groups"]["adapter"], summary["group_sha256"]["adapter"])
                self.assertEqual(self.entry[name]["tensor_groups"]["head_qk"], summary["group_sha256"]["head_qk"])
                self.assertEqual(self.entry[name]["tensor_groups"]["head_qk"], self.entry["A"]["tensor_groups"]["head_qk"])
        for name in SAME_STRUCTURE:
            self.assertEqual(self.entry[name]["tensor_groups"]["adapter"], self.entry["A"]["tensor_groups"]["adapter"])
        self.assertEqual(self.entry["POINTER_NOFFN"]["tensor_groups"]["adapter"],
                         self.entry["POOL_NOFFN"]["tensor_groups"]["adapter"])
        self.assertNotEqual(self.entry["POINTER_NOFFN"]["tensor_groups"]["adapter"],
                            self.entry["A"]["tensor_groups"]["adapter"])

    def test_the_same_seed_alone_does_not_match_a_scope_change(self):
        reference, natural = self.initial["A"], self.initial["PLAIN_NOFFN"]
        differing = [name for name, value in natural.items() if not same(value, reference[name])]
        self.assertEqual(sum(name.startswith("head.") for name in differing), 4)
        self.assertTrue(any(".lora_A." in name for name in differing))
        summary = self.receipt["POINTER_NOFFN"]["tensors"]
        self.assertEqual(summary["head"]["naturally_equal_before_copy"], 0)
        self.assertEqual(summary["adapter"]["copied_and_read_back_equal"] - summary["adapter"]["naturally_equal_before_copy"],
                         len(differing) - 4)
        for name in SAME_STRUCTURE + ("SAME_AS_REFERENCE",):          # same construction: the copy changed nothing
            summary = self.receipt[name]["tensors"]
            self.assertEqual(summary["head"]["naturally_equal_before_copy"], 4)
            self.assertEqual(summary["adapter"]["naturally_equal_before_copy"],
                             summary["adapter"]["copied_and_read_back_equal"])

    def test_rank28_keeps_its_ordinary_adapter_and_takes_only_the_head(self):
        reference, ordinary, arm = self.initial["A"], self.initial["PLAIN_R28"], self.initial["POINTER_ALL_R28"]
        head = sorted(name for name in arm if name.startswith("head."))
        adapter = sorted(name for name in arm if not name.startswith("head."))
        self.assertEqual(set(arm), set(reference))                     # the same modules as the reference
        for name in head:
            self.assertTrue(same(arm[name], reference[name]), name)
        self.assertFalse(any(same(ordinary[name], reference[name]) for name in head))
        for name in adapter:
            self.assertTrue(same(arm[name], ordinary[name]), name)     # the ordinary rank-28 values, untouched
            axis = 0 if ".lora_A." in name else 1
            self.assertEqual((arm[name].shape[axis], reference[name].shape[axis]), (28, 64))
            self.assertEqual(arm[name].shape[1 - axis], reference[name].shape[1 - axis])
            if ".lora_B." in name:
                self.assertEqual(torch.count_nonzero(arm[name]).item(), 0)
        receipt = self.receipt["POINTER_ALL_R28"]
        summary = receipt["tensors"]["adapter"]
        self.assertEqual((summary["mode"], summary["copied_and_read_back_equal"], summary["matched_to_reference"]),
                         ("ordinary", 0, False))
        self.assertIsNone(summary["naturally_equal_before_copy"])
        self.assertIn("kaiming_uniform_", summary["initialization"])
        self.assertEqual([row[0] for row in receipt["tensors"]["copied"]["rows"]], head)
        self.assertEqual(receipt["tensors"]["group_sha256"]["adapter"], self.entry["POINTER_ALL_R28"]["tensor_groups"]["adapter"])
        self.assertEqual(self.entry["POINTER_ALL_R28"]["tensor_groups"]["adapter"], self.entry["PLAIN_R28"]["tensor_groups"]["adapter"])
        for low, high in receipt["tensors"]["lora_A_std_over_bound_by_leaf"].values():
            self.assertTrue(.4 < low <= high < .75)                    # a uniform draw has sqrt(1/3)
        # alpha 56, alpha / rank = 2, nothing that rescales a module
        saved = read(self.out["POINTER_ALL_R28"] / "adapter_config.json")
        self.assertEqual((saved["r"], saved["lora_alpha"], saved["use_rslora"], saved["rank_pattern"], saved["alpha_pattern"]),
                         (28, 56, False, {}, {}))
        self.assertEqual(receipt["structure"]["scaling_values"], [2.0])
        self.assertEqual(receipt["structure"]["lora_configuration"]["lora_alpha"], 56)
        capture = read(self.out["POINTER_ALL_R28"] / "diagnostics/initial_capture.json")
        self.assertEqual(set(capture["adapter_scaling"].values()), {2.0})
        self.assertEqual(read(self.out["A"] / "adapter_config.json")["lora_alpha"], 128)

    def test_training_starts_from_the_reference_generator_state(self):
        entry = self.entry["A"]["rng_at_loop_entry"]
        self.assertIsNone(entry["torch_cuda_sha256"])                  # CPU test: the CUDA branch is not exercised
        for name, receipt in self.receipt.items():
            with self.subTest(arm=name):
                self.assertEqual(self.entry[name]["rng_at_loop_entry"], entry)
                generators = receipt["generators"]
                self.assertTrue(generators["read_back_equal"])
                self.assertEqual(generators["restored_digests_equal_reference_loop_entry"],
                                 {key: entry[key] for key in ("torch_cpu_sha256", "torch_cuda_sha256")})
                self.assertEqual(generators["natural_equal_to_reference"], [receipt["arm"]["same_structure_as_reference"]])
                self.assertEqual(receipt["loop_entry"]["data_order_rng_sha256"], entry["data_order_rng_sha256"])
                # Preserve the saved construction state.
                own = read(self.out[name] / "r3_control/model_to_device_complete.RNG.json")
                self.assertEqual(generators["natural_sha256_in_own_r3_capture"], [row["sha256"] for row in own["full_bytes"]])
        for name in ("PLAIN_NOFFN", "PLAIN_R28"):                      # without the restore, construction shifts it
            other = self.entry[name]["rng_at_loop_entry"]
            self.assertNotEqual(other["torch_cpu_sha256"], entry["torch_cpu_sha256"])
            self.assertEqual(other["data_order_rng_sha256"], entry["data_order_rng_sha256"])

    def test_input_stream_is_the_references_and_dropout_follows_the_structure(self):
        reference = self.done["A"]
        for name in self.receipt:
            self.assertEqual(self.done[name]["input_stream_sha256"], reference["input_stream_sha256"], name)
            self.assertEqual((self.out[name] / ORDER).read_bytes(), (self.out["A"] / ORDER).read_bytes(), name)
            self.assertTrue(self.done[name]["unseeded_python_numpy_rng_unchanged"])
        trace = {name: row["torch_rng_trace_sha256"] for name, row in self.done.items()}
        for name in SAME_STRUCTURE + ("SAME_AS_REFERENCE",):          # same modules, same draws
            self.assertEqual(trace[name], trace["A"], name)
        self.assertEqual(trace["POINTER_NOFFN"], trace["POOL_NOFFN"])   # fewer modules: another sequence, shared by both
        self.assertNotEqual(trace["POINTER_NOFFN"], trace["A"])
        self.assertNotEqual(trace["POINTER_NOFFN"], trace["PLAIN_NOFFN"])   # and not what the seed alone gives
        # LoRA dropout acts on the module input, whose shape does not depend on the rank. Observed here; the
        # controller records it for the rank-28 arm and does not require it.
        self.assertEqual(trace["POINTER_ALL_R28"], trace["A"])

    def test_copy_and_restore_alone_reproduce_the_reference_run(self):
        reference, repeat, through = (state(self.out[name], "final") for name in ("A", "A_REPEAT", "SAME_AS_REFERENCE"))
        self.assertEqual(set(reference), set(through))
        for name, value in reference.items():
            self.assertTrue(same(value, repeat[name]), name)
            self.assertTrue(same(value, through[name]), name)
        updates = "campaign/updates.jsonl"
        self.assertEqual((self.out["A"] / updates).read_bytes(), (self.out["SAME_AS_REFERENCE"] / updates).read_bytes())
        self.assertFalse(all(same(value, state(self.out["POOL_ALL"], "final")[name]) for name, value in reference.items()))

    def test_structure_and_element_counts_are_the_actual_ones(self):
        counts = {}
        for name in e458_arms.ARMS:
            with self.subTest(arm=name):
                arm, structure = e458_arms.arm(name), self.receipt[name]["structure"]
                recorded = read(self.out[name] / "training_config.json")["dse"]
                self.assertEqual(structure["actual_numel"], structure["planned_numel"])
                self.assertEqual({key: structure["actual_numel"][key] for key in ("head", "adapter")}, planned(name))
                self.assertEqual((recorded["parameter_counts"]["head"], recorded["parameter_counts"]["adapter_and_special_embeddings"],
                                  recorded["parameter_counts"]["initial_trainable"]),
                                 tuple(structure["actual_numel"][key] for key in ("head", "adapter", "trainable")))
                self.assertEqual(set(structure["by_leaf"]), set(arm["target_modules"]))
                self.assertEqual(recorded["lora_target_modules"], arm["target_modules"])
                self.assertEqual(structure["lora_tensors"], len(self.initial[name]) - 4)
                self.assertEqual((structure["lora_configuration"]["r"], structure["lora_configuration"]["lora_alpha"],
                                  structure["lora_configuration"]["use_rslora"], structure["scaling_values"]),
                                 (arm["rank"], 2 * arm["rank"], False, [2.0]))
                counts[name] = structure["actual_numel"]["adapter"]
        full = read(self.out["A"] / "training_config.json")["dse"]["parameter_counts"]["adapter_and_special_embeddings"]
        self.assertEqual(counts["POINTER_ALL_R28"] * 64, full * 28)
        self.assertEqual(counts["POOL_ALL"], full)
        self.assertEqual(counts["POINTER_NOFFN"], counts["POOL_NOFFN"])
        self.assertLess(counts["POINTER_NOFFN"], full)

    def test_last_content_checkpoints_carry_the_readout(self):
        out = self.out["POINTER_LAST_CONTENT"]
        for path in (out / "head.pt", out / "checkpoints/step_000002/head.pt"):
            meta = torch.load(path, map_location="cpu")
            self.assertEqual(normalize_head_config(meta["head_config"], head_dim=meta["head_dim"])["readout"], "last_span")
            self.assertEqual(meta["args"]["e458_arm"], "POINTER_LAST_CONTENT")
        self.assertEqual(read(out / "training_config.json")["head_config"], e458_arms.head_config("last_span"))
        self.assertEqual(self.entry["POINTER_LAST_CONTENT"]["head_config"]["readout"], "last_span")
        lengths = [row["option_content_token_lengths"] for row in read(out / "diagnostics/fixed_sample_initial.json")]
        self.assertTrue(any(0 in row for row in lengths) and any(max(row) > 1 for row in lengths))
        self.assertEqual(read(self.out["POOL_ALL"] / "training_config.json")["head_config"]["readout"], "mean_span")

    def test_reloaded_checkpoints_reproduce_the_training_side_logits_for_every_readout(self):
        """The evaluation loader reads the readout from the saved head metadata; nothing is passed or guessed."""
        import kev.evaluate as evaluate
        readouts = {"A": "end", "POINTER_NOFFN": "end", "POINTER_ALL_R28": "end", "POOL_ALL": "mean_span",
                    "POINTER_LAST_CONTENT": "last_span"}
        for name, readout in readouts.items():
            with self.subTest(run=name):
                out = self.out[name]
                with mock.patch.object(model_module, "AutoModelForCausalLM", FakeAuto), \
                        mock.patch.object(evaluate, "load_tokenizer", lambda base, revision=None: FakeTokenizer()):
                    tok, model = evaluate.load(str(out), "cpu")
                self.assertEqual(model.head_config["readout"], readout)
                sample = read(out / "diagnostics/fixed_sample.json")
                recorded = [row for row in read(out / "diagnostics/fixed_sample_final.json") if row["sample_variant"] == "original"]
                position, empty = 0, 0
                with torch.no_grad():
                    for entry in sample["records"]:
                        enc = model.encode(tok, entry["record"], max_state=2048, max_branch=2048, strict=True)
                        self.assertEqual("opt_spans" in enc, readout != "end")
                        for logits in model.forward(enc):
                            row = recorded[position]
                            self.assertTrue(torch.allclose(logits, torch.tensor(row["logits"]), atol=1e-5), (name, position))
                            empty += 0 in row["option_content_token_lengths"]
                            position += 1
                self.assertEqual(position, len(recorded))
                self.assertGreater(empty, 0)                               # questions with an empty option are among them

    def test_plain_runs_of_this_tree_do_not_take_the_reference_path(self):
        for name in ("A", "A_REPEAT", "PLAIN_NOFFN", "PLAIN_R28"):
            self.assertFalse((self.out[name] / "e458_reference").exists())
            args = read(self.out[name] / "training_config.json")["args"]
            self.assertFalse(any(args[key] for key in e458_arms.ARGUMENTS))

    # --- refusals: each stops before the diagnostics snapshot, so before any update ------------------------------

    def refuse(self, label, flags, error, text, shared=COMMON):
        out = self.root / label
        with self.assertRaises(error) as caught:
            train(out, self.data, flags, shared)
        self.assertIn(text, str(caught.exception))
        for member in ("e458_reference/receipt.json", "diagnostics", "campaign", "head.pt"):
            self.assertFalse((out / member).exists(), member)

    def damaged(self, label):
        copy = self.root / (label + "_reference")
        shutil.copytree(self.out["A"], copy)
        return copy

    def test_absent_or_inconsistent_reference_records_stop_the_run(self):
        copy = self.damaged("no_receipt")
        (copy / "campaign/initial_receipt.json").unlink()
        self.refuse("no_receipt", arm_flags("POOL_ALL", copy), ReferenceUnusable, "has no step-zero record")
        copy = self.damaged("no_generator_bytes")
        (copy / "r3_control/model_to_device_complete.cpu.bin").unlink()
        self.refuse("no_generator_bytes", arm_flags("POOL_ALL", copy), ReferenceUnusable, "generator record is unreadable")
        copy = self.damaged("binary_changed")
        path = copy / "diagnostics/adapter_initial.safetensors"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        self.refuse("binary_changed", arm_flags("POINTER_NOFFN", copy), ReferenceUnusable, "not the file its own receipts hashed")
        # A binary whose file digests were made consistent again still fails the per-tensor receipt.
        copy = self.damaged("tensor_changed")
        path = copy / "diagnostics/head_initial.safetensors"
        tensors = load_file(str(path))
        tensors["q.bias"] = tensors["q.bias"] + 1
        save_file(tensors, str(path))
        for name, key in (("diagnostics/initial_capture.json", "initial_head_sha256"),
                          ("r3_control/initial_complete_trainable_tensor_binary_map.json", "full_head_binary_sha256")):
            record = read(copy / name)
            record[key] = digest(path)
            (copy / name).write_text(json.dumps(record))
        self.refuse("tensor_changed", arm_flags("POINTER_ALL_R28", copy), ReferenceUnusable,
                    "head.q.bias is not the tensor its step-zero receipt hashed")

    def test_insufficient_generator_record_is_reported_not_reconstructed(self):
        copy = self.damaged("generator_moved")
        record = read(copy / "campaign/initial_receipt.json")
        record["rng_at_loop_entry"]["torch_cpu_sha256"] = "0" * 64     # as if something drew before the loop
        (copy / "campaign/initial_receipt.json").write_text(json.dumps(record))
        self.refuse("generator_moved", arm_flags("POINTER_NOFFN", copy), ReferenceUnusable,
                    "are not its training-entry state")

    def test_an_arm_that_is_not_the_declared_one_stops_the_run(self):
        reference = self.out["A"]
        self.refuse("wrong_count", arm_flags("POINTER_NOFFN", reference, adapter=planned("POINTER_NOFFN")["adapter"] + 64),
                    ValueError, "are not the planned")
        flags = arm_flags("POINTER_NOFFN", reference)
        flags[flags.index("--lora_targets") + 1] = "all"
        self.refuse("wrong_scope", flags, ValueError, "is not the declared arm")
        flags = arm_flags("POINTER_ALL_R28", reference)
        flags[flags.index("--lora") + 1] = "32"
        self.refuse("wrong_rank", flags, ValueError, "is not the declared arm")
        flags = arm_flags("POINTER_LAST_CONTENT", reference)
        flags[flags.index("--head_config") + 1] = json.dumps(e458_arms.head_config("mean_span"))
        self.refuse("wrong_readout", flags, ValueError, "is not the declared arm")
        flags = arm_flags("POOL_ALL", reference)
        flags[flags.index("--e458_arm") + 1] = "POOL_EVERYTHING"
        self.refuse("unknown_arm", flags, ValueError, "unknown E4a/E5/E8 arm")

    def test_a_reference_of_another_protocol_stops_the_run(self):
        reference = self.out["A"]
        self.refuse("other_rate", arm_flags("POOL_ALL", reference), ReferenceUnusable,
                    "arguments ['lr'] differ from the reference", shared=common(lr="2e-3"))
        self.refuse("other_seed", arm_flags("POOL_ALL", reference), ReferenceUnusable,
                    "arguments ['seed'] differ from the reference", shared=common(seed="45"))
        self.refuse("other_checkpoints", arm_flags("POOL_ALL", reference), ReferenceUnusable,
                    "arguments ['campaign_checkpoint_steps'] differ", shared=common(campaign_checkpoint_steps="3"))
        self.refuse("arm_as_reference", arm_flags("POOL_ALL", self.out["POINTER_LAST_CONTENT"]), ReferenceUnusable,
                    "is itself an E4a/E5/E8 arm")
        self.refuse("scope_as_reference", arm_flags("POOL_ALL", self.out["PLAIN_NOFFN"]), ReferenceUnusable,
                    "reference adapter is not rank 64 on every target module")
        other = self.root / "other.jsonl"
        write_training_file(other, records=28)
        with self.assertRaises(ReferenceUnusable) as caught:
            train(self.root / "other_data", other, arm_flags("POOL_ALL", reference))
        self.assertIn("reference trained on different data bytes", str(caught.exception))

    def test_timeout_is_not_part_of_the_compared_protocol(self):
        out = self.root / "longer_timeout"
        train(out, self.data, arm_flags("POINTER_NOFFN", self.out["A"]) + ["--campaign_timeout_seconds", "3600"])
        receipt = read(out / "e458_reference/receipt.json")
        self.assertEqual(receipt["reference"]["arguments"]["not_compared"]["campaign_timeout_seconds"], [3600.0, 0.0])
        self.assertEqual(receipt["reference"]["arguments"]["different"], ["lora_targets"])
        for name, value in state(out, "final").items():
            self.assertTrue(same(value, state(self.out["POINTER_NOFFN"], "final")[name]), name)

    def test_arm_flags_go_together(self):
        for flags in (["--e458_arm", "POOL_ALL"], ["--e458_reference_run", str(self.out["A"])],
                      ["--e458_expected_adapter_numel", "8"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                train(self.root / "rejected", self.data, structure_flags(64, "all", "mean_span") + flags)
        self.assertFalse((self.root / "rejected").exists())


class ArmTableTest(unittest.TestCase):
    def test_five_arms_each_one_declared_change_set(self):
        self.assertEqual(list(e458_arms.ARMS), ["POINTER_NOFFN", "POOL_ALL", "POOL_NOFFN", "POINTER_ALL_R28",
                                               "POINTER_LAST_CONTENT"])
        rows = {name: e458_arms.arm(name) for name in e458_arms.ARMS}
        self.assertEqual({name: sorted(row["argument_changes"]) for name, row in rows.items()},
                         {"POINTER_NOFFN": ["lora_targets"], "POOL_ALL": ["head_config"],
                          "POOL_NOFFN": ["head_config", "lora_targets"], "POINTER_ALL_R28": ["lora"],
                          "POINTER_LAST_CONTENT": ["head_config"]})
        self.assertEqual({name: (row["rank"], row["alpha"], row["targets"], row["readout"], row["adapter"])
                          for name, row in rows.items()},
                         {"POINTER_NOFFN": (64, 128, "attn", "end", "reference"),
                          "POOL_ALL": (64, 128, "all", "mean_span", "reference"),
                          "POOL_NOFFN": (64, 128, "attn", "mean_span", "reference"),
                          "POINTER_ALL_R28": (28, 56, "all", "end", "ordinary"),
                          "POINTER_LAST_CONTENT": (64, 128, "all", "last_span", "reference")})
        self.assertEqual({row["structure"] for row in rows.values()}, {"all/r64", "attn/r64", "all/r28"})
        self.assertEqual([name for name, row in rows.items() if row["same_structure_as_reference"]],
                         ["POOL_ALL", "POINTER_LAST_CONTENT"])
        self.assertEqual(rows["POINTER_NOFFN"]["target_modules"],
                         ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"])
        self.assertFalse(set(e458_arms.MLP) & set(rows["POOL_NOFFN"]["target_modules"]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
