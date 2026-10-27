"""CPU tests for the campaign diagnostics and the modified trainer loop.

Run in the training environment with CUDA_VISIBLE_DEVICES='' and PYTHONPATH at
this source tree. A tiny synthetic backbone stands in for NeoHorse; the real
kev.train loop, heads, supplement diagnostics, R3 capture and campaign
diagnostics run as they are. No backbone download or GPU.
"""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import load_file, save_file

from kev import heads
from kev import model as model_module
from kev.campaign_diagnostics import CampaignHeadObserver
from kev.supplement_diagnostics import HeadObserver, rng_equal, rng_state

POINTER = dict(version=1, kind="pointer", dim=256, readout="end", rng_anchor_dim=256)
RESIDUAL = dict(version=1, kind="residual_pair_mlp", dim=256, readout="end", rng_anchor_dim=256, mlp_hidden=128)
HIDDEN = 16


def observed_head(config, seed=7, trained_branch=False):
    torch.manual_seed(seed)
    head = heads.make_head(HIDDEN, head_config=config)
    if trained_branch:
        torch.manual_seed(seed + 1)
        with torch.no_grad():
            head.pair_mlp[-1].weight.normal_(0, .05)
            head.pair_mlp[-1].bias.fill_(.01)
    return head


class CampaignObserverTest(unittest.TestCase):
    def test_hooks_leave_outputs_gradients_and_rng_unchanged(self):
        for config, trained in ((POINTER, False), (RESIDUAL, False), (RESIDUAL, True)):
            with self.subTest(kind=config["kind"], trained=trained):
                plain = observed_head(config, trained_branch=trained)
                watched = observed_head(config, trained_branch=trained)
                observer = CampaignHeadObserver(watched)
                watched._diagnostic_observer = observer
                torch.manual_seed(123)
                query, options = torch.randn(HIDDEN), torch.randn(5, HIDDEN)
                q1, k1 = query.clone().requires_grad_(), options.clone().requires_grad_()
                q2, k2 = query.clone().requires_grad_(), options.clone().requires_grad_()
                state = rng_state()
                out1, out2 = plain(q1, k1), watched(q2, k2)
                self.assertTrue(torch.equal(out1, out2))
                weight = torch.arange(1., 6.)
                (out1 * weight).sum().backward()
                (out2 * weight).sum().backward()
                self.assertTrue(rng_equal(state, rng_state()))
                self.assertTrue(torch.equal(q1.grad, q2.grad))
                self.assertTrue(torch.equal(k1.grad, k2.grad))
                for name, value in plain.named_parameters():
                    self.assertTrue(torch.equal(value.grad, dict(watched.named_parameters())[name].grad), name)
                paths = observer.gradient_paths()
                self.assertEqual(paths["questions"], 1)
                self.assertEqual(observer.pending, [])
                for group in ("option_projection", "decision_projection"):
                    self.assertLessEqual(paths[group]["closure_sq"], 1e-9 * max(paths[group]["total_sq"], 1e-30))

    def test_split_matches_a_reference_backward_with_the_branch_input_detached(self):
        head = observed_head(RESIDUAL, trained_branch=True)
        observer = CampaignHeadObserver(head)
        head._diagnostic_observer = observer
        torch.manual_seed(5)
        h_decide, h_opts, weight = torch.randn(HIDDEN), torch.randn(6, HIDDEN), torch.randn(6)
        (head(h_decide, h_opts) * weight).sum().backward()
        paths = observer.gradient_paths()
        params = [head.q.weight, head.q.bias, head.k.weight, head.k.bias]
        total = [p.grad.clone().double() for p in params]
        head.zero_grad()
        k, q = head.k(h_opts), head.q(h_decide)
        kd, queries = k.detach(), q.detach().expand_as(k)
        features = torch.cat((queries, kd, queries * kd, (queries - kd).abs()), dim=-1)
        reference = (k @ q) * head.scale + head.pair_mlp(features).squeeze(-1)
        (reference * weight).sum().backward()
        residual = [t - p.grad.double() for t, p in zip(total, params)]
        residual_sq = float(sum(r.square().sum() for r in residual))
        dot = float(sum((t * r).sum() for t, r in zip(total, residual)))
        self.assertGreater(residual_sq, 0)
        self.assertAlmostEqual(paths["qk_parameters"]["residual_sq"] / residual_sq, 1, places=4)
        self.assertAlmostEqual(paths["qk_parameters"]["total_dot_residual"] / dot, 1, places=4)
        for group in ("option_projection", "decision_projection"):
            row = paths[group]
            rebuilt = row["base_sq"] + row["residual_sq"] + 2 * row["base_dot_residual"]
            self.assertAlmostEqual(rebuilt / row["total_sq"], 1, places=5)

    def test_frozen_zero_branch_sends_exactly_zero_gradient(self):
        head = observed_head(RESIDUAL)
        for parameter in head.pair_mlp.parameters():
            parameter.requires_grad_(False)
        observer = CampaignHeadObserver(head)
        head._diagnostic_observer = observer
        torch.manual_seed(9)
        logits = head(torch.randn(HIDDEN), torch.randn(4, HIDDEN))
        logits.square().sum().backward()
        paths = observer.gradient_paths()
        for group in ("option_projection", "decision_projection", "hidden_state", "qk_parameters"):
            self.assertEqual(paths[group]["residual_sq"], 0.0)
        summary = observer.update_summary()
        self.assertEqual(summary["residual_logits"]["max_abs"], 0.0)
        shape = observer.score_shape_summary()
        self.assertEqual((shape["full_base_argmax_flips"], shape["full_base_kl_sum"]), (0, 0.0))

    def test_score_magnitude_keeps_the_supplement_format_and_covers_pointer(self):
        head = observed_head(RESIDUAL, trained_branch=True)
        original, campaign = HeadObserver(), CampaignHeadObserver(head)
        torch.manual_seed(3)
        inputs = [(torch.randn(HIDDEN), torch.randn(n, HIDDEN)) for n in (2, 5, 3)]
        with torch.no_grad():
            for observer in (original, campaign):
                head._diagnostic_observer = observer
                for query, options in inputs:
                    head(query, options)
        self.assertEqual(original.update_summary(), campaign.update_summary())
        pointer = observed_head(POINTER)
        observer = CampaignHeadObserver(pointer)
        pointer._diagnostic_observer = observer
        with torch.no_grad():
            logits = pointer(*inputs[1])
        summary = observer.update_summary()
        self.assertEqual(set(summary), {"base_logits"})
        self.assertAlmostEqual(summary["base_logits"]["sum_sq"], float(logits.double().square().sum()))
        self.assertEqual(observer.score_shape_summary()["options"], 5)

    def test_pointer_and_residual_share_projection_initialization_and_rng(self):
        made = {}
        for label, config in (("pointer", POINTER), ("residual", RESIDUAL), ("residual_again", RESIDUAL)):
            torch.manual_seed(44)
            made[label] = (heads.make_head(HIDDEN, head_config=config), torch.get_rng_state().clone())
        for label in ("residual", "residual_again"):
            self.assertTrue(torch.equal(made["pointer"][1], made[label][1]))
            for name in ("q.weight", "q.bias", "k.weight", "k.bias"):
                self.assertTrue(torch.equal(made["pointer"][0].state_dict()[name], made[label][0].state_dict()[name]))
        for name, value in made["residual"][0].pair_mlp.state_dict().items():
            self.assertTrue(torch.equal(value, made["residual_again"][0].pair_mlp.state_dict()[name]))
        self.assertEqual(torch.count_nonzero(made["residual"][0].pair_mlp[-1].weight).item(), 0)


# --- the real kev.train loop on a synthetic backbone -------------------------------------------------------

SPECIAL_IDS = {token: i + 1 for i, token in enumerate(model_module.SPECIAL)}


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[10 + (byte % 53) for byte in text.encode()])

    def convert_tokens_to_ids(self, token):
        return SPECIAL_IDS[token]

    def save_pretrained(self, path):
        Path(path, "tokenizer_config.json").write_text("{}\n")


class LoraLinear(torch.nn.Module):
    def __init__(self, d, rank, dropout):
        super().__init__()
        self.base = torch.nn.Linear(d, d)
        self.lora_dropout = torch.nn.Dropout(dropout)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(d, rank, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(rank, d, bias=False)})
        self.scaling = {"default": 2.0}
        torch.nn.init.zeros_(self.lora_B["default"].weight)

    def forward(self, x):
        update = self.lora_B["default"](self.lora_A["default"](self.lora_dropout(x)))
        return self.base(x) + self.scaling["default"] * update


class TinyBackbone(torch.nn.Module):
    def __init__(self, d, rank, dropout):
        super().__init__()
        self.embed = torch.nn.Embedding(64, d)
        self.layers = torch.nn.ModuleList([LoraLinear(d, rank, dropout) for _ in range(2)])
        self.config = types.SimpleNamespace(use_cache=True, hidden_size=d)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_("lora_" in name)

    def forward(self, ids):
        h = self.embed(ids)
        for layer in self.layers:
            causal_mean = h.cumsum(0) / torch.arange(1, len(h) + 1, device=h.device)[:, None]
            h = h + torch.tanh(layer(h + causal_mean))
        return h

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        save_file({name: value.detach().cpu().contiguous() for name, value in self.named_parameters()
                   if "lora_" in name}, str(Path(path) / "adapter_model.safetensors"))
        Path(path, "adapter_config.json").write_text('{"synthetic": true}\n')


class TinyDecisionModel(torch.nn.Module):
    """Minimal DecisionModel interface using the production head constructor."""
    def __init__(self, name, tok, device, lora=None, revision=None, attn=None, head_dim=256, option_isolation=False,
                 special_embeddings=False, lora_targets="all", dtype=torch.float32, head_dropout=0.0,
                 lora_dropout=0.05, head_config=None):
        super().__init__()
        self.head_config = model_module.normalize_head_config(head_config, head_dim=head_dim)
        self.lm = TinyBackbone(HIDDEN, lora, lora_dropout)
        self.hybrid = self.option_isolation = False
        self.lora_target_modules = ["synthetic"]
        self.head = model_module.make_head(HIDDEN, head_dim=head_dim, dropout=head_dropout,
                                           head_config=self.head_config)
        self.device = device
        self.to(device)

    def encode(self, tok, rec, **kw):
        return model_module.encode(tok, rec, option_isolation=False, readout=self.head_config["readout"], **kw)

    def forward(self, enc):
        h = self.lm(torch.tensor(enc["ids"], device=self.device))
        return [self.head(h[decide], h[torch.tensor(options, device=h.device)])
                for decide, options in zip(enc["decide_idx"], enc["opt_idx"])]

    def forward_batch(self, encs):
        return [self.forward(enc) for enc in encs]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def write_training_file(path, records=24):
    rows = []
    for i in range(records):
        if i % 3 == 0:
            keys = ["alpha", "beta", "gamma", "delta"]
            question = {"type": "choice", "instructions": f"pick the label of item {i}",
                        "criteria": {"alpha": "the first kind", "beta": None, "gamma": "a third kind", "delta": "last"},
                        "label": keys[i % 4]}
        elif i % 3 == 1:
            question = {"type": "noul", "instructions": f"is item {i} even?", "label": i % 2 == 0}
        else:
            question = {"type": "score", "instructions": f"rate item {i}", "criteria": ["low", "middle", "high"],
                        "label": i % 3}
        rows.append({"state": f"record {i}: synthetic text {'x' * (i % 5)}", "questions": {"main": question}})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


COMMON = ["--base", "synthetic", "--epochs", "1", "--lr", "1e-3", "--head_lr", "2e-3", "--weight_decay", "0.01",
          "--lora", "2", "--accum", "4", "--device", "cpu", "--batch", "1", "--dtype", "fp32",
          "--weights_dtype", "fp32", "--checkpointing", "0", "--p_none_pair", "0.25", "--seed", "44",
          "--lora_dropout", "0.05", "--supplement_diagnostics", "1", "--r3_control_capture", "1"]
ARMS = {
    "pointer": ["--head_config", json.dumps(POINTER), "--r3_sham_residual", "0",
                "--campaign_diagnostics", "1", "--campaign_checkpoint_steps", "2"],
    "pointer_repeat": ["--head_config", json.dumps(POINTER), "--r3_sham_residual", "0",
                       "--campaign_diagnostics", "1", "--campaign_checkpoint_steps", "2"],
    "pointer_without_campaign": ["--head_config", json.dumps(POINTER), "--r3_sham_residual", "0"],
    "frozen_zero": ["--head_config", json.dumps(RESIDUAL), "--r3_sham_residual", "1",
                    "--campaign_diagnostics", "1", "--campaign_checkpoint_steps", "2"],
    "trainable": ["--head_config", json.dumps(RESIDUAL), "--r3_sham_residual", "0",
                  "--campaign_diagnostics", "1", "--campaign_checkpoint_steps", "2"],
}


def run_trainer(out, data, extra):
    import kev.train as train_module
    argv = ["kev.train", *COMMON, "--data", str(data), "--out", str(out), *extra]
    with mock.patch.object(train_module, "DecisionModel", TinyDecisionModel), \
            mock.patch.object(train_module, "load_tokenizer", lambda name, revision=None: FakeTokenizer()), \
            mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
        train_module.main()


def final_state(out):
    return {**load_file(str(out / "diagnostics/head_final.safetensors")),
            **load_file(str(out / "diagnostics/adapter_final.safetensors"))}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def read(path):
    return json.loads(path.read_text())


class TrainerLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path(tempfile.mkdtemp(prefix="jev-campaign-test-"))
        cls.data = cls.root / "train.jsonl"
        write_training_file(cls.data)
        cls.out = {}
        for arm, extra in ARMS.items():
            cls.out[arm] = cls.root / arm
            run_trainer(cls.out[arm], cls.data, extra)
        cls.steps = read(cls.out["pointer"] / "training_metrics.json")["optimizer_steps"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def same_tensors(self, left, right, names=None):
        for name in (names or left):
            self.assertTrue(torch.equal(left[name], right[name]), name)

    def test_campaign_records_do_not_change_the_training_computation(self):
        with_campaign, without = self.out["pointer"], self.out["pointer_without_campaign"]
        self.assertEqual(self.steps, 6)
        a, b = final_state(with_campaign), final_state(without)
        self.assertEqual(set(a), set(b))
        self.same_tensors(a, b)
        left = rows(with_campaign / "diagnostics/optimizer_updates.jsonl")
        right = rows(without / "diagnostics/optimizer_updates.jsonl")
        self.assertEqual([r["preclip_global_l2"] for r in left], [r["preclip_global_l2"] for r in right])
        self.assertTrue(any(r["preclip_global_l2"] != left[0]["preclip_global_l2"] for r in left))
        self.assertEqual(set(left[0]["h4_actual_same_forward"]), {"base_logits"})
        self.assertEqual(right[0]["h4_actual_same_forward"], {})

    def test_same_seed_pointer_repeat_has_identical_receipts_and_traces(self):
        first, second = self.out["pointer"], self.out["pointer_repeat"]
        a, b = read(first / "campaign/initial_receipt.json"), read(second / "campaign/initial_receipt.json")
        for key in ("tensor_sha256", "tensor_groups", "rng_at_loop_entry", "head_parameters", "optimizer"):
            self.assertEqual(a[key], b[key], key)
        done_a, done_b = read(first / "campaign/completion.json"), read(second / "campaign/completion.json")
        for key in ("input_stream_sha256", "torch_rng_trace_sha256"):
            self.assertEqual(done_a[key], done_b[key], key)
        self.assertTrue(done_a["unseeded_python_numpy_rng_unchanged"])
        trace = [r["torch_rng_sha256"] for r in rows(first / "campaign/updates.jsonl")]
        self.assertEqual(len(set(trace)), self.steps)   # dropout draws advance the seeded generator each update
        self.same_tensors(final_state(first), final_state(second))
        order = "r3_control/epoch_0_actual_original_data_order.json"
        self.assertEqual((first / order).read_bytes(), (second / order).read_bytes())

    def test_frozen_zero_arm_matches_pointer_on_every_shared_tensor(self):
        pointer, frozen = self.out["pointer"], self.out["frozen_zero"]
        a, b = final_state(pointer), final_state(frozen)
        self.same_tensors(a, b, names=list(a))
        receipt_p, receipt_f = read(pointer / "campaign/initial_receipt.json"), read(frozen / "campaign/initial_receipt.json")
        for key in ("adapter", "head_qk"):
            self.assertEqual(receipt_p["tensor_groups"][key], receipt_f["tensor_groups"][key])
        self.assertIsNone(receipt_p["tensor_groups"]["pair_mlp"])
        self.assertEqual(receipt_p["rng_at_loop_entry"], receipt_f["rng_at_loop_entry"])
        self.assertEqual(receipt_p["optimizer"], receipt_f["optimizer"])
        branch = [row for row in receipt_f["head_parameters"] if row["name"].startswith("pair_mlp.")]
        self.assertEqual(len(branch), 4)
        self.assertTrue(all(not row["requires_grad"] and not row["in_optimizer"] for row in branch))
        for row in rows(frozen / "campaign/updates.jsonl"):
            path = row["gradient_path"]
            for group in ("option_projection", "decision_projection", "hidden_state", "qk_parameters"):
                self.assertEqual(path[group]["residual_sq"], 0.0)
            self.assertEqual(row["score_magnitude"]["residual_logits"]["max_abs"], 0.0)
            self.assertEqual(row["score_shape"]["full_base_argmax_flips"], 0)
            self.assertIsNone(row["clip"]["group_l2"]["pair_mlp"])
            self.assertEqual(row["clip"]["coefficient_over_shared_only"], 1.0)
        done_p, done_f = read(pointer / "campaign/completion.json"), read(frozen / "campaign/completion.json")
        for key in ("input_stream_sha256", "torch_rng_trace_sha256"):
            self.assertEqual(done_p[key], done_f[key], key)
        sham = read(frozen / "r3_control/completion.json")
        self.assertTrue(sham["sham"] and sham["final_frozen_zero_verified"])

    def test_trainable_residual_arm_records_a_real_branch(self):
        pointer, frozen, trainable = self.out["pointer"], self.out["frozen_zero"], self.out["trainable"]
        receipt_p = read(pointer / "campaign/initial_receipt.json")
        receipt_f = read(frozen / "campaign/initial_receipt.json")
        receipt_t = read(trainable / "campaign/initial_receipt.json")
        self.assertEqual(receipt_f["tensor_groups"], receipt_t["tensor_groups"])
        self.assertEqual(receipt_p["rng_at_loop_entry"], receipt_t["rng_at_loop_entry"])
        head_group = receipt_t["optimizer"]["groups"][1]
        self.assertEqual(head_group["names"][:4], receipt_p["optimizer"]["groups"][1]["names"])
        self.assertEqual(len(head_group["names"]), 8)
        updates = rows(trainable / "campaign/updates.jsonl")
        self.assertEqual([r["optimizer_update"] for r in updates], list(range(1, self.steps + 1)))
        self.assertEqual(updates[0]["score_magnitude"]["residual_logits"]["max_abs"], 0.0)
        self.assertGreater(updates[-1]["score_magnitude"]["residual_logits"]["max_abs"], 0.0)
        self.assertGreater(updates[-1]["gradient_path"]["qk_parameters"]["residual_sq"], 0.0)
        for row in updates:
            clip, path = row["clip"], row["gradient_path"]
            self.assertGreater(clip["group_l2"]["pair_mlp"], 0.0)
            self.assertAlmostEqual(clip["recomputed_global_l2"] / clip["preclip_global_l2"], 1, places=5)
            self.assertAlmostEqual(clip["coefficient"], min(1.0, 1 / (clip["preclip_global_l2"] + 1e-6)), places=6)
            self.assertLessEqual(clip["coefficient_over_shared_only"], 1.0)
            for group in ("option_projection", "decision_projection"):
                self.assertLessEqual(path[group]["closure_sq"], 1e-8 * path[group]["total_sq"])
            self.assertEqual(row["objective"]["source_records"], 4)
            self.assertGreaterEqual(row["objective"]["records"], 4)
        done_p, done_t = read(pointer / "campaign/completion.json"), read(trainable / "campaign/completion.json")
        for key in ("input_stream_sha256", "torch_rng_trace_sha256"):
            self.assertEqual(done_p[key], done_t[key], key)
        self.assertFalse(all(torch.equal(v, final_state(trainable)[n]) for n, v in final_state(pointer).items()))

    def test_intermediate_checkpoint_uses_the_final_layout(self):
        out = self.out["trainable"]
        directory = out / "checkpoints/step_000002"
        self.assertEqual(sorted(p.name for p in directory.iterdir()),
                         ["adapter_config.json", "adapter_model.safetensors", "head.pt"])
        meta, final = torch.load(directory / "head.pt", map_location="cpu"), torch.load(out / "head.pt", map_location="cpu")
        self.assertEqual(set(final) | {"planned_optimizer_steps", "intermediate_checkpoint"}, set(meta))
        self.assertEqual((meta["optimizer_steps"], meta["stopped_early"], final["optimizer_steps"]), (2, True, self.steps))
        self.assertFalse(torch.equal(meta["head"]["q.weight"], final["head"]["q.weight"]))
        done = read(out / "campaign/completion.json")
        self.assertEqual([c["optimizer_update"] for c in done["checkpoints"]], [2])
        self.assertTrue((out / "campaign" / done["checkpoints"][0]["fixed_sample"]).is_file())
        self.assertEqual(done["optimizer_updates"], self.steps)

    def test_optional_timeout_stops_the_run(self):
        with self.assertRaises(TimeoutError):
            run_trainer(self.root / "timeout", self.data, ARMS["pointer"] + ["--campaign_timeout_seconds", "1e-9"])
        self.assertFalse((self.root / "timeout/head.pt").exists())

    def test_campaign_records_require_the_r3_capture(self):
        extra = ["--head_config", json.dumps(POINTER), "--campaign_diagnostics", "1", "--r3_control_capture", "0"]
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            run_trainer(self.root / "rejected", self.data, extra)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
