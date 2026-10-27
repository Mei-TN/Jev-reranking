"""Small CPU tests; no pretrained model, network, or pytest dependency."""
import io
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "trees" / "head1_v1"))
import torch
from torch import nn
from torch.nn import functional as F
from kev.heads import PointerHead, make_head, normalize_head_config


def configs():
    base = {"version": 1, "kind": "pointer", "dim": 256, "readout": "end", "rng_anchor_dim": 256}
    return [base, {**base, "dim": 64}, {**base, "kind": "tanh_pointer"},
            {**base, "kind": "cosine_pointer", "cosine_scale": 16},
            {**base, "kind": "residual_pair_mlp", "mlp_hidden": 128},
            {**base, "readout": "mean_span"}]


class OriginalPointer(nn.Module):
    """Independent reference implementation for compatibility tests."""
    def __init__(self, d, dp=256, dropout=0.0):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, options):
        if self.dropout.p > 0:
            return (self.dropout(self.k(options)) @ self.dropout(self.q(query))) * self.scale
        return (self.k(options) @ self.q(query)) * self.scale


class HeadTests(unittest.TestCase):
    def test_default_checkpoint_keys_outputs_gradients_and_rng_are_legacy(self):
        for width in (64, 256):
            for dropout in (0.0, 0.2):
                with self.subTest(width=width, dropout=dropout):
                    torch.manual_seed(71)
                    original = OriginalPointer(9, width, dropout)
                    original_rng = torch.get_rng_state().clone()
                    torch.manual_seed(71)
                    actual = make_head(9, head_dim=width, dropout=dropout)
                    self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
                    self.assertEqual(set(actual.state_dict()), {"q.weight", "q.bias", "k.weight", "k.bias"})
                    for key, value in original.state_dict().items():
                        self.assertTrue(torch.equal(value, actual.state_dict()[key]))
                    q, k = torch.randn(9), torch.randn(5, 9)
                    torch.manual_seed(102)
                    wanted = original(q, k)
                    wanted.sum().backward()
                    wanted_rng = torch.get_rng_state().clone()
                    torch.manual_seed(102)
                    got = actual(q, k)
                    got.sum().backward()
                    self.assertTrue(torch.equal(wanted, got))
                    self.assertTrue(torch.equal(wanted_rng, torch.get_rng_state()))
                    for (name, p), (other, op) in zip(original.named_parameters(), actual.named_parameters()):
                        self.assertEqual(name, other)
                        self.assertTrue(torch.equal(p.grad, op.grad))

    def test_campaign_anchor_shares_weights_and_preserves_following_rng(self):
        torch.manual_seed(111)
        legacy = OriginalPointer(11)
        expected_rng = torch.get_rng_state().clone()
        next_draw = torch.randn(23)
        for cfg in configs():
            with self.subTest(config=cfg):
                torch.manual_seed(111)
                head = make_head(11, head_config=cfg)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                self.assertTrue(torch.equal(next_draw, torch.randn(23)))
                for name in ("q", "k"):
                    self.assertTrue(torch.equal(getattr(head, name).weight, getattr(legacy, name).weight[:cfg["dim"]]))
                    self.assertTrue(torch.equal(getattr(head, name).bias, getattr(legacy, name).bias[:cfg["dim"]]))

    def test_variable_candidates_and_fixed_feature_permutation(self):
        torch.manual_seed(14)
        query = torch.randn(7)
        for cfg in configs():
            head = make_head(7, head_config=cfg).eval()
            for count in (1, 2, 5, 9):
                with self.subTest(config=cfg, count=count):
                    options = torch.randn(count, 7)
                    logits = head(query, options)
                    self.assertEqual(logits.shape, (count,))
                    self.assertTrue(bool(torch.isfinite(logits).all()))
                    order = torch.randperm(count)
                    torch.testing.assert_close(head(query, options[order]), logits[order], rtol=1e-5, atol=1e-6)

    def test_tanh_and_cosine_formulas(self):
        torch.manual_seed(42)
        q, k = torch.randn(7), torch.randn(4, 7)
        tanh = make_head(7, head_config=configs()[2])
        torch.testing.assert_close(tanh(q, k), (tanh.k(k).tanh() @ tanh.q(q).tanh()) / 16)
        cosine = make_head(7, head_config=configs()[3])
        expected = (F.normalize(cosine.k(k), dim=-1) @ F.normalize(cosine.q(q), dim=-1)) * 16
        torch.testing.assert_close(cosine(q, k), expected)
        self.assertLessEqual(float(cosine(q, k).abs().max()), 16.00001)
        self.assertEqual(set(cosine.state_dict()), {"q.weight", "q.bias", "k.weight", "k.bias"})
        with torch.no_grad():
            for parameter in cosine.parameters():
                parameter.zero_()
        self.assertTrue(torch.equal(cosine(q, k), torch.zeros(4)))

    def test_cosine_keeps_outer_autocast_dtype_and_finite_gradient(self):
        head = make_head(7, head_config=configs()[3])
        q, k = torch.randn(7), torch.randn(3, 7)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = head(q, k)
            loss = F.cross_entropy(logits[None], torch.tensor([1]))
        self.assertEqual(logits.dtype, torch.bfloat16)
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()))

    def test_residual_is_initially_exact_baseline_then_learns(self):
        torch.manual_seed(24)
        baseline = make_head(8, head_config=configs()[0])
        torch.manual_seed(24)
        head = make_head(8, head_config=configs()[4])
        q, k = torch.randn(8), torch.randn(5, 8)
        self.assertTrue(torch.equal(head(q, k), baseline(q, k)))
        self.assertTrue(torch.count_nonzero(head.pair_mlp[-1].weight).item() == 0)
        self.assertTrue(torch.count_nonzero(head.pair_mlp[-1].bias).item() == 0)
        optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
        F.cross_entropy(head(q, k)[None], torch.tensor([2])).backward()
        self.assertGreater(float(head.pair_mlp[-1].weight.grad.abs().sum()), 0)
        self.assertEqual(float(head.pair_mlp[0].weight.grad.abs().sum()), 0)
        optimizer.step()
        optimizer.zero_grad()
        F.cross_entropy(head(q, k)[None], torch.tensor([2])).backward()
        self.assertGreater(float(head.pair_mlp[0].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()))
        grouped_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(grouped_ids, {id(p) for p in head.parameters()})

    def test_checkpoint_schema_and_strict_weight_roundtrip(self):
        query, options = torch.randn(6), torch.randn(3, 6)
        for cfg in configs():
            with self.subTest(config=cfg):
                head = make_head(6, head_config=cfg).eval()
                buffer = io.BytesIO()
                torch.save({"head": head.state_dict(), "head_config": head.head_config}, buffer)
                buffer.seek(0)
                loaded = torch.load(buffer, map_location="cpu", weights_only=True)
                restored = make_head(6, head_config=loaded["head_config"]).eval()
                restored.load_state_dict(loaded["head"], strict=True)
                self.assertEqual(restored.head_config, normalize_head_config(cfg))
                self.assertTrue(torch.equal(head(query, options), restored(query, options)))
        wrong = make_head(6, head_config=configs()[4])
        with self.assertRaises(RuntimeError):
            wrong.load_state_dict(PointerHead(6).state_dict(), strict=True)

    def test_reject_unknown_or_ambiguous_schema(self):
        base = configs()[0]
        invalid = [{}, [], {**base, "unknown": 1}, {**base, "version": True}, {**base, "version": 2},
                   {**base, "kind": "mlp"}, {**base, "dim": True}, {**base, "dim": 128},
                   {**base, "readout": "mean"}, {**base, "rng_anchor_dim": 64},
                   {**base, "cosine_scale": 16}, {**configs()[3], "cosine_scale": float("nan")},
                   {**configs()[3], "cosine_scale": 8}, {**configs()[4], "mlp_hidden": 64}]
        for cfg in invalid:
            with self.subTest(config=cfg), self.assertRaises(ValueError):
                normalize_head_config(cfg)
        self.assertEqual(normalize_head_config(None, head_dim=37),
                         {"version": 1, "kind": "pointer", "dim": 37, "readout": "end"})


if __name__ == "__main__":
    unittest.main()
