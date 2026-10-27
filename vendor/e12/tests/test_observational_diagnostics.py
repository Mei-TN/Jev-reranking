"""CPU tests of behavioral parity, RNG preservation and actual intermediates.

Run in the existing training environment with CUDA_VISIBLE_DEVICES='' and
PYTHONPATH pointing at this supplement tree. No backbone download or GPU.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from kev import heads
from kev.supplement_diagnostics import (HeadObserver, TrainingDiagnostics, capture_sample,
                                       rng_state, rng_equal, update_norms)


def reference_module():
    path = Path(__file__).with_name("heads_original_reference.py")
    spec = importlib.util.spec_from_file_location("unchanged_head1_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ObservationalDiagnosticsTest(unittest.TestCase):
    def test_original_output_gradients_and_rng_with_observer(self):
        original = reference_module()
        for kind, readout in (("pointer", "end"), ("residual_pair_mlp", "end"), ("pointer", "mean_span")):
            with self.subTest(kind=kind, readout=readout):
                config = dict(version=1, kind=kind, dim=256, readout=readout, rng_anchor_dim=256)
                if kind == "residual_pair_mlp":
                    config["mlp_hidden"] = 128
                torch.manual_seed(45)
                before = original.make_head(16, head_config=config)
                original_rng = torch.get_rng_state().clone()
                torch.manual_seed(45)
                after = heads.make_head(16, head_config=config)
                self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
                for name, value in before.state_dict().items():
                    self.assertTrue(torch.equal(value, after.state_dict()[name]))
                torch.manual_seed(123)
                query = torch.randn(16)
                options = torch.randn(5, 16)
                q1, k1 = query.clone().requires_grad_(), options.clone().requires_grad_()
                q2, k2 = query.clone().requires_grad_(), options.clone().requires_grad_()
                observer = HeadObserver()
                observer.sample_rows = []
                after._diagnostic_observer = observer
                state = rng_state()
                out1, out2 = before(q1, k1), after(q2, k2)
                self.assertTrue(torch.equal(out1, out2))
                self.assertTrue(rng_equal(state, rng_state()))
                weight = torch.arange(1, 6)
                (out1*weight).sum().backward()
                (out2*weight).sum().backward()
                self.assertTrue(torch.equal(q1.grad, q2.grad))
                self.assertTrue(torch.equal(k1.grad, k2.grad))
                for name, value in before.named_parameters():
                    self.assertTrue(torch.equal(value.grad, dict(after.named_parameters())[name].grad))
                self.assertEqual(len(observer.sample_rows), 1)
                self.assertEqual(observer.sample_rows[0]["logits"], out2.detach().tolist())

    def test_h4_initial_zero_residual_and_actual_training_observation(self):
        config = dict(version=1, kind="residual_pair_mlp", dim=256,
                      readout="end", rng_anchor_dim=256, mlp_hidden=128)
        torch.manual_seed(44)
        head = heads.make_head(16, head_config=config)
        observer = HeadObserver()
        observer.sample_rows = []
        head._diagnostic_observer = observer
        query, options = torch.randn(16), torch.randn(4, 16)
        output = head(query, options)
        row = observer.sample_rows[0]
        self.assertEqual(row["residual_logits"], [0.0]*4)
        self.assertEqual(row["base_logits"], output.detach().tolist())
        opt = torch.optim.AdamW(head.parameters(), lr=2.5e-5)
        output.square().sum().backward()
        opt.step()
        observer.sample_rows = []
        final = head(query, options)
        row = observer.sample_rows[0]
        base = torch.tensor(row["base_logits"])
        residual = torch.tensor(row["residual_logits"])
        self.assertTrue(torch.equal(base+residual, final.detach()))
        self.assertTrue(bool(residual.abs().sum() > 0))
        observer.sample_rows = None
        head(query, options)
        aggregate = observer.update_summary()
        self.assertEqual(aggregate["base_logits"]["numel"], 4)
        self.assertAlmostEqual(aggregate["residual_logits"]["sum_sq"], float(residual.double().square().sum()))
        self.assertEqual(observer.update_summary(), {})

    def test_effective_adapter_delta_norm_matches_dense_construction(self):
        torch.manual_seed(7)
        initial = {"layer.lora_A.default.weight": torch.randn(3, 5),
                   "layer.lora_B.default.weight": torch.randn(7, 3),
                   "head.q.weight": torch.randn(2, 5)}
        final = {name: value + .1*torch.randn_like(value) for name, value in initial.items()}
        result = update_norms(initial, final, {"layer": 2.0})
        dense = 2*(final["layer.lora_B.default.weight"].double() @ final["layer.lora_A.default.weight"].double()
                   - initial["layer.lora_B.default.weight"].double() @ initial["layer.lora_A.default.weight"].double())
        self.assertAlmostEqual(result["adapter_effective_update_l2"], float(dense.norm()), places=11)

    def test_fixed_sample_restores_modes_without_drawing_rng(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = heads.PointerHead(4)
                self.query, self.options = torch.randn(4), torch.randn(3, 4)
            def encode(self, tokenizer, record, **kwargs):
                return {"ids": [1, 2, 3], "questions": record["questions"],
                        "opt_spans": [[(0, 1), (1, 2), (2, 3)] for _ in record["questions"]]}
            def forward(self, enc):
                return [self.head(self.query, self.options) for _ in enc["questions"]]
        model = TinyModel()
        model.train()
        model.head.eval()
        observer = HeadObserver()
        model.head._diagnostic_observer = observer
        sample = {"records": [{"record_id": "fixed", "record": {
            "state": "fixed", "questions": [{"qtype": "choice", "label": 0,
                                                "options": ["a", "b", "c"]}]}}]}
        before = rng_state()
        import contextlib
        rows = capture_sample(model, None, sample, observer, contextlib.nullcontext())
        self.assertTrue(rng_equal(before, rng_state()))
        self.assertTrue(model.training)
        self.assertFalse(model.head.training)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["new_to_original_option"], [2, 1, 0])
        self.assertEqual(rows[1]["question"]["label"], 2)
        self.assertIsNone(observer.sample_rows)

    def test_step_logging_does_not_mutate_optimizer_or_rng(self):
        parameter = torch.nn.Parameter(torch.tensor([2., -1.]))
        optimizer = torch.optim.AdamW([parameter], lr=2.5e-5)
        parameter.square().sum().backward()
        norm = torch.nn.utils.clip_grad_norm_([parameter], 1.0)
        optimizer.step()
        state = {name: value.clone() for name, value in optimizer.state[parameter].items()}
        weights = parameter.detach().clone()
        before = rng_state()
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrainingDiagnostics.__new__(TrainingDiagnostics)
            recorder.observer = HeadObserver()
            recorder.log = (Path(directory)/"steps.jsonl").open("xb")
            recorder.updates = 0
            recorder.record_update(1, norm, [optimizer.param_groups[0]["lr"]],
                                   [optimizer.param_groups[0]["betas"][0]], parameter.numel())
            recorder.log.close()
        self.assertTrue(rng_equal(before, rng_state()))
        self.assertTrue(torch.equal(weights, parameter))
        for name, value in state.items():
            self.assertTrue(torch.equal(value, optimizer.state[parameter][name]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
