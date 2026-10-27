"""CPU tests of phase scheduling using the production trainer update statements."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

from kev.reverse_schedule import (ReverseController, adapter_counts, reverse_counts,
                                  validate_reverse_arguments)
from kev.dse_options import DSEOptions, FreezeController
from kev.supplement_diagnostics import TrainingDiagnostics, HeadObserver


HERE = Path(__file__).parent
V1_TRAIN_SHA = "8065bc03fd0b2ba8abf345fe90806df0434649d5c49cd2c69d8da029907d7f66"
V1_DSE_SHA = "02e70888bbb4988ca686779ddd73ee03bb962a0ec8f9ea234c4db3554b326ee0"


def original_dse_module():
    path = HERE / "v1_dse_options_reference.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != V1_DSE_SHA:
        raise ValueError("immutable V1 DSE reference differs")
    spec = importlib.util.spec_from_file_location("pinned_v1_dse_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def update_kernel(path, *, original=False):
    """Compile the actual clipping/step/logging/scheduler/phase block only.

    Backbone/data loading is deliberately absent. The tested statement bytes
    come from the production trainer or its pinned V1 reference, not a parallel
    reimplementation of optimizer scheduling.
    """
    raw = Path(path).read_bytes()
    if original and hashlib.sha256(raw).hexdigest() != V1_TRAIN_SHA:
        raise ValueError("immutable V1 training reference differs")
    tree = ast.parse(raw)
    blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and any(isinstance(row, ast.Assign) and any(
                  isinstance(target, ast.Name) and target.id == "used_lrs"
                  for target in row.targets) for row in node.body)]
    if len(blocks) != 1:
        raise ValueError("production optimizer block is not unique")
    rows = []
    for row in blocks[0].body:
        if (isinstance(row, ast.If) and isinstance(row.test, ast.Compare)
                and isinstance(row.test.left, ast.Name) and row.test.left.id == "dev"):
            break
        rows.append(copy.deepcopy(row))
    wrapper = ast.parse("def update(model,opt,sched,diagnostics,step,steps,reverse=None,freeze=None):\n"
                        "    ema=teacher=adapter_restart=None\n").body[0]
    wrapper.body += rows + [ast.Return(value=ast.Name(id="step", ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    namespace = {"torch": torch, "adapter_counts": adapter_counts}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["update"]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.3))
        self.head = torch.nn.Linear(4, 1)

    def forward(self, value):
        return self.head(self.lm(value))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class Sink:
    def __init__(self):
        self.rows = []
        self.events = []

    def record_update(self, step, norm, lrs, beta1, count, **kwargs):
        self.rows.append((step, float(norm), tuple(lrs), tuple(beta1), count, kwargs))

    def record_reverse_boundary(self, event):
        self.events.append(dict(event))


def optimizer_pair(model, total, *, trainable_only=False):
    adapters = [p for p in model.lm.parameters() if not trainable_only or p.requires_grad]
    opt = torch.optim.AdamW([{"params": adapters, "lr": .03},
                            {"params": list(model.head.parameters()), "lr": .03}],
                           lr=.03, weight_decay=.2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[.03, .03], total_steps=total, pct_start=.1)
    return opt, sched


def tensor_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: tensor_tree(row) for key, row in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tensor_tree(row) for row in value)
    return value


class ReverseScheduleTest(unittest.TestCase):
    def assert_tree_equal(self, before, after):
        if isinstance(before, torch.Tensor):
            self.assertTrue(torch.equal(before, after))
        elif isinstance(before, dict):
            self.assertEqual(before.keys(), after.keys())
            for key in before:
                self.assert_tree_equal(before[key], after[key])
        elif isinstance(before, (list, tuple)):
            self.assertEqual(len(before), len(after))
            for a, b in zip(before, after):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(before, after)

    def test_exact_fold_boundaries(self):
        self.assertEqual(reverse_counts(1932), (1352, 580))
        self.assertEqual(reverse_counts(1931), (1351, 580))

    def test_actual_trainer_tail_skips_adamw_weights_decay_and_moments(self):
        kernel = update_kernel(HERE.parent / "kev/train.py")
        for total, joint in ((1932, 1352), (1931, 1351)):
            with self.subTest(total=total):
                torch.manual_seed(44)
                model = TinyModel().train()
                opt, sched = optimizer_pair(model, total)
                reverse, sink = ReverseController(model, total), Sink()
                fixed = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
                initial = tensor_tree(model.state_dict())
                phase_snapshot = None
                head_steps = 0
                for index in range(total):
                    reverse.before_forward()
                    output = model(fixed)
                    (output.square().mean() + output.mean()).backward()
                    head_steps += int(all(p.grad is not None for p in model.head.parameters()))
                    activity = adapter_counts(opt)
                    self.assertGreater(activity["group"], 0)
                    self.assertEqual(activity["requires_grad"], activity["group"] if index < joint else 0)
                    self.assertEqual(activity["non_none_grad"], activity["group"] if index < joint else 0)
                    result = kernel(model, opt, sched, sink, index, total, reverse)
                    self.assertEqual(result, index + 1)
                    if result == joint:
                        phase_snapshot = (tensor_tree(model.lm.state_dict()),
                                          [tensor_tree(opt.state[p]) for p in model.lm.parameters()],
                                          tensor_tree(model.head.state_dict()))
                        self.assertEqual(len(sink.events), 1)
                        self.assertEqual(sink.events[0]["first_head_only_optimizer_update"], joint + 1)
                    elif result > joint:
                        self.assert_tree_equal(phase_snapshot[0], model.lm.state_dict())
                        for old_state, parameter in zip(phase_snapshot[1], model.lm.parameters()):
                            self.assert_tree_equal(old_state, opt.state[parameter])
                        self.assertFalse(model.lm.training)
                        self.assertTrue(model.head.training)
                reverse.finish(total)
                self.assertEqual(head_steps, total)
                self.assertEqual(sched.total_steps, total)
                self.assertEqual(sched.last_epoch, total)
                self.assertTrue(any(not torch.equal(phase_snapshot[2][name], value)
                                    for name, value in model.head.state_dict().items()))
                self.assertTrue(any(not torch.equal(initial["lm." + name], value)
                                    for name, value in phase_snapshot[0].items()))
                for parameter in model.head.parameters():
                    self.assertEqual(int(opt.state[parameter]["step"]), total)
                # Independent uninterrupted OneCycle sequence has no phase reset.
                dummy = TinyModel()
                expected_opt, expected_sched = optimizer_pair(dummy, total)
                expected_lrs, expected_betas = [], []
                for _ in range(total):
                    expected_lrs.append(tuple(group["lr"] for group in expected_opt.param_groups))
                    expected_betas.append(tuple(group["betas"][0] for group in expected_opt.param_groups))
                    expected_opt.step()
                    expected_sched.step()
                self.assertEqual([row[2] for row in sink.rows], expected_lrs)
                self.assertEqual([row[3] for row in sink.rows], expected_betas)
                self.assertEqual(reverse.joint_observed, joint)
                self.assertEqual(reverse.tail_observed, 580)

    def test_disabled_actual_trainer_matches_pinned_v1_rng_output_gradient_update(self):
        original = update_kernel(HERE / "v1_train_reference.py", original=True)
        updated = update_kernel(HERE.parent / "kev/train.py")
        torch.manual_seed(45)
        model0 = TinyModel().train()
        model1 = copy.deepcopy(model0)
        optimizers = [optimizer_pair(model, 40) for model in (model0, model1)]
        sinks = [Sink(), Sink()]
        initial_rng = torch.get_rng_state().clone()
        fixed = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
        trajectories = []
        for model, pair, kernel, sink in zip((model0, model1), optimizers, (original, updated), sinks):
            torch.set_rng_state(initial_rng.clone())
            rows = []
            for index in range(40):
                out = model(fixed)
                (out.square().mean() + out.mean()).backward()
                gradient = [p.grad.detach().clone() for p in model.parameters()]
                before = torch.get_rng_state().clone()
                result = kernel(model, pair[0], pair[1], sink, index, 40, None)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertEqual(result, index + 1)
                rows.append((out.detach().clone(), gradient, tensor_tree(model.state_dict()),
                             tensor_tree(pair[0].state_dict()), tensor_tree(pair[1].state_dict()),
                             torch.get_rng_state().clone()))
            trajectories.append(rows)
        self.assert_tree_equal(trajectories[0], trajectories[1])
        self.assertEqual(sinks[0].events, [])
        self.assertEqual(sinks[1].events, [])

    def test_freeze_clears_stale_grad_and_retains_original_groups_and_state(self):
        torch.manual_seed(7)
        model = TinyModel().train()
        reverse = ReverseController(model, 2)
        opt = torch.optim.AdamW([{"params": list(model.lm.parameters())},
                                {"params": list(model.head.parameters())}], lr=.03, weight_decay=.9)
        model(torch.ones(2, 4)).sum().backward()
        opt.step()  # Deliberately leave gradients present at the phase boundary.
        groups = [tuple(map(id, group["params"])) for group in opt.param_groups]
        weights, states = tensor_tree(model.lm.state_dict()), [tensor_tree(opt.state[p]) for p in model.lm.parameters()]
        event = reverse.after_update(1, opt)
        self.assertEqual(event["non_none_grad_parameter_count"], 0)
        self.assertEqual(groups, [tuple(map(id, group["params"])) for group in opt.param_groups])
        reverse.before_forward()
        model(torch.ones(2, 4)).sum().backward()
        opt.step()
        self.assert_tree_equal(weights, model.lm.state_dict())
        for old, parameter in zip(states, model.lm.parameters()):
            self.assert_tree_equal(old, opt.state[parameter])

    def test_disabled_reverse_preserves_pinned_v1_D_freeze_thaw_trajectory(self):
        reference = original_dse_module()
        kernels = (update_kernel(HERE / "v1_train_reference.py", original=True),
                   update_kernel(HERE.parent / "kev/train.py"))
        torch.manual_seed(51)
        model0 = TinyModel().train()
        models = (model0, copy.deepcopy(model0))
        initial_rng = torch.get_rng_state().clone()
        fixed = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
        trajectories = []
        for model, kernel, options, controller_type in zip(
                models, kernels, (reference.DSEOptions(freeze_fraction=.3), DSEOptions(freeze_fraction=.3)),
                (reference.FreezeController, FreezeController)):
            freeze = controller_type(model, options, 40)
            opt, sched = optimizer_pair(model, 40, trainable_only=True)
            self.assertEqual(len(opt.param_groups[0]["params"]), 0)
            freeze.apply_modes()
            torch.set_rng_state(initial_rng.clone())
            rows, sink = [], Sink()
            for index in range(40):
                output = model(fixed)
                (output.square().mean() + output.mean()).backward()
                gradients = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
                kernel(model, opt, sched, sink, index, 40, None, freeze)
                rows.append((output.detach().clone(), gradients, tensor_tree(model.state_dict()),
                             tensor_tree(opt.state_dict()), tensor_tree(sched.state_dict()),
                             torch.get_rng_state().clone(), model.lm.training, model.head.training,
                             freeze.frozen, sum(p.numel() for p in opt.param_groups[0]["params"])))
            self.assertTrue(all(row[9] == 0 for row in rows[:11]))
            self.assertGreater(rows[11][9], 0)  # thaw after update12, before forward13
            self.assertTrue(all(row[6] is False for row in rows[:11]))
            self.assertTrue(all(row[6] is True for row in rows[11:]))
            self.assertEqual(sink.events, [])
            trajectories.append(rows)
        self.assert_tree_equal(trajectories[0], trajectories[1])

    def test_tail_mode_is_reapplied_after_external_model_train(self):
        model = TinyModel().train()
        reverse = ReverseController(model, 2)
        opt, _ = optimizer_pair(model, 2)
        reverse.after_update(1, opt)
        model.train()
        self.assertTrue(model.lm.training)
        reverse.before_forward()
        self.assertTrue(all(not module.training for module in model.lm.modules()))
        self.assertTrue(all(module.training for module in model.head.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.lm.parameters()))

    def test_invalid_compositions_and_disabled_validation(self):
        before, python_before = torch.get_rng_state().clone(), random.getstate()
        validate_reverse_arguments(SimpleNamespace(reverse_head_only_fraction=0.0), DSEOptions(), {})
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(python_before, random.getstate())
        with self.assertRaises(ValueError):
            validate_reverse_arguments(SimpleNamespace(reverse_head_only_fraction=.3),
                                       DSEOptions(freeze_fraction=.3), {})

    def test_actual_logger_distinguishes_retained_group_from_frozen_activity(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = TrainingDiagnostics.__new__(TrainingDiagnostics)
            logger.directory = Path(directory)
            logger.observer = HeadObserver()
            logger.log = (logger.directory / "optimizer_updates.jsonl").open("xb")
            logger.updates = 0
            logger.record_update(1, torch.tensor(.5), [.03, .03], [.9, .9], 20,
                                 adapter_active_count=0, adapter_non_none_grad_count=0,
                                 reverse_phase="head_only")
            logger.log.close()
            import json
            row = json.loads((logger.directory / "optimizer_updates.jsonl").read_bytes())
            self.assertEqual(row["adapter_group_parameter_count"], 20)
            self.assertEqual(row["active_adapter_parameter_count"], 0)
            self.assertEqual(row["adapter_requires_grad_parameter_count"], 0)
            self.assertEqual(row["adapter_non_none_grad_parameter_count"], 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
