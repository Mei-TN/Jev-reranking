"""Separate fixed joint-then-head-only supplement; disabled path has no actor.

The adapter optimizer group and state remain in place. Freezing requires grad
None, so AdamW skips both parameter/weight-decay and moment updates.
"""
from fractions import Fraction
import math

from .dse_options import DSEOptions


def add_reverse_argument(parser):
    parser.add_argument("--reverse_head_only_fraction", type=float, default=0.0,
                        choices=(0.0, 0.3),
                        help="separate reverse supplement: final ceil(.3T) updates head-only")


def validate_reverse_arguments(args, dse, head_config):
    fraction = args.reverse_head_only_fraction
    if not math.isfinite(fraction) or fraction not in (0.0, 0.3):
        raise ValueError("reverse_head_only_fraction must be 0 or .3")
    if fraction == 0:
        return
    if dse != DSEOptions():
        raise ValueError("reverse requires original native DSE defaults; no initial freeze/restart/composition")
    if (args.full_finetune or args.init_from or args.special_embeddings or args.option_isolation
            or args.lora_targets != "all" or args.perm_kl or args.anchor or args.anchor_w or args.ord_w
            or args.replay or args.synthetic_repeat != 1):
        raise ValueError("reverse requires fresh original all-target adapter/head recipe")
    if (not args.data or args.epochs != 1 or args.batch != 1 or args.accum != 8 or args.lora != 64
            or args.lr != 2.5e-5 or (args.head_lr or args.lr) != 2.5e-5
            or args.supplement_diagnostics != 1):
        raise ValueError("reverse fixed recipe requires exact data, epoch1/batch1/accum8/rank64/LR2.5e-5/diagnostics1")
    if (head_config["kind"] != "pointer" or head_config["readout"] != "end"
            or head_config["dim"] != 256 or head_config["rng_anchor_dim"] != 256):
        raise ValueError("reverse fixed supplement requires original H0 pointer/end")


def reverse_counts(total_steps, fraction=0.3):
    if type(total_steps) is not int or total_steps < 2:
        raise ValueError("reverse horizon must be an integer at least 2")
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("reverse fraction must be strictly between 0 and 1")
    tail = math.ceil(Fraction(str(fraction)) * total_steps)
    if not 0 < tail < total_steps:
        raise ValueError("reverse requires nonempty joint and head-only phases")
    return total_steps - tail, tail


def schedule_metadata(total_steps, fraction):
    if fraction == 0:
        return {"enabled": False, "fraction": 0.0}
    joint, tail = reverse_counts(total_steps, fraction)
    return {"enabled": True, "fraction": fraction,
            "joint_optimizer_updates": joint, "head_only_optimizer_updates": tail,
            "first_head_only_optimizer_update": joint + 1,
            "rounding": "T-ceil(Fraction(str(fraction))*T)",
            "head_only_mode": "lm.eval/head.train",
            "scheduler": "unchanged global full-horizon OneCycle",
            "adapter_optimizer_policy": "group/state retained; requires_gradFalse and gradNone"}


def adapter_counts(optimizer):
    parameters = optimizer.param_groups[0]["params"]
    return {"group": sum(p.numel() for p in parameters),
            "requires_grad": sum(p.numel() for p in parameters if p.requires_grad),
            "non_none_grad": sum(p.numel() for p in parameters if p.grad is not None)}


class ReverseController:
    def __init__(self, model, total_steps, fraction=0.3):
        self.model = model
        self.total_steps = total_steps
        self.joint_steps, self.tail_steps = reverse_counts(total_steps, fraction)
        self.parameters = [p for p in model.lm.parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError("reverse requires initially trainable adapter parameters")
        self.completed_steps = self.joint_observed = self.tail_observed = 0
        self.frozen = False
        self.boundary_event = None

    @property
    def phase(self):
        return "head_only" if self.frozen else "joint"

    def apply_modes(self):
        if self.frozen:
            self.model.lm.eval()
            self.model.head.train()

    def before_forward(self):
        if self.completed_steps >= self.total_steps:
            raise RuntimeError("forward after the fixed reverse horizon")
        if self.frozen:
            if any(p.requires_grad or p.grad is not None for p in self.parameters):
                raise RuntimeError("tail adapter must remain requires_gradFalse with gradNone")
            self.apply_modes()
        elif any(not p.requires_grad for p in self.parameters):
            raise RuntimeError("joint adapter unexpectedly frozen")

    def after_update(self, completed_steps, optimizer):
        if completed_steps != self.completed_steps + 1 or completed_steps > self.total_steps:
            raise RuntimeError("reverse completed-update sequence differs")
        if tuple(map(id, optimizer.param_groups[0]["params"])) != tuple(map(id, self.parameters)):
            raise RuntimeError("reverse adapter optimizer group changed")
        if self.frozen:
            if any(p.requires_grad or p.grad is not None for p in self.parameters):
                raise RuntimeError("tail optimizer step observed adapter gradient/activity")
            self.tail_observed += 1
        else:
            self.joint_observed += 1
        self.completed_steps = completed_steps
        if completed_steps != self.joint_steps:
            return None
        for parameter in self.parameters:
            parameter.requires_grad_(False)
            parameter.grad = None
        self.frozen = True
        self.apply_modes()
        self.boundary_event = {"completed_optimizer_update": completed_steps,
                               "first_head_only_optimizer_update": completed_steps + 1,
                               "joint_target": self.joint_steps, "tail_target": self.tail_steps,
                               "adapter_group_parameter_count": adapter_counts(optimizer)["group"],
                               "requires_grad_parameter_count": adapter_counts(optimizer)["requires_grad"],
                               "non_none_grad_parameter_count": adapter_counts(optimizer)["non_none_grad"],
                               "optimizer_group_retained": True,
                               "phase": self.phase, "lm_training": self.model.lm.training,
                               "head_training": self.model.head.training}
        return self.boundary_event

    def finish(self, completed_steps):
        if (completed_steps != self.total_steps or self.completed_steps != self.total_steps
                or self.joint_observed != self.joint_steps or self.tail_observed != self.tail_steps
                or not self.frozen or self.boundary_event is None):
            raise RuntimeError("reverse fixed phase counts are incomplete")
        if any(p.requires_grad or p.grad is not None for p in self.parameters):
            raise RuntimeError("reverse final adapter state is active")

    def metadata(self):
        return {"planned_joint_updates": self.joint_steps, "planned_head_only_updates": self.tail_steps,
                "observed_joint_updates": self.joint_observed,
                "observed_head_only_updates": self.tail_observed,
                "completed_optimizer_updates": self.completed_steps,
                "boundary_event": self.boundary_event, "current_phase": self.phase}
