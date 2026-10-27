"""E3 adapter-group-only OneCycle restart; never owns model optimizer state."""
import math

import torch


def validate_restart_horizon(total_steps, freeze_steps):
    if (type(total_steps) is not int or type(freeze_steps) is not int
            or not 0 < freeze_steps < total_steps):
        raise ValueError("adapter restart requires a nonempty frozen phase and an actual thaw before the full horizon")
    remaining = total_steps - freeze_steps
    # pct_start=.1 yields a zero-length warmup denominator at T=10, and an
    # absent warmup below 10. This diagnostic intentionally requires both phases.
    if remaining < 11:
        raise ValueError("adapter restart needs at least 11 remaining optimizer updates for a nondegenerate OneCycle warmup (pct_start=.1)")
    return remaining


class AdapterGroupRestart:
    """Override only group 0's LR/beta1 after each GLOBAL scheduler step.

    The original full-horizon scheduler still owns the real optimizer and head
    group. A parameter-free AdamW is solely a public-PyTorch OneCycle schedule
    carrier: it never updates a model parameter or shares optimizer state. Its
    LR and beta tuple are copied to real adapter group 0 after global scheduling.
    The global group's initial_lr/max_lr/min_lr metadata is intentionally intact.
    """
    def __init__(self, optimizer, *, total_steps, freeze_steps, max_lr):
        self.remaining = validate_restart_horizon(total_steps, freeze_steps)
        if not math.isfinite(max_lr) or max_lr <= 0:
            raise ValueError("adapter restart max_lr must be finite and positive")
        if len(optimizer.param_groups) != 2 or optimizer.param_groups[0]["params"]:
            raise ValueError("adapter restart requires the original empty adapter group and head group")
        if "betas" not in optimizer.param_groups[0]:
            raise ValueError("adapter restart requires AdamW beta scheduling")
        self.adapter_group = optimizer.param_groups[0]
        self.total_steps, self.freeze_steps, self.max_lr = total_steps, freeze_steps, max_lr
        self.completed_updates = self.local_updates = 0
        self.local_optimizer = self.local_scheduler = None
        self.first_used = self.last_used = None

    def _copy_schedule(self):
        group = self.local_optimizer.param_groups[0]
        self.adapter_group["lr"] = group["lr"]
        self.adapter_group["betas"] = group["betas"]

    def after_update(self, completed_optimizer_updates, *, thawed):
        """Call once AFTER successful real opt.step and global scheduler.step.

        FreezeController must have already thawed exactly at freeze_steps. The
        fresh local scheduler's constructor installs index 0 for the NEXT (first
        unfrozen) update. Later calls advance local index exactly once per real
        adapter update, including PyTorch's unused post-final scheduler state.
        """
        n = completed_optimizer_updates
        if type(n) is not int or n != self.completed_updates + 1 or n > self.total_steps:
            raise ValueError("adapter restart needs exactly the next successful global optimizer update")
        if type(thawed) is not bool:
            raise ValueError("thawed must be a boolean event")
        if n < self.freeze_steps:
            if thawed or self.adapter_group["params"]:
                raise ValueError("adapter parameters exposed before the frozen phase ended")
        elif n == self.freeze_steps:
            if not thawed or not self.adapter_group["params"] or self.local_scheduler is not None:
                raise ValueError("adapter restart must coincide with the actual one-time thaw")
            self.local_optimizer = torch.optim.AdamW([{"params": []}], lr=self.max_lr,
                                                     betas=self.adapter_group["betas"])
            self.local_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.local_optimizer, max_lr=self.max_lr, total_steps=self.remaining, pct_start=.1)
            self._copy_schedule()
        else:
            if thawed or self.local_scheduler is None or not self.adapter_group["params"]:
                raise ValueError("adapter restart lost its thawed parameter group")
            local = self.local_optimizer.param_groups[0]
            used = {"global_update": n, "local_update": self.local_updates + 1,
                    "lr": local["lr"], "beta1": local["betas"][0]}
            if self.first_used is None:
                self.first_used = used
            self.last_used = used
            # An empty step satisfies the public scheduler step-order contract;
            # this optimizer has no model parameters and therefore no state.
            self.local_optimizer.step()
            self.local_scheduler.step()
            self.local_updates += 1
            self._copy_schedule()
        self.completed_updates = n

    def metadata(self):
        return {"mode": "restart_adapter_group_only", "global_head_horizon": self.total_steps,
                "freeze_updates": self.freeze_steps, "local_adapter_horizon": self.remaining,
                "completed_global_updates": self.completed_updates, "completed_adapter_updates": self.local_updates,
                "thawed": self.local_scheduler is not None, "first_adapter_update_used": self.first_used,
                "last_adapter_update_used": self.last_used,
                "local_scheduler_last_epoch": None if self.local_scheduler is None else self.local_scheduler.last_epoch,
                "scheduler": {"implementation": "torch.optim.lr_scheduler.OneCycleLR", "pct_start": .1,
                              "anneal_strategy": "cos", "div_factor": 25., "final_div_factor": 10000.,
                              "cycle_momentum": True, "base_momentum": .85, "max_momentum": .95,
                              "max_lr": self.max_lr, "beta2": self.adapter_group["betas"][1]},
                "head_schedule": "unchanged original global full-horizon OneCycle LR and beta1",
                "optimizer_state": "real AdamW state retained; parameter-free local schedule carrier has no state",
                "post_final_state": "PyTorch step(T) is retained but unused; final optimizer update consumes local index T-1"}
