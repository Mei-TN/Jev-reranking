"""Adapter/head exponential moving average updates.

The teacher uses a separate module tree initialized from the verified frozen
base. Frozen tensor storage may be shared; trainable storage and module objects
must be separate so teacher inference does not change the student.
Call update after optimizer.step.
"""
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import numbers
import weakref

import torch


def _selected(name):
    return name.startswith('head.') or '.lora_A.' in name or '.lora_B.' in name


def adapter_head_names(student):
    """Select the fixed NeoHorse recipe's LoRA A/B and pointer head by name.

    Selection does not disappear during a temporary freeze. Other trainable
    parameter classes fail rather than silently remaining outside the teacher.
    """
    params = dict(student.named_parameters())
    names = tuple(sorted(name for name in params if _selected(name)))
    if not names or not any(name.startswith('head.') for name in names):
        raise ValueError('EMA requires named head parameters')
    if not any('.lora_A.' in n for n in names) or not any('.lora_B.' in n for n in names):
        raise ValueError('EMA requires unmerged LoRA A and B parameters')
    name_set = set(names)
    if any(p.requires_grad and name not in name_set for name,p in params.items()):
        raise ValueError('Unsupported trainable parameters outside LoRA A/B and head')
    return names


def ramp_weight(completed_updates, total_updates, *, maximum_weight, ramp_fraction=.2):
    """Weight for the next forward, ramping from zero over the full horizon."""
    if (isinstance(completed_updates,bool) or not isinstance(completed_updates,int)
            or isinstance(total_updates,bool) or not isinstance(total_updates,int)
            or total_updates < 1 or not 0 <= completed_updates <= total_updates):
        raise ValueError('Invalid completed/full optimizer-update counts')
    if (isinstance(maximum_weight,bool) or not isinstance(maximum_weight,numbers.Real)
            or not math.isfinite(maximum_weight) or maximum_weight < 0
            or isinstance(ramp_fraction,bool) or not isinstance(ramp_fraction,numbers.Real) or not math.isfinite(ramp_fraction)
            or not 0 <= ramp_fraction <= 1):
        raise ValueError('Invalid EMA weight or ramp fraction')
    ramp_steps = math.ceil(Fraction(str(ramp_fraction)) * total_updates)
    return float(maximum_weight) * (min(completed_updates / ramp_steps, 1.) if ramp_steps else 1.)


class AdapterHeadEMA:
    def __init__(self, student, decay):
        if isinstance(decay,bool) or not isinstance(decay,numbers.Real) or not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError('EMA decay must be finite and in [0,1)')
        self.decay = float(decay)
        self.names = adapter_head_names(student)
        self.student_ref = weakref.ref(student)
        params = dict(student.named_parameters())
        for name in self.names:
            if params[name].dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(params[name]).all()):
                raise ValueError('EMA requires finite fp32/fp64 master parameters')
        self.shadow = {name:params[name].detach().clone() for name in self.names}
        self.updates = 0
        self._revision = 0
        self._synchronized = weakref.WeakKeyDictionary()

    def _student_parameters(self):
        student = self.student_ref()
        if student is None:
            raise RuntimeError('EMA student no longer exists')
        if adapter_head_names(student) != self.names:
            raise ValueError('EMA parameter namespace changed')
        params = dict(student.named_parameters())
        for name,value in self.shadow.items():
            p=params[name]
            if p.shape != value.shape or p.dtype != value.dtype or p.device != value.device:
                raise ValueError('EMA student shape/dtype/device changed')
        return student,params

    @torch.no_grad()
    def update(self, completed_optimizer_updates):
        """Require the next completed optimizer-step index, never a microbatch.

        This rejects duplicate, skipped and out-of-order indices. The trainer
        must still supply the true successful optimizer count; the EMA cannot
        independently observe optimizer.step or an AMP-skipped update.
        """
        if (type(completed_optimizer_updates) is not int
                or completed_optimizer_updates != self.updates + 1):
            raise ValueError('EMA requires exactly the next completed optimizer-update count')
        _,params = self._student_parameters()
        if any(not bool(torch.isfinite(params[name]).all()) for name in self.names):
            raise ValueError('Nonfinite student parameter; EMA update refused')
        for name,value in self.shadow.items():
            if self.decay == 0:
                value.copy_(params[name])
            else:
                value.mul_(self.decay).add_(params[name],alpha=1-self.decay)
        self.updates += 1
        self._revision += 1

    def ramp_weight(self, total_updates, *, maximum_weight, ramp_fraction=.2):
        return ramp_weight(self.updates,total_updates,maximum_weight=maximum_weight,ramp_fraction=ramp_fraction)

    @torch.no_grad()
    def copy_to(self, teacher):
        """Copy shadows into a disjoint teacher, then freeze it and use eval mode."""
        student,source = self._student_parameters()
        if {id(m) for m in student.modules()} & {id(m) for m in teacher.modules()}:
            raise ValueError('Teacher and student share module objects')
        target = dict(teacher.named_parameters())
        if tuple(sorted(n for n in target if _selected(n))) != self.names:
            raise ValueError('Teacher parameter namespace differs')
        if set(target) != set(source):
            raise ValueError('Teacher full parameter namespace differs')
        for name,p in target.items():
            q = source[name]
            if p.shape != q.shape or p.dtype != q.dtype or p.device != q.device:
                raise ValueError('Teacher parameter shape/dtype/device differs')
        student_storages = {(p.device,p.untyped_storage().data_ptr()) for p in source.values()}
        selected_storages = {(source[n].device,source[n].untyped_storage().data_ptr()) for n in self.names}
        if any((p.device,p.untyped_storage().data_ptr()) in selected_storages for p in target.values()):
            raise ValueError('Teacher aliases student trainable tensor storage')
        copied_storages = set()
        name_set = set(self.names)
        frozen_target_storages = {(p.device,p.untyped_storage().data_ptr()) for n,p in target.items() if n not in name_set}
        shadow_storages = {(p.device,p.untyped_storage().data_ptr()) for p in self.shadow.values()}
        for name,value in self.shadow.items():
            p=target[name]
            storage = (p.device,p.untyped_storage().data_ptr())
            if storage in student_storages:
                raise ValueError('Teacher selected parameter aliases student tensor storage')
            if storage in copied_storages:
                raise ValueError('Teacher selected parameter storage is not independent')
            copied_storages.add(storage)
            if storage in frozen_target_storages:
                raise ValueError('Teacher selected parameter aliases its frozen backbone')
            if storage in shadow_storages:
                raise ValueError('Teacher selected parameter aliases EMA shadow storage')
        # All validation precedes mutations; frozen backbone tensors are untouched.
        for name,value in self.shadow.items():
            target[name].copy_(value)
        teacher.requires_grad_(False)
        teacher.eval()
        self._synchronized[teacher] = (self.updates,self._revision)

    @contextmanager
    def teacher_prediction(self, teacher):
        """Only a teacher synchronized to this EMA revision may predict."""
        if self._synchronized.get(teacher) != (self.updates,self._revision):
            raise ValueError('Teacher is unknown or stale; call EMA.copy_to before prediction')
        if any(p.requires_grad for p in teacher.parameters()) or any(m.training for m in teacher.modules()):
            raise ValueError('Teacher must be frozen and fully in eval mode')
        with torch.no_grad():
            yield teacher

    def state_dict(self):
        """EMA-only state. This is not a full optimizer/RNG training checkpoint."""
        return {'schema_version':1,'decay':self.decay,'updates':self.updates,
                'names':list(self.names),'shadow':{n:v.detach().clone() for n,v in self.shadow.items()}}

    @torch.no_grad()
    def load_state_dict(self, state):
        if not isinstance(state,dict) or set(state) != {'schema_version','decay','updates','names','shadow'}:
            raise ValueError('EMA checkpoint fields differ')
        if (type(state['schema_version']) is not int or state['schema_version'] != 1
                or isinstance(state['decay'],bool) or state['decay'] != self.decay
                or state['names'] != list(self.names)):
            raise ValueError('EMA checkpoint identity differs')
        if isinstance(state['updates'],bool) or not isinstance(state['updates'],int) or state['updates'] < 0:
            raise ValueError('EMA checkpoint update count invalid')
        if set(state['shadow']) != set(self.names):
            raise ValueError('EMA checkpoint tensor names differ')
        for name,value in self.shadow.items():
            incoming=state['shadow'][name]
            if (not isinstance(incoming,torch.Tensor) or incoming.shape != value.shape
                    or incoming.dtype != value.dtype or not bool(torch.isfinite(incoming).all())):
                raise ValueError('EMA checkpoint tensor invalid')
        for name,value in self.shadow.items():
            value.copy_(state['shadow'][name].to(value.device))
        self.updates=state['updates']
        # A load with the SAME count may change weights: invalidate teachers too.
        self._revision += 1

    def metadata(self):
        structure={n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in self.shadow.items()}
        return {'decay':self.decay,'completed_ema_updates':self.updates,
                'initialization':'copy of initial student; no zero-start bias correction',
                'update_timing':'after student optimizer.step; caller must enforce one update per step',
                'update_count_check':'exact next completed optimizer step; optimizer success remains trainer responsibility',
                'ramp_steps':'ceil(decimal ramp_fraction * full total_updates); starts at completed_updates=0',
                'parameter_structure':structure,
                'parameter_structure_sha256':hashlib.sha256(json.dumps(structure,sort_keys=True).encode()).hexdigest(),
                'full_training_resume_supported_by_this_state':False}
