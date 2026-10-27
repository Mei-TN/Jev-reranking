"""Record model-construction and parameter state, including a zero-branch control.

Observation preserves RNG state, model parameters, and optimizer state.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch


def canonical(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))+'\n').encode()


def snapshot_rng():
    initialized = torch.cuda.is_initialized()
    return dict(cpu=torch.get_rng_state().clone(),
                cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if initialized else [],
                cuda_initialized=initialized, python=random.getstate())


def same_rng(a, b):
    return (a['cuda_initialized']==b['cuda_initialized'] and a['python']==b['python']
            and torch.equal(a['cpu'],b['cpu']) and len(a['cuda'])==len(b['cuda'])
            and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda'])))


class R3ControlCapture:
    def __init__(self, out_dir, data_rng, *, sham, probe_validate=False, deadline_monotonic=None):
        # Campaign copy: no publication stop. The only stop is the optional per-run campaign timeout.
        self.deadline_monotonic=deadline_monotonic
        self.gate();self.directory=Path(out_dir)/'r3_control'
        self.directory.mkdir(exist_ok=False)
        self.data_rng=data_rng;self.sham=bool(sham);self.head_calls=0;self.updates=0
        self.probe_validate=bool(probe_validate);self.probe_verified=False
        self.original_make_head=None;self.model_module=None
        self.model=None;self.pending_update=None;self.sham_initial_sha256={}
        self.gate()
        self.log=(self.directory/'optimizer_invariants.jsonl').open('xb')

    def gate(self):
        if self.deadline_monotonic is not None and time.monotonic()>=self.deadline_monotonic:
            raise TimeoutError('campaign timeout reached')

    def put(self, name, raw):
        self.gate()
        with (self.directory/name).open('xb') as f:
            if f.write(raw)!=len(raw):raise IOError('short R3 observational write')
            self.gate();f.flush();self.gate();os.fsync(f.fileno())
        fd=os.open(self.directory,os.O_RDONLY|os.O_DIRECTORY)
        try:self.gate();os.fsync(fd)
        finally:os.close(fd)
        self.gate()

    def save(self, name, value):self.put(name,canonical(value))

    def save_rng(self, label, state):
        rows=[]
        tensors=[('cpu',state['cpu'])]+[(f'cuda_visible_{i}',x) for i,x in enumerate(state['cuda'])]
        for name,value in tensors:
            self.gate();raw=value.numpy().tobytes();filename=label+'.'+name+'.bin'
            self.put(filename,raw)
            rows.append(dict(role=name,file=filename,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        self.save(label+'.RNG.json',dict(full_bytes=rows,python_global_state=state['python'],
            CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
            cuda_was_initialized=state['cuda_initialized'],
            cuda_uninitialized_scope='NOT_INITIALIZED_NO_LAZY_INIT_NO_BEFORE_BYTES' if not state['cuda_initialized'] else None,
            no_random_draw_or_state_restore=True,old_dropout_masks='UNKNOWN_NOT_RECORDED'))

    def attach_head_constructor(self, model_module):
        self.gate();self.model_module=model_module;self.original_make_head=model_module.make_head
        original=self.original_make_head
        def observed(*args,**kwargs):
            self.gate();self.head_calls+=1
            if self.head_calls!=1:raise ValueError('exactly one actual R3 head constructor allowed')
            before=snapshot_rng()
            head=original(*args,**kwargs)  # Original canonical/fork_rng code and its draws.
            after=snapshot_rng()
            if self.probe_validate:
                # One explicit bounded-probe assay, isolated in CPU fork_rng.
                # It uses the genuinely captured before-state, never seed reconstruction.
                from .heads import PointerHead
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(before['cpu'])
                    canonical_head=PointerHead(args[0],dp=256,dropout=kwargs.get('dropout',0.0))
                    canonical_after=torch.get_rng_state().clone()
                    if not torch.equal(canonical_after,after['cpu']):
                        raise ValueError('actual head construction did not retain canonical CPU RNG advancement')
                    for projection in ('q','k'):
                        for parameter in ('weight','bias'):
                            actual=getattr(getattr(head,projection),parameter).detach().cpu().contiguous()
                            expected=getattr(getattr(canonical_head,projection),parameter).detach().cpu().contiguous()
                            if not torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)):
                                raise ValueError('actual q/k bytes differ from canonical construction under captured before-state')
                if not same_rng(after,snapshot_rng()):raise ValueError('bounded probe CPU fork failed to preserve actual post-head RNG')
                self.probe_verified=True
                self.save('bounded_probe_canonical_qk_rng_assay.json',dict(verified=True,
                    actual_before_state_used=True,nominal_seed_reconstruction=False,
                    extra_CPU_head_construction_within_explicit_probe_only=True,
                    no_CUDA_draw_or_lazy_initialization=True,production_R3_extra_draws=False,
                    old_actual_dropout_masks='UNKNOWN_NOT_RECORDED'))
            self.save_rng('head_construct_before',before)
            self.save_rng('head_construct_after',after)
            if not same_rng(after,snapshot_rng()):raise ValueError('head receipts changed actual RNG')
            return head
        model_module.make_head=observed

    def detach_head_constructor(self):
        if self.model_module is not None:
            self.model_module.make_head=self.original_make_head

    def after_model(self, model):
        self.gate();self.model=model
        if self.head_calls!=1:raise ValueError('actual head construction witness missing')
        state=snapshot_rng();self.save_rng('model_to_device_complete',state)
        if self.sham:
            if model.head_config['kind']!='residual_pair_mlp' or model.head_config['readout']!='end':
                raise ValueError('sham only permits the unchanged residual/end head')
            final=model.head.pair_mlp[-1]
            if not isinstance(final,torch.nn.Linear):raise ValueError('actual residual last layer differs')
            for parameter in (final.weight,final.bias):
                if torch.count_nonzero(parameter.detach()).item()!=0:
                    raise ValueError('original residual final layer is not zero initialized')
            parameters=dict(model.head.pair_mlp.named_parameters())
            if set(parameters)!={'0.weight','0.bias','2.weight','2.bias'}:
                raise ValueError('exact four original pair_mlp parameters required')
            for name,parameter in parameters.items():
                raw=parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                self.sham_initial_sha256[name]=hashlib.sha256(raw).hexdigest()
                parameter.requires_grad_(False)
            self.save('sham_four_initial_frozen_tensor_SHA.json',dict(initial_sha256=self.sham_initial_sha256,
                first_layer_weights_preserve_original_initialization=True,last_layer_weight_bias_initially_zero=True,
                whole_pair_mlp_frozen=True,parameter_count=4))
        if not same_rng(state,snapshot_rng()):raise ValueError('sham flag or receipts changed RNG')
        self.save('actual_head_identity.json',dict(head_config=model.head_config,sham=self.sham,
            original_complete_forward_retained=True,only_sham_whole_pair_mlp_four_tensors_frozen=self.sham,
            no_output_shortcut=True,old_actual_dropout_masks='UNKNOWN_NOT_RECORDED'))

    def bind_initial(self, diagnostics):
        self.gate();state=snapshot_rng();rows=[]
        if diagnostics is None:raise ValueError('R3 requires original complete initial tensor capture')
        captured=diagnostics.initial
        for name,value in self.model.named_parameters():
            if value.requires_grad:
                if name not in captured:raise ValueError('initial binary does not cover a trainable tensor: '+name)
                initial=captured[name]
                if initial.shape!=value.shape or initial.dtype!=value.dtype:
                    raise ValueError('initial snapshot tensor shape/dtype differs: '+name)
                group='head' if name.startswith('head.') else 'adapter'
                rows.append(dict(parameter=name,shape=list(initial.shape),dtype=str(initial.dtype),
                    numel=initial.numel(),binary='diagnostics/'+group+'_initial.safetensors',
                    binary_key=name.removeprefix('head.') if group=='head' else name))
        if not rows:raise ValueError('empty complete trainable snapshot map')
        binding=json.loads((diagnostics.directory/'initial_capture.json').read_bytes())
        self.save('initial_complete_trainable_tensor_binary_map.json',dict(optimizer_steps=0,
            genuine_training_capture=True,nominal_seed_reconstruction=False,trainable_tensors=rows,
            full_head_binary_sha256=binding['initial_head_sha256'],
            full_adapter_binary_sha256=binding['initial_adapter_sha256'],
            original_initial_capture_path='diagnostics/initial_capture.json',
            sham_all_four_frozen_tensors_still_present_in_complete_head_binary=self.sham,
            baseline_tensor_comparison='PENDING_BOUNDED_ACTUAL_BINARY_COMPARISON_NO_POSTHOC_CORRECTION'))
        if not same_rng(state,snapshot_rng()):raise ValueError('initial tensor index changed RNG')

    def before_shuffle(self, epoch):
        self.gate();self.shuffle_before=self.data_rng.getstate()

    def after_shuffle(self, epoch, requests):
        self.gate();state=snapshot_rng()
        self.save('epoch_'+str(epoch)+'_actual_original_data_order.json',dict(epoch=epoch,
            original_data_rng_state_before=self.shuffle_before,
            original_data_rng_state_after=self.data_rng.getstate(),
            record_ids=[x['_meta']['id'] for x in requests],
            sources=[x['_meta']['source'] for x in requests],
            permutation_source='original independent random.Random(seed).shuffle with no reseed or replacement',
            no_new_draw=True,actual_dropout_masks='NOT_CAPTURED_NO_BITWISE_HISTORY_CLAIM'))
        if not same_rng(state,snapshot_rng()):raise ValueError('data order receipt changed global RNG')

    def first_update_batch(self, epoch, microbatch, chunk, accum):
        if epoch==0 and microbatch<accum:
            self.save('first_update_micro_'+str(microbatch)+'.json',dict(epoch=epoch,microbatch=microbatch,
                original_source_ids=[x['_meta']['id'] for x in chunk],
                original_source_names=[x['_meta']['source'] for x in chunk],
                before_original_augmentation=True,no_record_or_candidate_changes=True))

    def sham_check(self, opt):
        self.gate()
        if not self.sham:return dict(sham=False)
        ids={id(p) for group in opt.param_groups for p in group['params']}
        parameters=dict(self.model.head.pair_mlp.named_parameters());rows=[]
        if set(parameters)!=set(self.sham_initial_sha256) or len(parameters)!=4:
            raise ValueError('four frozen pair_mlp parameter coverage differs')
        for name,p in parameters.items():
            if p.requires_grad or p.grad is not None or id(p) in ids:
                raise ValueError('sham whole pair_mlp gradient or optimizer exclusion veto')
            raw=p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes();digest=hashlib.sha256(raw).hexdigest()
            if digest!=self.sham_initial_sha256[name]:raise ValueError('frozen pair_mlp bytes changed from original initialization')
            final_zero=name.startswith('2.')
            if final_zero and torch.count_nonzero(p.detach()).item()!=0:raise ValueError('sham final layer changed from exact zero')
            rows.append(dict(parameter='head.pair_mlp.'+name,bytes=len(raw),sha256=digest,
                initial_sha256=self.sham_initial_sha256[name],initial_bytes_unchanged=True,
                requires_grad=False,gradient_is_none=True,in_any_optimizer_group=False,
                final_layer=final_zero,final_layer_zero_verified=final_zero))
        return dict(sham=True,frozen_parameters=rows,weight_decay_cannot_update_an_excluded_parameter=True,
                    complete_original_pair_mlp_forward_retained=True)

    def before_update(self, opt, update):
        self.gate()
        if update!=self.updates+1 or self.pending_update is not None:raise ValueError('R3 invariant optimizer order differs')
        self.pending_update=dict(optimizer_update=update,before_optimizer=self.sham_check(opt))

    def after_update(self, opt, update):
        self.gate()
        if self.pending_update is None or self.pending_update['optimizer_update']!=update:
            raise ValueError('actual R3 pre-step invariant missing')
        row=dict(self.pending_update,after_optimizer=self.sham_check(opt),
                 actual_optimizer_step_returned=True,no_optimizer_or_gradient_mutation_by_observer=True)
        raw=canonical(row)
        if self.log.write(raw)!=len(raw):raise IOError('short R3 invariant log write')
        self.gate();self.log.flush();self.gate();os.fsync(self.log.fileno());self.gate()
        self.pending_update=None;self.updates=update

    def finish(self, optimizer_steps):
        self.gate()
        if self.updates!=optimizer_steps or self.pending_update is not None:
            raise ValueError('R3 observed and actual optimizer steps differ')
        self.log.flush();os.fsync(self.log.fileno());self.log.close()
        if self.sham:
            for name,p in self.model.head.pair_mlp.named_parameters():
                raw=p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                if p.requires_grad or p.grad is not None or hashlib.sha256(raw).hexdigest()!=self.sham_initial_sha256[name]:
                    raise ValueError('final four frozen sham initial-byte or gradient invariant failed')
                if name.startswith('2.') and torch.count_nonzero(p.detach()).item()!=0:
                    raise ValueError('final sham last layer zero invariant failed')
        files={}
        for path in sorted(self.directory.iterdir()):
            self.gate()
            if path.is_file():
                with path.open('rb') as stream:files[path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
                self.gate()
        self.save('completion.json',dict(status='R3_CAPTURE_COMPLETE_PENDING_INDEPENDENT_REVIEW',
            files_sha256=files,
            actual_optimizer_updates=self.updates,sham=self.sham,final_frozen_zero_verified=self.sham,
            frozen_pair_mlp_parameter_count=4 if self.sham else 0,final_pair_mlp_initial_bytes_unchanged_verified=self.sham,
            probe_validation_enabled=self.probe_validate,probe_canonical_qk_and_cpu_rng_verified=self.probe_verified,
            campaign_timeout_enforced=self.deadline_monotonic is not None,
            initial_tensor_comparison='PENDING_SEPARATE_ACTUAL_BINARY_READ',
            old_dropout_masks='UNKNOWN_NOT_RECORDED',scientific_acceptance=None))
