"""Load an FP32, unmerged EMA model and validate checkpoint pairing."""
import hashlib
import io
import json
import math
import os
from pathlib import Path

import torch

from .dse_ema import AdapterHeadEMA


PINNED_REVISION = '56f0584bb40578a2c33b1b40a08ccd17243ad710'
STUDENT_FILES = ('head.pt', 'adapter_config.json', 'adapter_model.safetensors')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(body):
    return hashlib.sha256(body).hexdigest()


def _count(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f'Invalid {name}')
    return value


def _observe_model(model):
    """Check the loaded representation and report observed precision/adapter state."""
    parameters = dict(model.named_parameters())
    floating_dtypes = sorted({str(p.dtype) for p in parameters.values() if p.is_floating_point()})
    _require(floating_dtypes == ['torch.float32'], 'EMA evaluation requires observed fp32 parameters')
    a_names = [n for n in parameters if '.lora_A.' in n]
    b_names = [n for n in parameters if '.lora_B.' in n]
    _require(a_names and b_names and not any(bool(getattr(m, 'merged', False)) for m in model.modules()),
             'EMA evaluation requires observed unmerged LoRA A/B parameters')
    return {'floating_parameter_dtypes_observed': floating_dtypes,
            'buffer_dtypes_observed': sorted({str(b.dtype) for b in model.buffers()}),
            'lora_A_parameter_count_observed': len(a_names),
            'lora_B_parameter_count_observed': len(b_names), 'merged_observed': False}


def verify_ema_checkpoint(run_dir, *, allow_stopped_early=False):
    """Read and bind saved EMA to the exact student artifacts before loading base.

    Legacy shadows without student file hashes fail. Pairing hashes protect
    against accidental cross-run mixing, not maliciously rewritten metadata.
    All torch containers are loaded with weights_only=True from captured bytes.
    """
    _require(type(allow_stopped_early) is bool,'allow_stopped_early must be bool')
    directory = Path(run_dir).resolve(strict=True)
    _require(directory.is_dir(),'EMA evaluation requires an existing local run directory')
    bodies = {}
    for name in (*STUDENT_FILES,'ema_shadow.pt'):
        path = directory/name
        _require(path.is_file(),f'Missing EMA checkpoint artifact: {name}')
        bodies[name] = path.read_bytes()
    head = torch.load(io.BytesIO(bodies['head.pt']),map_location='cpu',weights_only=True)
    package = torch.load(io.BytesIO(bodies['ema_shadow.pt']),map_location='cpu',weights_only=True)
    _require(isinstance(head,dict) and isinstance(package,dict) and set(package)=={'ema','metadata'},
             'Invalid EMA/head checkpoint container')
    state,meta = package['ema'],package['metadata']
    _require(isinstance(state,dict) and isinstance(meta,dict),'Invalid EMA state/metadata')
    _require(type(meta.get('checkpoint_schema_version')) is int and meta['checkpoint_schema_version']==1,
             'EMA checkpoint lacks versioned student-file pairing')
    hashes = meta.get('student_artifacts_sha256')
    _require(isinstance(hashes,dict) and set(hashes)==set(STUDENT_FILES),'EMA student artifact binding is incomplete')
    for name in STUDENT_FILES:
        _require(hashes[name]==_sha(bodies[name]),f'EMA/student artifact pairing mismatch: {name}')
    args = head.get('args')
    _require(isinstance(args,dict) and isinstance(meta.get('args'),dict) and meta['args']==args,
             'EMA/student complete training arguments differ')
    # Ensure the purported argument/config provenance is strict JSON metadata.
    json.dumps(args,allow_nan=False)
    _require(head.get('full_finetune') is False and head.get('special_embeddings') is False
             and head.get('weights_dtype')=='fp32','EMA evaluator only supports the fp32 adapter recipe')
    _require(head.get('init_source') is None,'EMA evaluation currently requires a run initialized from the pinned base')
    for key in ('base','base_revision','suite_sha256','dse','optimizer_steps','stopped_early'):
        _require(key in head and key in meta and head[key]==meta[key],f'EMA/student metadata differs: {key}')
    _require(isinstance(head['base'],str) and Path(head['base']).is_dir(),
             'EMA evaluation requires the existing local base directory; no model download is permitted')
    _require(head['base_revision']==PINNED_REVISION,'EMA base revision differs from the registered recipe')
    mode = meta.get('mode')
    _require(mode in ('mean_teacher','eval_only') and mode==args.get('ema_mode'),'EMA mode is absent or inconsistent')
    dse = head['dse']
    _require(isinstance(dse,dict) and dse.get('ema_mode')==mode,'EMA DSE mode differs')
    completed = _count(head['optimizer_steps'],'optimizer steps',1)
    planned = _count(meta.get('planned_optimizer_steps'),'planned optimizer steps',1)
    _require(_count(state.get('updates'),'EMA updates')==completed
             and _count(meta.get('completed_ema_updates'),'metadata EMA updates')==completed
             and _count(dse.get('planned_optimizer_steps'),'DSE horizon',1)==planned
             and completed<=planned,'EMA update/horizon counters differ')
    stopped = head['stopped_early']
    _require(type(stopped) is bool and stopped==(completed<planned),'EMA stopped_early marker is inconsistent')
    _require(allow_stopped_early or not stopped,'Refusing stopped_early EMA except for explicit diagnostic loading')
    decay = state.get('decay')
    _require(isinstance(decay,(int,float)) and not isinstance(decay,bool) and math.isfinite(decay)
             and 0<=decay<1 and decay==meta.get('decay')==args.get('ema_decay'),
             'EMA decay metadata differs')
    _require(set(state)=={'schema_version','decay','updates','names','shadow'}
             and type(state['schema_version']) is int and state['schema_version']==1,
             'Invalid EMA tensor-state schema')
    names,shadow=state['names'],state['shadow']
    _require(isinstance(names,list) and names and all(isinstance(n,str) and n for n in names)
             and names==sorted(set(names)) and isinstance(shadow,dict) and set(shadow)==set(names),
             'EMA tensor namespace invalid')
    for name,value in shadow.items():
        _require(isinstance(value,torch.Tensor) and value.dtype==torch.float32
                 and bool(torch.isfinite(value).all()),'EMA checkpoint tensor invalid')
    structure={name:{'shape':list(value.shape),'dtype':str(value.dtype)} for name,value in shadow.items()}
    _require(meta.get('parameter_structure')==structure
             and meta.get('parameter_structure_sha256')==_sha(json.dumps(structure,sort_keys=True).encode()),
             'EMA parameter structure metadata differs')
    adapter = json.loads(bodies['adapter_config.json'])
    _require(isinstance(adapter, dict), 'Invalid adapter configuration container')
    rank = _count(head.get('lora'),'LoRA rank',1)
    _require(_count(adapter.get('r'),'adapter rank',1)==rank
             and _count(adapter.get('lora_alpha'),'adapter alpha',1)==2*rank,
             'Adapter rank/alpha differs from registered architecture')
    _require(not adapter.get('trainable_token_indices'),'EMA evaluator rejects trainable token embeddings')
    evidence = {'weight_kind':'ema','student_weight_kind':'ordinary head.pt and adapter remain student',
                'checkpoint_pairing_verified':True,'run_dir':str(directory),
                'artifact_sha256':{n:_sha(b) for n,b in bodies.items()},
                'mode':mode,'decay':float(decay),'optimizer_steps':completed,
                'planned_optimizer_steps':planned,'stopped_early':stopped,'diagnostic_only':stopped,
                'base':head['base'],'base_revision':head['base_revision'],
                'suite_sha256':head['suite_sha256'],'seed':args.get('seed'),
                'training_args_sha256':_sha(json.dumps(args,sort_keys=True,allow_nan=False).encode()),
                'required_dtype':'fp32','required_merged':False,
                'runtime_base_and_environment_attested':False,'scoring_authorized_by_loader':False,
                'full_training_resume_supported':False}
    return head,package,evidence


def load_ema_for_evaluation(run_dir, device, *, allow_stopped_early=False, attn='eager'):
    """Return (tokenizer, frozen EMA model, evidence), leaving student files intact.

    The ordinary evaluator still loads the student by default. This entry point
    explicitly forces fp32 and merge=False and refuses LoRA scaling overrides.
    The caller must set and record the evaluation precision/backend policy and
    validate model/data lineage separately before producing experiment metrics.
    """
    head,package,evidence=verify_ema_checkpoint(run_dir,allow_stopped_early=allow_stopped_early)
    _require(float(os.environ.get('KEV_LORA_SCALE','1'))==1.,'EMA evaluation forbids a KEV_LORA_SCALE override')
    _require(attn in ('eager','sdpa'),'EMA evaluation requires an explicit supported attention backend')
    from .evaluate import load
    tokenizer,student=load(evidence['run_dir'],device,dtype=torch.float32,merge=False,attn=attn)
    # Detect files changed while the loader reread the student artifacts.
    for name,digest in evidence['artifact_sha256'].items():
        _require(_sha((Path(evidence['run_dir'])/name).read_bytes())==digest,
                 f'Checkpoint changed during model loading: {name}')
    ema=AdapterHeadEMA(student,package['ema']['decay'])
    ema.load_state_dict(package['ema'])
    teacher=student.make_ema_teacher(ema.names)
    ema.copy_to(teacher)
    evidence.update(_observe_model(teacher))
    evidence['attention_backend_requested']=attn
    evidence['selected_parameters']=list(ema.names)
    teacher._jev_ema_checkpoint_sha256=evidence['artifact_sha256']['ema_shadow.pt']
    # No closure to student or EMA is retained; the frozen teacher owns its
    # selected parameters and its shared frozen storage through new Parameters.
    return tokenizer,teacher,evidence


@torch.no_grad()
def predict_ema_logits(model, encodings):
    """No-grad batched logits from a model returned by the explicit EMA loader.

    Frozen parameters alone do not disable PEFT input-requires-grad hooks. Use
    this entry point (or an explicit no_grad context) for raw-logit evaluation.
    This is a model forward only, not a data-lineage or scoring authorization.
    """
    _require(isinstance(getattr(model,'_jev_ema_checkpoint_sha256',None),str),
             'Explicit EMA model-load provenance is required')
    _require(not any(p.requires_grad for p in model.parameters())
             and not any(m.training for m in model.modules()),'EMA prediction requires a frozen eval model')
    _observe_model(model)
    _require(isinstance(encodings,(list,tuple)) and bool(encodings),'EMA prediction requires nonempty encodings')
    return model.forward_batch(encodings)
