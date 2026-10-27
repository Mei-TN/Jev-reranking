"""Standalone loader using the bundled DecisionModel and its actual head factory."""
import json
from pathlib import Path
import sys
from .bundle_contract import config_identity, validate_manifest


def load_bundle(bundle, device='cuda', expected_config=None, *, require_validation=True):
    root = Path(bundle).resolve()
    meta = validate_manifest(root, expected_config, require_validation=require_validation)
    code = root / 'code'
    if Path(__file__).resolve().parent != code/'runtime':
        raise ValueError('load_bundle must run from this bundle/code runtime in a fresh interpreter')
    # Never silently resolve a training/official runtime already imported elsewhere.
    for name, module in list(sys.modules.items()):
        if name == 'kev' or name.startswith('kev.'):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(code / 'kev'):
                raise ValueError('foreign kev runtime already imported: ' + str(path))
    sys.path.insert(0, str(code))
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer
    from kev.model import DecisionModel
    from kev.heads import make_head, normalize_head_config
    c = normalize_head_config(meta['head_config'])
    if config_identity(c) != meta['head_config']:
        raise ValueError('head factory/config normalization disagreement')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    tok = AutoTokenizer.from_pretrained(root / 'tokenizer', local_files_only=True)
    backbone, info = AutoModel.from_pretrained(root / 'backbone', dtype=torch.bfloat16,
        attn_implementation='sdpa' if str(device).startswith('cuda') else 'eager',
        local_files_only=True, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('incomplete backbone reload: ' + repr(info))
    if type(backbone).__name__ != meta['backbone_class']:
        raise ValueError('backbone class mismatch')
    # Same constructor bypass as the official merged loader; no original base reload.
    model = DecisionModel.__new__(DecisionModel)
    torch.nn.Module.__init__(model)
    model.lm = backbone.to(device=device, dtype=torch.bfloat16)
    model.head_config = c
    model.head = make_head(backbone.config.hidden_size, head_dim=c['dim'], head_config=c)
    model.head.load_state_dict(load_file(str(root / 'classification_head.safetensors')), strict=True)
    model.head = model.head.to(device=device, dtype=torch.float32)
    model.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    model.hybrid = 'linear_attention' in set(getattr(backbone.config, 'layer_types', None) or [])
    model.option_isolation = meta['option_isolation']
    model.device = device
    if model.hybrid != meta['hybrid'] or (model.hybrid and model.option_isolation):
        raise ValueError('invalid model layout')
    model.eval()
    return tok, model


def predict(tok, model, request, max_state=2048, max_branch=8192):
    import torch
    from kev.api import SystemOneRequest, to_record
    rec, metadata = to_record(SystemOneRequest.model_validate(request))
    enc = model.encode(tok, rec, strict=True, max_state=max_state, max_branch=max_branch)
    with torch.inference_mode():
        probabilities = [p.tolist() for p in model.probs(enc)]
    answers = {}
    for p, m in zip(probabilities, metadata):
        if m['type'] == 'choice':
            answers[m['id']] = dict(type='choice', choice=m['keys'][max(range(len(p)), key=p.__getitem__)], probabilities=dict(zip(m['keys'], p)))
        elif m['type'] == 'noul':
            answers[m['id']] = dict(type='noul', noul=p[1], probabilities={'false': p[0], 'true': p[1]})
        else:
            answers[m['id']] = dict(type='score', score=sum(i*v for i,v in enumerate(p)), legend=m['legend'], probabilities={str(i):v for i,v in enumerate(p)})
    return dict(answers=answers, input_tokens=len(enc['ids']))
