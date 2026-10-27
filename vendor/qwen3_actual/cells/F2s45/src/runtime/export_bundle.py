"""Export the actual new kev.evaluate.load path; never borrow a fixed PointerHead loader."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .bundle_contract import SCHEMA, config_identity, digest, read_config
from .load_bundle import predict


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--name', required=True)
    p.add_argument('--base', required=True, type=Path)
    p.add_argument('--base-revision', required=True)
    p.add_argument('--head-config', required=True, type=Path)
    p.add_argument('--verification-examples', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    sys.dont_write_bytecode = True
    source = Path(__file__).resolve().parents[1]
    run, out = a.run.resolve(), a.out.resolve()
    expected = read_config(a.head_config)
    if out.exists(): raise ValueError('never overwrite an earlier export')
    if not a.base.is_dir(): raise ValueError('base must be a local directory')
    os.environ.update(KEV_MERGE='1', KEV_LORA_SCALE='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, str(source))
    import torch
    from safetensors.torch import save_file
    import kev.evaluate as kev_eval
    from kev.heads import normalize_head_config
    from peft import PeftModel
    if Path(kev_eval.__file__).resolve().parent != source / 'kev':
        raise ValueError('wrong training/evaluation package imported')
    recorded = torch.load(run / 'head.pt', map_location='cpu', weights_only=False)
    actual = config_identity(normalize_head_config(recorded.get('head_config'), head_dim=recorded.get('head_dim', 256)))
    if actual != expected: raise ValueError('checkpoint is not the requested head variant')
    if recorded.get('base_revision') != a.base_revision: raise ValueError('base revision mismatch')
    if recorded.get('weights_dtype') == 'bf16' or recorded.get('full_finetune'):
        raise ValueError('this exporter is scoped to FP32 LoRA merge checkpoints')
    checkpoint_names = ['head.pt', 'adapter_model.safetensors', 'adapter_config.json']
    source_hashes = {str(path.relative_to(source)): digest(path) for package in ['kev', 'runtime'] for path in sorted((source/package).glob('*.py'))}
    checkpoint_hashes = {name: digest(run/name) for name in checkpoint_names}
    original_load = torch.load
    def load_with_local_base(path, *args, **kwargs):
        value = original_load(path, *args, **kwargs)
        if isinstance(path, (str, os.PathLike)) and Path(path).resolve() == run / 'head.pt':
            value = dict(value); value['base'] = str(a.base.resolve())
        return value
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    out.mkdir(parents=True, exist_ok=False)
    torch.load = load_with_local_base
    try:
        tok, model = kev_eval.load(str(run), a.device, dtype=torch.bfloat16, merge=True)
    finally:
        torch.load = original_load
    if isinstance(model.lm, PeftModel) or any('lora_' in name for name,_ in model.lm.named_parameters()):
        raise ValueError('LoRA was not completely merged')
    if config_identity(model.head_config) != expected: raise ValueError('loaded head config mismatch')
    if any(p.dtype != torch.bfloat16 for p in model.lm.parameters()) or any(p.dtype != torch.float32 for p in model.head.parameters()):
        raise ValueError('wrong backbone/head precision after merge')
    model.lm.save_pretrained(out/'backbone', safe_serialization=True, max_shard_size='4GB')
    tok.save_pretrained(out/'tokenizer')
    save_file({k:v.detach().cpu().contiguous() for k,v in model.head.state_dict().items()}, str(out/'classification_head.safetensors'))
    dump(out/'head_config.json', expected)
    for package in ('kev', 'runtime'):
        target = out/'code'/package; target.mkdir(parents=True)
        for path in sorted((source/package).glob('*.py')): shutil.copyfile(path, target/path.name)
    examples = json.loads(a.verification_examples.read_text())
    if not examples: raise ValueError('verification examples are empty')
    examples = [dict(id=e['id'], request=e['request'], expected=predict(tok, model, e['request'])) for e in examples]
    dump(out/'verification_examples.json', examples)
    packages = {}
    for name in ('torch', 'transformers', 'safetensors', 'peft', 'pydantic', 'tokenizers', 'triton'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    dump(out/'environment.json', dict(packages=packages, python=sys.version, cuda=torch.version.cuda))
    manifest = dict(schema=SCHEMA, name=a.name, probability_source='classification_head', head_kind=expected['kind'], head_config=expected,
        source_checkpoint=str(run), format='merged text backbone + classification head; not a chat causal LM',
        backbone_class=type(model.lm).__name__, backbone_dtype='bfloat16', head_dtype='float32', merge_dtype='float32',
        option_isolation=model.option_isolation, hybrid=model.hybrid, temperature=1.0, lora_scale=1.0,
        recorded_base=recorded.get('base'), local_base=str(a.base.resolve()), base_revision=a.base_revision,
        source_hashes=source_hashes, checkpoint_hashes=checkpoint_hashes, created_unix=time.time(),
        verification_input_sha256=digest(a.verification_examples),
        files={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*')) if path.is_file()})
    dump(out/'model_manifest.json', manifest)
    del model, tok
    gc.collect()
    if str(a.device).startswith('cuda'): torch.cuda.empty_cache()
    # Fresh interpreter: bundled source plus the explicitly verified role overlay.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, (str(out/'code'), os.environ.get('JEV_VERIFIED_RUNTIME_OVERLAY', '')))), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    completed = subprocess.run([sys.executable, '-m', 'runtime.verify_bundle', '--bundle', str(out), '--device', a.device], env=env, cwd=out, check=False)
    if completed.returncode != 0: raise SystemExit(completed.returncode)
    for name, sha in source_hashes.items():
        if digest(source/name) != sha: raise ValueError('source changed during export: '+name)
    for name, sha in checkpoint_hashes.items():
        if digest(run/name) != sha: raise ValueError('checkpoint changed during export: '+name)
    report = json.loads((out/'reload_verification.json').read_text())
    report.update(source_unchanged=True, checkpoint_unchanged=True)
    dump(out/'validation_report.json', report)
    with (out/'SHA256SUMS').open('x') as stream:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name != 'SHA256SUMS' and '__pycache__' not in path.parts:
                stream.write(f'{digest(path)}  {path.relative_to(out)}\n')
    print(json.dumps(dict(output=str(out), head_config=expected, validation='PASS')), flush=True)


if __name__ == '__main__': main()
