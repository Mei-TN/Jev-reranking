"""Export a Kev LoRA run as a NeoHorse merged-BF16 bundle.

Load with KEV_MERGE=1 and KEV_LORA_SCALE=1, merge LoRA weights in FP32,
then cast the backbone to BF16 with TF32 disabled. Save backbone shards of up
to 4 GB, the tokenizer, and the FP32 head as pointer_head.safetensors.
Remap the recorded base-model path to a verified local copy of the same revision.
Compare saved verification requests before export and after offline reload
with an absolute probability tolerance of 1e-5.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='Kev LoRA run (adapter_model.safetensors, adapter_config.json, head.pt)')
    p.add_argument('--out', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--base', required=True, help='local copy of the base model recorded in head.pt')
    p.add_argument('--base-revision', required=True)
    p.add_argument('--kev-runtime', required=True, help='directory containing the authoritative kev package')
    p.add_argument('--template-bundle', required=True, help='official bundle providing load_merged.py etc.')
    a = p.parse_args()
    run, out, tmpl = Path(a.run), Path(a.out), Path(a.template_bundle)
    assert not out.exists(), 'Never overwrite an earlier export'
    os.environ.update(KEV_MERGE='1', KEV_LORA_SCALE='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, a.kev_runtime)
    import torch
    from safetensors.torch import save_file
    import kev.evaluate as kev_eval
    from peft import PeftModel

    recorded = torch.load(run / 'head.pt', map_location='cpu')
    if recorded.get('base_revision') != a.base_revision:
        raise SystemExit(f"base revision mismatch: {recorded.get('base_revision')} != {a.base_revision}")
    original_load = torch.load

    def load_with_local_base(path, *args, **kwargs):
        obj = original_load(path, *args, **kwargs)
        if Path(str(path)).resolve() == (run / 'head.pt').resolve():
            obj = dict(obj)
            obj['base'] = a.base
        return obj

    torch.load = load_with_local_base
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    out.mkdir(parents=True)
    print('Loading and merging the LoRA run in FP32, then casting backbone to BF16', flush=True)
    t0 = time.time()
    tok, model = kev_eval.load(str(run), 'cuda', dtype=torch.bfloat16, merge=True)
    torch.load = original_load
    assert not isinstance(model.lm, PeftModel)
    assert not any('lora_' in n for n, _ in model.lm.named_parameters())
    assert next(model.lm.parameters()).dtype == torch.bfloat16
    assert next(model.head.parameters()).dtype == torch.float32
    print(f'merged in {time.time() - t0:.1f}s; saving backbone, head, tokenizer', flush=True)
    model.lm.save_pretrained(out / 'backbone', safe_serialization=True, max_shard_size='4GB')
    tok.save_pretrained(out / 'tokenizer')
    save_file({k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()},
              str(out / 'pointer_head.safetensors'))
    (out / 'runtime').mkdir()
    for name in ['__init__.py', 'model.py', 'api.py']:
        shutil.copy2(Path(a.kev_runtime) / 'kev' / name, out / 'runtime' / name)
    for name in ['load_merged.py', 'verify_bundle.py']:
        shutil.copy2(tmpl / name, out / name)
    shutil.copy2(Path(__file__), out / 'export_merged_bundle.py')
    versions = {}
    for name in ['torch', 'transformers', 'safetensors', 'pydantic', 'peft', 'triton',
                 'flash-linear-attention', 'tokenizers', 'huggingface-hub', 'einops']:
        versions[name] = importlib.metadata.version(name)
    dump(out / 'environment.json', dict(packages=versions, python=sys.version, cuda=torch.version.cuda,
                                        gpu=torch.cuda.get_device_name(0)))
    manifest = dict(name=a.name, source_checkpoint=str(run),
                    format='merged text backbone + independent pointer head; NOT chat causal LM',
                    backbone_class=type(model.lm).__name__, backbone_dtype='bfloat16', head_dtype='float32',
                    merge_dtype='float32', lora_scale=1.0, head_dim=model.head.q.out_features,
                    option_isolation=model.option_isolation, hybrid=model.hybrid, temperature=1.0,
                    runtime_origin='https://github.com/jaredpalmer/kev',
                    recorded_base=recorded.get('base'), local_base=a.base, base_revision=a.base_revision,
                    source_hashes={name: digest(run / name) for name in
                                   ['head.pt', 'adapter_model.safetensors', 'adapter_config.json']} |
                                  {f'kev/{name}': digest(Path(a.kev_runtime) / 'kev' / name) for name in ['model.py', 'api.py', 'evaluate.py']},
                    exporter='export_merged_bundle.py (follows official export_bundle.py)',
                    created_unix=time.time())
    dump(out / 'model_manifest.json', manifest)
    sys.path.insert(0, str(out))
    from load_merged import load_bundle, predict
    examples = []
    for e in json.loads((tmpl / 'verification_examples.json').read_text()):
        examples.append(dict(id=e['id'], request=e['request'], expected=predict(tok, model, e['request'])))
    shutil.copy2(tmpl / 'example_request.json', out / 'example_request.json')
    dump(out / 'verification_examples.json', examples)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    print('Offline reload from exported folder only', flush=True)
    tok, model = load_bundle(out)
    max_delta = 0.0
    for e in examples:
        new = predict(tok, model, e['request'])
        for q, old in e['expected']['answers'].items():
            if 'choice' in old:
                assert old['choice'] == new['answers'][q]['choice'], e['id']
            for k, v in old['probabilities'].items():
                delta = abs(new['answers'][q]['probabilities'][k] - v)
                max_delta = max(max_delta, delta)
                assert delta <= 1e-5, (e['id'], delta)
    report = dict(status='PASS', requests=len(examples), max_probability_delta=max_delta, offline_reload=True,
                  checks=['merged backbone, no LoRA modules', 'BF16 backbone, FP32 head',
                          'strict standalone reload without missing/unexpected keys',
                          'official verification requests re-scored after reload'])
    for name, sha in manifest['source_hashes'].items():
        src = run / name if not name.startswith('kev/') else Path(a.kev_runtime) / name
        assert digest(src) == sha, name
    dump(out / 'validation_report.json', report)
    files = sorted(q for q in out.rglob('*') if q.is_file() and '__pycache__' not in q.parts)
    (out / 'SHA256SUMS').write_text(''.join(f'{digest(q)}  {q.relative_to(out)}\n' for q in files))
    print(json.dumps(dict(output=str(out), validation=report,
                          total_bytes=sum(q.stat().st_size for q in files)), indent=2), flush=True)


if __name__ == '__main__':
    main()
