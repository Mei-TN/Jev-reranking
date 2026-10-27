"""Isolated text-only native decision endpoint; no generation or calibration fit."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from .bundle_contract import digest, read_config, validate_manifest, validate_probabilities
from .load_bundle import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--head-config', required=True, type=Path)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--log-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-state', type=int, default=8192)
    parser.add_argument('--max-branch', type=int, default=8192)
    parser.add_argument('--max-packed', type=int, default=8192)
    args = parser.parse_args()
    if min(args.max_state, args.max_branch, args.max_packed) < 1:
        raise ValueError('token limits must be positive')
    expected = read_config(args.head_config)
    manifest = validate_manifest(args.bundle, expected)
    if manifest.get('name') != args.model_id:
        raise ValueError('service model-id must match exported bundle name')
    import torch
    tok, model = load_bundle(args.bundle, args.device, expected)
    from kev.api import SystemOneRequest, to_record
    torch.set_num_threads(2)
    args.log_dir.mkdir(parents=True, exist_ok=False)
    lock = threading.Lock()
    identity = dict(model=args.model_id, probability_source='classification_head',
                    head_kind=expected['kind'], head_config=expected)
    config = dict(**identity, checkpoint=str(args.bundle.resolve()), port=args.port,
                  gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), tp=1,
                  dtype=str(next(model.lm.parameters()).dtype),
                  head_dtype=str(next(model.head.parameters()).dtype),
                  merge='premerged_fp32_then_bf16', lora_scale=1, temperature=1,
                  bundle_loader='runtime.load_bundle.load_bundle', prefix_cache=False,
                  max_state=args.max_state, max_branch=args.max_branch,
                  max_packed=args.max_packed, strict=True, supports_vision=False,
                  head_sha256=digest(args.bundle/'classification_head.safetensors'),
                  manifest_sha256=digest(args.bundle/'model_manifest.json'))
    (args.log_dir/'effective_config.json').write_text(json.dumps(config, indent=2)+'\n')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload):
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self.reply(200, config) if self.path in ('/health', '/v1/models') else self.reply(404, {})

        def do_POST(self):
            if self.path not in ('/v1/decision', '/v1/validate'):
                self.reply(404, {}); return
            started = time.time()
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if not isinstance(body, dict) or body.get('model') != args.model_id:
                    raise ValueError('Model identity mismatch')
                if set(body)-{'model', 'state', 'questions', 'image'}:
                    raise ValueError('Unexpected request fields')
                if 'image' in body:
                    raise ValueError('This text-only ablation endpoint does not support images')
                for q in body['questions'].values():
                    if set(q)-{'type', 'instructions', 'criteria'}:
                        raise ValueError('Labels/metadata are not model inputs')
                rec, metadata = to_record(SystemOneRequest.model_validate(body))
                enc = model.encode(tok, rec, max_state=args.max_state,
                                   max_branch=args.max_branch, strict=True)
                if len(enc['ids']) > args.max_packed:
                    raise ValueError(f'packed request exceeds {args.max_packed}-token limit')
                lengths = dict(input_tokens=len(enc['ids']), state_tokens=enc['seg'].count(0))
                if self.path == '/v1/validate':
                    self.reply(200, dict(**identity, eligible=True, **lengths)); return
                with lock, torch.inference_mode():
                    if str(args.device).startswith('cuda'): torch.cuda.synchronize()
                    tick = time.perf_counter()
                    probabilities = model.probs(enc)
                    if str(args.device).startswith('cuda'): torch.cuda.synchronize()
                    seconds = time.perf_counter()-tick
                if len(probabilities) != len(metadata):
                    raise RuntimeError('Model question count mismatch')
                result = dict(**identity, probabilities={}, forward_seconds=seconds, **lengths)
                for p, meta in zip(probabilities, metadata):
                    keys = (meta['keys'] if meta['type']=='choice' else ['false','true']
                            if meta['type']=='noul' else list(meta['legend']))
                    values = p.tolist()
                    if len(keys) != len(values): raise RuntimeError('Model option count mismatch')
                    result['probabilities'][meta['id']] = dict(zip(keys, values))
                validate_probabilities(result, body)
                event = dict(started_at=started, request=body, response=result,
                             elapsed_s=time.time()-started)
                with lock, (args.log_dir/'requests.jsonl').open('a') as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
                self.reply(200, result)
            except (ValueError, KeyError, TypeError) as error:
                self.reply(422, dict(error=str(error), **identity))
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.reply(500, dict(error=str(error), **identity))

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(json.dumps(dict(ready=True, **config)), flush=True)
    server.serve_forever()


if __name__ == '__main__': main()
