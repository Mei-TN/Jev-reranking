"""New local replica; original immutable bundle loader/full forward, no export.

Import performs no model load. Actual entry is owned by inference_model.py.
Branch envelopes are checked against a separate gold-free frozen rerank ledger.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import traceback

from paired_branch import encoded_identity, paired_probs


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--replica', required=True, type=int, choices=range(4))
    args = parser.parse_args()
    stop_e = float(os.environ['JEV_WORKER_STOP_EPOCH'])
    stop_m = float(os.environ['JEV_WORKER_STOP_MONO'])
    def gate():
        if time.time() >= stop_e or time.monotonic() >= stop_m:
            raise TimeoutError('original service/model stop')
    remaining = min(stop_e-time.time(), stop_m-time.monotonic())
    if remaining <= 0: os._exit(124)
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, remaining)  # Original supplied bound, never reset.
    plan = json.loads(args.plan.read_bytes())
    if plan['schema'] != 'jev-review-inference-single-model-actual-plan-v1' or plan['admitted'] is not True:
        raise ValueError('new actual inference plan required')
    if plan['kind'] not in ('external', 'branch'):
        raise ValueError('unknown new inference scope')
    bundle = Path(plan['bundle']).resolve()
    if sha(bundle/'model_manifest.json') != plan['bundle_manifest_sha256']:
        raise ValueError('original bundle manifest differs')
    sys.path.insert(0, str(bundle/'code'))
    contract = importlib.import_module('runtime.bundle_contract')
    loader = importlib.import_module('runtime.load_bundle')
    for module in (contract, loader):
        if not Path(module.__file__).resolve().is_relative_to(bundle/'code/runtime'):
            raise ValueError('foreign bundle loader imported')
    expected = contract.config_identity(plan['head_config'])
    manifest = contract.validate_manifest(bundle, expected, verify_files=False)
    if manifest['name'] != plan['model_id']:
        raise ValueError('bundle logical model differs')
    if manifest['files'] != plan['bundle_member_sha256']:
        raise ValueError('root member witness differs from actual manifest')
    # load_bundle verifies original members and validation_report; no model tensor edit.
    tok, model = loader.load_bundle(bundle, 'cuda', expected)
    from kev.api import SystemOneRequest, to_record
    import torch
    torch.set_num_threads(2)
    if plan['kind'] == 'branch' and expected['kind'] != 'residual_pair_mlp':
        raise ValueError('branch arm requires unchanged H4 residual head')
    ledger = {}
    if plan['kind'] == 'branch':
        gate()
        path = Path(plan['branch_requests'])
        if sha(path) != plan['branch_requests_sha256']:
            raise ValueError('frozen gold-free rerank ledger differs')
        with path.open('rb') as f:
            for ordinal, raw in enumerate(f):
                row = json.loads(raw)
                if (set(row) != {'ordinal', 'task', 'episode_id', 'candidate_id',
                                 'request_id', 'role', 'payload', 'payload_sha256',
                                 'source_planned_ordinal','source_planned_request_sha256'}
                        or row['ordinal'] != ordinal or row['role'] != 'rerank'
                        or row['request_id'] in ledger):
                    raise ValueError('branch ledger occurrence/type differs')
                body = row['payload']
                if body['model'] != '__MODEL_ID__' or set(body) != {'model', 'state', 'questions'}:
                    raise ValueError('branch ledger payload contains extra/gold fields')
                if hashlib.sha256(wire(body)).hexdigest() != row['payload_sha256']:
                    raise ValueError('ordered branch payload hash differs')
                ledger[row['request_id']] = (ordinal, row['payload_sha256'])
        if len(ledger) != 191276:
            raise ValueError('branch full rerank denominator differs')
    gate()
    lock = threading.Lock(); log_lock = threading.Lock(); state_lock=threading.Lock()
    failed = threading.Event(); consumed = set()
    active_handlers=0;completed_responses=0;memory_first_failure=None
    teardown_requested=False;teardown_ack_flushed=False;shutdown_thread=None
    expected_completed=([3571,3570,3570,3570] if plan['kind']=='external' else [47819]*4)[args.replica]
    log_dir = Path(plan['output'])/('service'+str(args.replica))
    log_dir.mkdir(exist_ok=False)
    # Establish the journal entry before the ready/config directory fsync.
    with (log_dir/'requests.jsonl').open('xb') as f:f.flush();os.fsync(f.fileno())
    identity = dict(model=plan['model_id'],cohort=plan['cohort'],probability_source='classification_head',
                    head_kind=expected['kind'], head_config=expected)
    config = dict(**identity, kind=plan['kind'], replica=args.replica,
                  bundle=str(bundle), manifest_sha256=sha(bundle/'model_manifest.json'),
                  head_sha256=sha(bundle/'classification_head.safetensors'),
                  loader_source_sha256=sha(loader.__file__),
                  head_source_sha256=sha(bundle/'code/kev/heads.py'),
                  tokenizer_member_sha256={k:v for k,v in manifest['files'].items() if k.startswith('tokenizer/')},
                  backbone_dtype=str(next(model.lm.parameters()).dtype),
                  head_dtype=str(next(model.head.parameters()).dtype),
                  temperature=1, lora_scale=1, prefix_cache=False,
                  strict=True, max_state=8192, max_branch=8192, max_packed=8192,
                  option_isolation=model.option_isolation,
                  actual_runtime=dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,
                                      effective_device=str(model.device),device_name=torch.cuda.get_device_name(0),
                                      torch_module=str(Path(torch.__file__).resolve()),
                                      matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                      cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                      flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
                                      memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
                                      math_sdp=torch.backends.cuda.math_sdp_enabled(),
                                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                      CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),
                  port=plan['ports'][args.replica],gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                  physical_gpu_index=plan['gpu'],
                  service_source_sha256=sha(__file__), paired_source_sha256=sha(Path(__file__).with_name('paired_branch.py')))
    if config['backbone_dtype'] != 'torch.bfloat16' or config['head_dtype'] != 'torch.float32' or config['option_isolation'] is not False:
        raise ValueError('unchanged inference dtype/layout differs')
    def publish(name, value):
        gate()
        with (log_dir/name).open('xb') as f:
            f.write(wire(value)+b'\n'); f.flush(); os.fsync(f.fileno())
        fd = os.open(log_dir, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        gate()
    publish('effective_config.json', config)
    def failure_record(exc,operation,request_id=None,body=None):
        nonlocal memory_first_failure
        # Memory is authoritative for the settled handshake even when error
        # publication itself fails; a missing file never means no first error.
        with state_lock:
            if memory_first_failure is None:
                memory_first_failure=dict(type=type(exc).__name__,message=str(exc),
                                          operation=operation,request_id=request_id,
                                          epoch=time.time(),monotonic=time.monotonic())
            failed.set();first=dict(memory_first_failure)
        with log_lock:
            if not (log_dir/'FIRST_FAILURE.json').exists():
                try:publish('FIRST_FAILURE.json',dict(first=first,request=body,traceback=traceback.format_exc()))
                except BaseException:pass
    def service_state():
        with state_lock:
            return dict(active_handlers=active_handlers,completed_responses=completed_responses,
                        first_failure=memory_first_failure,
                        settled=active_handlers==0 and memory_first_failure is None)

    def teardown(handler):
        nonlocal teardown_requested,teardown_ack_flushed,shutdown_thread
        gate();body=None
        try:
            length=int(handler.headers['Content-Length'])
            if not 0<length<=1024:raise ValueError('bounded teardown control length required')
            raw=handler.rfile.read(length)
            if len(raw)!=length:raise ValueError('incomplete teardown control request')
            body=json.loads(raw)
            wanted=dict(model=plan['model_id'],replica=args.replica,expected_completed_responses=expected_completed)
            if body!=wanted:raise ValueError('fixed teardown control identity/count differs')
            with state_lock:
                if teardown_requested:raise ValueError('teardown control already consumed')
                teardown_requested=True
            ack=dict(teardown_ack=True,model=plan['model_id'],replica=args.replica,
                     expected_completed_responses=expected_completed,
                     service_source_sha256=config['service_source_sha256'])
            handler.reply(200,ack)
            publish('TEARDOWN_CONTROL_ACK.json',dict(request=body,ack=ack,response_write_flush_complete=True,
                                                    epoch=time.time(),monotonic=time.monotonic()))
            with state_lock:teardown_ack_flushed=True
        except BaseException as exc:
            failure_record(exc,'teardown',body=body)
            raise
        finally:
            # shutdown must run outside both serve_forever and its request handler.
            # server_close below joins every non-daemon GET/POST handler before
            # terminal memory is sampled or natural native0 is possible.
            with state_lock:
                if shutdown_thread is None:
                    shutdown_thread=threading.Thread(target=server.shutdown,name='original-teardown',daemon=False)
                    shutdown_thread.start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def reply(self, code, value):
            gate(); raw = wire(value)
            self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw))); self.end_headers()
            self.wfile.write(raw); self.wfile.flush(); gate()
        def do_GET(self):
            try:
                gate()
                self.reply(200,dict(config,service_state=service_state())) if self.path=='/health' else self.reply(404,{'error':'unregistered path'})
            except BaseException as exc:failure_record(exc,'health')
        def do_POST(self):
            nonlocal active_handlers,completed_responses
            if self.path=='/v1/teardown':
                try:teardown(self)
                except BaseException:pass  # first failure is authoritative; terminal exit is nonzero.
                return
            began = time.time(); body = None; request_id = None
            with state_lock:active_handlers+=1
            try:
                gate()
                if failed.is_set(): raise RuntimeError('original first service failure stop')
                with state_lock:
                    if teardown_requested:raise RuntimeError('scientific request after teardown control')
                path = '/v1/decision' if plan['kind'] == 'external' else '/v1/paired-rerank'
                if self.path != path: raise ValueError('unregistered inference role')
                length = int(self.headers['Content-Length'])
                if not 0 < length <= 1 << 20: raise ValueError('bounded input length required')
                raw = self.rfile.read(length)
                if len(raw) != length: raise ValueError('incomplete actual HTTP request')
                incoming = json.loads(raw)
                if plan['kind'] == 'branch':
                    if set(incoming) != {'request_id', 'ordinal', 'payload'}:
                        raise ValueError('branch transport envelope differs')
                    request_id = incoming['request_id']; body = incoming['payload']
                    normalized = dict(body, model='__MODEL_ID__')
                    item = ledger[request_id]
                    if (incoming['ordinal'] != item[0] or item[0] % 4 != args.replica
                            or hashlib.sha256(wire(normalized)).hexdigest() != item[1]):
                        raise ValueError('actual ordered rerank occurrence differs')
                    with log_lock:
                        if request_id in consumed: raise ValueError('a rerank occurrence was dispatched twice')
                        consumed.add(request_id)
                else:
                    body = incoming
                if (set(body) != {'model', 'state', 'questions'} or body['model'] != plan['model_id']
                        or any(set(q) != {'type', 'instructions', 'criteria'} for q in body['questions'].values())):
                    raise ValueError('gold/metadata or actual model input differs')
                rec, metadata = to_record(SystemOneRequest.model_validate(body))
                enc = model.encode(tok, rec, strict=True, max_state=8192, max_branch=8192)
                if len(enc['ids']) > 8192: raise ValueError('packed encoding exceeds original cap')
                tick = time.monotonic()
                if plan['kind'] == 'branch':
                    if any(m['type'] != 'choice' or set(m['keys']) != {'entailment','neutral','contradiction'} for m in metadata):
                        raise ValueError('branch permits original rerank NLI only')
                    paired = paired_probs(model, enc, lock)
                    result = dict(**identity, probabilities={}, branch_zeroed={}, capture=paired,
                                  input_tokens=len(enc['ids']))
                    for i, meta in enumerate(metadata):
                        for target, arm in (('probabilities','full'),('branch_zeroed','branch_zeroed')):
                            result[target][meta['id']] = dict(zip(meta['keys'], paired[arm][i]))
                else:
                    with lock, torch.inference_mode():
                        values = model.probs(enc)
                    if len(values) != len(metadata): raise ValueError('original question occurrence differs')
                    result = dict(**identity, probabilities={}, input_tokens=len(enc['ids']),
                                  encoding=encoded_identity(enc))
                    for probs, meta in zip(values, metadata):
                        keys = meta['keys'] if meta['type']=='choice' else ['false','true'] if meta['type']=='noul' else list(meta['legend'])
                        if len(keys) != len(probs): raise ValueError('original option occurrence differs')
                        result['probabilities'][meta['id']] = dict(zip(keys, probs.tolist()))
                contract.validate_probabilities(result, body); gate()
                result.update(manifest_sha256=config['manifest_sha256'], head_sha256=config['head_sha256'],
                              service_source_sha256=config['service_source_sha256'], forward_seconds=time.monotonic()-tick)
                event = dict(request_id=request_id, request_wire_sha256=hashlib.sha256(raw).hexdigest(),
                             response_sha256=hashlib.sha256(wire(result)).hexdigest(),
                             started_epoch=began, finished_epoch=time.time(), input_tokens=len(enc['ids']),
                             encoding_sha256=encoded_identity(enc)['sha256'])
                self.reply(200, result)
                # A successful response is settled only after the real
                # write/flush AND durable journal, never at client reception.
                event['response_write_flush_complete']=True
                with log_lock,(log_dir/'requests.jsonl').open('ab') as f:
                    gate();f.write(wire(event)+b'\n');f.flush();os.fsync(f.fileno());gate()
                with state_lock:completed_responses+=1
            except BaseException as exc:
                failure_record(exc,'decision',request_id,body)
                try: self.reply(500, dict(**identity, error=repr(exc)))
                except BaseException: pass
            finally:
                with state_lock:active_handlers-=1

    gate()
    server = ThreadingHTTPServer(('127.0.0.1', plan['ports'][args.replica]), Handler)
    server.daemon_threads = False
    server.block_on_close = True
    try:
        print(json.dumps(dict(ready=True, **config)), flush=True); gate()
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close()
    if shutdown_thread is not None:
        gate();shutdown_thread.join(timeout=min(stop_e-time.time(),stop_m-time.monotonic()));gate()
        if shutdown_thread.is_alive():raise RuntimeError('original shutdown thread not drained')
    gate();terminal=service_state()
    terminal.update(schema='jev-review-inference-service-terminal-state-v1',model=plan['model_id'],
        kind=plan['kind'],cohort=plan['cohort'],replica=args.replica,
        expected_completed_responses=expected_completed,teardown_requested=teardown_requested,
        teardown_control_ack_write_flush_complete=teardown_ack_flushed,
        all_handler_threads_joined=True,service_source_sha256=config['service_source_sha256'],
        owner=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
                   host=socket.gethostname(),
                   boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()),
        finished_epoch=time.time(),finished_monotonic=time.monotonic())
    complete=(terminal['active_handlers']==0 and terminal['completed_responses']==expected_completed
              and terminal['first_failure'] is None and teardown_requested and teardown_ack_flushed)
    terminal['requested_exit_native']=0 if complete else 1
    publish('TERMINAL_STATE.json',terminal);gate()
    if not complete:raise RuntimeError('original service terminal count/failure/control veto')


if __name__ == '__main__': main()
