"""Read a fixed evaluation population on CPU.

Capture the native eval_mc request and scorer inputs before model prediction.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

from branch_heads_eval import COUNTS
from external_heads_eval import wire

NATIVE_SHA='0217e9565a033ddb85cb725ce9cef707fe41d4ce6edb38052340b13601514380'
BRIDGE_SHA='ed03ce2a16e665cb4f4674f9d36b124744587a0c4e27af8636f988b48285aa49'
POPULATION_SHA='369ba7b5f7b8184d168140494c9f1df255da9a223314602fb5c35a547e27051f'
POINTER_SHA='17de4110a8068501d96215fb08d1c3bb57fcc6c53cdd7b3c74dbecfb71d9b97e'


class CaptureStop(Exception):pass


def main():
    entry_e,entry_m=time.time(),time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,300)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True,type=Path)
    a=parser.parse_args();plan_raw=a.plan.read_bytes();plan=json.loads(plan_raw)
    if (plan['schema']!='jev-review-branch-population-build-actual-plan-v1'
            or plan['admitted'] is not True or plan['max_seconds']!=300
            or tuple(plan['task_order'])!=tuple(COUNTS)
            or tuple(plan['model_ids'])!=('SUPP_WINH4_H4_f2_s44','SUPP_WINH4_H4_f2_s45',
                                         'SUPP_WINH4_H4_f3_s44','SUPP_WINH4_H4_f3_s45')):
        raise ValueError('new fixed native population plan required')
    stop_e=min(entry_e+300,plan['root_deadline_epoch']);stop_m=entry_m+stop_e-entry_e
    def gate():
        if time.time()>=stop_e or time.monotonic()>=stop_m:raise TimeoutError('original300 population stop')
    signal.setitimer(signal.ITIMER_REAL,min(stop_e-time.time(),stop_m-time.monotonic()))
    total_bytes=0
    def checked(path,expected):
        nonlocal total_bytes
        gate();path=Path(path);before=path.stat()
        if before.st_size!=expected['bytes'] or before.st_size>2<<30:
            raise ValueError('finite immutable source size differs: '+str(path))
        h=hashlib.sha256();n=0
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''):
                gate();n+=len(chunk);total_bytes+=len(chunk)
                if n>expected['bytes'] or total_bytes>8<<30:raise ValueError('original source byte cap reached')
                h.update(chunk)
        after=path.stat()
        fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if fields(before)!=fields(after) or n!=expected['bytes'] or h.hexdigest()!=expected['sha256']:
            raise ValueError('immutable source differs: '+str(path))
        return dict(bytes=n,sha256=h.hexdigest(),generation=list(fields(after)))
    out=Path(plan['output']);gate();out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        gate()
        with (out/name).open('xb') as f:f.write(wire(value)+b'\n');f.flush();os.fsync(f.fileno())
        fd=os.open(out,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        gate()
    save('ENTRY_ONCE.json',dict(plan_sha256=hashlib.sha256(plan_raw).hexdigest(),started_epoch=entry_e,
                               started_monotonic=entry_m,stop_epoch=stop_e,stop_monotonic=stop_m,retry=0))
    source_pre={p:checked(p,w) for p,w in plan['source_pins'].items()}
    source=Path(plan['rw_root'])/'src';sys.path.insert(0,str(source))
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      RUNWEAVER_OPENJEV_CACHE=plan['runtime_cache'])
    from agentic_evaluate.benchmarks import openjev_bridge as bridge,openjev_pointer as pointer
    import importlib.util
    spec=importlib.util.spec_from_file_location('new_branch_population',plan['population_source'])
    population=importlib.util.module_from_spec(spec);spec.loader.exec_module(population)
    if (source_pre[str(Path(bridge.__file__))]['sha256']!=BRIDGE_SHA
            or source_pre[plan['population_source']]['sha256']!=POPULATION_SHA
            or source_pre[str(Path(pointer.__file__))]['sha256']!=POINTER_SHA):
        raise ValueError('original bridge/pointer/population source not bound')
    total_q=0;total_r=0;tasks={};all_request_sha=hashlib.sha256()
    requests_out=(out/'rerank.requests.jsonl').open('xb');questions_out=(out/'rerank.questions.jsonl').open('xb')
    try:
        for task,(nq,nr) in COUNTS.items():
            gate();task_root=out/task;task_root.mkdir(exist_ok=False)
            records=plan['tasks'][task]
            if set(records)!=set(plan['model_ids']):raise ValueError('four original head cells missing')
            reference=records[plan['model_ids'][0]]
            for rid,r in records.items():
                for key in ('effective_config','native_argv','planned_requests','dataset_inventory'):
                    if key=='dataset_inventory' and task=='chess':
                        if r[key] is not None:raise ValueError('native chess must not invent inventory')
                        continue
                    checked(r[key]['path'],r[key])
            config=json.loads(Path(reference['effective_config']['path']).read_bytes())
            original_argv=json.loads(Path(reference['native_argv']['path']).read_bytes())
            # Same argument contract as pinned native main, with no call to main.
            arg=argparse.ArgumentParser(add_help=False)
            arg.add_argument('--models',nargs='+',required=True);arg.add_argument('--out',required=True)
            arg.add_argument('--gsm8k-cache',default='data/gsm8k_cands.jsonl');arg.add_argument('--gsm8k-n',type=int,default=200)
            arg.add_argument('--gsm8k-k',type=int,default=4);arg.add_argument('--mnli-n',type=int,default=None)
            arg.add_argument('--tasks',nargs='+');arg.add_argument('--mc-n',type=int,default=None)
            arg.add_argument('--chess-n',type=int,default=500);arg.add_argument('--fewshot',type=int,default=5)
            arg.add_argument('--bs',type=int,default=32);arg.add_argument('--max-len',type=int,default=4096)
            arg.add_argument('--gen-only',action='store_true')
            native_args=arg.parse_args(original_argv[1:])
            if (native_args.tasks!=[task] or native_args.models!=[plan['model_ids'][0]]
                    or native_args.gen_only or config['scope']!='static' or config['backend']!='pointer'
                    or config['task_ids']!=[task] or config['model']!=plan['model_ids'][0]
                    or config.get('sample_limit') is not None):
                raise ValueError('original native task/argument scope differs')
            native=bridge.load_native('eval')
            if source_pre[str(Path(native.__file__))]['sha256']!=NATIVE_SHA:
                raise ValueError('native eval source differs')
            # No network fallback: only the declared revisions and offline cache.
            original_dataset_load=native.load_dataset
            def offline_declared(path,*args,**kwargs):
                if path not in config['data_revisions']:raise ValueError('undeclared native dataset revision')
                gate();return original_dataset_load(path,*args,**kwargs)
            native.load_dataset=offline_declared
            bridge.install_dataset_tracking(native,config,task_root)
            items=native.MC_TASKS[task](native_args);gate()
            if len(items)!=nq:raise ValueError('full native questions differ')
            captured={}
            class CaptureOnly:
                def predict(self,pairs):
                    frame=inspect.currentframe().f_back
                    if frame.f_code is not native.eval_mc.__code__:
                        raise ValueError('capture is not the original native eval_mc first call')
                    if pairs is not frame.f_locals['rerank_pairs']:
                        raise ValueError('capture is not actual original rerank_pairs')
                    captured.update(rerank=list(pairs),grade=list(frame.f_locals['grade_pairs']),
                                    offsets=list(frame.f_locals['offsets']))
                    raise CaptureStop('before any native probability/scoring')
            try:native.eval_mc(CaptureOnly(),items)
            except CaptureStop:pass
            if set(captured)!={'rerank','grade','offsets'} or len(captured['rerank'])!=nr or len(captured['grade'])!=nr:
                raise ValueError('actual native role population differs')
            expected=[(role,pointer.pointer_request('__MODEL_ID__',pair)) for role in ('rerank','grade') for pair in captured[role]]
            expected_digests=[population.request_digest(body) for role,body in expected]
            original_join={}
            for rid,r in records.items():
                gate();h=hashlib.sha256();n=0
                with Path(r['planned_requests']['path']).open('rb') as f:
                    for ordinal,raw in enumerate(f):
                        gate();h.update(raw);row=json.loads(raw)
                        if (ordinal>=len(expected) or row['ordinal']!=ordinal
                                or population.request_digest(row['request'])!=expected_digests[ordinal]
                                or row['request_sha256']!=expected_digests[ordinal]
                                or row['request']['model']!=rid):
                            raise ValueError('original full planned occurrence/body differs: '+rid+'/'+task+'/'+str(ordinal))
                        n+=1
                if n!=nr*2 or h.hexdigest()!=r['planned_requests']['sha256']:
                    raise ValueError('original complete planned sequence differs')
                original_join[rid]=dict(full_calls=n,planned_sha256=h.hexdigest(),native_role_sequence_verified=True)
            if task!='chess':
                new_rows=[json.loads(l) for l in (task_root/'dataset_inventory.jsonl').read_text().splitlines()]
                new_identity=hashlib.sha256(population.dataset_image(new_rows)).hexdigest()
                for rid,r in records.items():
                    old_rows=[json.loads(l) for l in Path(r['dataset_inventory']['path']).read_text().splitlines()]
                    if hashlib.sha256(population.dataset_image(old_rows)).hexdigest()!=new_identity:
                        raise ValueError('native dataset/revision/sample/fingerprint order differs')
            else:
                if (task_root/'dataset_inventory.jsonl').exists() or native_args.chess_n!=500:
                    raise ValueError('fixed native chess generator scope differs')
                new_identity=None
            for occurrence,item in enumerate(items):
                gate();start,count=captured['offsets'][occurrence];opts=item['opts'];gold=item['gold']
                if count!=len(opts) or type(gold) is not int or not 0<=gold<count:
                    raise ValueError('native full option/gold position differs')
                # IDs are NEW derived positions, never asserted as old log fields.
                episode=task+':'+str(occurrence)
                candidates=[episode+':option:'+str(j) for j in range(count)]
                request_ids=[episode+':rerank:'+str(j) for j in range(count)]
                for j,(candidate,request_id) in enumerate(zip(candidates,request_ids)):
                    role,body=expected[start+j]
                    if role!='rerank':raise ValueError('actual native offset is not rerank')
                    row=dict(ordinal=total_r,task=task,episode_id=episode,candidate_id=candidate,
                             request_id=request_id,role='rerank',payload=body,
                             payload_sha256=hashlib.sha256(wire(body)).hexdigest(),
                             source_planned_ordinal=start+j,source_planned_request_sha256=expected_digests[start+j])
                    raw=wire(row)+b'\n';requests_out.write(raw);all_request_sha.update(raw);total_r+=1
                questions_out.write(wire(dict(task=task,episode_id=episode,occurrence=occurrence,
                                             question=item['q'],options=opts,candidate_ids=candidates,
                                             gold=candidates[gold],gold_index=gold,request_ids=request_ids,
                                             identifiers='NEW_DERIVED_FROM_NATIVE_POSITION'))+b'\n');total_q+=1
            tasks[task]=dict(questions=nq,rerank_calls=nr,original_complete_role_join=original_join,
                            dataset_inventory_identity_sha256=new_identity,native_args=vars(native_args))
        if total_q!=46475 or total_r!=191276:raise ValueError('complete rerank denominator differs')
        for stream in (requests_out,questions_out):stream.flush();os.fsync(stream.fileno())
    finally:requests_out.close();questions_out.close()
    source_after={p:checked(p,w) for p,w in plan['source_pins'].items()}
    if source_pre!=source_after:raise ValueError('immutable native source changed')
    outputs={}
    for name in ('rerank.requests.jsonl','rerank.questions.jsonl'):
        path=out/name;expected=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        outputs[name]=expected
    save('RESULT.json',dict(status='NATIVE_RERANK_POPULATION_COMPLETE_PENDING_INDEPENDENT_REVIEW',
                           questions=total_q,rerank_calls=total_r,tasks=tasks,outputs=outputs,
                           source_pre=source_pre,source_after=source_after,checked_control_bytes_read=total_bytes,
                           no_model_or_probabilities=True,native_result_file_created=False,
                           finished_epoch=time.time(),scientific_acceptance=None))
    gate()


if __name__=='__main__':main()
