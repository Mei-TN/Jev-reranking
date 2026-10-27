"""Evaluate paired residual-head and baseline probabilities from one forward pass.

Read fixed reranking requests and separately stored gold answers.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import traceback
import urllib.error
import urllib.request
from external_heads_eval import NoRedirect, checked_probabilities, digest, wire
from paired_branch import rerank_effects

COUNTS = {'mmlu_fewshot':(14042,56168), 'mmlu':(14042,56168),
          'hellaswag':(10042,40168), 'gsm8k_mc10':(1319,13190),
          'arc_easy':(2376,9501), 'gsm8k_mc4':(1319,5276),
          'winogrande':(1267,2534), 'arc_challenge':(1172,4687),
          'chess':(500,2000), 'gpqa_fewshot':(198,792), 'gpqa':(198,792)}


def evaluate(plan):
    if (plan['schema'] != 'jev-review-inference-single-model-actual-plan-v1'
            or plan['kind'] != 'branch' or plan['admitted'] is not True
            or plan['head_config']['kind'] != 'residual_pair_mlp'
            or plan['max_seconds'] != 10800 or len(plan['endpoints']) != 4):
        raise ValueError('new fixed branch actual plan required')
    # Use the producer's stop deadlines.
    stop_e, stop_m = plan['client_stop_epoch'], plan['client_stop_monotonic']
    stop = threading.Event(); error_lock = threading.Lock(); first_error = None
    def gate():
        if stop.is_set() or time.time() >= stop_e or time.monotonic() >= stop_m:
            raise TimeoutError('original paired client stop/first failure')
    out = Path(plan['output'])/'branch_client'; out.mkdir(exist_ok=False)
    def save(name, value):
        gate()
        with (out/name).open('xb') as f:
            f.write(wire(value)+b'\n'); f.flush(); os.fsync(f.fileno())
        fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        gate()
    try:
        save('ENTRY_ONCE.json', dict(stop_epoch=stop_e, stop_monotonic=stop_m, retry=0,
                                    plan_sha256=digest(plan['actual_plan_path'])))
        paths = {'requests':(plan['branch_requests'], plan['branch_requests_sha256']),
                 'questions':(plan['branch_questions'],plan['branch_questions_sha256'])}
        for path, expected in paths.values():
            gate()
            if digest(path) != expected: raise ValueError('frozen native branch input differs')
        with Path(paths['requests'][0]).open('rb') as f: requests = [json.loads(l) for l in f]
        with Path(paths['questions'][0]).open('rb') as f: questions = [json.loads(l) for l in f]
        if len(requests) != 191276 or len(questions) != 46475:
            raise ValueError('full fixed branch denominator differs')
        if len({r['request_id'] for r in requests}) != len(requests):
            raise ValueError('native rerank request occurrence collapsed')
        task_counts = {task:[0,0] for task in COUNTS}
        by_request = {r['request_id']:r for r in requests}
        referenced = []
        for q in questions:
            task = q['task']; ids = q['candidate_ids']; reqs = q['request_ids']
            if (len(ids) != len(reqs) or len(ids) != len(set(ids))
                    or q['gold'] not in ids or len(ids) < 2):
                raise ValueError('native branch candidate/gold association differs')
            for candidate, request_id in zip(ids, reqs):
                row = by_request[request_id]
                if row['task'] != task or row['episode_id'] != q['episode_id'] or row['candidate_id'] != candidate:
                    raise ValueError('native branch episode/request association differs')
            referenced.extend(reqs); task_counts[task][0] += 1; task_counts[task][1] += len(ids)
        if (task_counts != {k:list(v) for k,v in COUNTS.items()}
                or len(referenced) != len(set(referenced)) or set(referenced) != set(by_request)):
            raise ValueError('task denominator or complete native association differs')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        for endpoint in plan['endpoints']:
            gate()
            with opener.open(endpoint+'/health', timeout=min(30,stop_e-time.time(),stop_m-time.monotonic())) as response:
                health = json.loads(response.read(1 << 20))
            for k,v in plan['expected_service'].items():
                if health[k] != v: raise ValueError('actual branch service identity differs: '+k)
        def call(item):
            nonlocal first_error
            ordinal, row = item; raw = b''; http_status = None
            try:
                gate()
                if (row['ordinal'] != ordinal or row['role'] != 'rerank'
                        or hashlib.sha256(wire(row['payload'])).hexdigest() != row['payload_sha256']):
                    raise ValueError('native ordered rerank ledger differs')
                body = dict(row['payload'], model=plan['model_id'])
                if set(body) != {'model','state','questions'} or set(body['questions']) != {'relation'}:
                    raise ValueError('original rerank NLI input differs')
                sent = wire(dict(request_id=row['request_id'], ordinal=ordinal, payload=body))
                req = urllib.request.Request(plan['endpoints'][ordinal%4]+'/v1/paired-rerank', sent,
                                             {'Content-Type':'application/json'}, method='POST')
                began_e, began_m = time.time(), time.monotonic()
                with opener.open(req, timeout=min(60,stop_e-time.time(),stop_m-time.monotonic())) as response:
                    raw = response.read((1<<20)+1); http_status=response.status
                if len(raw)>1<<20: raise ValueError('bounded paired response exceeded')
                value = json.loads(raw); checked_probabilities(value,body,plan['expected_service'])
                checked_probabilities(dict(value,probabilities=value['branch_zeroed']),body,plan['expected_service'])
                capture=value['capture']; fields=capture['encoding']['fields']
                encoded=json.dumps(fields,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                if (capture['same_forward'] is not True or capture['model_probs_calls'] != 1
                        or capture['autocast_added'] is not False
                        or capture['historical_full_required'] is not False
                        or len(capture['captures']) != 1
                        or capture['captures'][0]['full_capture_exact'] is not True
                        or hashlib.sha256(encoded).hexdigest() != capture['encoding']['sha256']
                        or len(fields['ids']) != value['input_tokens']):
                    raise ValueError('actual full/base same-forward/token witness differs')
                for k in ('manifest_sha256','head_sha256','service_source_sha256'):
                    if value[k] != plan['expected_service'][k]: raise ValueError('response source identity differs: '+k)
                gate()
                return dict(ordinal=ordinal,task=row['task'],episode_id=row['episode_id'],
                            candidate_id=row['candidate_id'],request_id=row['request_id'],
                            original_payload_sha256=row['payload_sha256'],request=body,
                            request_wire_sha256=hashlib.sha256(sent).hexdigest(),HTTP_status=http_status,
                            response_raw_base64=base64.b64encode(raw).decode(),
                            response_sha256=hashlib.sha256(raw).hexdigest(),response=value,
                            started_epoch=began_e,elapsed_s=time.monotonic()-began_m)
            except BaseException as exc:
                with error_lock:
                    if first_error is None and not stop.is_set():
                        first_error=dict(ordinal=ordinal,task=row['task'],request_id=row['request_id'],
                                         type=type(exc).__name__,message=str(exc),HTTP_status=http_status,
                                         response_raw_base64=base64.b64encode(raw).decode())
                        stop.set()
                if isinstance(exc,urllib.error.HTTPError):
                    error_raw=exc.read(1<<20)
                    with error_lock:
                        if first_error and first_error['ordinal']==ordinal:
                            first_error['HTTP_status']=exc.code
                            first_error['HTTP_error_raw_base64']=base64.b64encode(error_raw).decode()
                raise
        events={}; token_order=hashlib.sha256()
        with (out/'ordered_paired_calls.jsonl').open('xb') as log:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for event in pool.map(call,enumerate(requests)):
                    gate(); log.write(wire(event)+b'\n'); events[event['request_id']]=event
                    token_order.update(event['response']['capture']['encoding']['sha256'].encode()+b'\n')
            log.flush(); os.fsync(log.fileno())
        gate()
        if len(events)!=191276: raise ValueError('missing actual paired native calls')
        totals={task:dict(questions=0,calls=0,full_correct=0,base_correct=0,
                         choice_changes=0,top_tie_set_changes=0,unordered_pairs=0,
                         strict_reversals=0,ties_created=0,ties_broken=0,
                         full_uniform_tie_credit=0.,base_uniform_tie_credit=0.) for task in COUNTS}
        with (out/'scored_questions.jsonl').open('xb') as log:
            for q in questions:
                gate(); es=[events[r] for r in q['request_ids']]
                full={c:e['response']['probabilities']['relation']['entailment'] for c,e in zip(q['candidate_ids'],es)}
                base={c:e['response']['branch_zeroed']['relation']['entailment'] for c,e in zip(q['candidate_ids'],es)}
                effect=rerank_effects(q['candidate_ids'],full,base,q['gold'])
                log.write(wire(dict(question=q,full=full,base=base,effect=effect))+b'\n')
                t=totals[q['task']]; t['questions']+=1; t['calls']+=len(es)
                for target,key in (('full_correct','full_correct'),('base_correct','base_correct'),
                                   ('choice_changes','choice_changed'),('top_tie_set_changes','top_tie_set_changed'),
                                   ('unordered_pairs','unordered_pair_denominator'),('strict_reversals','strict_reversals'),
                                   ('ties_created','tie_created'),('ties_broken','tie_broken'),
                                   ('full_uniform_tie_credit','full_uniform_tie_credit'),('base_uniform_tie_credit','base_uniform_tie_credit')):
                    t[target]+=effect[key]
            log.flush(); os.fsync(log.fileno())
        for path,expected in paths.values():
            gate()
            if digest(path)!=expected: raise ValueError('native branch input changed')
        save('RESULT.json',dict(status='PAIRED_RERANK_COMPLETE_PENDING_INDEPENDENT_REVIEW',
                               native_calls=191276,questions=46475,tasks=totals,
                               macro11_full=math.fsum(t['full_correct']/t['questions'] for t in totals.values())/11,
                               macro11_branch_zeroed=math.fsum(t['base_correct']/t['questions'] for t in totals.values())/11,
                               ordered_encoder_sha256=token_order.hexdigest(),
                               historical_full_reproduction='NOT_READ_BY_THIS_CONSUMER_NO_EXTRA_ON',
                               finished_epoch=time.time(),scientific_acceptance=None))
    except BaseException as exc:
        if time.time()<stop_e and time.monotonic()<stop_m:
            raw=wire(dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                          first_request_failure=first_error))+b'\n'
            if time.time()<stop_e and time.monotonic()<stop_m:
                with (out/'FIRST_FAILURE.json').open('xb') as f:
                    f.write(raw); f.flush(); os.fsync(f.fileno())
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    a=p.parse_args();evaluate(json.loads(Path(a.plan).read_bytes()))
