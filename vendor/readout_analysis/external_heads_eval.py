"""Independent three-source head12 HTTP consumer. Import performs no IO/inference.

Uses the six frozen materialized files directly; does not call materialize.py.
Services and GPU lifecycles are owned by a separately admitted root controller.
This single-model consumer requires an actual new plan/admission, never old once.
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
from urllib.parse import urlsplit

SOURCE_ORDER = ('piqa', 'commonsenseqa', 'clinc150')
HEAD12_IDS=tuple('SUPP_WINH4_'+head+'_f'+str(fold)+'_s'+str(seed)
                for fold in (2,3) for seed in (44,45) for head in ('H0','H4','H5'))
DELAY4_IDS=tuple('SUPP_WINH4_D_f'+str(fold)+'_s'+str(seed)
                for fold in (2,3) for seed in (44,45))
EXTERNAL_COHORTS={'head12':HEAD12_IDS,'delay4':DELAY4_IDS}
COUNTS = {'piqa': (1838, 3676), 'commonsenseqa': (1221, 6105),
          'clinc150': (4500, 4500)}
PINS = {
    'piqa.questions.jsonl': '82984e4b2d84200d70b1f72abeb67199a88423570365be4fba1736b14f22736c',
    'piqa.requests.jsonl': '376e00fea0085ef852a54c6a732e061d18ab7a6b5be1b80bcfcee27c1ac60184',
    'commonsenseqa.questions.jsonl': '67584d3f780811c6e11086073d2e59dac4c92bae6866b1a502959c788a328176',
    'commonsenseqa.requests.jsonl': '2c19625485da1da8f811853c8168a85e23472ff2d4649baf14f582c4cf5c47cc',
    'clinc150.questions.jsonl': 'eb72eabbe027e8e8a3fb7d7181c5b164238d7b5601ac59e53289f1b0853ddcc4',
    'clinc150.requests.jsonl': 'a166d367f315d343af8e70c7a148834aa49f44c5f2571c38ac4e62d40f548889'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def wire(value):
    # Keep actual criteria insertion order. Sorted hashing is NOT dispatch order.
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def checked_probabilities(response, body, expected_identity):
    for key in ('model', 'probability_source', 'head_kind', 'head_config'):
        if response[key] != expected_identity[key]:
            raise ValueError('response identity differs: '+key)
    if set(response['probabilities']) != set(body['questions']):
        raise ValueError('question response coverage differs')
    for qid, q in body['questions'].items():
        actual = response['probabilities'][qid]
        if set(actual) != set(q['criteria']):
            raise ValueError('complete native option keys differ')
        values = list(actual.values())
        if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('invalid native probabilities')
        if abs(math.fsum(values)-1) > 1e-5:
            raise ValueError('native probabilities not normalized')
    if type(response['input_tokens']) is not int or not 0 < response['input_tokens'] <= 8192:
        raise ValueError('actual strict encoding token count missing')


def score_source(source, questions, events):
    by_id = {e['request_id']: e for e in events}
    if len(by_id) != len(events):
        raise ValueError('duplicate original request occurrence')
    scored = []
    for q in questions:
        ids = q['frozen_candidate_ids']
        if ids != [c['id'] for c in q['candidates']] or len(ids) != len(set(ids)):
            raise ValueError('native candidate occurrence/order differs')
        reqs = [by_id[x] for x in q['request_ids']]
        if any(e['sample_id'] != q['id'] or e['source'] != source for e in reqs):
            raise ValueError('question/request source association differs')
        if source == 'clinc150':
            if len(reqs) != 1 or reqs[0]['candidate_id'] is not None or len(ids) != 150:
                raise ValueError('CLINC must remain one native150-way request')
            values = reqs[0]['response']['probabilities']['intent']
        else:
            if len(reqs) != len(ids) or [e['candidate_id'] for e in reqs] != ids:
                raise ValueError('original candidate/request association differs')
            values = {e['candidate_id']: e['response']['probabilities']['relation']['entailment'] for e in reqs}
        if set(values) != set(ids) or q['gold'] not in ids:
            raise ValueError('native candidate/gold coverage differs')
        ties = [i for i in ids if values[i] == max(values.values())]
        scored.append({'sample_id': q['id'], 'candidate_ids': ids, 'gold': q['gold'],
                       'values': values, 'choice': ties[0], 'ties': ties,
                       'correct': int(ties[0] == q['gold']),
                       'uniform_tie_credit': 1/len(ties) if q['gold'] in ties else 0})
    n = COUNTS[source][0]
    if len(scored) != n:
        raise ValueError('full original question denominator differs')
    result = {'source': source, 'N': n, 'correct': sum(s['correct'] for s in scored),
              'max_tie_questions': sum(len(s['ties']) > 1 for s in scored),
              'uniform_tie_credit_sum': math.fsum(s['uniform_tie_credit'] for s in scored)}
    if source == 'clinc150':
        overlap = [s for s in scored if s['sample_id'] == 'test:003240']
        if len(overlap) != 1:
            raise ValueError('fixed CLINC overlap stratum missing')
        result['O'] = {'N': 1, 'correct': overlap[0]['correct']}
        result['U'] = {'N': 4499, 'correct': result['correct']-overlap[0]['correct']}
    return result, scored


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('unregistered endpoint redirect')


def evaluate(plan):
    if plan['schema'] != 'jev-review-three-external-single-model-actual-plan-v1' or plan['admitted'] is not True:
        raise ValueError('new actual root admission required')
    if plan['cohort'] not in EXTERNAL_COHORTS or plan['model_id'] not in EXTERNAL_COHORTS[plan['cohort']]:
        raise ValueError('outside separately fixed head12/delay4 external stratum')
    if tuple(plan['sources']) != SOURCE_ORDER or plan['max_seconds'] != 7200 or len(plan['endpoints']) != 4:
        raise ValueError('fixed sources, four replicas or two-hour cap differs')
    started_e, started_m = time.time(), time.monotonic()
    stop_e = min(started_e+7200, plan['root_deadline_epoch'])
    stop_m = started_m+(stop_e-started_e)
    stop = threading.Event()
    first_error = None
    error_lock = threading.Lock()
    def gate():
        if stop.is_set() or time.time() >= stop_e or time.monotonic() >= stop_m:
            raise TimeoutError('original model/root budget or first failure stop')
    out = Path(plan['output']); out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        gate()
        with (out/name).open('xb') as f:
            f.write(wire(value)+b'\n'); f.flush(); os.fsync(f.fileno())
        fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        gate()
    results = []
    try:
        save('ENTRY_ONCE.json', {'plan': plan, 'started_epoch': started_e,
                                'started_monotonic': started_m, 'stop_epoch': stop_e,
                                'stop_monotonic': stop_m, 'retry': 0})
        source_root = Path(plan['materialized_root'])
        pins = {n: digest(source_root/n) for n in PINS}
        if pins != PINS:
            raise ValueError('frozen complete materialized source bytes differ')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        for url in plan['endpoints']:
            parsed = urlsplit(url)
            if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.path not in ('', '/'):
                raise ValueError('service endpoint must be a registered local replica')
            gate()
            with opener.open(url.rstrip('/')+'/health', timeout=min(30, stop_e-time.time(), stop_m-time.monotonic())) as r:
                health = json.loads(r.read(1 << 20))
            for k, v in plan['expected_service'].items():
                if health[k] != v:
                    raise ValueError('original bundle/runtime identity differs: '+k)
        for source in SOURCE_ORDER:
            gate()
            questions = jsonl(source_root/(source+'.questions.jsonl'))
            requests = jsonl(source_root/(source+'.requests.jsonl'))
            if (len(questions), len(requests)) != COUNTS[source]:
                raise ValueError('full frozen source population differs')
            if len({r['request_id'] for r in requests}) != len(requests):
                raise ValueError('request occurrences collapsed')
            log = (out/(source+'.ordered_calls.jsonl')).open('xb')
            events = []
            def call(item):
                nonlocal first_error
                ordinal, row = item; raw = b''
                try:
                    gate()
                    if row['source'] != source or hashlib.sha256(canonical(row['payload'])).hexdigest() != row['payload_sha256']:
                        raise ValueError('original payload hash/source differs')
                    body = json.loads(wire(row['payload']))
                    if body['model'] != '__MODEL_ID__' or set(body) != {'model', 'state', 'questions'}:
                        raise ValueError('gold/extra metadata or model placeholder differs')
                    for q in body['questions'].values():
                        if set(q) != {'type', 'instructions', 'criteria'}:
                            raise ValueError('unexpected model-side fields')
                    body['model'] = plan['model_id']; sent = wire(body)
                    req = urllib.request.Request(plan['endpoints'][ordinal % 4].rstrip('/')+'/v1/decision', sent,
                                                 {'Content-Type': 'application/json'}, method='POST')
                    began = time.time(); tick = time.monotonic()
                    with opener.open(req, timeout=min(30, stop_e-time.time(), stop_m-time.monotonic())) as response:
                        raw = response.read((1 << 20)+1); status = response.status
                    if len(raw) > 1 << 20: raise ValueError('response exceeds finite cap')
                    value = json.loads(raw); checked_probabilities(value, body, plan['expected_service'])
                    gate()
                    return {'ordinal': ordinal, 'source': source, 'request_id': row['request_id'],
                            'sample_id': row['sample_id'], 'candidate_id': row['candidate_id'],
                            'payload_sha256': row['payload_sha256'], 'request': body,
                            'request_wire_sha256': hashlib.sha256(sent).hexdigest(),
                            'HTTP_status': status, 'response_raw_base64': base64.b64encode(raw).decode(),
                            'response_sha256': hashlib.sha256(raw).hexdigest(), 'response': value,
                            'started_epoch': began, 'ended_epoch': time.time(), 'elapsed_s': time.monotonic()-tick}
                except BaseException as exc:
                    with error_lock:
                        if first_error is None and not stop.is_set():
                            first_error = {'source': source, 'ordinal': ordinal,
                                           'request_id': row['request_id'],
                                           'type': type(exc).__name__, 'message': str(exc),
                                           'response_raw_base64': base64.b64encode(raw).decode()}
                            if isinstance(exc, urllib.error.HTTPError):
                                first_error['HTTP_status'] = exc.code
                            stop.set()  # No later request may start inference.
                    if isinstance(exc, urllib.error.HTTPError):
                        error_raw = exc.read(1 << 20)
                        with error_lock:
                            if first_error is not None and first_error['source'] == source and first_error['ordinal'] == ordinal:
                                first_error['HTTP_error_raw_base64'] = base64.b64encode(error_raw).decode()
                    raise
            try:
                start_ordinal = 0
                if source == 'clinc150':
                    # The first formal request is consumed exactly once. Its full
                    #150-way checked_probabilities must succeed before the rest.
                    event = call((0, requests[0]))
                    log.write(wire(event)+b'\n'); events.append(event)
                    start_ordinal = 1
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for event in pool.map(call, enumerate(requests[start_ordinal:], start_ordinal)):
                        log.write(wire(event)+b'\n'); events.append(event)
                log.flush(); os.fsync(log.fileno())
            finally:
                log.close()
            gate(); summary, scored = score_source(source, questions, events)
            save(source+'.scored_questions.json', scored); save(source+'.SUMMARY.json', summary)
            results.append(summary)
        gate()
        if {n: digest(source_root/n) for n in PINS} != pins:
            raise ValueError('immutable input bytes changed during evaluation')
        save('RESULT.json', {'status': 'THREE_SOURCES_COMPLETE_PENDING_INDEPENDENT_REVIEW',
                             'cohort':plan['cohort'],'model_id':plan['model_id'],
                             'results': results, 'finished_epoch': time.time(),
                             'finished_monotonic': time.monotonic(), 'scientific_acceptance': None})
        return results
    except BaseException as exc:
        # The stop event blocks inference, not a timely bounded failure receipt.
        if time.time() < stop_e and time.monotonic() < stop_m:
            failure = {'type': type(exc).__name__, 'message': str(exc),
                       'traceback': traceback.format_exc(), 'completed_sources': results,
                       'first_request_failure': first_error}
            raw = wire(failure)+b'\n'
            if time.time() < stop_e and time.monotonic() < stop_m:
                with (out/'FIRST_FAILURE.json').open('xb') as f:
                    f.write(raw); f.flush(); os.fsync(f.fileno())
                if time.time() < stop_e and time.monotonic() < stop_m:
                    fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
                    try: os.fsync(fd)
                    finally: os.close(fd)
                # Late publication never changes failure into success.
                if time.time() >= stop_e or time.monotonic() >= stop_m:
                    raise TimeoutError('failure publication exceeded original budget') from exc
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    evaluate(json.loads(args.plan.read_bytes()))


if __name__ == '__main__':
    main()
