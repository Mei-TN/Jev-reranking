#!/usr/bin/env python3
"""Run one fixed inference population against four already-owned local bundle services.

This worker never starts/stops services, exports models, or retries failed requests.
--validate-only checks the complete materialized population on CPU without endpoints.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.parse import urlsplit
import urllib.error
import urllib.request

import prepare_inputs as prep
from materialize_population import file_sha
from token_preflight import read_json, read_jsonl

SERVER_SOURCE_SHA = 'df130e5a5e643a9801d25d1016a199b6434d8a30b07743548aa5accd4e4e20b6'
CONTRACT_SOURCE_SHA = '95fd30e4adcf1a9cc89e33f81abe28ab6a4bbbd057f72b797aca58328a369d11'
# The rest of what a service imports from bundle/code: the head itself and the loader that health only names.
LOADER_SOURCE_SHA = {
    'kev/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'kev/heads.py': 'c31edc946d2616cd8857f8eac917eb5b61525c41e49cf83920df44084c287e46',
    'runtime/__init__.py': '9af52b6bd52fd546e3f54fa3986c3a2c488f999ae9cdfe980732e900d1de0691',
    'runtime/load_bundle.py': '351a5f8e0cd5012f07165e74bbe131fa3514ff2374fe0a818b77c5f6a1e0b1db'}


def cache_identity(manifest_sha, population_sha, protocol_version, description_id, ordered_body):
    """Recorded identity only: this worker has no prediction cache or deduplication."""
    return prep.digest(prep.wire([manifest_sha, population_sha, protocol_version, description_id, ordered_body]))


def validate_population(root, run_id):
    root = Path(root)
    population = read_json(root / 'POPULATION.json')
    if population['status'] != 'COMPLETE_CPU_TOKEN_QUALIFIED' or population['total_requests'] != 69590:
        raise ValueError('incomplete or unqualified population')
    for name, expected in population['files'].items():
        if file_sha(root / name) != expected:
            raise ValueError('population member changed: ' + name)
    models = read_json(root / 'checkpoints.json')['models']
    matches = [model for model in models if model['run_id'] == run_id]
    if len(models) != 8 or len(matches) != 1:
        raise ValueError('not one of the eight fixed checkpoint identities')
    rows = read_jsonl(root / 'requests.jsonl')
    counts, occurrences, model_wire = Counter(), set(), hashlib.sha256()
    order_groups = {}
    for i, row in enumerate(rows):
        if i != row['ordinal'] or row['request_id'] in occurrences:
            raise ValueError('request order/source occurrence changed')
        if prep.digest(prep.wire(row['request'])) != row['wire_sha256']:
            raise ValueError('request wire differs from token-qualified witness')
        question = next(iter(row['request']['questions'].values()))
        if list(question['criteria']) != row['semantic_keys']:
            raise ValueError('semantic order changed')
        if not 0 < row['state_tokens'] <= row['input_tokens'] <= 8192:
            raise ValueError('request length not qualified')
        if row['condition'].startswith('piqa_nli_'):
            key = (row['question_id'], row['candidate_id'], row['description_id'])
            order_groups.setdefault(key, {})[row['permutation_id']] = row['input_token_ids_sha256']
        occurrences.add(row['request_id'])
        counts[row['condition']] += 1
        model_wire.update(prep.wire(dict(row['request'], model=run_id)) + b'\n')
    if (len(rows) != 69590 or dict(counts) != population['condition_requests'] or
            model_wire.hexdigest() != population['ordered_model_wire_sha256'][run_id] or
            len(order_groups) != 11028 or any(len(d) != 6 or len(set(d.values())) != 6 for d in order_groups.values())):
        raise ValueError('complete condition/order/model-wire population differs')
    return population, matches[0], rows


def validate_health(raw, run_id, head, manifest_sha, head_sha, bundle, port):
    expected = dict(model=run_id, probability_source='classification_head', head_kind=head['kind'], head_config=head,
        manifest_sha256=manifest_sha, head_sha256=head_sha, checkpoint=str(Path(bundle).resolve()), port=port,
        temperature=1, lora_scale=1, prefix_cache=False, strict=True,
        max_state=8192, max_branch=8192, max_packed=8192, supports_vision=False,
        dtype='torch.bfloat16', head_dtype='torch.float32', tp=1, merge='premerged_fp32_then_bf16',
        bundle_loader='runtime.load_bundle.load_bundle')
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError('service health differs: '+key)


def write_line(path, value):
    with Path(path).open('ab') as stream:
        stream.write(prep.wire(value) + b'\n')
        stream.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--population', type=Path, required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--bundle', type=Path)
    ap.add_argument('--endpoints', nargs=4)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--request-timeout', type=float, default=300)
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('output must be new; failed runs are never overwritten or resumed automatically')
    population, model, rows = validate_population(args.population, args.run_id)
    args.output.mkdir(parents=True, exist_ok=False)
    population_sha = file_sha(args.population/'POPULATION.json')
    if args.validate_only:
        prep.write_json(args.output/'CPU_VALIDATION.json', dict(status='PASS', run_id=args.run_id,
            population_sha256=population_sha, requests=len(rows), checkpoint_head_config=model['head_config'],
            predictions_generated=False, endpoint_accessed=False))
        print(json.dumps(dict(status='CPU_VALIDATION_PASS', requests=len(rows), output=str(args.output))))
        return
    if socket.gethostname() != os.environ.get('JEV_EVALUATION_HOST', socket.gethostname()) or os.getuid() != int(os.environ.get('JEV_EVALUATION_UID', str(os.getuid()))):
        raise RuntimeError('prediction worker is restricted to WIN3 user')
    if args.bundle is None or args.endpoints is None or len(set(args.endpoints)) != 4:
        raise ValueError('four distinct local owned endpoints and an actual bundle are required')
    ports = []
    for endpoint in args.endpoints:
        parsed = urlsplit(endpoint)
        if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password or parsed.port is None:
            raise ValueError('only exact http://127.0.0.1:port endpoints are supported')
        ports.append(parsed.port)
    if not 1 <= args.request_timeout <= 1800:
        raise ValueError('request timeout outside 1..1800 seconds')
    source = args.bundle/'code'
    for name, expected in dict(population['native_runtime_sources'], **LOADER_SOURCE_SHA,
                              **{'runtime/serve_bundle.py':SERVER_SOURCE_SHA, 'runtime/bundle_contract.py':CONTRACT_SOURCE_SHA}).items():
        if file_sha(source/name) != expected:
            raise ValueError('service/encoder source changed from qualified source: '+name)
    sys.path.insert(0, str(source))
    from runtime.bundle_contract import validate_manifest, validate_response, validate_probabilities
    manifest = validate_manifest(args.bundle, model['head_config'], verify_files=False)
    manifest_sha = file_sha(args.bundle/'model_manifest.json')
    if manifest['name'] != args.run_id or manifest['option_isolation']:
        raise ValueError('bundle does not match the fixed checkpoint/layout')
    head_sha = file_sha(args.bundle/'classification_head.safetensors')
    if head_sha != manifest['files']['classification_head.safetensors']:
        raise ValueError('actual exported head differs from its bound manifest')
    health = []
    for endpoint, port in zip(args.endpoints, ports):
        with urllib.request.urlopen(endpoint+'/health', timeout=15) as response:
            raw = json.load(response)
        validate_health(raw, args.run_id, model['head_config'], manifest_sha, head_sha, args.bundle, port)
        health.append(dict(endpoint=endpoint, config=raw))
    prep.write_json(args.output/'RUN_BINDING.json', dict(run_id=args.run_id, head_config=model['head_config'],
        population_sha256=population_sha, manifest_sha256=manifest_sha, health=health,
        worker_sha256=file_sha(__file__), service_source_sha256=SERVER_SOURCE_SHA,
        cache='no worker prediction cache; no request deduplication; service prefix_cache=false',
        request_timeout=args.request_timeout, retries=0, started_epoch=time.time()))
    lock = threading.Lock()
    started = time.monotonic()

    def call(item):
        row, endpoint = item
        body = dict(row['request'], model=args.run_id)
        tick = time.monotonic()
        event = dict(ordinal=row['ordinal'], request_id=row['request_id'], condition=row['condition'],
            question_id=row['question_id'], candidate_id=row['candidate_id'], endpoint=endpoint,
            request=body, request_wire_sha256=prep.digest(prep.wire(body)),
            protocol_version=population['protocol_version'], semantic_keys=row['semantic_keys'],
            input_token_ids_sha256=row['input_token_ids_sha256'],
            cache_identity=cache_identity(manifest_sha, population_sha, population['protocol_version'], row.get('description_id'), body))
        try:
            request = urllib.request.Request(endpoint+'/v1/decision', data=prep.wire(body), headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
                raw = json.load(response)
            event['response'] = raw
            validate_response(raw, args.run_id, model['head_config'], False)
            probabilities = validate_probabilities(raw, body)
            qid = next(iter(body['questions']))
            if list(probabilities[qid]) != row['semantic_keys']:
                raise ValueError('live service did not preserve requested semantic order')
            if (type(raw.get('input_tokens')) is not int or raw['input_tokens'] != row['input_tokens'] or
                    type(raw.get('state_tokens')) is not int or raw['state_tokens'] != row['state_tokens']):
                raise ValueError('live service token counts differ from native CPU witness')
        except Exception as error:
            event['error'] = dict(type=type(error).__name__, message=str(error))
            if isinstance(error, urllib.error.HTTPError):
                event['error']['http_response'] = error.read().decode('utf-8', errors='replace')
        finally:
            event['elapsed_s'] = time.monotonic()-tick
            with lock:
                write_line(args.output/'completion_calls.jsonl', event)
        return event

    completed = []
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            # Each endpoint has at most one in-flight request. A failed batch finishes
            # its already-owned requests, then stops before submitting another batch.
            for start in range(0, len(rows), 4):
                batch = rows[start:start+4]
                for row in batch:
                    write_line(args.output/'planned_calls.jsonl', dict(ordinal=row['ordinal'], request_id=row['request_id'],
                        request=dict(row['request'], model=args.run_id)))
                events = list(pool.map(call, zip(batch, args.endpoints)))
                for expected, event in zip(batch, events):
                    if event['ordinal'] != expected['ordinal'] or event['request_id'] != expected['request_id']:
                        raise ValueError('response order/identity differs')
                    write_line(args.output/'ordered_calls.jsonl', event)
                    completed.append(event)
                failures = [e for e in events if e.get('error')]
                if failures:
                    raise RuntimeError('request failure preserved; first failed ordinal '+str(failures[0]['ordinal']))
                if len(completed) % 1000 == 0 or len(completed) == len(rows):
                    print(json.dumps(dict(stage='inference', model=args.run_id, completed=len(completed), total=len(rows))), flush=True)
        from inference_scores import score_complete
        scores, item_scores = score_complete(rows, completed, read_jsonl(args.population/'scoring_population.jsonl'), args.population/'native_eval.py')
        prep.write_jsonl(args.output/'question_scores.jsonl', item_scores)
        prep.write_json(args.output/'metrics.json', scores)
        prep.write_json(args.output/'COMPLETE.json', dict(status='COMPLETE', run_id=args.run_id,
            requests=len(completed), conditions=population['condition_requests'], questions=population['questions'],
            head_config=model['head_config'], population_sha256=population_sha, manifest_sha256=manifest_sha,
            elapsed_s=time.monotonic()-started, predictions_generated=True,
            files={p.name:file_sha(p) for p in args.output.iterdir() if p.is_file()}))
    except BaseException as error:
        # A failed batch's calls are in the ledgers too; count only the ones that actually succeeded.
        failed = sum(1 for event in completed if event.get('error'))
        prep.write_json(args.output/'FAILED.json', dict(status='FAILED_PRESERVED', run_id=args.run_id,
            error_type=type(error).__name__, error=str(error), completed_requests=len(completed)-failed,
            failed_requests=failed, attempted_requests=len(completed),
            required_requests=len(rows), automatic_retry=False, service_ownership='external controller; worker stops none'))
        raise


if __name__ == '__main__':
    main()
