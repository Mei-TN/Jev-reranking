#!/usr/bin/env python3
"""Send the sealed 7,816-request SocialIQA population of one model to four already-owned bundle services.

1,954 direct and 5,862 D0 NLI requests, each occurrence once, in source order. This worker never starts
or stops a service, exports a model, retries, deduplicates or skips a request. It keeps the planned,
completion-ordered and source-ordered ledgers with every raw response or error. After the first failed
batch it stops and writes FAILED.json; a missing response is never scored. --validate-only checks the
sealed population on CPU and opens no endpoint. Stdlib only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # Sibling modules import by name even under PYTHONSAFEPATH or -P.

import e10_contract as c
import e10_models as models

WIN3_HOST = os.environ.get('JEV_EVALUATION_HOST', socket.gethostname())
OUTPUTS = ('RUN_BINDING.json', 'planned_calls.jsonl', 'completion_calls.jsonl', 'ordered_calls.jsonl',
           'question_scores.jsonl', 'metrics.json')


def expected_health(model, served_model_id, bundle, port, manifest_sha, head_sha):
    """Everything a replica reports about itself except the GPU it was given, which the controller checks."""
    head = model['head_config']
    return dict(model=served_model_id, probability_source='classification_head', head_kind=head['kind'], head_config=head,
                checkpoint=str(Path(bundle).resolve()), port=port, tp=1, dtype='torch.bfloat16', head_dtype='torch.float32',
                merge='premerged_fp32_then_bf16', lora_scale=1, temperature=1, bundle_loader='runtime.load_bundle.load_bundle',
                prefix_cache=False, max_state=c.LIMIT, max_branch=c.LIMIT, max_packed=c.LIMIT, strict=True,
                supports_vision=False, head_sha256=head_sha, manifest_sha256=manifest_sha)


def checked_response(raw, body, row, model, served_model_id, contract):
    """Identity, head, semantic-key order, distribution and the qualified token counts of one live response."""
    contract.validate_response(raw, served_model_id, model['head_config'], False)
    probabilities = contract.validate_probabilities(raw, body)
    qid, = body['questions']
    c.probability_distribution(probabilities[qid], row['semantic_keys'])
    if (type(raw.get('input_tokens')) is not int or raw['input_tokens'] != row['input_tokens'] or
            type(raw.get('state_tokens')) is not int or raw['state_tokens'] != row['state_tokens']):
        raise ValueError('live service token counts differ from the qualified token witness')


def write_line(path, value):
    with Path(path).open('ab') as stream:
        stream.write(c.wire(value) + b'\n')
        stream.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--population', type=Path, required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--served-model-id', help='the bundle name the services answer to: the run id, or <run id>_eval for a reused bundle')
    ap.add_argument('--bundle', type=Path)
    ap.add_argument('--endpoints', nargs=4)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--request-timeout', type=float, default=300)
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args(argv)
    if args.output.exists():
        raise ValueError('output must be new; a failed run is never overwritten or resumed')
    model = models.model_map().get(args.run_id)
    if model is None:
        raise ValueError('not one of the sixteen fixed model identities')
    population, rows, scoring = c.load_population(args.population, c.population_pin())
    population_sha = c.file_sha(args.population / 'POPULATION.json')
    if args.validate_only:
        args.output.mkdir(parents=True)
        c.write_json(args.output / 'CPU_VALIDATION.json', dict(
            status='PASS', run_id=args.run_id, cohort=model['cohort'], population_sha256=population_sha, requests=len(rows),
            questions=len(scoring), mask_counts=population['mask_counts'], predictions_generated=False, endpoint_accessed=False))
        print(json.dumps(dict(status='CPU_VALIDATION_PASS', requests=len(rows), output=str(args.output))))
        return 0
    if socket.gethostname() != WIN3_HOST or os.getuid() != int(os.environ.get('JEV_EVALUATION_UID', str(os.getuid()))):
        raise RuntimeError('prediction worker is restricted to WIN3 user')
    served = args.served_model_id
    if served not in (args.run_id, args.run_id + '_eval') or args.bundle is None or args.endpoints is None or len(set(args.endpoints)) != 4:
        raise ValueError('a served model id of this run, its bundle and four distinct endpoints are required')
    ports = []
    for endpoint in args.endpoints:
        parsed = urlsplit(endpoint)
        if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.path or parsed.query or parsed.fragment or
                parsed.username or parsed.password or parsed.port is None):
            raise ValueError('only exact http://127.0.0.1:port endpoints are supported')
        ports.append(parsed.port)
    if not 1 <= args.request_timeout <= 1800:
        raise ValueError('request timeout outside 1..1800 seconds')
    code = args.bundle / 'code'
    for name, expected in models.SOURCES[model['source_identity']].items():
        if name not in models.EXPORT_ONLY and c.file_sha(code / name) != expected:
            raise ValueError('served source differs from the qualified ' + model['source_identity'] + ' source: ' + name)
    sys.path.insert(0, str(code))
    from runtime import bundle_contract as contract  # stdlib only; the same rules the service applies
    manifest = contract.validate_manifest(args.bundle, model['head_config'], verify_files=False)
    manifest_sha, head_sha = c.file_sha(args.bundle / 'model_manifest.json'), c.file_sha(args.bundle / 'classification_head.safetensors')
    if manifest['name'] != served or manifest['option_isolation'] or head_sha != manifest['files']['classification_head.safetensors']:
        raise ValueError('bundle does not match the served identity, layout or exported head')
    health = []
    for endpoint, port in zip(args.endpoints, ports):
        with urllib.request.urlopen(endpoint + '/health', timeout=15) as response:
            raw = json.load(response)
        for key, value in expected_health(model, served, args.bundle, port, manifest_sha, head_sha).items():
            if raw.get(key) != value:
                raise ValueError('service health differs: ' + key)
        health.append(dict(endpoint=endpoint, config=raw))
    args.output.mkdir(parents=True)
    c.write_json(args.output / 'RUN_BINDING.json', dict(
        run_id=args.run_id, cohort=model['cohort'], served_model_id=served, head_config=model['head_config'],
        population_sha256=population_sha, manifest_sha256=manifest_sha, head_sha256=head_sha, health=health,
        worker_sha256=c.file_sha(__file__), contract_sha256=c.file_sha(HERE / 'e10_contract.py'),
        cache='no worker prediction cache; no request deduplication; service prefix_cache=false',
        request_timeout=args.request_timeout, retries=0, started_epoch=time.time()))
    lock, started = threading.Lock(), time.monotonic()

    def call(item):
        row, endpoint = item
        body = dict(row['request'], model=served)
        tick = time.monotonic()
        event = dict(ordinal=row['ordinal'], request_id=row['request_id'], interface=row['interface'], item_id=row['item_id'],
                     source_ordinal=row['source_ordinal'], candidate_id=row['candidate_id'], endpoint=endpoint, request=body,
                     request_wire_sha256=c.digest(c.wire(body)), request_image_sha256=c.digest(c.request_image(body)),
                     semantic_keys=row['semantic_keys'], input_token_ids_sha256=row['input_token_ids_sha256'],
                     qualified_input_tokens=row['input_tokens'], qualified_state_tokens=row['state_tokens'])
        try:
            if event['request_image_sha256'] != row['request_image_sha256']:
                raise ValueError('sent request differs from the token-qualified request')
            request = urllib.request.Request(endpoint + '/v1/decision', data=c.wire(body), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
                raw = json.load(response)
            event['response'] = raw
            checked_response(raw, body, row, model, served, contract)
        except Exception as error:
            event['error'] = dict(type=type(error).__name__, message=str(error))
            if isinstance(error, urllib.error.HTTPError):
                event['error']['http_response'] = error.read().decode('utf-8', errors='replace')
        finally:
            event['elapsed_s'] = time.monotonic() - tick
            with lock:
                write_line(args.output / 'completion_calls.jsonl', event)
        return event

    completed = []
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            # At most one request in flight per endpoint. A failed batch finishes the calls it
            # already owns, then nothing further is submitted.
            for start in range(0, len(rows), 4):
                batch = rows[start:start + 4]
                for row in batch:
                    write_line(args.output / 'planned_calls.jsonl', dict(ordinal=row['ordinal'], request_id=row['request_id'],
                                                                        request=dict(row['request'], model=served)))
                events = list(pool.map(call, zip(batch, args.endpoints)))
                for expected, event in zip(batch, events):
                    if event['ordinal'] != expected['ordinal'] or event['request_id'] != expected['request_id']:
                        raise ValueError('response order/identity differs')
                    write_line(args.output / 'ordered_calls.jsonl', event)
                    completed.append(event)
                failures = [event for event in events if event.get('error')]
                if failures:
                    raise RuntimeError('request failure preserved; first failed ordinal ' + str(failures[0]['ordinal']))
                if len(completed) % 1000 == 0 or len(completed) == len(rows):
                    print(json.dumps(dict(stage='e10', model=args.run_id, completed=len(completed), total=len(rows))), flush=True)
        scores = c.question_rows(rows, completed, scoring)
        c.write_jsonl(args.output / 'question_scores.jsonl', scores)
        c.write_json(args.output / 'metrics.json', c.summarize(scores))
        c.write_json(args.output / 'COMPLETE.json', dict(
            status='COMPLETE', run_id=args.run_id, cohort=model['cohort'], served_model_id=served, requests=len(completed),
            interface_requests=population['interface_requests'], questions=len(scoring), head_config=model['head_config'],
            population_sha256=population_sha, manifest_sha256=manifest_sha, elapsed_s=time.monotonic() - started,
            predictions_generated=True, files={p.name: c.file_sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}))
    except BaseException as error:
        # The failed batch's calls are in the ledgers too; only the successful ones are counted as completed.
        failed = sum(1 for event in completed if event.get('error'))
        c.write_json(args.output / 'FAILED.json', dict(
            status='FAILED_PRESERVED', run_id=args.run_id, error_type=type(error).__name__, error=str(error),
            completed_requests=len(completed) - failed, failed_requests=failed, attempted_requests=len(completed),
            required_requests=len(rows), automatic_retry=False, metrics_written=(args.output / 'metrics.json').exists(),
            service_ownership='external controller; worker stops none'))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
