#!/usr/bin/env python3
"""Aggregate matched SocialIQA evaluations for one or two training epochs.

The run-map JSON contains cohort, population_dir, and runs; each run provides
run_id, result, and training_acceptance paths relative to the run-map file.
Require all eight models of the selected cohort and all 7,816 requests per
model. Validate checkpoint and bundle files, controller completion, service
request logs, and training records before computing metrics from ordered
responses. Write results only after every model passes validation.
"""

import os
import socket
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # Sibling modules import by name even under PYTHONSAFEPATH or -P.

import e10_contract as c
import e10_models as models
import run_e10  # the controller's own occurrence comparison and exact comparison, reused unchanged

RESULT_SCHEMA, PLAN_SCHEMA = 'jev-e10-socialiqa-result-v1', 'jev-e10-socialiqa-single-model-plan-v1'
EXIT_SCHEMA = 'jev-e10-socialiqa-controller-exit-v1'
COMPLETE = 'E10_MODEL_COMPLETE_PENDING_INDEPENDENT_REVIEW'
STAGES = dict(fresh=['export', 'bundle', 'services', 'e10_worker', 'request_witness'],
              reuse=['bundle', 'services', 'e10_worker', 'request_witness'])
WORKER_FILES = {'RUN_BINDING.json', 'metrics.json', 'question_scores.jsonl', 'ordered_calls.jsonl',
                'completion_calls.jsonl', 'planned_calls.jsonl'}
# The campaign record that verifies one model's training. The E4a/E5/E8 controller writes it for its POOL_ALL arms
# and for their E1 LL_A references alike; the E11 controller writes its own, which also records the epoch count
# and every source-record occurrence. A record of the other cohort's schema is refused.
ACCEPTANCE_SCHEMA = dict(e10_one_epoch='jev-e458-training-acceptance-v1', e11_two_epoch='jev-e11-training-acceptance-v1')
# The files each producer hashes into `evidence_sha256`, relative to the checkpoint directory
# (campaign_e458/run_e458.py and campaign_e11/run_e11.py, acceptance_records). A record must name exactly these.
_CAMPAIGN_EVIDENCE = ('training_config.json', 'training_metrics.json', 'campaign/initial_receipt.json', 'campaign/completion.json')
EVIDENCE = dict(e10_one_epoch=_CAMPAIGN_EVIDENCE,
                e11_two_epoch=_CAMPAIGN_EVIDENCE + ('e11_reference/receipt.json', 'e11_reference/epochs.jsonl'))
# Training inputs included in the overlap scan.
AUDITED_TRAINING_DATA_SHA256 = dict(F2='9253f1c937978174b2ff2e2959ce732c06fe3c0a220154fb962be1c2a1478017',
                                    F3='2a1cfac79cd64d9cb22f43c4b69c3283fd44c73c92a4468c661d27ac9c897e2a')
INITIAL_RECEIPT_SCHEMA, EPOCH_SCHEMA = 'jev-e1e2-campaign-initial-receipt-v1', 'jev-e11-epoch-record-v1'
# What the evaluation PLAN bound from the training records when the model was prepared; all must still hold.
BOUND_TRAINING = ('plan', 'plan_sha256', 'campaign', 'checkpoint', 'checkpoint_sha256', 'evidence_sha256',
                  'trained_source_sha256', 'epochs', 'optimizer_steps', 'source_records_seen')
SCOPES = ('full_source', 'auxiliary_mask')
WIN3_HOST = os.environ.get('JEV_EVALUATION_HOST', socket.gethostname())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def close(a, b, message):
    if a is None or b is None:
        require(a is None and b is None, message)
    else:
        require(type(a) in (int, float) and math.isfinite(a) and math.isclose(a, b, abs_tol=1e-12, rel_tol=0), message)


def members(root, identities):
    for name, expected in identities.items():
        require('/' not in name and '\\' not in name and name not in ('.', '..'), 'unsafe completion member')
        require((root / name).is_file() and c.file_sha(root / name) == expected, 'completed member changed: ' + name)


def mapped_runs(path):
    path = Path(path).resolve()
    mapping = c.read_json(path)
    cohort, rows = mapping.get('cohort'), mapping.get('runs')
    require(cohort in models.COHORTS, 'run-map must select exactly one cohort: ' + ' or '.join(models.COHORTS))
    fixed = {run_id for run_id, model in models.model_map().items() if model['cohort'] == cohort}
    require(isinstance(rows, list) and len(rows) == 8 and {row.get('run_id') for row in rows} == fixed,
            'exactly the eight fixed identities of this cohort are required; no partial or mixed cohort')

    def resolve(value):
        require(isinstance(value, str) and value, 'explicit evidence path required')
        value = Path(value).expanduser()
        return (value if value.is_absolute() else path.parent / value).resolve()
    resolved = []
    for row in rows:
        require(set(row) == {'run_id', 'result', 'training_acceptance'}, 'run-map evidence fields differ')
        resolved.append(dict(run_id=row['run_id'], result=resolve(row['result']), training_acceptance=resolve(row['training_acceptance'])))
    require(len({row['result'] for row in resolved}) == 8, 'result reused across models')
    return cohort, resolve(mapping.get('population_dir')), resolved


def training_acceptance(path, cohort, model, plan):
    """The campaign's own verification of this model's training, bound to the checkpoint that was evaluated.

    A recorded status alone is never accepted. Every evidence file the acceptance producer hashed is re-read; the
    training records are re-read and must still be exactly what the evaluation PLAN bound when the model was
    prepared; and the training data is traced through the trainer's own completion and initial records to the
    input the overlap audit scanned."""
    record, recorded, run_id = c.read_json(path), plan['training'], model['run_id']
    evaluated = recorded['checkpoint_sha256']
    wanted = dict(optimizer_steps=model['optimizer_steps'], planned_optimizer_steps=model['optimizer_steps'], status='complete')
    if cohort == 'e11_two_epoch':  # Both epochs and all their records, as that controller records them.
        wanted.update(epochs=model['epochs'], stopped_early=False, source_records_seen=model['source_records'],
                      requested_source_records=model['source_records'])
    require(record.get('schema') == ACCEPTANCE_SCHEMA[cohort] and record.get('status') == 'VERIFIED' and
            record.get('not_verified_because') == [] and
            (record.get('run_id'), record.get('fold'), record.get('seed')) == (model['run_id'], model['fold'], model['seed']) and
            record.get('checkpoint_sha256') == evaluated and Path(record.get('checkpoint', '')) == Path(plan['checkpoint']) and
            all(record['training'].get(key) == value for key, value in wanted.items()) and
            record['structure']['head_config'] == model['head_config'],
            'training acceptance record does not verify this evaluated checkpoint: ' + run_id)
    out, evidence = Path(plan['checkpoint']), record.get('evidence_sha256')
    require(isinstance(evidence, dict) and set(evidence) == set(EVIDENCE[cohort]),
            'training acceptance record does not name the evidence its producer hashes: ' + run_id)
    for name, digest in evidence.items():
        require((out / name).is_file() and c.file_sha(out / name) == digest,
                'campaign evidence is missing or changed since training acceptance: ' + name + ' of ' + run_id)
    state, now = models.training_state(model, plan['campaign_plan'])
    require(state == 'VERIFIED', 'training records no longer verify this evaluated checkpoint: ' + run_id)
    for key in BOUND_TRAINING:
        require(now[key] == recorded[key], 'training ' + key + ' differs from what the evaluation PLAN bound: ' + run_id)
    require(all(evidence[name] == recorded['evidence_sha256'][name] for name in ('training_config.json', 'training_metrics.json')),
            'the training acceptance record and the evaluation PLAN bind different training records: ' + run_id)
    return dict(training_acceptance_sha256=c.file_sha(path), training_evidence_sha256=now['evidence_sha256'],
                campaign_evidence_sha256=evidence, checkpoint_sha256=evaluated,
                **training_data(out, cohort, model, plan, evidence))


def training_data(out, cohort, model, plan, evidence):
    """The data and the horizon as the trainer itself recorded them, traced to the input the overlap audit scanned.

    completion.json, written when training ended, hashes every campaign record, including initial_receipt.json,
    whose training_data_sha256 is the trainer's own digest of the data file it opened (training_config args.data).
    Two equal plan strings would say nothing about the bytes that were trained on."""
    run_id, fold = model['run_id'], model['fold']
    receipt, completion = c.read_json(out / 'campaign/initial_receipt.json'), c.read_json(out / 'campaign/completion.json')
    config, campaign_plan = c.read_json(out / 'training_config.json'), c.read_json(plan['campaign_plan'])
    files = completion.get('files_sha256')
    require(completion.get('status') == 'complete' and completion.get('optimizer_updates') == model['optimizer_steps'] and
            completion.get('planned_optimizer_steps') == model['optimizer_steps'] and isinstance(files, dict) and
            files.get('initial_receipt.json') == evidence['campaign/initial_receipt.json'],
            'the completion record does not bind a complete run and its initial receipt: ' + run_id)
    for name, digest in files.items():
        require(PurePosixPath(name).name == name and (out / 'campaign' / name).is_file() and
                c.file_sha(out / 'campaign' / name) == digest, 'a campaign record changed since training ended: ' + name + ' of ' + run_id)
    data = receipt.get('training_data_sha256')
    require(receipt.get('schema') == INITIAL_RECEIPT_SCHEMA and receipt.get('planned_optimizer_steps') == model['optimizer_steps'] and
            receipt.get('head_config') == model['head_config'],
            'the initial receipt does not record this model, horizon and head: ' + run_id)
    require(data == AUDITED_TRAINING_DATA_SHA256[fold] == campaign_plan['data'][fold]['sha256'] and
            config['dse']['requested_source_records'] == model['source_records'],
            'the training data the trainer recorded is not the audited ' + fold + ' input over the full horizon: ' + run_id)
    if cohort == 'e11_two_epoch':
        # Cumulative counts after each epoch, as kev.e11_reference wrote them; both epochs flush their last update.
        per_epoch = model['optimizer_steps'] // model['epochs']
        rows = [(row.get('schema'), row.get('epoch'), row.get('optimizer_updates'), row.get('source_records_seen'))
                for row in c.read_jsonl(out / 'e11_reference/epochs.jsonl')]
        require(rows == [(EPOCH_SCHEMA, 0, per_epoch, model['records_per_epoch']),
                         (EPOCH_SCHEMA, 1, model['optimizer_steps'], model['source_records'])],
                'the E11 epoch records do not show two complete epochs: ' + run_id)
    return dict(training_data_sha256=data, training_data_path=config['args']['data'],
                requested_source_records=config['dse']['requested_source_records'])


def bundle_contract(code_tree, identity):
    """The code tree's own stdlib runtime/bundle_contract.py, loaded only after the tree's export and serving
    files are the pinned bytes of the campaign that trained the model."""
    models.source_state(code_tree, identity)
    spec = importlib.util.spec_from_file_location('e10_aggregate_bundle_contract_' + identity,
                                                  Path(code_tree) / 'runtime/bundle_contract.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_bundle(root, plan, result, model):
    """The served bundle revalidated now by the controller's own bundle_state(): every file the manifest binds is
    rehashed, the exporter's reload verification must pass, the manifest must name this checkpoint and source,
    and the served head tensors must be bit-identical to the trained head.pt. A reused bundle must also still be
    the one its finished full evaluation exported, with the result and exit receipt the PLAN bound."""
    run_id, code_tree = model['run_id'], Path(plan['code_tree'])
    require(plan['source_sha256'] == models.SOURCES[model['source_identity']], 'evaluation PLAN pins another source: ' + run_id)
    contract = bundle_contract(code_tree, model['source_identity'])
    if plan['bundle_mode'] == 'reuse':
        bound = plan['reused_bundle']
        state, detail = models.reuse_state(Path(bound['evaluation']), Path(bound['evaluation_exit_receipt']), model,
                                           plan['training'], code_tree, contract)
        require(state == 'VERIFIED' and run_e10.exact(detail) == run_e10.exact(bound),
                'the reused bundle or its full evaluation changed since the evaluation PLAN bound them: ' + run_id)
    else:
        require(Path(result['bundle']['path']) == (root / 'bundle').resolve(), 'a fresh export must be inside its run: ' + run_id)
        detail = models.bundle_state(root / 'bundle', model, plan['served_model_id'], plan['training'], code_tree, contract)
    require(set(result['bundle']) <= set(detail) and
            run_e10.exact({key: detail[key] for key in result['bundle']}) == run_e10.exact(result['bundle']),
            'the served bundle is not the one the controller verified: ' + run_id)
    return dict(bundle_manifest_sha256=detail['manifest_sha256'], bundle_files_rehashed=detail['bound_files'],
                head_tensors_equal_trained_head=detail['head_tensors_equal_trained_head'])


def verify_services(root, plan, result):
    """The four replicas' own request logs, replayed through the controller's occurrence comparison against the
    bound worker ledger. A missing log, or a missing, changed, extra or moved occurrence fails, and the replay
    must reproduce the witness the controller recorded."""
    logs = [root / 'services' / ('replica' + str(i)) / 'requests.jsonl' for i in range(4)]
    require(all(path.is_file() for path in logs), 'a service request log is missing: ' + plan['run_id'])
    require(isinstance(plan.get('ports'), list) and len(plan['ports']) == 4, 'evaluation PLAN does not record four service ports')
    endpoints = ['http://127.0.0.1:' + str(port) for port in plan['ports']]
    witness = run_e10.request_witness(root / 'worker/ordered_calls.jsonl', logs, endpoints, plan['served_model_wire_sha256'])
    require(run_e10.exact(witness) == run_e10.exact(result['served_requests']),
            'replayed service occurrences differ from the witness the controller recorded: ' + plan['run_id'])
    return dict(service_request_logs_sha256=[c.file_sha(path) for path in logs], per_replica=witness['per_replica'])


def verify_parent(path, model, pin):
    """The controller's result and the external receipt of its exit. Services are stopped by cleanup, so their
    native exits need not be zero; every other owned stage must have exited zero on its own."""
    run_id, path = model['run_id'], Path(path)
    require(path.name == 'RESULT.json' and path.is_file(), 'mandatory parent RESULT is absent: ' + str(path))
    root = path.parent
    require((root / 'CONTROLLER_EXIT.json').is_file(), 'external controller exit receipt is absent: ' + run_id)
    receipt, result, plan = c.read_json(root / 'CONTROLLER_EXIT.json'), c.read_json(path), c.read_json(root / 'PLAN.json')
    require(receipt.get('schema') == EXIT_SCHEMA and receipt.get('run_id') == run_id and type(receipt.get('native_exit')) is int and
            receipt['native_exit'] == 0 and receipt.get('result_sha256') == c.file_sha(path),
            'external controller exit/result binding differs: ' + run_id)
    mode = result.get('bundle_mode')
    require(result.get('schema') == RESULT_SCHEMA and result.get('status') == COMPLETE and result.get('run_id') == run_id and
            result.get('cohort') == model['cohort'] and type(result.get('native_exit')) is int and result['native_exit'] == 0 and
            mode in STAGES and result.get('completed') == STAGES[mode], 'parent result is not complete: ' + run_id)
    require('failure' not in result and result.get('worker_failure', 'missing') is None and 'worker_files_error' not in result and
            result.get('scientific_acceptance', 'missing') is None, 'parent reports a failure or a scientific verdict')
    require(c.file_sha(root / 'PLAN.json') == result['plan_sha256'], 'parent PLAN changed')
    served = plan.get('served_model_id')
    require(plan.get('schema') == PLAN_SCHEMA and all(plan.get(key) == value for key, value in model.items()
                                                      if key not in ('campaign_plan', 'code_tree')) and
            plan.get('bundle_mode') == mode and served == (run_id if mode == 'fresh' else plan['reused_bundle']['name']) and
            served in (run_id, run_id + '_eval') and plan['population_pin']['population_sha256'] == pin and
            plan.get('requests') == c.REQUESTS and plan.get('replicas') == 4 and plan.get('retries') == 0 and
            plan['service_limits'] == dict(max_state=c.LIMIT, max_branch=c.LIMIT, max_packed=c.LIMIT) and
            plan['source_sha256'] == models.SOURCES[model['source_identity']] and
            {gate: row['state'] for gate, row in plan['gates'].items()} ==
            dict(data='SEALED', training='VERIFIED', bundle='VERIFIED' if mode == 'reuse' else 'FRESH_EXPORT_AT_LAUNCH'),
            'parent PLAN identity, population or gates differ: ' + run_id)
    training = plan['training']
    require((training['epochs'], training['optimizer_steps'], training['source_records_seen']) ==
            (model['epochs'], model['optimizer_steps'], model['source_records']) and training['checkpoint'] == plan['checkpoint'] and
            training['trained_source_sha256'] == {name: plan['source_sha256']['kev/' + name] for name in models.TRAINED_KEV},
            'parent PLAN does not record the complete %d-epoch training: %s' % (model['epochs'], run_id))
    names = {'e10_worker'} | {'service' + str(i) for i in range(4)} | ({'export'} if mode == 'fresh' else set())
    processes = {p['name']: p for p in result['processes']}
    require(len(result['processes']) == len(names) and set(processes) == names, 'parent process coverage differs')
    for name, process in processes.items():
        require(type(process.get('native_exit')) is int, 'missing native process exit: ' + name)
        if name.startswith('service'):
            require(process.get('service') is True and process.get('stopped_by_cleanup') is True,
                    'service was not alive until owned cleanup: ' + name)
        else:
            native = c.read_json(root / (name + '.EXIT.json'))
            require(process['native_exit'] == 0 and process.get('service') is False and process.get('stopped_by_cleanup') is False and
                    native.get('name') == name and all(process.get(k) == v for k, v in native.items()),
                    'nonzero, interrupted or unreceipted native stage: ' + name)
    cleanup, gpu = result['cleanup'], result['gpu_post']
    require(cleanup.get('complete') is True and cleanup.get('residual') == [] and
            {p['name'] for p in cleanup['processes']} == names and len(cleanup['processes']) == len(names) and
            all(p.get(k) == processes[p['name']].get(k) for p in cleanup['processes'] for k in ('pid', 'native_exit', 'stopped_by_cleanup')),
            'parent cleanup is incomplete')
    require('error' not in gpu and gpu.get('uuid') == plan['gpu_uuid'] and gpu.get('index') == plan['gpu_index'] and
            gpu.get('host') == WIN3_HOST and gpu.get('account') == 'user' and 'H20' in gpu.get('name', '') and
            type(gpu.get('memory_mib')) is int and 0 <= gpu['memory_mib'] < 500 and isinstance(gpu.get('compute_query'), str) and
            not any(line.split(',')[0].strip() == plan['gpu_uuid'] for line in gpu['compute_query'].splitlines()),
            'parent GPU release is not verified')
    served_requests = result['served_requests']
    require(served_requests.get('requests') == c.REQUESTS and served_requests.get('exact_request_response_occurrences_equal') is True and
            served_requests.get('interface_requests') == dict(direct=c.QUESTIONS, nli=3 * c.QUESTIONS) and
            served_requests.get('served_model_wire_sha256') == plan['served_model_wire_sha256'] and
            len(served_requests['per_replica']) == 4 and sum(served_requests['per_replica']) == c.REQUESTS,
            'parent served request population differs')
    state, bundle = result['bundle'], Path(result['bundle']['path'])
    manifest, verification = c.read_json(bundle / 'model_manifest.json'), c.read_json(bundle / 'validation_report.json')
    require((mode == 'fresh') == (bundle == (root / 'bundle').resolve()) and
            c.file_sha(bundle / 'model_manifest.json') == result['bundle_manifest_sha256'] == state['manifest_sha256'] and
            manifest['name'] == served and manifest['head_config'] == model['head_config'] and manifest['option_isolation'] is False and
            manifest['base_revision'] == models.BASE_REVISION and manifest['checkpoint_hashes'] == training['checkpoint_sha256'] and
            manifest['files']['classification_head.safetensors'] == result['bundle_head_sha256'] and
            state.get('head_tensors_equal_trained_head') is True and
            all(manifest['source_hashes'].get(name) == value for name, value in plan['source_sha256'].items()),
            'parent bundle does not bind this checkpoint, source and trained head: ' + run_id)
    require(verification.get('status') == 'PASS' and verification.get('head_config') == model['head_config'] and
            verification.get('manifest_sha256') == result['bundle_manifest_sha256'] and
            type(verification.get('max_probability_delta')) in (int, float) and 0 <= verification['max_probability_delta'] <= 1e-5 and
            all(verification.get(k) is True for k in ('argmax_equal', 'source_unchanged', 'checkpoint_unchanged')),
            'parent export/reload verification differs')
    return root, plan, result


def verify_worker(root, plan, result, model, requests):
    """The worker's ledgers against the sealed population; returns the complete ordered events."""
    worker, run_id, served = root / 'worker', model['run_id'], plan['served_model_id']
    require(not (worker / 'FAILED.json').exists(), 'failed worker cannot be aggregated: ' + run_id)
    complete, binding = c.read_json(worker / 'COMPLETE.json'), c.read_json(worker / 'RUN_BINDING.json')
    pin = plan['population_pin']['population_sha256']
    require(complete.get('status') == 'COMPLETE' and complete.get('run_id') == run_id and complete.get('requests') == c.REQUESTS and
            complete.get('served_model_id') == served and complete.get('predictions_generated') is True and
            complete.get('population_sha256') == pin and complete.get('manifest_sha256') == result['bundle_manifest_sha256'] and
            complete.get('head_config') == model['head_config'] and WORKER_FILES <= set(complete['files']) and
            c.file_sha(worker / 'COMPLETE.json') == result['worker'].get('complete_sha256') and
            result['worker'].get('status') == 'COMPLETE' and result['worker'].get('requests') == c.REQUESTS,
            'worker completion does not bind this model, bundle and population: ' + run_id)
    members(worker, complete['files'])
    files = result['worker_files']
    require(set(files) == set(complete['files']) | {'COMPLETE.json'} == {p.name for p in worker.iterdir() if p.is_file()} and
            all(identity == dict(bytes=(worker / name).stat().st_size, sha256=c.file_sha(worker / name)) for name, identity in files.items()),
            'parent worker file coverage differs')
    require(binding['run_id'] == run_id and binding['served_model_id'] == served and binding['retries'] == 0 and
            binding['worker_sha256'] == plan['package_sha256']['e10_worker.py'] and binding['head_config'] == model['head_config'] and
            binding['manifest_sha256'] == complete['manifest_sha256'] and len(binding['health']) == 4 and
            len({h['endpoint'] for h in binding['health']}) == 4, 'worker binding mismatch: ' + run_id)
    for endpoint in binding['health']:
        health = endpoint['config']
        require(health['model'] == served and health['head_config'] == model['head_config'] and
                health['manifest_sha256'] == complete['manifest_sha256'] and health['prefix_cache'] is False and
                health['strict'] is True and [health['max_state'], health['max_branch'], health['max_packed']] == [c.LIMIT] * 3,
                'service identity, context or cache differs')
    events, identities, stream, counts = [], [], hashlib.sha256(), Counter()
    for ordinal, event in enumerate(jsonl(worker / 'ordered_calls.jsonl')):
        require(ordinal < len(requests), 'extra ordered call')
        expected = requests[ordinal]
        body = dict(expected['request'], model=served)
        require(event['ordinal'] == ordinal and event['request_id'] == expected['request_id'] and not event.get('error'),
                'missing, duplicate or failed ordered call: ' + run_id)
        require(c.wire(event['request']) == c.wire(body) and (event['interface'], event['item_id'], event['candidate_id']) ==
                (expected['interface'], expected['item_id'], expected['candidate_id']), 'ordered request identity differs')
        response = event['response']
        require(response['model'] == served and response['head_config'] == model['head_config'] and
                response['head_kind'] == 'pointer' and response['probability_source'] == 'classification_head',
                'response identity differs')
        require(response['input_tokens'] == expected['input_tokens'] and response['state_tokens'] == expected['state_tokens'],
                'actual tokens differ from the qualified witness')
        stream.update(c.wire(body) + b'\n')
        identities.append(c.digest(c.wire(event)))
        counts[event['interface']] += 1
        events.append(event)
    require(len(events) == c.REQUESTS and dict(counts) == dict(direct=c.QUESTIONS, nli=3 * c.QUESTIONS) and
            stream.hexdigest() == plan['served_model_wire_sha256'] == c.stream_sha(requests, served),
            'ordered calls do not cover the complete sealed population: ' + run_id)
    seen = set()
    for event in jsonl(worker / 'completion_calls.jsonl'):  # Completion order may differ; every event exactly once.
        ordinal = event['ordinal']
        require(type(ordinal) is int and 0 <= ordinal < c.REQUESTS and ordinal not in seen and
                c.digest(c.wire(event)) == identities[ordinal], 'completion event missing, duplicated or changed')
        seen.add(ordinal)
    planned = 0
    for ordinal, event in enumerate(jsonl(worker / 'planned_calls.jsonl')):
        require(ordinal < c.REQUESTS and event['ordinal'] == ordinal and event['request_id'] == requests[ordinal]['request_id'] and
                c.wire(event['request']) == c.wire(dict(requests[ordinal]['request'], model=served)), 'planned call differs')
        planned += 1
    require(len(seen) == c.REQUESTS and planned == c.REQUESTS, 'incomplete completion or planned ledger: ' + run_id)
    return events, dict(worker_complete_sha256=c.file_sha(worker / 'COMPLETE.json'),
                        ordered_calls_sha256=complete['files']['ordered_calls.jsonl'])


def recompute(requests, events, scoring):
    """Independent rescoring of every question from the raw responses: entailment by semantic key, first exact
    maximum in source A/B/C order, no renormalization, equal texts kept as separate occurrences."""
    values = {name: {} for name in c.INTERFACES}
    for expected, event in zip(requests, events):
        (qid, distribution), = event['response']['probabilities'].items()
        require(qid == next(iter(expected['request']['questions'])) and list(distribution) == expected['semantic_keys'],
                'live semantic keys or order differ')
        raw = list(distribution.values())
        require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in raw) and abs(sum(raw) - 1) <= 1e-5,
                'served values are not a probability distribution')
        slot = values[expected['interface']].setdefault(expected['item_id'], {})
        if expected['interface'] == 'direct':
            require(not slot, 'duplicate direct response')
            slot.update(distribution)
        else:
            require(expected['candidate_id'] not in slot, 'duplicate candidate occurrence')
            slot[expected['candidate_id']] = distribution['entailment']
    rows = {}
    for interface in c.INTERFACES:
        for question in scoring:
            slot = values[interface].get(question['item_id'], {})
            require(set(slot) == set(c.CANDIDATES), 'missing scored candidate identity')
            scores = [slot[cid] for cid in c.CANDIDATES]
            ties = [cid for cid, score in zip(c.CANDIDATES, scores) if score == max(scores)]
            rows[(interface, question['item_id'])] = dict(
                source_ordinal=question['source_ordinal'], candidate_order=list(c.CANDIDATES), candidate_scores=scores,
                gold=question['gold'], prediction=ties[0], correct=int(ties[0] == question['gold']), exact_tie=len(ties) > 1,
                maximum_candidates=ties, mask_status=question['mask_status'], in_auxiliary_subset=question['in_auxiliary_subset'],
                mask_reason=question['mask_reason'], duplicate_text_groups=question['duplicate_text_groups'])
    metrics = {}
    for interface in c.INTERFACES:
        own = [row for (name, _), row in rows.items() if name == interface]
        require(len(own) == c.QUESTIONS, 'fixed question denominator changed')

        def part(selected):
            return dict(n_questions=len(selected), correct=sum(r['correct'] for r in selected),
                        accuracy=sum(r['correct'] for r in selected) / len(selected) if selected else None)
        metrics[interface] = dict(full_source=part(own), auxiliary_mask=part([r for r in own if r['in_auxiliary_subset']]),
                                  excluded_detected=part([r for r in own if r['mask_status'] == c.DETECTED]),
                                  excluded_unresolved=part([r for r in own if r['mask_status'] == c.UNRESOLVED]),
                                  exact_tie_questions=sum(r['exact_tie'] for r in own))
    return rows, metrics


def compare_worker(root, rows, metrics):
    """The worker's own scores must be the recomputed ones, question by question."""
    reported = c.read_json(root / 'worker/metrics.json')
    require(set(reported['interfaces']) == set(metrics) and reported['primary'] == 'full_source', 'worker metric coverage differs')
    for interface, parts in metrics.items():
        for name, value in parts.items():
            other = reported['interfaces'][interface][name]
            if name == 'exact_tie_questions':
                require(other == value, 'reported exact ties differ')
            else:
                require((other['n_questions'], other['correct']) == (value['n_questions'], value['correct']), 'reported counts differ: ' + name)
                close(other['accuracy'], value['accuracy'], 'reported accuracy differs: ' + interface + '/' + name)
    scored = set()
    for record in jsonl(root / 'worker/question_scores.jsonl'):
        key = (record['interface'], record['item_id'])
        require(key in rows and key not in scored, 'question score coverage differs')
        require(all(record[field] == value for field, value in rows[key].items()), 'question-level score differs: ' + ':'.join(key))
        scored.add(key)
    require(scored == set(rows), 'incomplete question scoring ledger')


def descriptive(values):
    require(len(values) == 4, 'exactly four settings required; no partial mean')
    if any(v is None for v in values):
        return None
    sd = statistics.stdev(values)
    return dict(n_settings=4, mean=statistics.mean(values), sample_sd=sd, descriptive_se=sd / math.sqrt(4))


def aggregate_scores(cohort, metrics):
    """Pure arithmetic over eight already-verified models of one cohort. Accuracies are fractions, differences
    are percentage points: pooling minus pointer per setting and interface, and their NLI-minus-direct interaction."""
    runs = models.cohort_runs(cohort)
    require(set(metrics) == {run_id for pair in runs.values() for run_id in pair.values()}, 'fixed eight models required; no partial mean')
    settings, series = [], {}
    for (fold, seed), pair in runs.items():
        row = dict(fold=fold, seed=seed, pointer=pair['pointer'], pooling=pair['pooling'])
        for scope in SCOPES:
            cell, deltas = {}, {}
            for interface in c.INTERFACES:
                pointer, pooling = (metrics[pair[arm]][interface][scope] for arm in ('pointer', 'pooling'))
                require(pointer['n_questions'] == pooling['n_questions'], 'matched pair uses different denominators')
                delta = None if pointer['accuracy'] is None else 100 * (pooling['accuracy'] - pointer['accuracy'])
                cell[interface] = dict(n_questions=pointer['n_questions'], pointer_correct=pointer['correct'],
                                       pooling_correct=pooling['correct'], pointer_accuracy=pointer['accuracy'],
                                       pooling_accuracy=pooling['accuracy'], pooling_minus_pointer_pp=delta)
                deltas[interface] = delta
                for name in ('pointer_accuracy', 'pooling_accuracy', 'pooling_minus_pointer_pp'):
                    series.setdefault((scope, interface, name), []).append(cell[interface][name])
            cell['interaction_nli_minus_direct_pp'] = None if None in deltas.values() else deltas['nli'] - deltas['direct']
            series.setdefault((scope, 'interaction_nli_minus_direct_pp'), []).append(cell['interaction_nli_minus_direct_pp'])
            row[scope] = cell
        settings.append(row)
    summary = {scope: dict({interface: {name: descriptive(series[(scope, interface, name)])
                                        for name in ('pointer_accuracy', 'pooling_accuracy', 'pooling_minus_pointer_pp')}
                            for interface in c.INTERFACES},
                           interaction_nli_minus_direct_pp=descriptive(series[(scope, 'interaction_nli_minus_direct_pp')]))
               for scope in SCOPES}
    return dict(status='COMPLETE_EIGHT_MODEL_COHORT_AGGREGATE', cohort=cohort, epochs=models.COHORTS[cohort]['epochs'],
                models=8, settings=4, primary='full_source',
                statistical_units='Four specified fold/seed settings with overlapping folds. Mean and sample SD/sqrt(4) are '
                                  'descriptive, not independent-confirmation uncertainty.',
                auxiliary_mask='One predeclared mask for every model and interface; excluded questions are reported by reason.',
                cohorts_not_mixed='E10 one-epoch and E11 two-epoch cohorts are separate aggregates and are never averaged.',
                matched_settings=settings, four_setting_summary=summary, per_model=metrics)


def aggregate(run_map):
    """-> the complete aggregate, or ValueError. Reads only."""
    cohort, population_root, rows = mapped_runs(run_map)
    pin = c.population_pin()
    population, requests, scoring = c.load_population(population_root, pin)
    metrics, evidence, packages = {}, [], set()
    catalog = models.model_map()
    for row in rows:
        model = catalog[row['run_id']]
        root, plan, result = verify_parent(row['result'], model, pin['population_sha256'])
        proof = training_acceptance(row['training_acceptance'], cohort, model, plan)
        bundle = verify_bundle(root, plan, result, model)
        events, worker = verify_worker(root, plan, result, model, requests)
        services = verify_services(root, plan, result)
        scores, metrics[row['run_id']] = recompute(requests, events, scoring)
        compare_worker(root, scores, metrics[row['run_id']])
        packages.add(json.dumps(plan['package_sha256'], sort_keys=True))
        evidence.append(dict(run_id=row['run_id'], bundle_mode=plan['bundle_mode'], served_model_id=plan['served_model_id'],
                             result_sha256=c.file_sha(row['result']), controller_exit_sha256=c.file_sha(root / 'CONTROLLER_EXIT.json'),
                             accepted_occurrences=c.REQUESTS, **bundle, **services, **worker, **proof))
    require(len(packages) == 1, 'the eight runs were not produced by one version of this package')
    result = aggregate_scores(cohort, metrics)
    result.update(run_map_sha256=c.file_sha(run_map), population_sha256=pin['population_sha256'],
                  mask_counts=population['mask_counts'], package_sha256=json.loads(packages.pop()), verified_evidence=evidence)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-map', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    require(not args.output.exists(), 'output must be new')
    result = aggregate(args.run_map)
    # No output directory or mean exists before all eight runs and their proofs pass.
    args.output.mkdir(parents=True)
    c.write_json(args.output / ('SOCIALIQA_AGGREGATE_' + result['cohort'] + '.json'), result)
    print(json.dumps(dict(status=result['status'], cohort=result['cohort'], output=str(args.output.resolve()))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
