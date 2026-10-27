"""Native MC scorer with explicit submission, completion and ordered response witnesses."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import urllib.request
from common import durable, read, sha, worker_alarm
from population import MC_COUNTS, canonical, counter_digest, dataset_image, request_digest, request_token_digest


def main():
    worker_alarm()
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--task', required=True, choices=MC_COUNTS)
    p.add_argument('--output', required=True)
    p.add_argument('--endpoints', nargs=4, required=True)
    a = p.parse_args()
    plan = read(a.plan)
    source, rw = Path(plan['code_tree']), Path(plan['rw_root'])
    sys.path[:0] = [str(source), str(rw/'src')]
    os.environ['RUNWEAVER_OPENJEV_CACHE'] = plan['runtime_cache']
    from agentic_evaluate.benchmarks import openjev_adapter as adapter_module
    from agentic_evaluate.benchmarks import openjev_bridge as bridge
    from agentic_evaluate.benchmarks import openjev_pointer as pointer
    from agentic_evaluate.benchmarks.native_task_runtime import locked
    from runtime.openjev_head import ClassificationNLI
    from runtime.bundle_contract import validate_probabilities, validate_response
    import numpy as np
    for module in (adapter_module, bridge, pointer):
        if not Path(module.__file__).resolve().is_relative_to(rw.resolve()/'src'):
            raise ValueError('wrong native module location')
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    expected = read(plan['expected_mc_population'])['tasks'][a.task]
    revisions = read(rw/'scripts/openjev/data_revisions.json')
    native = SimpleNamespace(environment='static', user=None, seed=None,
        task_ids=[a.task], output_dir=str(output.resolve()), max_concurrency=4,
        agent=SimpleNamespace(model=plan['run_id'], endpoint=a.endpoints[0], api_key_env=None, args={}),
        options=dict(backend='pointer', pointer_endpoints=a.endpoints,
                     data_revisions=revisions, request_timeout_s=300))
    adapter = adapter_module.OpenJevAdapter()
    config = adapter.effective_config(native)
    if a.task.startswith('gpqa') and not config.get('gpqa_path'):
        raise ValueError('frozen GPQA absent')
    config.update(head_config=plan['head_config'], head_kind=plan['head_config']['kind'],
                  allow_legacy_h0=False, transport='native_decision_http',
                  protocol_status='classification_head_nli_adapter')

    class IndexedNLI(ClassificationNLI):
        def __init__(self):
            super().__init__(config, output)
            self.next_ordinal = 0
            self.request_counts, self.token_counts = Counter(), Counter()
            self.order_sha = hashlib.sha256()
            self.token_order_sha = hashlib.sha256()

        def indexed(self, item):
            ordinal, pair = item
            body = pointer.pointer_request(config['model'], pair)
            event = dict(ordinal=ordinal, request=body, template='kev-mnli-v1',
                         probability_source=self.probability_source,
                         expected_head_config=self.expected_config, started_at=time.time())
            tick = time.monotonic()
            try:
                endpoint = self.endpoints.get()
                event['endpoint'] = endpoint
                try:
                    req = urllib.request.Request(endpoint+'/v1/decision', canonical(body), self.headers)
                    with urllib.request.urlopen(req, timeout=config['request_timeout_s']) as response:
                        raw = json.load(response)
                finally:
                    self.endpoints.put(endpoint)
                event['response'] = raw
                validate_response(raw, config['model'], self.expected_config, False)
                if raw['probability_source'] != self.probability_source:
                    raise ValueError('probability source changed')
                probabilities = validate_probabilities(raw, body)
                event['probabilities'] = [probabilities['relation'][label] for label in bridge.LABELS]
                return event
            except Exception as exc:
                event['error_type'] = type(exc).__name__
                raise
            finally:
                event['elapsed_s'] = time.monotonic()-tick
                with self.lock:
                    self.calls += 1; self.secs += event['elapsed_s']
                    with (output/'model_calls.jsonl').open('a') as f:
                        f.write(canonical(event).decode()+'\n')

        def predict(self, pairs):
            batch = []
            with (output/'planned_requests.jsonl').open('a') as f:
                for pair in pairs:
                    ordinal = self.next_ordinal
                    self.next_ordinal += 1
                    body = pointer.pointer_request(config['model'], pair)
                    f.write(canonical(dict(ordinal=ordinal, request=body,
                                           request_sha256=request_digest(body))).decode()+'\n')
                    batch.append((ordinal, pair))
            values = []
            with (output/'ordered_model_calls.jsonl').open('a') as f:
                for event, planned in zip(bridge.ordered_parallel(self.indexed, batch, 4), batch):
                    if event['ordinal'] != planned[0] or request_digest(event['request']) != request_digest(pointer.pointer_request(config['model'], planned[1])):
                        raise ValueError('native ordered response does not match planned ordinal/body')
                    f.write(canonical(event).decode()+'\n')
                    key = request_digest(event['request'])
                    self.order_sha.update(key.encode()+b'\n')
                    self.request_counts[key] += 1
                    token_key=request_token_digest(event['request'], event['response']['input_tokens'])
                    self.token_counts[token_key] += 1
                    self.token_order_sha.update(token_key.encode()+b'\n')
                    values.append(event['probabilities'])
            if len(values) != len(batch):
                raise ValueError('missing ordered native response')
            return np.array(values)

    with locked(adapter.cache/'install.lock', exclusive=False), locked(output/'.run.lock', nonblocking=True):
        remote = IndexedNLI()
        config['probability_source'] = remote.probability_source
        durable(output/'effective_config.json', config)
        durable(output/'source_binding.json', dict(
            native_modules={str(Path(m.__file__).resolve()):sha(m.__file__) for m in (adapter_module, bridge, pointer)},
            eval_worker_sha256=sha(__file__), training_overlay={str(p.relative_to(source)):sha(p)
              for d in ('kev','runtime') for p in sorted((source/d).glob('*.py'))},
            native_source=adapter_module.verify_source(), data_revisions_sha256=sha(rw/'scripts/openjev/data_revisions.json'),
            executable=sys.executable, model=plan['run_id'], head_config=plan['head_config']))
        original, old_argv = pointer.PointerNLI, sys.argv
        def isolated(bound_config, bound_output):
            if bound_config != config or Path(bound_output).resolve() != output.resolve():
                raise ValueError('wrong native bridge binding')
            return remote
        try:
            pointer.PointerNLI = isolated
            sys.argv = ['openjev_bridge', str(output.resolve())]
            bridge.main()
            raw = read(output/'native_result.json')
            observed = raw[plan['run_id']]
            n = observed[a.task].get('n', observed[a.task].get('n_questions'))
            if set(observed) != {a.task} or n != expected['questions']:
                raise ValueError('native full question count differs')
            if remote.calls != expected['calls'] or remote.next_ordinal != expected['calls']:
                raise ValueError('native complete request count differs')
            if counter_digest(remote.request_counts) != expected['request_occurrence_sha256'] or counter_digest(remote.token_counts) != expected['request_encoder_tokens_occurrence_sha256']:
                raise ValueError('request/option/duplicate/token population differs from accepted reference')
            if expected['dataset_identity']['kind'] == 'native_dataset_loader':
                rows = [json.loads(l) for l in (output/'dataset_inventory.jsonl').read_text().splitlines()]
                dataset_sha = hashlib.sha256(dataset_image(rows)).hexdigest()
                if dataset_sha != expected['dataset_identity']['identity_sha256']:
                    raise ValueError('native ordered dataset inventory differs')
            else:
                # Chess is the fixed native synthetic load_chess(500,seed=0), with all
                # request bodies checked above; never invent a loader inventory.
                if a.task != 'chess' or (output/'dataset_inventory.jsonl').exists():
                    raise ValueError('unexpected non-loader population')
                current = adapter_module.verify_source()
                if current['revision'] != expected['dataset_identity']['source_revision'] or current['files']['code/eval.py'] != expected['dataset_identity']['source_sha256']:
                    raise ValueError('native chess generator source differs')
                dataset_sha = None
            durable(output/'summary.json', dict(complete=True, task=a.task, questions=n,
                calls=remote.calls, model=plan['run_id'], head_config=plan['head_config'],
                metrics=adapter_module.flatten_metrics(raw), native_result_sha256=sha(output/'native_result.json'),
                ordered_request_sha256=remote.order_sha.hexdigest(),
                ordered_request_encoder_token_sha256=remote.token_order_sha.hexdigest(),
                ordered_call_file_sha256=sha(output/'ordered_model_calls.jsonl'),
                planned_call_file_sha256=sha(output/'planned_requests.jsonl'),
                completion_call_file_sha256=sha(output/'model_calls.jsonl'),
                request_occurrence_sha256=counter_digest(remote.request_counts),
                request_encoder_tokens_occurrence_sha256=counter_digest(remote.token_counts),
                dataset_inventory_identity_sha256=dataset_sha,
                expected_reference_sha256=sha(plan['expected_mc_population']),
                started_epoch=started, finished_epoch=time.time()))
        finally:
            pointer.PointerNLI, sys.argv = original, old_argv


if __name__ == '__main__':
    main()
