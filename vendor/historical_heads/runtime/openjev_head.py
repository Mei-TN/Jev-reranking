"""Isolated native NLI transport; OpenJev task loaders and scorers are unchanged."""
import json
import os
from pathlib import Path
from queue import Queue
import time
import urllib.error
import urllib.request
from agentic_evaluate.benchmarks.openjev_bridge import LABELS, RemoteNLI
from agentic_evaluate.benchmarks.openjev_pointer import pointer_request
from .bundle_contract import config_identity, validate_probabilities, validate_response


class ClassificationNLI(RemoteNLI):
    def __init__(self, config, output):
        if config.get('vision') or config.get('replay_model_calls'):
            raise ValueError('This new MC11 run is text-only and does not replay old model calls')
        super().__init__(config, output)
        self.expected_config = config_identity(config['head_config'])
        self.allow_legacy = config.get('allow_legacy_h0', False)
        self.endpoints = Queue()
        self.headers = {'Content-Type':'application/json'}
        if config.get('api_key_env'):
            self.headers['Authorization'] = 'Bearer '+os.environ[config['api_key_env']]
        self.identities = []
        keys = ('model','probability_source','head_kind','head_config','manifest_sha256',
                'head_sha256','dtype','head_dtype','merge','lora_scale','temperature',
                'prefix_cache','max_state','max_branch','max_packed','strict')
        for endpoint in config['pointer_endpoints']:
            endpoint = endpoint.rstrip('/')
            req = urllib.request.Request(endpoint+'/health', headers=self.headers)
            with urllib.request.urlopen(req, timeout=config['request_timeout_s']) as response:
                identity = json.load(response)
            validate_response(identity, config['model'], self.expected_config, self.allow_legacy)
            if identity.get('strict') is not True or identity.get('temperature') != 1 or identity.get('lora_scale') != 1:
                raise ValueError('Expected strict native temperature-1/scale-1 service')
            if self.identities and any(identity.get(k) != self.identities[0]['config'].get(k) for k in keys):
                raise ValueError('Classification replicas have different effective configurations')
            self.identities.append(dict(endpoint=endpoint, config=identity))
            self.endpoints.put(endpoint)
        if not self.identities: raise ValueError('No classification endpoint')
        self.probability_source = self.identities[0]['config']['probability_source']
        (Path(output)/'classification_service_configs.json').write_text(json.dumps(self.identities, indent=2)+'\n')

    def one(self, pair, image=None):
        if image is not None: raise ValueError('MC11 does not use image input')
        # Reuse the frozen actual native template and criterion order verbatim.
        body = pointer_request(self.config['model'], pair)
        event = dict(started_at=time.time(), request=body,
                     probability_source=self.probability_source,
                     expected_head_config=self.expected_config, template='kev-mnli-v1')
        tick = time.monotonic()
        try:
            endpoint = self.endpoints.get()
            event['endpoint'] = endpoint
            try:
                req = urllib.request.Request(endpoint+'/v1/decision',
                    json.dumps(body, allow_nan=False).encode(), self.headers)
                with urllib.request.urlopen(req, timeout=self.config['request_timeout_s']) as response:
                    raw = json.load(response)
            finally:
                self.endpoints.put(endpoint)
            event['response'] = raw
            validate_response(raw, self.config['model'], self.expected_config, self.allow_legacy)
            if raw['probability_source'] != self.probability_source:
                raise ValueError('Probability source changed after endpoint identity check')
            distributions = validate_probabilities(raw, body)
            result = [distributions['relation'][label] for label in LABELS]
            event['probabilities'] = result
            return result  # Full precision, same label order; no calibration/rounding.
        except Exception as exc:
            if isinstance(exc, urllib.error.HTTPError):
                event['http_status'] = exc.code
                event['error_body'] = exc.read().decode(errors='replace')[:4096]
            event['error_type'] = type(exc).__name__
            raise
        finally:
            event['elapsed_s'] = time.monotonic()-tick
            with self.lock:
                self.calls += 1
                self.secs += event['elapsed_s']
                with (self.output/'model_calls.jsonl').open('a') as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
