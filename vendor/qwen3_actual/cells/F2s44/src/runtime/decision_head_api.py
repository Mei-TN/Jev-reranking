"""Process-local OpenCompass overlay; the official client/scorers stay unchanged."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from queue import Queue
import urllib.request
from opencompass.registry import MODELS
from opencompass.models.base_api import BaseAPIModel
from opencompass.datasets.decision_native import NativeJevBenchEvaluator, NativeKevEvaluator
from .bundle_contract import config_identity, native_prediction


class HeadJevBenchEvaluator(NativeJevBenchEvaluator):
    probability_source = 'classification_head'


class HeadKevEvaluator(NativeKevEvaluator):
    probability_source = 'classification_head'


@MODELS.register_module()
class DecisionHeadAPI(BaseAPIModel):
    def __init__(self, path, endpoints, head_config, api_key_env=None,
                 allow_legacy_h0=False, **kwargs):
        super().__init__(path=path, **kwargs)
        if not endpoints: raise ValueError('At least one decision endpoint required')
        self.expected_config = config_identity(head_config)
        self.allow_legacy_h0 = allow_legacy_h0
        self.headers = {'Content-Type': 'application/json'}
        if api_key_env: self.headers['Authorization'] = 'Bearer '+os.environ[api_key_env]
        self.endpoints = Queue()
        for endpoint in endpoints: self.endpoints.put(endpoint.rstrip('/'))
        self.workers = self.max_workers = len(endpoints)

    def generate(self, inputs, max_out_len=1, **kwargs):
        def one(value):
            if not isinstance(value, str): value = ''.join(item['prompt'] for item in value)
            envelope = json.loads(value)
            body = {**envelope['request'], 'model': self.path}
            endpoint = self.endpoints.get()
            try:
                req = urllib.request.Request(endpoint+'/v1/decision',
                    json.dumps(body, allow_nan=False).encode(), self.headers)
                with urllib.request.urlopen(req, timeout=300) as response: result = json.load(response)
                # The unchanged upstream parser requires the exact one-key
                # envelope. Identity was checked above and lives in run/service
                # evidence; scorer metadata is supplied by Head*Evaluator.
                return native_prediction(result, body, self.path, self.expected_config,
                                         envelope['kind'], self.allow_legacy_h0)
            finally:
                self.endpoints.put(endpoint)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(one, inputs))

    def get_ppl(self, *args, **kwargs):
        raise NotImplementedError('Classification heads do not expose language-model perplexity')
