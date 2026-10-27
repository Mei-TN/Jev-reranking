"""Test worker loading, serialization, and failure handling with a mocked model.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from jev.config import root, sha
from jev.evaluate import score_files


FAKE_TORCH = r'''
from contextlib import contextmanager
import json
import os
from pathlib import Path

bfloat16 = 'fixture-bfloat16'

def event(name):
    with (Path(os.environ['JEV_WORKER_FIXTURE'])/'events.jsonl').open('a') as out:
        out.write(json.dumps(name)+'\n')

def load(path, *args, **kwargs):
    event('torch.load')
    return json.load(path) if hasattr(path, 'read') else json.loads(Path(path).read_text())

@contextmanager
def inference_mode():
    event('inference-enter')
    yield
    event('inference-exit')

class _CUDA:
    @staticmethod
    def synchronize(): event('synchronize')
cuda = _CUDA()
'''

FAKE_EVALUATOR = r'''
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch

class Vector:
    def __init__(self, values): self.values = values
    def tolist(self): return self.values

class KnownModel:
    def __init__(self, metadata):
        self.head = SimpleNamespace(q=SimpleNamespace(out_features=metadata.get('head_dim',256)))
        if 'head_config' in metadata: self.head_config = metadata['head_config']
    def encode(self, tokenizer, record, **kwargs):
        folder=Path(os.environ['JEV_WORKER_FIXTURE'])
        (folder/'encoded_record.json').write_text(json.dumps(record))
        (folder/'encode_limits.json').write_text(json.dumps(kwargs))
        return {'ids':[11,22,33,44], 'seg':[0,0,1,1]}
    def probs(self, encoded):
        torch.event('forward')
        return [Vector(value) for value in json.loads(os.environ['JEV_WORKER_PROBABILITIES'])]

def load(run, device, dtype=None, merge=True):
    metadata=torch.load(str(Path(run)/'head.pt'),map_location='cpu')
    if os.environ.get('JEV_FILELIKE_LOAD'):
        torch.load(io.StringIO('{"auxiliary":true}'))
    (Path(os.environ['JEV_WORKER_FIXTURE'])/'loaded_metadata.json').write_text(json.dumps({
        'metadata':metadata,'device':device,'dtype':dtype,'merge':merge}))
    return object(), KnownModel(metadata)
'''


class PredictionWorkerContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name).resolve()
        self.dependencies = self.folder / "fixture-dependencies"
        self.dependencies.mkdir()
        (self.dependencies / "torch.py").write_text(FAKE_TORCH, encoding="utf-8")
        self.source = self.folder / "selected-source"
        (self.source / "kev").mkdir(parents=True)
        (self.source / "kev/__init__.py").write_text("", encoding="utf-8")
        shutil.copyfile(root() / "vendor/head1_v1/kev/api.py", self.source / "kev/api.py")
        (self.source / "kev/evaluate.py").write_text(FAKE_EVALUATOR, encoding="utf-8")
        self.checkpoint = self.folder / "checkpoint"
        self.checkpoint.mkdir()
        self.metadata = dict(base="/historical/base", head_dim=1024, head={"fixture": True})
        (self.checkpoint / "head.pt").write_text(json.dumps(self.metadata), encoding="utf-8")
        self.base = self.folder / "local-base"
        self.base.mkdir()
        self.requests, self.output = self.folder / "requests.jsonl", self.folder / "predictions.jsonl"
        self.rows = [dict(id="fixture-id", request=dict(state="fixture state", questions={
            "choice": dict(type="choice", instructions="Choose", criteria={"z": "last letter", "a": "first letter", "m": "middle letter"}),
            "binary": dict(type="noul", instructions="True?"),
            "ordinal": dict(type="score", instructions="Level?", criteria=["low", "mid", "high"]),
        }))]
        self.probabilities = [[0.10000000001, 0.69999999999, 0.2], [0.25, 0.75], [0.1, 0.3, 0.6]]

    def invoke(self, device="cpu", **extraenv):
        self.requests.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(self.dependencies), CUDA_VISIBLE_DEVICES="",
            JEV_WORKER_FIXTURE=str(self.folder), JEV_WORKER_PROBABILITIES=json.dumps(self.probabilities), **extraenv)
        script = Path(__file__).resolve().parents[1] / "src/jev/predict_worker.py"
        return subprocess.run([sys.executable, "-P", str(script), "--checkpoint", str(self.checkpoint),
            "--source-root", str(self.source), "--base", str(self.base), "--model-name", "fixture-model",
            "--requests", str(self.requests), "--output", str(self.output), "--device", device],
            env=env, capture_output=True, text=True, encoding="utf-8", timeout=20)

    def events(self):
        return [json.loads(line) for line in (self.folder / "events.jsonl").read_text().splitlines()]

    def test_legacy_metadata_is_rebound_in_memory_and_real_native_order_is_preserved(self):
        original = sha(self.checkpoint / "head.pt")
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        response = json.loads(self.output.read_text())["response"]
        self.assertEqual(response["head_config"], dict(version=1, kind="pointer", dim=1024, readout="end"))
        self.assertEqual(response["model"], "fixture-model")
        self.assertEqual(list(response["probabilities"]["choice"]), ["z", "a", "m"])
        self.assertEqual(list(response["probabilities"]["binary"]), ["false", "true"])
        self.assertEqual(list(response["probabilities"]["ordinal"]), ["0", "1", "2"])
        self.assertEqual(response["probabilities"]["choice"]["z"], self.probabilities[0][0])
        loaded = json.loads((self.folder / "loaded_metadata.json").read_text())
        self.assertEqual(loaded["metadata"]["base"], str(self.base))
        self.assertTrue(loaded["merge"])
        self.assertEqual(loaded["dtype"], "fixture-bfloat16")
        self.assertEqual(sha(self.checkpoint / "head.pt"), original)
        limits = json.loads((self.folder / "encode_limits.json").read_text())
        self.assertEqual(limits, dict(max_state=8192, max_branch=8192, strict=True))
        record = json.loads((self.folder / "encoded_record.json").read_text())
        self.assertEqual(record["state"], "fixture state")
        self.assertEqual(record["questions"][0]["options"], ["z: last letter", "a: first letter", "m: middle letter"])
        # The real serializer's internal placeholder label is always zero; it
        # does not consume a reference target or change with expected answers.
        self.assertEqual([question["label"] for question in record["questions"]], [0, 0, 0])

    def test_versioned_head_readout_metadata_is_not_replaced_by_legacy_pointer_defaults(self):
        head = dict(version=1, kind="pointer", dim=256, readout="mean_span", rng_anchor_dim=256)
        (self.checkpoint / "head.pt").write_text(json.dumps(dict(self.metadata, head_config=head)), encoding="utf-8")
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.output.read_text())["response"]["head_config"], head)

    def test_reference_labels_cannot_enter_native_model_inputs(self):
        self.rows[0]["request"]["questions"]["choice"]["label"] = "a"
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("labels/metadata cannot be model inputs", result.stderr)
        self.assertNotIn("forward", self.events())

    def test_cuda_timing_synchronization_brackets_forward_without_claiming_real_latency(self):
        result = self.invoke(device="cuda")
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        before, forward, after = events.index("synchronize"), events.index("forward"), len(events) - 1 - events[::-1].index("synchronize")
        self.assertLess(before, forward)
        self.assertLess(forward, after)
        response = json.loads(self.output.read_text())["response"]
        self.assertGreaterEqual(response["forward_seconds"], 0)
        self.assertEqual(response["input_tokens"], 4)

    def test_missing_question_or_option_probabilities_cannot_report_native_success(self):
        for index, probabilities in enumerate(([[0.1, 0.7, 0.2]], [[0.5, 0.5], [0.25, 0.75], [0.1, 0.3, 0.6]])):
            self.probabilities = probabilities
            self.output = self.folder / f"missing-{index}.jsonl"
            with self.subTest(probabilities=probabilities):
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0, "standalone prediction silently wrote incomplete native question/option output")
                self.assertEqual(self.output.stat().st_size, 0)

    def test_finite_invalid_distributions_cannot_report_native_success(self):
        self.probabilities[0] = [-0.1, 0.6, 0.5]
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0, "standalone prediction accepted out-of-range probabilities")

    def test_probability_tolerance_matches_evaluator_without_normalizing_raw_values(self):
        self.probabilities[0] = [0.1, 0.2, 0.700005]
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        probabilities = json.loads(self.output.read_text())["response"]["probabilities"]["choice"]
        self.assertEqual(list(probabilities.values()), self.probabilities[0])
        labels = self.folder / "labels.jsonl"
        labels.write_text(json.dumps(dict(id="fixture-id", targets=dict(choice="a", binary=True, ordinal=2))) + "\n", encoding="utf-8")
        self.assertEqual(score_files(self.requests, labels, self.output, self.folder / "scores")["questions"], 3)
        self.output = self.folder / "invalid-sum.jsonl"
        self.probabilities[0] = [0.1, 0.2, 0.70002]
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.stat().st_size, 0)

    def test_torch_load_filelike_inputs_remain_compatible_during_local_base_rebinding(self):
        result = self.invoke(JEV_FILELIKE_LOAD="1")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
