"""JevBench public decisions and Kev frozen suites, with upstream scoring.

Probability outputs are verbalized JSON for the generation path. This module
never synthesizes probabilities from a label or silently drops malformed answers.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from datasets import Dataset, DatasetDict
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset
from ._decision_upstream.jevbench.tasks import Task
from ._decision_upstream.jevbench.adapters.openai_compat import OpenAICompatAdapter
from ._decision_upstream.jevbench.scoring import score_task
from ._decision_upstream.jevbench.summarize import summarize as jev_summarize
from ._decision_upstream import kev_scoring

SOURCES = json.loads((Path(__file__).parent / '_decision_upstream/SOURCES.json').read_text())
KEV_SUITES = {'decision-v7': 'v7', 'transfer-v4': 'v4', 'transfer-v9': 'v9'}


def source_file(project, relative, data_root=None):
    """Fetch immutable bytes to a shared cache, or verify an explicit checkout."""
    spec = SOURCES[project]
    expected = spec['datasets'][relative]
    if data_root:
        path = Path(data_root) / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Dataset checksum mismatch: {path}')
        return path
    cache = Path(os.environ.get('OPENCOMPASS_DECISION_CACHE',
                                str(Path.home() / '.cache/opencompass/decisions')))
    path = cache / project / spec['revision'] / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return path
        url = spec['repository'].replace('https://github.com/',
                                       'https://raw.githubusercontent.com/')
        url += '/' + spec['revision'] + '/' + relative
        data = urllib.request.urlopen(url, timeout=120).read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Download checksum mismatch: {relative}')
        fd, temporary = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return path


def parse_json(prediction):
    # Match the native JevBench parser: the last complete probability object.
    candidates = []
    for i, char in enumerate(str(prediction)):
        if char == '{':
            try:
                value, _ = json.JSONDecoder().raw_decode(str(prediction)[i:])
                if isinstance(value, dict) and set(value) == {'probabilities'}:
                    candidates.append(value)
            except json.JSONDecodeError:
                pass
    if not candidates:
        raise ValueError('No probability object in final content')
    return candidates[-1]


@LOAD_DATASET.register_module()
class JevBenchDataset(BaseDataset):
    """231 repository-public items; not the private 534-item leaderboard."""
    @staticmethod
    def load(data_root=None, subsets=('easy', 'original', 'hard')):
        if not subsets or len(set(subsets)) != len(subsets) or set(subsets) - {'easy', 'original', 'hard'}:
            raise ValueError('Unknown/duplicate JevBench public subsets')
        rows = []
        adapter = OpenAICompatAdapter('', '')
        for subset in subsets:
            path = source_file('jevbench', f'datasets/public/{subset}.jsonl', data_root)
            for line in path.read_text().splitlines():
                raw = json.loads(line)
                task = Task.from_dict(raw)
                messages = adapter.build_request(task)['messages']
                rows.append(dict(id=task.id, system_prompt=messages[0]['content'],
                                 question_prompt=messages[1]['content'],
                                 reference=json.dumps(raw, ensure_ascii=False),
                                 subset=subset))
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate JevBench task IDs')
        return DatasetDict(test=Dataset.from_list(rows))


class JevBenchEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references) or not references:
            raise ValueError('Empty or mismatched prediction population')
        tasks, records, details = [], [], []
        for pred, ref in zip(predictions, references):
            task = Task.from_dict(json.loads(ref))
            tasks.append(task)
            try:
                result = score_task(parse_json(pred)['probabilities'], task)
            except (ValueError, TypeError):
                result = score_task(None, task)
            records.append(dict(task_id=task.id, ok=result['valid'],
                                probs_source=getattr(self,'probability_source','verbalized'), model='opencompass',
                                latency_s=None, cost_usd=None, **result))
            details.append(dict(id=task.id, pred=pred, **result))
        native = jev_summarize(tasks, records)
        result = dict(accuracy=100 * native['accuracy'],
                      macro_accuracy=100 * native['macro_accuracy'],
                      schema_validity=100 * native['schema_validity'],
                      schema_validity_strict=100 * native['schema_validity_strict'],
                      calibration_n=native['calibration_n'],
                      n_planned=native['n_planned'],
                      details=details, native_report=native,
                      probability_source=getattr(self,'probability_source','verbalized'),
                      population='repository_public_only', upstream_revision=SOURCES['jevbench']['revision'])
        for key, value in [('brier', native['brier_mean']),
                           ('ece', native['ece']['ece'] if native['ece'] else None),
                           ('ordinal_mae', native['ordinal_mae'])]:
            result[key] = value
        return result


@LOAD_DATASET.register_module()
class KevDataset(BaseDataset):
    """One request per record; keep packed questions and all variant metadata."""
    @staticmethod
    def load(suite='transfer-v9', partition='development', data_root=None):
        if suite not in KEV_SUITES or partition not in {'development', 'test'}:
            raise ValueError('Unknown Kev suite/partition; training is not an eval split')
        path = source_file('kev', f'evals/{KEV_SUITES[suite]}/{suite}/{partition}.jsonl', data_root)
        rows = []
        for line in path.read_text().splitlines():
            raw = json.loads(line)
            questions = {qid: {k: q[k] for k in ('type', 'instructions', 'criteria') if k in q}
                         for qid, q in raw['questions'].items()}
            keys = {qid: kev_scoring.question_keys(q['type'], q.get('criteria'))
                    for qid, q in questions.items()}
            rows.append(dict(id=raw['_meta']['id'],
                system_prompt='Return only JSON: {"probabilities": {question_id: {option_key: probability}}}. '
                    'Include every question and every option, with finite probabilities in [0,1] summing to 1 per question.',
                question_prompt=json.dumps(dict(state=raw['state'], questions=questions,
                                               probability_keys=keys), ensure_ascii=False),
                reference=json.dumps(raw, ensure_ascii=False)))
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate Kev record IDs')
        return DatasetDict(test=Dataset.from_list(rows))


class KevEvaluator(BaseEvaluator):
    """Preserve native refusal to publish official metrics on invalid outputs."""
    def score(self, predictions, references):
        if len(predictions) != len(references) or not references:
            raise ValueError('Empty or mismatched prediction population')
        rows, details, failures = [], [], []
        for pred, ref in zip(predictions, references):
            record = json.loads(ref)
            try:
                scored = kev_scoring.prediction_rows(record, parse_json(pred))
                rows.extend(scored)
                details.append(dict(id=record['_meta']['id'], pred=pred, rows=scored))
            except (ValueError, TypeError, KeyError) as error:
                failures.append(dict(id=record['_meta']['id'], error=str(error), pred=pred))
        result = dict(requested_records=len(references), invalid_records=len(failures),
                      schema_validity=100 * (len(references)-len(failures))/len(references),
                      probability_source=getattr(self,'probability_source','verbalized'), upstream_revision=SOURCES['kev']['revision'], details=details + failures)
        if failures:
            # No probability imputation and no deceptively improved valid-only score.
            result.update(accuracy=None, brier=None, ece=None,
                          error='Native Kev report unavailable: invalid probability responses')
            return result
        try:
            report = kev_scoring.summarize(rows)
        except (ValueError, KeyError) as error:
            result.update(accuracy=None, brier=None, ece=None,
                          error=f'Incomplete native scoring groups (e.g. smoke subset): {error}')
            return result
        clean = report['clean']
        result.update(accuracy=100*clean['acc'], brier=clean['brier'], ece=clean['ece'],
                      nll=clean['nll'], coverage_at_5pct_error=clean['coverage_at_5pct_error'],
                      coverage_at_1pct_error=clean['coverage_at_1pct_error'], native_report=report)
        return result
