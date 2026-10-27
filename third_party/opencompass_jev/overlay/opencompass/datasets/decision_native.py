"""Shared frozen selection and native requests for the existing official scorers."""
import json
from pathlib import Path
from datasets import Dataset,DatasetDict
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset
from .decision_benchmarks import JevBenchDataset,KevDataset,JevBenchEvaluator,KevEvaluator


def native_request(kind,raw):
    questions={'decision':raw['question']} if kind=='jevbench' else raw['questions']
    return {'state':raw['state'],'questions':{qid:{key:value for key,value in question.items()
        if key in ('type','instructions','criteria')} for qid,question in questions.items()}}


@LOAD_DATASET.register_module()
class NativeDecisionDataset(BaseDataset):
    @staticmethod
    def load(kind,selection,suite=None,partition='development'):
        data=JevBenchDataset.load()['test'] if kind=='jevbench' else KevDataset.load(suite=suite,partition=partition)['test']
        frozen=json.loads(Path(selection).read_text())
        selected=set(frozen['selected_ids']);rows=[]
        for row in data:
            if row['id'] in selected:
                raw=json.loads(row['reference'])
                rows.append({'id':row['id'],'request_json':json.dumps({'kind':kind,'request':native_request(kind,raw)},ensure_ascii=False),
                             'reference':row['reference']})
        if len(rows)!=len(selected):raise ValueError('Frozen selection does not match benchmark IDs')
        return DatasetDict(test=Dataset.from_list(rows))


class NativeJevBenchEvaluator(JevBenchEvaluator):
    probability_source='pointer_head'


class NativeKevEvaluator(KevEvaluator):
    probability_source='pointer_head'
