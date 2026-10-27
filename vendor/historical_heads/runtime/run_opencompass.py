"""Frozen seven-dataset JevBench/Kev run with the isolated classification-head client."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
from .bundle_contract import config_identity, digest, read_config, validate_response


def check_endpoints(model, endpoints, head_config, allow_legacy_h0=False):
    if not endpoints or any(not e.startswith(('http://','https://')) for e in endpoints):
        raise ValueError('Provide at least one HTTP(S) endpoint')
    identities = []
    for endpoint in endpoints:
        with urllib.request.urlopen(endpoint.rstrip('/')+'/health', timeout=30) as response:
            result = json.load(response)
        validate_response(result, model, head_config, allow_legacy_h0)
        if result.get('temperature') != 1 or result.get('lora_scale') != 1:
            raise ValueError('endpoint calibration differs from fixed native procedure')
        identities.append(result)
    if len({x.get('manifest_sha256') for x in identities}) != 1:
        raise ValueError('replicas use different bundle manifests')
    if len({x['probability_source'] for x in identities}) != 1:
        raise ValueError('replicas use different probability sources')
    return identities


def build_config(model, endpoints, selection_root, head_config, allow_legacy_h0=False,
                 probability_source='classification_head'):
    expected = config_identity(head_config)
    root = Path(selection_root).resolve()
    specs = [('jevbench_public','jevbench',None,'development')]
    specs += [(f'kev_{s}_{p}','kev',s,p) for s in ('decision-v7','transfer-v4','transfer-v9') for p in ('development','test')]
    source = '''from runtime.decision_head_api import DecisionHeadAPI, HeadJevBenchEvaluator, HeadKevEvaluator
from opencompass.datasets.decision_native import NativeDecisionDataset, NativeJevBenchEvaluator, NativeKevEvaluator
from opencompass.openicl.icl_inferencer import ParallelGenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
reader = dict(input_columns=['request_json'], output_column='reference', train_split='test', test_split='test')
native_infer_cfg = dict(prompt_template=dict(type=PromptTemplate, template='{request_json}'), retriever=dict(type=ZeroRetriever), inferencer=dict(type=ParallelGenInferencer, max_out_len=1, save_every=1))
datasets = []
'''
    selections = {}
    for abbr, kind, suite, partition in specs:
        selection = root/f'{abbr}.selection.json'
        frozen = json.loads(selection.read_text())
        if frozen['kind'] != kind or frozen.get('suite') != suite or frozen['partition'] != partition:
            raise ValueError('Selection identity mismatch: '+str(selection))
        selections[selection.name] = digest(selection)
        prefix = 'Native' if probability_source=='pointer_head' else 'Head'
        evaluator = prefix+('JevBenchEvaluator' if kind=='jevbench' else 'KevEvaluator')
        source += f"datasets.append(dict(abbr={abbr!r}, type=NativeDecisionDataset, kind={kind!r}, suite={suite!r}, partition={partition!r}, selection={str(selection)!r}, reader_cfg=reader, infer_cfg=native_infer_cfg, eval_cfg=dict(evaluator=dict(type={evaluator}))))\n"
    model_args = dict(path=model, endpoints=endpoints, head_config=expected,
                      allow_legacy_h0=allow_legacy_h0, batch_size=len(endpoints), max_seq_len=65536, max_out_len=1)
    source += f'models = [dict(type=DecisionHeadAPI, **{model_args!r})]\n'
    return source, selections


def collect_results(work_dir, selection_root, probability_source):
    """Require all seven frozen populations; never call partial output complete."""
    result_files = {}
    for selection in sorted(Path(selection_root).glob('*.selection.json')):
        name = selection.name.removesuffix('.selection.json')
        if name != 'jevbench_public' and not (name.startswith('kev_') and
                name in {f'kev_{s}_{p}' for s in ('decision-v7','transfer-v4','transfer-v9') for p in ('development','test')}):
            continue
        candidates = [p for p in Path(work_dir).rglob(name+'.json') if 'results' in p.parts]
        if len(candidates) != 1:
            raise ValueError('missing or ambiguous OpenCompass result for '+name)
        result = json.loads(candidates[0].read_text())
        selected = json.loads(selection.read_text())['selected_ids']
        n = len(set(selected))
        if n == 0 or len(selected) != n:
            raise ValueError('empty or duplicate frozen selection: '+name)
        count_key = 'n_planned' if name=='jevbench_public' else 'requested_records'
        # OpenCompass aggregates nonnumeric replica metadata into a list.
        # This fixed run has one replica; reject missing/multiple/mixed values.
        source = result.get('probability_source')
        source_matches = (source == probability_source or
                          source == [probability_source])
        if (result.get(count_key) != n or result.get('schema_validity') != 100 or
                not source_matches or
                (name!='jevbench_public' and result.get('invalid_records') != 0)):
            raise ValueError('incomplete/invalid OpenCompass population: '+name)
        result_files[name] = dict(path=str(candidates[0].resolve()), sha256=digest(candidates[0]), records=n)
    if len(result_files) != 7: raise ValueError('expected seven JevBench/Kev results')
    return result_files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--endpoints', required=True, nargs='+')
    p.add_argument('--head-config', required=True, type=Path)
    p.add_argument('--selection-root', required=True, type=Path)
    p.add_argument('--oc-root', required=True, type=Path)
    p.add_argument('--work-dir', required=True, type=Path)
    p.add_argument('--allow-legacy-h0', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    expected = read_config(a.head_config)
    identities = [] if a.dry_run else check_endpoints(a.model, a.endpoints, expected, a.allow_legacy_h0)
    source, selections = build_config(a.model, a.endpoints, a.selection_root, expected, a.allow_legacy_h0,
                                     identities[0]['probability_source'] if identities else 'classification_head')
    a.work_dir.mkdir(parents=True, exist_ok=False)
    config = a.work_dir.resolve()/'head_native_config.py'
    config.write_text(source)
    if a.dry_run: print(config); return
    (a.work_dir/'evaluation_identity.json').write_text(json.dumps(dict(
        model=a.model, head_config=expected, endpoints=identities, selections=selections,
        config_sha256=digest(config), scope='JevBench + six Kev selections'), indent=2)+'\n')
    implementation = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=implementation+os.pathsep+str(a.oc_root.resolve()), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, str(a.oc_root.resolve()/'run.py'), str(config),
                    '--work-dir', str(a.work_dir.resolve()), '--max-num-workers','1'],
                   env=env, cwd=a.oc_root, check=True)
    results = collect_results(a.work_dir, a.selection_root, identities[0]['probability_source'])
    (a.work_dir/'completion.json').write_text(json.dumps(dict(
        complete=True, model=a.model, head_config=expected, results=results), indent=2)+'\n')


if __name__ == '__main__': main()
