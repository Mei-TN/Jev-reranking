"""Read native scores and their evaluation denominators."""
from decimal import Decimal as D
from pathlib import Path

from common import read, sha
from jk_audit import JK_COUNTS
from population import MC_COUNTS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(not isinstance(value, bool), 'boolean score')
    result = D(str(value))
    require(result.is_finite(), 'nonfinite score')
    return result


def close(a, b, message):
    require(abs(number(a) - number(b)) <= D('1e-10'), 'arithmetic mismatch: ' + message)


def jk_value(task, native, n):
    """JevBench family macro; Kev clean known-source question accuracy."""
    require(len(native['details']) == n and native['schema_validity'] == 100 and
            native['probability_source'] in ('classification_head', ['classification_head']),
            'invalid native JK output: ' + task)
    report = native['native_report'][0]
    if task == 'jevbench_public':
        require(native['n_planned'] == n, 'JevBench planned count')
        families = report['per_family']
        require(len(families) == 18 and all(0 <= f['n_correct'] <= f['n_scorable'] and
                f['n_scorable'] > 0 for f in families.values()), 'JevBench family counts')
        require(sum(f['n_scorable'] for f in families.values()) == report['n_scorable'] == n and
                sum(f['n_correct'] for f in families.values()) == report['n_correct'] ==
                sum(sum(d['correct']) for d in native['details']), 'JevBench detail counts')
        value = D(100) * sum((D(f['n_correct']) / D(f['n_scorable']) for f in families.values()), D(0)) / D(18)
        close(value, native['macro_accuracy'], task)
        close(value, D(100) * number(report['macro_accuracy']), task + ' report')
        denominator = {'selected_records': n, 'scorable_questions': report['n_scorable'], 'families': 18}
    else:
        require(native['requested_records'] == n and native['invalid_records'] == 0, 'Kev counts')
        rows = [row for detail in native['details'] for group in detail['rows'] for row in group
                if row['variant'] == 'clean' and row['source'] != 'unknowable']
        require(rows and len(rows) == report['clean']['n'], 'Kev clean denominator')
        require(all(len(r['p']) == len(r['keys']) and type(r['label']) is int and
                0 <= r['label'] < len(r['p']) and all(D(0) <= number(p) <= D(1) for p in r['p'])
                for r in rows), 'Kev clean detail schema')
        correct = sum(max(range(len(r['p'])), key=r['p'].__getitem__) == r['label'] for r in rows)
        value = D(100) * D(correct) / D(len(rows))
        close(value, native['accuracy'], task)
        close(value, D(100) * number(report['clean']['acc']), task + ' report')
        denominator = {'selected_records': n, 'clean_known_source_questions': len(rows), 'correct': correct}
    return float(value), denominator


def collect_scores(root, plan):
    root = Path(root)
    run_id, head = plan['run_id'], plan['head_config']
    jk = read(root / 'jk_audit/summary.json')
    completion = read(root / 'opencompass/completion.json')
    require(jk['complete'] is True and jk['records'] == 6667 and
            set(jk['populations']) == set(JK_COUNTS), 'seven JK populations incomplete')
    require(completion['complete'] is True and completion['model'] == run_id and
            completion['head_config'] == head and set(completion['results']) == set(JK_COUNTS),
            'JK completion identity')
    tasks, denominators = {}, {}
    for task, n in JK_COUNTS.items():
        record = completion['results'][task]
        population = jk['populations'][task]
        require(population['result'] == record and population['records'] == record['records'] == n,
                'JK audit/native population mismatch: ' + task)
        require(Path(record['path']).resolve().is_relative_to(root.resolve()), 'native result escapes output')
        require(sha(record['path']) == record['sha256'], 'native result changed: ' + task)
        tasks[task], denominators[task] = jk_value(task, read(record['path']), n)
    expected = read(plan['expected_mc_population'])
    for task, (n, calls) in MC_COUNTS.items():
        directory = root / 'mc11' / task
        summary = read(directory / 'summary.json')
        reference = expected['tasks'][task]
        require(summary['complete'] is True and summary['model'] == run_id and summary['head_config'] == head and
                summary['task'] == task and summary['questions'] == reference['questions'] == n and
                summary['calls'] == reference['calls'] == calls, 'MC identity/count: ' + task)
        require(summary['expected_reference_sha256'] == sha(plan['expected_mc_population']), 'MC reference changed')
        require(summary['dataset_inventory_identity_sha256'] == reference['dataset_identity'].get('identity_sha256'),
                'MC dataset inventory mismatch: ' + task)
        for key in ('request_occurrence_sha256', 'request_encoder_tokens_occurrence_sha256'):
            require(summary[key] == reference[key], 'MC population mismatch: ' + task + '/' + key)
        require(sha(directory / 'native_result.json') == summary['native_result_sha256'], 'MC native result changed')
        native = read(directory / 'native_result.json')
        require(set(native) == {run_id} and set(native[run_id]) == {task}, 'MC native model/task identity')
        metric = native[run_id][task]
        require(metric.get('n', metric.get('n_questions')) == n, 'MC native denominator: ' + task)
        accuracy = number(metric['rerank_acc'])
        require(D(0) <= accuracy <= D(1), 'MC accuracy range')
        close(accuracy, summary['metrics'][run_id + '/' + task + '/rerank_acc'], task + ' summary')
        tasks[task] = float(D(100) * accuracy)
        denominators[task] = {'questions': n, 'native_calls': calls, 'score': 'rerank_acc'}
    require(set(tasks) == set(JK_COUNTS) | set(MC_COUNTS), 'incomplete 18-task population')
    return dict(tasks_percent=tasks, denominators=denominators, endpoints_percent={
        'JevBench': tasks['jevbench_public'],
        'Kev': sum(tasks[t] for t in JK_COUNTS if t != 'jevbench_public') / 6,
        'MC-11': sum(tasks[t] for t in MC_COUNTS) / 11},
        aggregation='JevBench: equal macro over 18 families; Kev: equal mean over six clean-known task accuracies; MC-11: equal mean over eleven native rerank_acc values.')
