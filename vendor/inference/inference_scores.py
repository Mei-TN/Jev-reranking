"""Read full-precision decision-head scores under the frozen source-slot protocol."""
import ast
from collections import defaultdict
import math
from pathlib import Path

import prepare_inputs as prep


def pick(candidates, scores, gold):
    if len(scores) != len(candidates) or gold not in candidates or len(set(candidates)) != len(candidates):
        raise ValueError('candidate identity/score coverage differs')
    if any(type(s) not in (int, float) or not math.isfinite(s) for s in scores):
        raise ValueError('invalid decision score')
    maximum = max(scores)
    tied = [cid for cid, score in zip(candidates, scores) if score == maximum]
    selected = tied[0]
    return dict(prediction=selected, correct=int(selected == gold),
                exact_tie=len(tied) > 1, maximum_candidates=tied,
                uniform_tie_credit=(1.0/len(tied) if gold in tied else 0.0))


def native_scorer(eval_source):
    # Execute the original scoring function only; no model-loading module imports.
    import numpy as np
    from sklearn.metrics import f1_score
    tree = ast.parse(Path(eval_source).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_mc')
    scope = dict(np=np, f1_score=f1_score, CON=0, ENT=1, NEU=2)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(eval_source), 'exec'), scope)
    return scope['eval_mc']


def score_complete(request_rows, response_rows, scoring_rows, eval_source):
    if len(request_rows) != 69590 or len(response_rows) != len(request_rows):
        raise ValueError('cannot score an incomplete request population')
    questions = {(r['dataset'], r['question_id']):r for r in scoring_rows}
    if len(questions) != 2036:
        raise ValueError('scoring population changed')
    grouped = defaultdict(dict)
    gpqa_events = defaultdict(list)
    for expected, event in zip(request_rows, response_rows):
        if event['ordinal'] != expected['ordinal'] or event['request_id'] != expected['request_id'] or event.get('error'):
            raise ValueError('response occurrence does not match planned request')
        meta = expected
        distribution = event['response']['probabilities']
        group = grouped[(meta['condition'], meta['question_id'])]
        if meta['condition'] == 'piqa_direct':
            if group:
                raise ValueError('duplicate direct question result')
            group.update(distribution['solution'])
        else:
            if meta['candidate_id'] in group:
                raise ValueError('candidate source occurrence duplicated')
            group[meta['candidate_id']] = distribution['relation']['entailment']
        if meta['dataset'] == 'gpqa_diamond':
            gpqa_events[meta['condition']].append(event)
    item_scores = []
    condition_values = defaultdict(list)
    for (condition, qid), distribution in grouped.items():
        dataset = 'piqa' if condition.startswith('piqa_') else 'gpqa_diamond'
        question = questions[(dataset, qid)]
        candidates, gold = question['candidates'], question['gold']
        if set(distribution) != set(candidates):
            raise ValueError('missing/extra source candidates in final score')
        values = [distribution[cid] for cid in candidates]
        result = pick(candidates, values, gold)
        row = dict(condition=condition, dataset=dataset, question_id=qid,
                   candidate_order=candidates, candidate_scores=values, gold=gold, **result)
        if dataset == 'gpqa_diamond':
            hashes = question['exact_text_sha256']
            row['exact_text_equivalent_correct'] = int(hashes[row['prediction']] == hashes[gold])
        condition_values[condition].append(row)
        item_scores.append(row)
    conditions = {}
    for condition, rows in condition_values.items():
        expected_n = 1838 if condition.startswith('piqa_') else 198
        if len(rows) != expected_n:
            raise ValueError('question denominator mismatch')
        conditions[condition] = dict(n_questions=expected_n, correct=sum(r['correct'] for r in rows),
            accuracy=sum(r['correct'] for r in rows)/expected_n,
            exact_tie_questions=sum(r['exact_tie'] for r in rows),
            uniform_tie_accuracy=sum(r['uniform_tie_credit'] for r in rows)/expected_n)
        if condition.startswith('gpqa_'):
            conditions[condition]['exact_text_equivalent_accuracy'] = sum(r['exact_text_equivalent_correct'] for r in rows)/expected_n
    if len(conditions) != 21:
        raise ValueError('missing fixed conditions')
    # Run the native complete scorer against the recorded responses, with its unchanged
    # rerank/grade pair builders and C/E/N vector semantics. This issues no network calls.
    items = []
    for row in scoring_rows:
        if row['dataset'] == 'gpqa_diamond':
            items.append(dict(row['native_item'], hyp=lambda text:'Answer: '+text))

    class RecordedNLI:
        calls = 0

        def predict(self, pairs):
            import numpy as np
            conditions_in_order = ('gpqa_disjoint5_rerank', 'gpqa_disjoint5_grade')
            if self.calls >= 2:
                raise ValueError('unexpected native score pass')
            events = gpqa_events[conditions_in_order[self.calls]]
            self.calls += 1
            if len(pairs) != len(events) or len(events) != 792:
                raise ValueError('native scorer pair count differs')
            values = []
            for (premise, hypothesis), event in zip(pairs, events):
                body = event['request']
                if (body['state'] != premise or body['questions']['relation']['instructions'] !=
                        f'Hypothesis: "{hypothesis}" How does it relate to the premise?'):
                    raise ValueError('native scorer pairs differ from recorded requests')
                values.append(prep.native_label_vector(event['response']['probabilities']['relation']))
            return np.array(values)

    remote = RecordedNLI()
    native = native_scorer(eval_source)(remote, items)
    if (remote.calls != 2 or native['n_questions'] != 198 or
            native['rerank_acc'] != conditions['gpqa_disjoint5_rerank']['accuracy'] or
            native['grade_rank_acc'] != conditions['gpqa_disjoint5_grade']['accuracy']):
        raise ValueError('native scorer agreement failed')
    # PIQA E7 uses P0D0 from the already-run E9 matrix; no extra calls.
    # The grade condition saw the reference answer in its premise: it is native grade_rank_acc, never a primary score.
    return dict(conditions=conditions, gpqa_native=native,
        primary_conditions=sorted(c for c in conditions if c != 'gpqa_disjoint5_grade'),
        gold_conditioned_native_grade_condition='gpqa_disjoint5_grade',
        interface_difference=conditions['piqa_nli_P0D0']['accuracy'] - conditions['piqa_direct']['accuracy'],
        note='This is one checkpoint. H5-H0 pairing and min_t(H5_t-H0_t) require the four matched pairs.'), item_scores
