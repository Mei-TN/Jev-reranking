#!/usr/bin/env python3
"""Materialize the already-qualified complete request population; CPU only, no scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import prepare_inputs as prep
from token_preflight import read_json, read_jsonl, source_functions


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def materialize(prepared, assets, gpqa_archive, token_report, output):
    prepared, assets, token_report, output = map(Path, (prepared, assets, token_report, output))
    if output.exists():
        raise ValueError('preserve existing population: output must be new')
    protocol = read_json(prepared / 'proposed_protocol.json')
    preparation = read_json(prepared / 'PREPARE_RECEIPT.json')
    for name, expected in preparation['files'].items():
        if file_sha(prepared / name) != expected:
            raise ValueError('fixed input identity changed: ' + name)
    annotation = read_json(prepared / 'POST_FREEZE_AUDIT.json')
    if annotation['request_matrix_changed'] or annotation['primary_denominator'] != 198 or annotation['exact_duplicate_questions'] != 2:
        raise ValueError('expected accepted text-collision clarification')
    summary = read_json(token_report / 'TOKEN_PREFLIGHT_SUMMARY.json')
    report_receipt = read_json(token_report / 'TOKEN_PREFLIGHT_RECEIPT.json')
    for name, expected in report_receipt['files'].items():
        if file_sha(token_report / name) != expected:
            raise ValueError('original token qualification changed: ' + name)
    if (summary['status'] != 'PASS' or summary['total_request_templates'] != 69590 or
            summary['intended_service_overflows'] != 0 or summary['predictions_generated'] or
            summary['prepared_receipt_sha256'] != file_sha(prepared / 'PREPARE_RECEIPT.json')):
        raise ValueError('not the complete matching token qualification')
    source = source_functions(assets / 'openjev_pointer.py', assets / 'native_eval.py')
    if source['CRITERIA'] != protocol['descriptions']['D0']:
        raise ValueError('D0 differs')
    for name, expected in prep.PIQA_PINS.items():
        if file_sha(assets / name) != expected:
            raise ValueError('PIQA source differs: ' + name)
    for name, expected in summary['native_input_builder_sha256'].items():
        if file_sha(assets / name) != expected:
            raise ValueError('input builder differs: ' + name)
    questions = read_jsonl(assets / 'piqa.questions.jsonl')
    tables, _ = prep.read_gpqa(gpqa_archive, None)
    selected = read_json(prepared / 'gpqa_demo_selection.json')['selected']
    main = {r['Record ID']: r for r in tables['main']}
    demos = [(main[r['record_id']]['Question'].strip(), main[r['record_id']]['Correct Answer'].strip()) for r in selected]
    gpqa = read_jsonl(prepared / 'gpqa_diamond_identity_ledger.jsonl')
    items, pair_meta, scoring = [], [], []
    for q in questions:
        scoring.append(dict(dataset='piqa', question_id=q['id'], candidates=[c['id'] for c in q['candidates']], gold=q['gold']))
    for row, ledger in zip(tables['diamond'], gpqa):
        if row['Record ID'] != ledger['record_id']:
            raise ValueError('GPQA order differs')
        candidates = {c['candidate_id']: c for c in ledger['candidates']}
        ordered = [candidates[cid] for cid in ledger['frozen_candidate_ids']]
        opts = [row[c['source_field']].strip() for c in ordered]
        item = dict(q=source['with_demos'](row['Question'].strip(), demos), opts=opts,
                    gold=ledger['gold_sorted_index'], hyp=source['FEWSHOT_HYP'])
        items.append(item)
        pair_meta.extend(dict(question_id=row['Record ID'], candidate_id=c['candidate_id'],
                             source_slot=c['source_slot'], candidate_sorted_index=i) for i, c in enumerate(ordered))
        scoring.append(dict(dataset='gpqa_diamond', question_id=row['Record ID'], candidates=ledger['frozen_candidate_ids'],
            gold=ledger['gold_candidate_id'], gold_sorted_index=ledger['gold_sorted_index'],
            exact_text_sha256={c['candidate_id']: prep.digest(text.encode()) for c, text in zip(ordered, opts)},
            native_item=dict(q=item['q'], opts=opts, gold=item['gold'])))
    rerank, grade, grade_gold, offsets = source['native_pairs'](None, items)
    if len(scoring) != 2036 or len(rerank) != 792 or len(grade) != 792 or len(offsets) != 198:
        raise ValueError('population cardinality differs')

    def bodies():
        for stage, pairs in (('rerank', rerank), ('grade', grade)):
            for meta, pair in zip(pair_meta, pairs):
                yield dict(dataset='gpqa_diamond', condition='gpqa_disjoint5_' + stage, **meta), source['pointer_request']('__MODEL_ID__', pair)
        for q in questions:
            yield dict(dataset='piqa', condition='piqa_direct', question_id=q['id'], candidate_id=None), prep.direct_request(q)
        for did, descriptions in protocol['descriptions'].items():
            for p in protocol['permutations']:
                for q in questions:
                    for c in q['candidates']:
                        yield dict(dataset='piqa', condition='piqa_nli_' + p['id'] + did, question_id=q['id'], candidate_id=c['id'],
                                   description_id=did, permutation_id=p['id']), prep.nli_request(q, c, descriptions, p['order'])

    output.mkdir(parents=True, exist_ok=False)
    conditions, model_wires = Counter(), {rid:hashlib.sha256() for rid in summary['checkpoint_model_wire_stream_sha256']}
    total = 0
    with (token_report / 'request_token_ledger.jsonl').open(encoding='utf-8') as witness, (output / 'requests.jsonl').open('xb') as stream:
        for ordinal, (meta, body) in enumerate(bodies()):
            row = json.loads(next(witness))
            if any(row[k] != v for k, v in meta.items()) or row['wire_sha256'] != prep.digest(prep.wire(body)):
                raise ValueError('materialized request differs from token-qualified witness')
            for rid, accumulator in model_wires.items():
                accumulator.update(prep.wire(dict(body, model=rid)) + b'\n')
            stream.write(prep.wire(dict(ordinal=ordinal, **meta, request_id=row['request_id'], request=body,
                wire_sha256=row['wire_sha256'], input_tokens=row['packed_tokens'], state_tokens=row['state_tokens'],
                input_token_ids_sha256=row['input_token_ids_sha256'], semantic_keys=row['semantic_keys'])) + b'\n')
            conditions[meta['condition']] += 1
            total += 1
        if witness.readline():
            raise ValueError('leftover token witness requests')
    if total != 69590 or {rid:h.hexdigest() for rid,h in model_wires.items()} != summary['checkpoint_model_wire_stream_sha256']:
        raise ValueError('materialized complete model wire streams differ')
    prep.write_jsonl(output / 'scoring_population.jsonl', scoring)
    (output / 'native_eval.py').write_bytes((assets / 'native_eval.py').read_bytes())
    (output / 'proposed_protocol.json').write_bytes((prepared / 'proposed_protocol.json').read_bytes())
    (output / 'POST_FREEZE_AUDIT.json').write_bytes((prepared / 'POST_FREEZE_AUDIT.json').read_bytes())
    (output / 'checkpoints.json').write_bytes((prepared / 'checkpoints.json').read_bytes())
    prep.write_json(output / 'POPULATION.json', dict(status='COMPLETE_CPU_TOKEN_QUALIFIED',
        protocol_version=protocol['version'], protocol_sha256=file_sha(prepared / 'proposed_protocol.json'),
        preparation_receipt_sha256=file_sha(prepared / 'PREPARE_RECEIPT.json'),
        token_report_sha256=file_sha(token_report / 'TOKEN_PREFLIGHT_SUMMARY.json'),
        text_audit_sha256=file_sha(prepared / 'POST_FREEZE_AUDIT.json'),
        total_requests=total, total_checkpoint_requests=total*8, condition_requests=dict(conditions),
        questions={'piqa':1838, 'gpqa_diamond':198},
        ordered_model_wire_sha256=summary['checkpoint_model_wire_stream_sha256'],
        native_runtime_sources=summary['native_source_identities'],
        files={p.name:file_sha(p) for p in output.iterdir() if p.is_file()}, predictions_generated=False))
    print(json.dumps(dict(status='COMPLETE_CPU_TOKEN_QUALIFIED', total_requests=total, output=str(output))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'assets', 'gpqa-archive', 'token-report', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    materialize(a.prepared, a.assets, a.gpqa_archive, a.token_report, a.output)


if __name__ == '__main__':
    main()
