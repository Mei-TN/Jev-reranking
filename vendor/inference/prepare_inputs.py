#!/usr/bin/env python3
"""Inspect source assets and prepare an inference protocol for validation."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import tarfile
import unicodedata
import zipfile

REPO = Path(__file__).resolve().parents[2]
GPQA_ZIP_SHA = '461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc'
GPQA_CSV_SHA = {
    'main': 'acdeeac8f622267f2cd727d7d474202ea08dec80f7d3c3593b3ef8644f19b8e3',
    'diamond': '41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305'}
PIQA_PINS = {
    'piqa.questions.jsonl': '82984e4b2d84200d70b1f72abeb67199a88423570365be4fba1736b14f22736c',
    'piqa.requests.jsonl': '376e00fea0085ef852a54c6a732e061d18ab7a6b5be1b80bcfcee27c1ac60184'}
QUESTION_FIELDS = ('Question', 'Pre-Revision Question', 'Extra Revised Question')
ANSWER_FIELDS = ('Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3')
DEMO_NAMESPACE = 'jev-e12-gpqa-disjoint-v1'
CANDIDATE_NAMESPACE = 'jev-e12-candidate-order-v1'
EXCLUDED_VERSION_COLLISION = 'recSl0g9xnRRB2e0L'
EXPECTED_DEMOS = ('recUZuPWogSLnPBaw', 'rech8ob03J69QNAxU', 'recQ0eDZk1sxnKVqh',
                  'recQM36StGVD7wwP2', 'recp6DecZYavGF0TW')
LABELS = ('entailment', 'neutral', 'contradiction')
PERMUTATIONS = tuple(itertools.permutations(LABELS))
DIRECT_INSTRUCTION = 'Select the solution that best achieves the goal.'


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def norm(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def source_criteria(path):
    """Read the authoritative literal without importing its runtime dependencies."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    matches = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'CRITERIA' for t in node.targets)]
    if len(matches) != 1 or tuple(matches[0]) != LABELS:
        raise ValueError('expected one frozen E/N/C criteria literal')
    return matches[0]


def description_sets(d0):
    return {
        'D0': dict(d0),
        'D1': {
            'entailment': 'The premise supports the hypothesis as true.',
            'neutral': 'The premise does not determine whether the hypothesis is true.',
            'contradiction': 'The premise supports the hypothesis as false.'},
        'D2': {
            'entailment': 'The hypothesis must be true if the premise is true.',
            'neutral': 'The hypothesis could be either true or false if the premise is true.',
            'contradiction': 'The hypothesis must be false if the premise is true.'}}


def piqa_state(question):
    return 'Goal: ' + question['stem'] + '\nWhich solution best achieves this goal?'


def direct_request(question):
    return dict(model='__MODEL_ID__', state=piqa_state(question), questions={'solution': dict(
        type='choice', instructions=DIRECT_INSTRUCTION,
        criteria={c['id']: c['text'] for c in question['candidates']})})


def nli_request(question, candidate, descriptions, order):
    hypothesis = 'The correct solution is: ' + candidate['text']
    return dict(model='__MODEL_ID__', state=piqa_state(question), questions={'relation': dict(
        type='choice', instructions=f'Hypothesis: "{hypothesis}" How does it relate to the premise?',
        criteria={label: descriptions[label] for label in order})})


def native_label_vector(probabilities):
    """Native eval.py wants C/E/N regardless of the request insertion order."""
    return [probabilities[label] for label in ('contradiction', 'entailment', 'neutral')]


def question_aliases(row):
    return {field: norm(row.get(field, '')) for field in QUESTION_FIELDS if row.get(field, '').strip()}


def gpqa_partition(main, diamond):
    eval_ids = {row['Record ID'] for row in diamond}
    eval_current = {norm(row['Question']) for row in diamond}
    alias_index = defaultdict(list)
    for row in diamond:
        for field, value in question_aliases(row).items():
            alias_index[value].append({'record_id': row['Record ID'], 'field': field})
    ledger, eligible = [], []
    for ordinal, row in enumerate(main):
        reasons = []
        if row['Record ID'] in eval_ids:
            reasons.append('record_id_in_evaluation')
        if norm(row['Question']) in eval_current:
            reasons.append('current_question_in_evaluation')
        collisions = []
        for field, value in question_aliases(row).items():
            for match in alias_index.get(value, []):
                collisions.append(dict(source_field=field, normalized_question_sha256=digest(value.encode()), **match))
        if collisions:
            reasons.append('question_revision_alias_in_evaluation')
        accepted = not reasons
        if accepted:
            eligible.append(row)
        ledger.append(dict(source='gpqa_main', source_ordinal=ordinal, record_id=row['Record ID'],
            eligible=accepted, exclusion_reasons=reasons,
            current_question_sha256=digest(row['Question'].strip().encode()),
            normalized_aliases={field: dict(sha256=digest(value.encode()), chars=len(value))
                                for field, value in question_aliases(row).items()}, collisions=collisions))
    return eligible, ledger


def demo_rank(row):
    return digest(wire([DEMO_NAMESPACE, row['Record ID']]))


def candidate_ledger(row):
    """Retain each original slot, even if answer strings coincide; sort without gold."""
    occurrence = Counter()
    candidates = []
    for slot, field in enumerate(ANSWER_FIELDS):
        text = row[field].strip()
        duplicate_ordinal = occurrence[text]
        occurrence[text] += 1
        cid = 'answer:' + digest(wire([row['Record ID'], text, duplicate_ordinal]))
        candidates.append(dict(source_slot=slot, source_field=field, candidate_id=cid,
            answer_sha256=digest(text.encode()), normalized_answer_sha256=digest(norm(text).encode()),
            duplicate_occurrence=duplicate_ordinal,
            order_key=digest(wire([CANDIDATE_NAMESPACE, row['Record ID'], cid]))))
    ordered = sorted(candidates, key=lambda c: (c['order_key'], c['candidate_id']))
    return dict(record_id=row['Record ID'], candidates=candidates,
        frozen_candidate_ids=[c['candidate_id'] for c in ordered],
        gold_candidate_id=candidates[0]['candidate_id'],
        gold_sorted_index=[c['source_slot'] for c in ordered].index(0))


def read_gpqa(archive, env_tar):
    if archive:
        blob = Path(archive).read_bytes()
        origin = dict(path=str(Path(archive).resolve()), kind='zip')
    else:
        path = Path(env_tar)
        with tarfile.open(path, 'r|gz') as stream:
            for member in stream:
                if member.name == 'openjev-runtime/data/gpqa-official/dataset.zip':
                    blob = stream.extractfile(member).read()
                    origin = dict(path=str(path.resolve()), kind='nested_zip', member=member.name)
                    break
            else:
                raise ValueError('official GPQA nested zip absent')
    if digest(blob) != GPQA_ZIP_SHA:
        raise ValueError('not the retained official GPQA archive')
    tables, identities = {}, {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zipped:
        for name, expected in GPQA_CSV_SHA.items():
            member = 'dataset/gpqa_' + name + '.csv'
            data = zipped.read(member, pwd=b'deserted-untie-orchid')
            if digest(data) != expected:
                raise ValueError('GPQA member differs: ' + member)
            rows = list(csv.DictReader(io.StringIO(data.decode('utf-8'))))
            tables[name] = rows
            identities[name] = dict(member=member, rows=len(rows), bytes=len(data), sha256=expected)
    return tables, dict(**origin, zip_sha256=digest(blob), zip_bytes=len(blob), members=identities)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def write_jsonl(path, rows):
    with Path(path).open('xb') as stream:
        for row in rows:
            stream.write(wire(row) + b'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', type=Path, default=REPO)
    ap.add_argument('--gpqa-archive', type=Path)
    ap.add_argument('--gpqa-env-tar', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo_root.resolve()
    if args.output.exists():
        raise ValueError('preserve existing preparation: output must be new')
    pointer = repo / 'closeout_20261011/assets/private_staging/extracted/runweaver/runweaver/src/agentic_evaluate/benchmarks/openjev_pointer.py'
    materialized = repo / 'arr_paper/reproduction/round5/source_qualification/materialized'
    d0 = source_criteria(pointer)
    descriptions = description_sets(d0)
    data = {}
    for name, expected in PIQA_PINS.items():
        blob = (materialized / name).read_bytes()
        if digest(blob) != expected:
            raise ValueError('PIQA original population differs: ' + name)
        data[name] = [json.loads(line) for line in blob.decode('utf-8').splitlines()]
    questions, requests = data['piqa.questions.jsonl'], data['piqa.requests.jsonl']
    if len(questions) != 1838 or len(requests) != 3676:
        raise ValueError('PIQA population count mismatch')
    by_request = {row['request_id']: row for row in requests}
    if len(by_request) != 3676 or len({q['id'] for q in questions}) != 1838:
        raise ValueError('collapsed PIQA identity')
    piqa_ledger = []
    for question in questions:
        ids = question['frozen_candidate_ids']
        if len(ids) != 2 or len(set(ids)) != 2 or ids != [c['id'] for c in question['candidates']] or question['gold'] not in ids:
            raise ValueError('PIQA candidate mapping differs')
        for candidate, request_id in zip(question['candidates'], question['request_ids']):
            observed = by_request[request_id]
            generated = nli_request(question, candidate, d0, LABELS)
            if (observed['sample_id'] != question['id'] or observed['candidate_id'] != candidate['id'] or
                    wire(observed['payload']) != wire(generated)):
                raise ValueError('D0/P0 is not the exact historical PIQA wire')
        piqa_ledger.append(dict(sample_id=question['id'], source_revision=question['source_revision'],
            stem_sha256=digest(question['stem'].encode()), original_candidate_ids=question['original_candidate_ids'],
            frozen_candidate_ids=ids, gold_candidate_id=question['gold'],
            candidates=[dict(candidate_id=c['id'], text_sha256=digest(c['text'].encode())) for c in question['candidates']],
            original_request_ids=question['request_ids'], direct_wire_sha256=digest(wire(direct_request(question)))))
    print('PIQA 1838 identities and all 3676 exact historical D0/P0 wires verified', flush=True)
    default_tar = repo / 'arr_paper/experiment_supplement_20261012/lowrate_noffn_four_20261013_v1/H20_EVAL_ENV_EXACT_SOURCE_V2.tar.gz'
    tables, gpqa_identity = read_gpqa(args.gpqa_archive, args.gpqa_env_tar or default_tar)
    main_rows, diamond = tables['main'], tables['diamond']
    if len(main_rows) != 448 or len(diamond) != 198:
        raise ValueError('GPQA population counts differ')
    for rows in (main_rows, diamond):
        if len({r['Record ID'] for r in rows}) != len(rows) or len({norm(r['Question']) for r in rows}) != len(rows):
            raise ValueError('GPQA source identity duplicates')
    eligible, partition = gpqa_partition(main_rows, diamond)
    extra_collisions = [r['record_id'] for r in partition if r['exclusion_reasons'] == ['question_revision_alias_in_evaluation']]
    if len(eligible) != 249 or extra_collisions != [EXCLUDED_VERSION_COLLISION]:
        raise ValueError('GPQA revised-question exclusion population changed')
    ranked = sorted(eligible, key=lambda r: (demo_rank(r), r['Record ID']))
    selected = ranked[:5]
    if tuple(r['Record ID'] for r in selected) != EXPECTED_DEMOS:
        raise ValueError('fixed five-demo selection differs')
    demo_selection = dict(namespace=DEMO_NAMESPACE, eligible_pool=249, selected_n=5,
        normalization='Unicode NFKC, casefold, whitespace folding; punctuation retained',
        order_rule='SHA256(UTF8 compact JSON [namespace, RecordID]), then RecordID; first five',
        ranked_eligible=[dict(rank=i + 1, record_id=r['Record ID'], rank_hash=demo_rank(r), selected=i < 5) for i, r in enumerate(ranked)],
        selected=[dict(record_id=r['Record ID'], question_sha256=digest(r['Question'].strip().encode()),
                       answer_sha256=digest(r['Correct Answer'].strip().encode()), domain=r['High-level domain'],
                       question_chars=len(r['Question'].strip()), answer_chars=len(r['Correct Answer'].strip())) for r in selected])
    eval_ledger = [dict(source_ordinal=i, current_question_sha256=digest(r['Question'].strip().encode()),
        normalized_aliases={f: digest(v.encode()) for f, v in question_aliases(r).items()}, **candidate_ledger(r))
        for i, r in enumerate(diamond)]
    queue_path = repo / 'arr_paper/experiment_supplement_20261012/QUEUE.json'
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    models = []
    for fold, seed, head in itertools.product((2, 3), (44, 45), ('H0', 'H5')):
        rid = f'SUPP_WINH4_{head}_f{fold}_s{seed}'
        rows = [r for r in queue['runs'] if r['run_id'] == rid]
        if len(rows) != 1:
            raise ValueError('missing/duplicate fixed historical run: ' + rid)
        argv = rows[0]['argv']
        get = lambda flag: argv[argv.index(flag) + 1]
        if get('--lr') != '2.5e-05' or get('--head_lr') != '2.5e-05' or get('--lora_targets') != 'all':
            raise ValueError('unexpected historical recipe')
        models.append(dict(run_id=rid, fold=fold, seed=seed, method=head,
            head_config=json.loads(get('--head_config')), original_checkpoint=get('--out'),
            original_code_tree=queue['code_tree'], h20_checkpoint=None,
            availability='original path absent on WIN3; migrated copy or transfer requires verification'))
    protocol = dict(status='PROPOSED_CPU_INPUTS_PREPARED_PENDING_PRO_REVIEW', version='inference-block-proposal-v1',
        models=8, cells=4, piqa_questions=1838, piqa_candidates=2, gpqa_evaluation_questions=198,
        permutations=[dict(id='P' + str(i), order=list(order)) for i, order in enumerate(PERMUTATIONS)],
        descriptions=descriptions, direct_instruction=DIRECT_INSTRUCTION,
        shared_baseline='E7 NLI = E9 P0D0; same calls counted once',
        gpqa_protocol='gpqa_disjoint5_v1; legacy zero/five-shot results retained separately',
        tokenization_checked=False, predictions_generated=False,
        requests_per_model=dict(piqa_direct=1838, piqa_nli_18=66168, gpqa_new_fewshot_rerank=792,
                                gpqa_new_fewshot_grade=792, total=69590), requests_all_eight=556720)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'proposed_protocol.json', protocol)
    write_json(args.output / 'checkpoints.json', dict(source_queue=str(queue_path), source_queue_sha256=digest(queue_path.read_bytes()), models=models))
    write_json(args.output / 'assets.json', dict(piqa_root=str(materialized), piqa_members=PIQA_PINS,
        d0_source=str(pointer), d0_source_sha256=digest(pointer.read_bytes()), gpqa=gpqa_identity))
    write_jsonl(args.output / 'piqa_identity_ledger.jsonl', piqa_ledger)
    write_jsonl(args.output / 'gpqa_main_partition_ledger.jsonl', partition)
    write_jsonl(args.output / 'gpqa_diamond_identity_ledger.jsonl', eval_ledger)
    write_json(args.output / 'gpqa_demo_selection.json', demo_selection)
    write_json(args.output / 'PREPARE_RECEIPT.json', dict(status=protocol['status'], predictions_generated=False,
        piqa_original_wire_equal=3676, main_records=448, diamond_records=198,
        main_minus_diamond_before_version_check=250, extra_version_collisions=extra_collisions,
        final_demo_pool=249, selected_demo_ids=list(EXPECTED_DEMOS),
        files={p.name:digest(p.read_bytes()) for p in args.output.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=protocol['status'], output=str(args.output.resolve()), final_demo_pool=249,
                          models=8, requests_all_eight=556720, tokenization_checked=False)), flush=True)


if __name__ == '__main__':
    main()
