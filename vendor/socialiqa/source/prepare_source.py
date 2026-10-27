"""Freeze official SocialIQA dev and a reversible audit-only JEV messages view.

Never changes source records, training releases, or benchmark population.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


def wire(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_bytes(canonical(obj) + b'\n')


def write_rows(path, rows):
    with Path(path).open('xb') as f:
        for row in rows:
            f.write(canonical(row) + b'\n')


def source_criteria(path):
    nodes = ast.parse(Path(path).read_text(encoding='utf-8')).body
    values = [ast.literal_eval(n.value) for n in nodes if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'CRITERIA' for t in n.targets)]
    if len(values) != 1 or list(values[0]) != ['entailment', 'neutral', 'contradiction']:
        raise ValueError('actual D0 criteria/order unavailable')
    return values[0]


def freeze_eval(root, d0_file):
    acquisition = root / 'acquisition_v1'
    archive = acquisition / 'official_v1.4.tgz'
    if sha(archive) != '99a399d50f1a996388e94ab435ba0ceadc6b8ad9d90e6420358c962a4850d217':
        raise ValueError('official v1.4 archive identity differs')
    output = root / 'frozen_v1'
    output.mkdir(exist_ok=False)
    with tarfile.open(archive) as tf:
        for name in ('socialIQa_v1.4_dev.jsonl', 'README.md', 'LICENCE-CC-BY-4.txt'):
            (output / name).write_bytes(tf.extractfile(name).read())
    rows = [json.loads(x) for x in (output / 'socialIQa_v1.4_dev.jsonl').read_bytes().splitlines()]
    with zipfile.ZipFile(acquisition / 'modelscope_referenced_official.zip') as z:
        mirrored = [json.loads(x) for x in z.read('socialiqa-train-dev/dev.jsonl').splitlines()]
        labels = z.read('socialiqa-train-dev/dev-labels.lst').decode().splitlines()
    if len(rows) != len(mirrored) or len(rows) != len(labels):
        raise ValueError('separate gold alignment differs')
    required = {'context', 'question', 'answerA', 'answerB', 'answerC', 'correct'}
    items, collisions = [], []
    for ordinal, (row, mirror, label) in enumerate(zip(rows, mirrored, labels)):
        if set(row) != required or any(not isinstance(v, str) or not v.strip() for v in row.values()):
            raise ValueError(f'invalid full three-candidate source row {ordinal}')
        if set(mirror) != required - {'correct'} or any(row[k] != mirror[k] for k in mirror):
            raise ValueError('ModelScope-referenced official dev text differs')
        if label not in ('1', '2', '3') or row['correct'] != 'ABC'[int(label)-1]:
            raise ValueError('ModelScope-referenced official gold differs')
        item_id = f'socialiqa-v1.4-dev:{ordinal:04d}'
        item = dict(item_id=item_id, source_ordinal=ordinal, context=row['context'], question=row['question'],
                    premise=row['context']+'\n\n'+row['question'], gold=row['correct'],
                    answer=row['answer'+row['correct']], candidates=[dict(id=c, text=row['answer'+c]) for c in 'ABC'],
                    raw_row_sha256=hashlib.sha256(canonical(row)).hexdigest())
        texts = [c['text'] for c in item['candidates']]
        if len(set(texts)) != 3:
            collisions.append(dict(item_id=item_id, candidate_groups=[[c['id'] for c in item['candidates'] if c['text']==t]
                                                                     for t in dict.fromkeys(texts) if texts.count(t)>1]))
        items.append(item)
    write_rows(output/'items.jsonl', items)
    criteria = source_criteria(d0_file)
    protocol = dict(schema='jev-socialiqa-input-v1', status='FROZEN_BEFORE_MODEL_SCORING',
        study_freeze='../E10_E11_PROSPECTIVE_FREEZE.md', full_questions=len(items), candidates_per_question=3,
        premise='context + "\n\n" + question, no strip/rewrite', candidate_order=list('ABC'),
        direct_instructions='Select the correct answer to the question.',
        nli_instructions='Hypothesis: "{candidate answer}" How does it relate to the premise?',
        nli_criteria=criteria, nli_order=list(criteria), nli_score='probabilities[entailment]',
        tie_rule='first maximum in original A/B/C candidate order, independent of gold',
        gold_on_wire=False, direct_head='actual trained classification head; no generative LM path',
        full_requests_per_model=len(items)*4, E10_models=8, E11_models=8,
        E10_requests=len(items)*4*8, E11_requests=len(items)*4*8,
        d0_source_sha256=sha(d0_file), duplicate_candidate_items=collisions,
        local_item_identity='release/split/zero-based source ordinal; NOT an upstream item ID',
        model_selection='fixed eight one-epoch and separate eight two-epoch identities in prospective freeze')
    write_json(output/'PROTOCOL.json', protocol)
    config = {'assets':[dict(benchmark='socialiqa', split='dev', revision='official-v1.4; archive-sha256:'+sha(archive),
        path='items.jsonl', sha256=sha(output/'items.jsonl'), format='jsonl', id_field='item_id',
        fields={'question':['premise'], 'answer':['answer']}, license='CC-BY-4.0 (official archive LICENCE-CC-BY-4.txt)',
        protocol='protected full official gold dev; exact source A/B/C identities; no evaluation mutation')]}
    write_json(output/'eval-assets.json', config)
    write_json(output/'SOURCE_FREEZE.json', dict(source_archive_sha256=sha(archive), source_archive_url='https://maartensap.com/social-iqa/data/socialIQa_v1.4.tgz',
        source_version='v1.4', split='dev', rows=len(items), gold_counts=dict(Counter(i['gold'] for i in items)),
        gold_alignment='all rows and A/B/C gold exactly match ModelScope loader referenced AI2 archive',
        modelscope_repository_revision='3d6830da625ebbd330ed943e3082684032c021ad',
        modelscope_asset_finding='only loader/card/metadata in repo; loader downloads official AI2 public zip',
        modelscope_license_conflict='top-level Apache2.0; card and actual official dataset archive CC-BY-4.0; not conflated',
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()}, predictions_generated=False))
    return len(items)


def string_values(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from string_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from string_values(v)


def adapt_record(row):
    request = row.get('request', row)
    if not isinstance(request.get('state'), str) or not isinstance(request.get('questions'), dict):
        raise ValueError('unsupported JEV structure; no silent skipping')
    question = [request['state']]
    answers = []
    for name, q in request['questions'].items():
        if not isinstance(q, dict):
            raise ValueError('invalid question')
        if q.get('instructions'):
            question.append(q['instructions'])
        criteria = q.get('criteria')
        if not isinstance(criteria, (dict, list)):
            raise ValueError('unsupported criteria')
        question.extend(string_values(criteria))
        gold = row.get('gold', {}).get(name, q.get('label'))
        if isinstance(criteria, dict):
            key = ('true' if gold else 'false') if isinstance(gold, bool) else str(gold)
            if key in criteria:
                answers.extend(string_values(criteria[key]))
        elif isinstance(gold, int) and not isinstance(gold, bool) and 0 <= gold < len(criteria):
            answers.extend(string_values(criteria[gold]))
    return {'messages':[{'role':'user','content':'\n'.join(question)},
                        {'role':'assistant','content':'\n'.join(answers)}]}


def adapt_sources(root, reconstruction):
    # The physical union includes actual inputs and their retained derivation chain.
    paths = {
        'actual_F2': ('actual_v1/F2/train.jsonl', '9253f1c937978174b2ff2e2959ce732c06fe3c0a220154fb962be1c2a1478017'),
        'actual_F3': ('actual_v1/F3/train.jsonl', '2a1cfac79cd64d9cb22f43c4b69c3283fd44c73c92a4468c661d27ac9c897e2a'),
        'base_nimble_kev': ('retained/data-round9/mixtures/nimble/train.jsonl','fc1a7d37c884451c4f99eb3ae8a86eceda8359725798382af803f433e3f1fa0f'),
        'night2': ('retained/kev/evals/night2/all.jsonl','b8692830579a1b501aed66a4b5490c943a6fc47d5eca2ac7d1fa132eda0a3319'),
        'jevforge_raw': ('retained/data-round10/raw/jevforge/data/train.jsonl','aefbce0389e3450ceb91c83d6d6bff20c3daf8d50f4b6fcdf42e29e420eb5fae'),
        'jevforge_projection': ('retained/data-round10/components/jevforge/train.jsonl',None),
        'merged_pool': ('actual_v1/train.jsonl','a99af30c53da506afb16dcd43ec900fd6302ac51a732ab663bba2ad645207368')}
    output = root/'audit_release_v1'
    (output/'messages').mkdir(parents=True, exist_ok=False)
    (output/'ledgers').mkdir()
    sources = []
    originals = set()
    with (output/'ledgers/provenance.jsonl').open('xb') as prov, (output/'ledgers/reversible_rows.jsonl').open('xb') as ledger:
        for split, (relative, expected) in paths.items():
            source = reconstruction/relative
            actual = sha(source)
            if expected is not None and actual != expected:
                raise ValueError('actual retained source hash mismatch: '+relative)
            count = 0
            with source.open('rb') as src, (output/'messages'/f'{split}.jsonl').open('xb') as dest:
                for line, raw in enumerate(src, 1):
                    row = json.loads(raw)
                    adapted = adapt_record(row)
                    encoded = canonical(adapted)
                    dest.write(encoded+b'\n')
                    stable = f'{actual}:{line}'
                    original_sha = hashlib.sha256(canonical(row)).hexdigest()
                    originals.add(original_sha)
                    meta = row.get('_meta', row.get('meta', {}))
                    identity = meta.get('id', row.get('id'))
                    provenance = dict(stable_key=stable, split=split, canonical_row_sha256=hashlib.sha256(encoded).hexdigest(),
                        original_file_sha256=actual, original_line=line, original_row_sha256=original_sha,
                        source_metadata={'item_id':identity} if identity is not None else {})
                    if meta.get('group_id'):
                        provenance['lineage_group_id'] = meta['group_id']
                    prov.write(canonical(provenance)+b'\n')
                    ledger.write(canonical(dict(stable_key=stable, split=split, source_path=str(source), source_line=line,
                        source_raw_line_sha256=hashlib.sha256(raw).hexdigest(), original_row=row))+b'\n')
                    count += 1
            sources.append(dict(split=split, path=str(source), sha256=actual, rows=count))
    write_json(output/'SOURCE_MANIFEST.json', dict(schema='jev-reversible-audit-view-v1', sources=sources,
        physical_rows=sum(x['rows'] for x in sources), unique_original_canonical_rows=len(originals),
        mapping='one canonical audit row per physical source row; exact original row and physical line preserved',
        protected_originals=True, purpose='audit only; never a training release',
        answer_mapping='gold/label selected criterion text; candidate texts also retained in user channel',
        identity='only explicit original meta.id/id/group_id; eval generated IDs are NOT upstream IDs',
        uncovered=['earliest public raw not fully reconstructed', 'Nimble upstream repository URL unresolved',
                   'untraceable additional materials', 'pretraining exposure', 'deep paraphrases', 'cross-language leakage']))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--d0-source', type=Path, required=True)
    ap.add_argument('--reconstruction', type=Path, required=True)
    a=ap.parse_args()
    n=freeze_eval(a.root,a.d0_source)
    adapt_sources(a.root,a.reconstruction)
    print(json.dumps(dict(status='FROZEN_AND_ADAPTED_NO_SCAN_YET', full_questions=n)))


if __name__=='__main__': main()
