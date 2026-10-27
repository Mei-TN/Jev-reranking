"""Create a reversible messages view of JEV training inputs for overlap checks.

Pass each source row through kev.data.materialize and kev.api.to_record,
then emit model-visible text in encoder packing order. Preserve source rows,
option names, and ordering.

  build   write a new output directory
  verify  rebuild output files from the ledger and compare their bytes
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # siblings stay importable under python -I
import jev_renderer

SCHEMA = 'jev-reversible-audit-view-v2'
JEV, JEVFORGE = 'jev_labelled_request', 'jevforge_released_gold_record'
ACTUAL, RETAINED = 'actual_training_input', 'retained_derivation_input'
# Pinned physical inputs, relative to corpus_reconstruction_v1. The projection pin is the
# train_sha256 already recorded by retained/data-round10/components/jevforge/manifest.json.
SOURCES = (
    dict(split='actual_F2', path='actual_v1/F2/train.jsonl', schema=JEV, role=ACTUAL,
         sha256='9253f1c937978174b2ff2e2959ce732c06fe3c0a220154fb962be1c2a1478017'),
    dict(split='actual_F3', path='actual_v1/F3/train.jsonl', schema=JEV, role=ACTUAL,
         sha256='2a1cfac79cd64d9cb22f43c4b69c3283fd44c73c92a4468c661d27ac9c897e2a'),
    dict(split='base_nimble_kev', path='retained/data-round9/mixtures/nimble/train.jsonl', schema=JEV, role=RETAINED,
         sha256='fc1a7d37c884451c4f99eb3ae8a86eceda8359725798382af803f433e3f1fa0f'),
    dict(split='night2', path='retained/kev/evals/night2/all.jsonl', schema=JEV, role=RETAINED,
         sha256='b8692830579a1b501aed66a4b5490c943a6fc47d5eca2ac7d1fa132eda0a3319'),
    dict(split='jevforge_raw', path='retained/data-round10/raw/jevforge/data/train.jsonl', schema=JEVFORGE, role=RETAINED,
         sha256='aefbce0389e3450ceb91c83d6d6bff20c3daf8d50f4b6fcdf42e29e420eb5fae'),
    dict(split='jevforge_projection', path='retained/data-round10/components/jevforge/train.jsonl', schema=JEV, role=RETAINED,
         sha256='4beaaa491ae2d1daac82e44f99fe89b5d16562d59a73533040adc00e9ad68fa9'),
    dict(split='merged_pool', path='actual_v1/train.jsonl', schema=JEV, role=RETAINED,
         sha256='a99af30c53da506afb16dcd43ec900fd6302ac51a732ab663bba2ad645207368'))
# Released JevForge records carry no per-question `src`; materialize copies the field but never renders it.
NO_ORIGINAL_SRC = '<e10-audit-adapter: no src in released record>'
JEVFORGE_KEYS = {'id', 'schema_version', 'split', 'source_group', 'request', 'targets', 'target_kinds', 'gold', 'meta'}
UNCOVERED = (
    'earliest public raw sources of the Kev+Nimble base are not reconstructed; the converted base is scanned instead',
    'Nimble upstream repository URL is unresolved',
    'Mind2Web parquet before the released JevForge gold file is not scanned',
    'materials that are not traceable to these seven files',
    'training-time augmentation (choice-order permutation, inserted none/distractor options, none-pairs) is not rendered; '
    'rows are scanned in their clean un-augmented order',
    'pretraining exposure of the backbone', 'deep paraphrase', 'cross-language leakage')


def wire(value):
    """Insertion-ordered compact JSON: option order is model-visible, so it must survive hashing."""
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def _pairs(items):
    value = dict(items)
    if len(value) != len(items):
        raise ValueError('duplicate JSON object key')
    return value


def _constant(token):
    raise ValueError('non-standard JSON constant ' + token)


def parse_row(text):
    row = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    if not isinstance(row, dict):
        raise ValueError('source row is not an object')
    return row


def physical_lines(blob):
    """(text, terminator) per physical line; b'\\n' is the only row separator, as in the writers."""
    start = 0
    while start < len(blob):
        end = blob.find(b'\n', start)
        end = len(blob) if end < 0 else end + 1
        raw, start = blob[start:end], end
        terminator = '\r\n' if raw.endswith(b'\r\n') else '\n' if raw.endswith(b'\n') else ''
        body = raw[:len(raw) - len(terminator)]
        if not body.strip():
            raise ValueError('blank source row')
        yield body.decode('utf-8'), terminator


def labelled_request(row, schema):
    """The request exactly as the trainer labels it; released JevForge gold/targets are mapped explicitly."""
    if schema == JEV:
        if set(row) != {'state', 'questions', '_meta'} or not isinstance(row['_meta'], dict):
            raise ValueError('unexpected JEV row fields ' + repr(sorted(row)))
        request = row
    elif schema == JEVFORGE:
        if set(row) != JEVFORGE_KEYS or set(row['request']) != {'state', 'questions'}:
            raise ValueError('unexpected released JevForge fields ' + repr(sorted(row)))
        questions = row['request']['questions']
        if not (list(questions) == list(row['gold']) == list(row['targets']) == list(row['target_kinds'])):
            raise ValueError('released JevForge question identities differ between request/gold/targets')
        labelled = {}
        for qid, q in questions.items():
            gold = row['gold'][qid]
            if set(q) != {'type', 'instructions', 'criteria'} or \
                    type(gold) is not {'noul': bool, 'choice': str, 'score': int}[q['type']]:
                raise ValueError(f'released JevForge question {qid!r} has an unexpected shape or gold type')
            labelled[qid] = {**q, 'label': gold, 'target': row['targets'][qid], 'src': NO_ORIGINAL_SRC}
        request = {'state': row['request']['state'], 'questions': labelled}
    else:
        raise ValueError('unknown source schema ' + schema)
    if not isinstance(request['questions'], dict) or not request['questions']:
        raise ValueError('row has no questions')
    for qid, q in request['questions'].items():
        if not isinstance(q, dict) or 'label' not in q or 'src' not in q:
            raise ValueError(f'question {qid!r} lacks the label/src the trainer requires')
    return request


def target_kind(target):
    if target is None:
        return 'hard_label'
    positive = [mass for mass in target if mass > 0]
    if len(positive) == 1:
        return 'onehot'
    if max(positive) - min(positive) > 1e-9:
        return 'nonuniform'
    return 'uniform_all' if len(positive) == len(target) else 'uniform_subset'


def adapt_record(materialize, row, schema):
    """-> (canonical message row bytes, render facts). User text is the packed input, in encoder order."""
    record = materialize(labelled_request(row, schema))
    segments, answers, questions = [record['state']], [], []
    for q in record['questions']:
        options, target = q['options'], q.get('target')
        if len(q['keys']) != len(options) or not 0 <= q['label'] < len(options) or \
                (target is not None and len(target) != len(options)):
            raise ValueError('renderer returned an inconsistent option/label/target layout')
        # question_loss trains on the soft target when present, otherwise on the hard label.
        supervised = [i for i, mass in enumerate(target) if mass > 0] if target is not None else [q['label']]
        segments.append(q['instr'])
        segments.extend(options)
        answers.extend(options[i] for i in supervised)
        questions.append(dict(qid=q['qid'], qtype=q['qtype'], options=len(options), label_index=q['label'],
                              label_key=q['keys'][q['label']], supervised_indices=supervised,
                              target_kind=target_kind(target), src=q['src']))
    if any(type(text) is not str for text in segments):
        raise ValueError('renderer returned non-text model input')
    message = {'messages': [{'role': 'user', 'content': '\n'.join(segments)},
                            {'role': 'assistant', 'content': '\n'.join(answers)}]}
    return canonical(message), dict(segment_chars=[len(text) for text in segments], questions=questions)


def split_segments(user, segment_chars):
    """Inverse of the '\\n' join: the recorded lengths recover every rendered segment exactly."""
    segments, position = [], 0
    for index, size in enumerate(segment_chars):
        segments.append(user[position:position + size])
        position += size
        if index + 1 < len(segment_chars):
            if user[position:position + 1] != '\n':
                raise ValueError('segment boundary is not the adapter separator')
            position += 1
    if position != len(user):
        raise ValueError('segment lengths do not cover the user text')
    return segments


def original_metadata(row, schema):
    return row['_meta'] if schema == JEV else {key: value for key, value in row.items() if key != 'request'}


def explicit_identity(row, schema):
    """Only identifiers present in the source row. Evaluation ordinals are never looked up or written here."""
    meta = row['_meta'] if schema == JEV else row
    group = meta.get('group_id') if schema == JEV else meta.get('source_group')
    fields = ('source', 'repo', 'revision', 'split', 'row') if schema == JEV else ('schema_version', 'split')
    identity = {key: meta[key] for key in fields if meta.get(key) is not None}
    if isinstance(meta.get('id'), str) and meta['id']:
        identity['item_id'] = meta['id']
    if isinstance(group, str) and group:
        identity['group_id'] = group
    if schema == JEVFORGE and isinstance(row['meta'].get('annotation_id'), str):
        identity['annotation_id'] = row['meta']['annotation_id']
    return identity


def adapted_source(materialize, blob, source):
    """One (message bytes, provenance, ledger, row) per physical row, in file order."""
    for line, (text, terminator) in enumerate(physical_lines(blob), 1):
        try:
            row = parse_row(text)
            encoded, facts = adapt_record(materialize, row, source['schema'])
        except Exception as error:
            raise ValueError(f"{source['split']} line {line}: {error}") from error
        shared = dict(stable_key=f"{source['split']}:{source['sha256']}:{line}", split=source['split'],
                      source_role=source['role'], source_schema=source['schema'])
        # No lineage_group_id: the explicit group is the grouped-split unit (one row up to a whole
        # website), so letting the scanner propagate a red decision through it would overstate evidence.
        provenance = dict(shared, canonical_row_sha256=digest(encoded), original_file_sha256=source['sha256'],
                          original_line=line, original_raw_line_sha256=digest(text.encode('utf-8')),
                          original_row_ordered_sha256=digest(wire(row)),
                          source_metadata=explicit_identity(row, source['schema']))
        ledger = dict(shared, source_path=source['path'], source_file_sha256=source['sha256'], source_line=line,
                      source_line_terminator=terminator, source_raw_line_sha256=provenance['original_raw_line_sha256'],
                      original_raw_line=text, original_row_ordered_sha256=provenance['original_row_ordered_sha256'],
                      original_metadata=original_metadata(row, source['schema']),
                      adapted_row_sha256=provenance['canonical_row_sha256'], render=facts)
        yield encoded, provenance, ledger, row


def loader_lines_equal(blob):
    """kev.data.load_records reads str.splitlines(); it must see exactly these physical rows."""
    text = blob.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')
    return [line for line in text.splitlines() if line.strip()] == [body for body, _ in physical_lines(blob)]


def file_facts(path):
    blob = Path(path).read_bytes()
    return dict(bytes=len(blob), rows=blob.count(b'\n'), sha256=digest(blob))


def release_files(root):
    """Facts of exactly the files build writes; anything else in a release is foreign."""
    names = ['ledgers/provenance.jsonl', 'ledgers/reversible_rows.jsonl'] + [
        f"messages/{source['split']}.jsonl" for source in SOURCES]
    return {name: file_facts(Path(root) / name) for name in sorted(names)}


class Accounting:
    """Everything MANIFEST.json states about the rows. build and verify fill it the same way."""

    def __init__(self):
        self.sources, self.rendered, self.ordered, self.ids, self.groups = [], {}, set(), {}, {}
        self.counts, self.rows = Counter(), 0

    def add(self, source, provenance, ledger, row):
        state = row['state'] if source['schema'] == JEV else row['request']['state']
        self.counts['state:' + type(state).__name__] += 1
        for q in ledger['render']['questions']:
            self.counts['question:' + q['qtype']] += 1
            self.counts['target:' + q['target_kind']] += 1
            self.counts['label_outside_supervised_set'] += q['label_index'] not in q['supervised_indices']
        identity, rendered = provenance['source_metadata'], provenance['canonical_row_sha256']
        self.counts['explicit_item_id'] += 'item_id' in identity
        self.counts['explicit_group_id'] += 'group_id' in identity
        self.rendered.setdefault(rendered, set()).add(source['split'])
        self.ordered.add(provenance['original_row_ordered_sha256'])
        if 'item_id' in identity:
            self.ids.setdefault((source['schema'], identity['item_id']), set()).add(rendered)
            if 'group_id' in identity:
                self.groups.setdefault((source['schema'], identity['group_id']), set()).add(identity['item_id'])
        self.rows += 1

    def close(self, source, size, loader_equal):
        self.sources.append(dict(split=source['split'], path=source['path'], sha256=source['sha256'], bytes=size,
                                 rows=self.rows, schema=source['schema'], role=source['role'],
                                 trainer_loader_sees_same_rows=loader_equal, counts=dict(sorted(self.counts.items()))))
        self.counts, self.rows = Counter(), 0

    def manifest(self, identities, files):
        actual = {s['split'] for s in SOURCES if s['role'] == ACTUAL}
        return dict(
            schema=SCHEMA, purpose='audit only; never a training release; no source row removed, reordered or merged',
            supersedes='audit_release_v1 (stopped before any row at actual_F2 line 1; preserved unchanged)',
            renderer=identities,
            adapter_sha256={name: jev_renderer.sha256_file(Path(__file__).with_name(name))
                            for name in ('jev_audit_adapter.py', 'jev_renderer.py')},
            sources=self.sources, physical_rows=sum(s['rows'] for s in self.sources),
            channels=dict(
                user='rendered state, then per question its rendered instructions and every rendered option, joined '
                     'by "\\n" in encoder packing order; delimiter tokens are not written',
                assistant='per question the rendered option(s) carrying supervised mass: every option with target>0 '
                          'when a soft target exists, otherwise the hard-label option',
                auxiliary='none'),
            identity=dict(
                stable_key='split:source_file_sha256:one_based_line, one per physical row',
                item_id='only _meta.id (JEV rows) or id (released JevForge records) present in the source row',
                group_id='_meta.group_id / source_group kept as descriptive metadata only. It is the grouped-split '
                         'unit, from one row up to a whole website, so it is not written as lineage_group_id and '
                         'never propagates a decision',
                evaluation_ids='generated socialiqa-v1.4-dev ordinals never enter this release'),
            logical_accounting=dict(
                basis='identical canonical message row = identical model-visible text and supervised option text',
                unique_rendered_rows=len(self.rendered),
                unique_rendered_rows_in_actual_training_inputs=sum(1 for s in self.rendered.values() if s & actual),
                unique_rendered_rows_only_in_retained_inputs=sum(1 for s in self.rendered.values() if not s & actual),
                unique_original_rows_including_metadata=len(self.ordered),
                unique_explicit_item_ids=len(self.ids),
                explicit_item_ids_with_more_than_one_rendering=sum(1 for shas in self.ids.values() if len(shas) > 1),
                explicit_groups=len(self.groups),
                largest_explicit_group_distinct_item_ids=max((len(v) for v in self.groups.values()), default=0)),
            files=files, uncovered=list(UNCOVERED))


def build(reconstruction, code_tree, output):
    output = Path(output).resolve()
    incomplete = output.with_name(output.name + '.incomplete')
    if output.exists() or incomplete.exists():
        raise ValueError('output or its .incomplete sibling already exists; use a new versioned path')
    _, materialize, identities = jev_renderer.load_renderer(code_tree)
    (incomplete / 'messages').mkdir(parents=True)
    (incomplete / 'ledgers').mkdir()
    accounting = Accounting()
    with (incomplete / 'ledgers/provenance.jsonl').open('xb') as prov_file, \
            (incomplete / 'ledgers/reversible_rows.jsonl').open('xb') as ledger_file:
        for source in SOURCES:
            blob = (Path(reconstruction) / source['path']).read_bytes()
            if digest(blob) != source['sha256']:
                raise ValueError('pinned source differs: ' + source['path'])
            loader_equal = loader_lines_equal(blob)
            if source['role'] == ACTUAL and not loader_equal:
                raise ValueError('trainer loader would not see these physical rows: ' + source['path'])
            with (incomplete / 'messages' / (source['split'] + '.jsonl')).open('xb') as message_file:
                for encoded, provenance, ledger, row in adapted_source(materialize, blob, source):
                    message_file.write(encoded + b'\n')
                    prov_file.write(canonical(provenance) + b'\n')
                    ledger_file.write(canonical(ledger) + b'\n')
                    accounting.add(source, provenance, ledger, row)
            accounting.close(source, len(blob), loader_equal)
    manifest = accounting.manifest(identities, release_files(incomplete))
    (incomplete / 'MANIFEST.json').write_bytes(canonical(manifest) + b'\n')
    os.replace(incomplete, output)
    return manifest


def native_materialize(code_tree):
    """kev.data imported whole (needs the training environment), to compare with the extracted function."""
    sys.dont_write_bytecode = True
    for relative in ('kev/api.py', 'kev/data.py'):
        jev_renderer.pinned(code_tree, relative)
    sys.path.insert(0, str(code_tree))
    import kev.data as native_data
    if Path(native_data.__file__).resolve() != (Path(code_tree) / 'kev/data.py').resolve():
        raise ValueError('a different kev.data was imported')
    return native_data.materialize


def verify(release, code_tree, native_code_tree=None):
    """Second pass with no access to the sources: the raw lines kept in the ledger must reproduce the
    pinned source bytes, and re-adapting them must reproduce every stored row and the manifest."""
    release = Path(release)
    _, materialize, identities = jev_renderer.load_renderer(code_tree)
    native = native_materialize(native_code_tree) if native_code_tree else None
    accounting, rows, native_rows, reproduced, failures = Accounting(), Counter(), 0, [], []
    message_shas, provenance_shas = Counter(), Counter()

    def fail(text):
        failures.append(text)
        if len(failures) > 50:
            raise ValueError('verification stopped after more than 50 differences')

    try:
        with (release / 'ledgers/provenance.jsonl').open('rb') as prov_file, \
                (release / 'ledgers/reversible_rows.jsonl').open('rb') as ledger_file:
            pending = ledger_file.readline()
            for source in SOURCES:
                split, stored, parts = source['split'], [], []
                while pending and json.loads(pending).get('split') == split:
                    record = json.loads(pending)
                    stored.append(pending)
                    parts.append((record['original_raw_line'] + record['source_line_terminator']).encode('utf-8'))
                    pending = ledger_file.readline()
                blob = b''.join(parts)
                if digest(blob) == source['sha256']:
                    reproduced.append(split)
                else:
                    fail('ledger does not reproduce the pinned source bytes: ' + split)
                with (release / 'messages' / (split + '.jsonl')).open('rb') as message_file:
                    for stored_ledger, (encoded, provenance, ledger, row) in zip(
                            stored, adapted_source(materialize, blob, source), strict=True):
                        rows[split] += 1
                        stored_message, stored_provenance = message_file.readline(), prov_file.readline()
                        for name, got, want in (('message', stored_message, encoded + b'\n'),
                                                ('provenance', stored_provenance, canonical(provenance) + b'\n'),
                                                ('ledger', stored_ledger, canonical(ledger) + b'\n')):
                            if got != want:
                                fail(f'{split} line {rows[split]}: stored {name} row is not the row rebuilt from the ledger')
                        message_shas[(split, digest(stored_message.rstrip(b'\n')))] += 1
                        kept = json.loads(stored_provenance)
                        provenance_shas[(kept['split'], kept['canonical_row_sha256'])] += 1
                        accounting.add(source, provenance, ledger, row)
                        if native is not None:
                            native_rows += 1
                            request = labelled_request(row, source['schema'])
                            if canonical(native(request)) != canonical(materialize(request)):
                                fail(f'{split} line {rows[split]}: extracted and imported materialize differ')
                    if message_file.readline():
                        fail('message rows without a ledger row: ' + split)
                accounting.close(source, len(blob), loader_lines_equal(blob) if blob else False)
            if pending or prov_file.readline():
                fail('ledger or provenance rows outside the pinned source order')
        expected = accounting.manifest(identities, release_files(release))
        if canonical(expected) + b'\n' != (release / 'MANIFEST.json').read_bytes():
            fail('MANIFEST.json is not the manifest rebuilt from the ledgers with this adapter')
        present = {p.relative_to(release).as_posix() for p in release.rglob('*') if not p.is_dir()}
        if present != set(expected['files']) | {'MANIFEST.json'}:
            fail('release holds files that build does not write: ' + repr(sorted(present - set(expected['files']))[:5]))
        if message_shas != provenance_shas:
            fail('message/provenance canonical SHA-256 multisets differ')
    except (ValueError, KeyError, OSError) as error:
        failures.append(f'{type(error).__name__}: {error}')
    return dict(schema='jev-reversible-audit-view-verification-v2', status='PASS' if not failures else 'FAIL',
                manifest_sha256=jev_renderer.sha256_file(release / 'MANIFEST.json'),
                physical_rows=sum(rows.values()), rows_by_split=dict(sorted(rows.items())),
                source_files_reproduced_from_ledger=sorted(reproduced),
                every_stored_row_and_the_manifest_rebuilt_byte_identically=not failures,
                message_provenance_sha256_multiset_closed=message_shas == provenance_shas and bool(message_shas),
                rows_compared_with_imported_kev_data_materialize=native_rows,
                extracted_materialize_sha256=identities['extracted_materialize_sha256'], failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('build', 'verify'):
        command = commands.add_parser(name)
        command.add_argument('--release', type=Path, required=True)
        command.add_argument('--code-tree', type=Path, required=True, help='directory holding the pinned kev/ sources')
    commands.choices['build'].add_argument('--reconstruction', type=Path, required=True)
    commands.choices['verify'].add_argument('--native-code-tree', type=Path,
                                            help='also import kev.data here (needs the training environment)')
    commands.choices['verify'].add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.command == 'build':
        manifest = build(args.reconstruction, args.code_tree, args.release)
        print(json.dumps(dict(status='ADAPTED_NO_SCAN_YET', physical_rows=manifest['physical_rows'],
                              logical_accounting=manifest['logical_accounting'])))
        return 0
    report = verify(args.release, args.code_tree, args.native_code_tree)
    if args.report:
        with args.report.open('xb') as stream:
            stream.write(canonical(report) + b'\n')
    print(json.dumps(dict(report, failures=report['failures'][:20])))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
