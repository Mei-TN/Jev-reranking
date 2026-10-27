#!/usr/bin/env python3
"""Check fixed inputs with the native tokenizer and encoder on CPU.

The short context profiles are diagnostic; bundle_8192 is the serving profile.
"""
import argparse
import ast
from collections import Counter, defaultdict
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import prepare_inputs as prep

SOURCE_PINS = {
    'kev/model.py': '902427ae67c6cc6d07da4bf0143643cfd1c5d80e58a0f2b3dc1fbb765be0dfc7',
    'kev/api.py': 'c9eebbdb6c625563d33f030299ce0fbe8b50493dd1e92592df9514c30f5720af'}
PROFILES = {
    'bundle_8192': {'max_state': 8192, 'max_branch': 8192, 'max_packed': 8192},
    'historical_serve_decision_defaults': {'max_state': 384, 'max_branch': 1024, 'max_packed': 2048},
    'load_bundle_predict_defaults': {'max_state': 2048, 'max_branch': 8192, 'max_packed': 8192}}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def source_functions(pointer_path, eval_path):
    """Execute only the trusted original input builders, not module-level model imports."""
    pointer = ast.parse(Path(pointer_path).read_text(encoding='utf-8'))
    native = ast.parse(Path(eval_path).read_text(encoding='utf-8'))
    selected = []
    for tree, names in ((pointer, {'CRITERIA', 'pointer_request'}),
                        (native, {'FEWSHOT_HYP', 'with_demos'})):
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                selected.append(node)
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
                selected.append(node)
    original = next(n for n in native.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_mc')
    prefix = []
    for node in original.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pr' for t in node.targets):
            break
        prefix.append(node)
    else:
        raise ValueError('native eval_mc pair prefix not found')
    prefix.append(ast.parse('return rerank_pairs, grade_pairs, grade_gold, offsets').body[0])
    builder = ast.FunctionDef(name='native_pairs', args=original.args, body=prefix,
                              decorator_list=[], returns=None, type_comment=None)
    selected.append(builder)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    scope = {}
    exec(compile(module, '<selected-original-input-builders>', 'exec'), scope)
    if set(('CRITERIA', 'pointer_request', 'FEWSHOT_HYP', 'with_demos', 'native_pairs')) - scope.keys():
        raise ValueError('missing selected original input builder')
    return scope


def profile_failures(state, branches, packed, profile):
    failures = []
    if state > profile['max_state']:
        failures.append('state')
    if any(state + branch > profile['max_branch'] for branch in branches):
        failures.append('state_plus_branch')
    if packed > profile['max_packed']:
        failures.append('packed')
    return failures


def quantiles(values):
    values = sorted(values)
    return dict(n=len(values), min=values[0], median=values[(len(values) - 1) // 2],
                p95=values[math.ceil(len(values) * .95) - 1],
                p99=values[math.ceil(len(values) * .99) - 1], max=values[-1],
                mean=sum(values) / len(values), total=sum(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--gpqa-archive', type=Path, required=True)
    ap.add_argument('--code-tree', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if socket.gethostname() != os.environ.get('JEV_EVALUATION_HOST', socket.gethostname()) or os.getuid() != int(os.environ.get('JEV_EVALUATION_UID', str(os.getuid()))):
        raise RuntimeError('preflight is restricted to WIN3 user')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('launch with CUDA_VISIBLE_DEVICES empty for this CPU-only preflight')
    if args.output.exists():
        raise ValueError('output must be a new directory')
    t0 = time.monotonic()
    receipt = read_json(args.prepared / 'PREPARE_RECEIPT.json')
    for name, expected in receipt['files'].items():
        if prep.digest((args.prepared / name).read_bytes()) != expected:
            raise ValueError('prepared input changed: ' + name)
    protocol = read_json(args.prepared / 'proposed_protocol.json')
    models = read_json(args.prepared / 'checkpoints.json')['models']
    if len(models) != 8 or Counter(m['head_config']['readout'] for m in models) != Counter(end=4, mean_span=4):
        raise ValueError('expected eight matched pointer/pooling checkpoints')
    for relative, expected in SOURCE_PINS.items():
        if prep.digest((args.code_tree / relative).read_bytes()) != expected:
            raise ValueError('native runtime source differs: ' + relative)
    for name, expected in prep.PIQA_PINS.items():
        if prep.digest((args.assets / name).read_bytes()) != expected:
            raise ValueError('PIQA source differs: ' + name)
    frozen_assets = read_json(args.prepared / 'assets.json')
    if prep.digest((args.assets / 'openjev_pointer.py').read_bytes()) != frozen_assets['d0_source_sha256']:
        raise ValueError('original pointer request source differs')
    functions = source_functions(args.assets / 'openjev_pointer.py', args.assets / 'native_eval.py')
    if prep.wire(functions['CRITERIA']) != prep.wire(protocol['descriptions']['D0']):
        raise ValueError('D0 changed from original source')
    questions = read_jsonl(args.assets / 'piqa.questions.jsonl')
    historical = {r['request_id']: r for r in read_jsonl(args.assets / 'piqa.requests.jsonl')}
    piqa_ledger = read_jsonl(args.prepared / 'piqa_identity_ledger.jsonl')
    if len(questions) != 1838 or len(historical) != 3676 or len(piqa_ledger) != 1838:
        raise ValueError('PIQA incomplete')
    for q, ledger in zip(questions, piqa_ledger):
        if q['id'] != ledger['sample_id'] or prep.digest(prep.wire(prep.direct_request(q))) != ledger['direct_wire_sha256']:
            raise ValueError('PIQA direct wire differs from prepared identity')
        for c, rid in zip(q['candidates'], q['request_ids']):
            if prep.wire(prep.nli_request(q, c, functions['CRITERIA'], prep.LABELS)) != prep.wire(historical[rid]['payload']):
                raise ValueError('PIQA original D0/P0 wire differs')
    tables, gpqa_origin = prep.read_gpqa(args.gpqa_archive, None)
    eligible, partition = prep.gpqa_partition(tables['main'], tables['diamond'])
    if prep.wire(partition) != prep.wire(read_jsonl(args.prepared / 'gpqa_main_partition_ledger.jsonl')):
        raise ValueError('GPQA full alias partition differs')
    selection = read_json(args.prepared / 'gpqa_demo_selection.json')
    selected_ids = [r['record_id'] for r in selection['selected']]
    ranked = sorted(eligible, key=lambda r: (prep.demo_rank(r), r['Record ID']))
    if len(eligible) != 249 or selected_ids != [r['Record ID'] for r in ranked[:5]]:
        raise ValueError('fixed five GPQA demos changed')
    demos = [(r['Question'].strip(), r['Correct Answer'].strip()) for r in ranked[:5]]
    eval_ledger = read_jsonl(args.prepared / 'gpqa_diamond_identity_ledger.jsonl')
    items, gpqa_meta, duplicates = [], [], []
    for row, ledger in zip(tables['diamond'], eval_ledger):
        rebuilt = prep.candidate_ledger(row)
        if any(ledger[k] != rebuilt[k] for k in rebuilt):
            raise ValueError('GPQA candidate lineage differs')
        by_id = {c['candidate_id']: c for c in ledger['candidates']}
        candidates = [by_id[cid] for cid in ledger['frozen_candidate_ids']]
        opts = [row[c['source_field']].strip() for c in candidates]
        item = dict(q=functions['with_demos'](row['Question'].strip(), demos), opts=opts,
                    gold=ledger['gold_sorted_index'], hyp=functions['FEWSHOT_HYP'])
        items.append(item)
        gpqa_meta.extend(dict(question_id=row['Record ID'], candidate_id=c['candidate_id'],
                             source_slot=c['source_slot'], candidate_sorted_index=i) for i, c in enumerate(candidates))
        groups = defaultdict(list)
        for c, text in zip(candidates, opts):
            groups[prep.norm(text)].append(c['candidate_id'])
        repeated = [v for v in groups.values() if len(v) > 1]
        if repeated:
            gold_text = opts[item['gold']]
            duplicates.append(dict(record_id=row['Record ID'], normalized_equivalence_groups=repeated,
                gold_candidate_id=ledger['gold_candidate_id'],
                gold_in_normalized_duplicate_group=any(ledger['gold_candidate_id'] in g for g in repeated),
                exact_gold_text_equivalent_ids=[c['candidate_id'] for c, text in zip(candidates, opts) if text == gold_text],
                candidates=[dict(**c, exact_text_sha256=prep.digest(text.encode())) for c, text in zip(candidates, opts)],
                tie_rule='first candidate in frozen gold-independent candidate order; no source-slot preference',
                auxiliary_rule='chosen stripped answer text equals stripped gold answer text'))
    if len(items) != 198 or len(gpqa_meta) != 792 or len(duplicates) != 3:
        raise ValueError('GPQA complete/duplicate count differs')
    rerank_pairs, grade_pairs, grade_gold, offsets = functions['native_pairs'](None, items)
    if len(rerank_pairs) != 792 or len(grade_pairs) != 792 or len(offsets) != 198 or sum(grade_gold) != 198:
        raise ValueError('native GPQA pair construction incomplete')
    args.output.mkdir(parents=True, exist_ok=False)
    prep.write_json(args.output / 'gpqa_duplicate_review.json', duplicates)
    print(json.dumps(dict(stage='input_identity_verified', piqa=1838, gpqa=198, demos=selected_ids,
                          duplicate_gold=[dict(record_id=r['record_id'], gold_duplicate=r['gold_in_normalized_duplicate_group']) for r in duplicates])), flush=True)
    sys.path.insert(0, str(args.code_tree))
    import torch
    from transformers import AutoTokenizer
    from kev.api import SystemOneRequest, to_record
    from kev.model import encode, SPECIAL
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA must remain uninitialized')
    native_tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)

    class CachedTokenizer:
        # Repeated strings are tokenized once by the actual native tokenizer. No estimates.
        @functools.lru_cache(maxsize=100000)
        def __call__(self, text, add_special_tokens=False):
            return native_tokenizer(text, add_special_tokens=add_special_tokens)

        def convert_tokens_to_ids(self, token):
            return native_tokenizer.convert_tokens_to_ids(token)

    tokenizer = CachedTokenizer()
    special_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in SPECIAL}
    if len(set(special_ids.values())) != 5 or any(v is None or v == native_tokenizer.unk_token_id for v in special_ids.values()):
        raise ValueError('missing delimiter vocabulary')

    def requests():
        # Long GPQA inputs are examined first so genuine intended-service overflow is visible early.
        for stage, pairs in (('rerank', rerank_pairs), ('grade', grade_pairs)):
            for meta, pair in zip(gpqa_meta, pairs):
                yield dict(dataset='gpqa_diamond', condition='gpqa_disjoint5_' + stage, **meta), functions['pointer_request']('__MODEL_ID__', pair)
        for q in questions:
            yield dict(dataset='piqa', condition='piqa_direct', question_id=q['id'], candidate_id=None), prep.direct_request(q)
        for did, descriptions in protocol['descriptions'].items():
            for permutation in protocol['permutations']:
                for q in questions:
                    for c in q['candidates']:
                        yield dict(dataset='piqa', condition='piqa_nli_' + permutation['id'] + did,
                                   question_id=q['id'], candidate_id=c['id'], description_id=did,
                                   permutation_id=permutation['id']), prep.nli_request(q, c, descriptions, permutation['order'])

    stats = defaultdict(lambda: dict(state=[], branch=[], packed=[], failures=Counter()))
    expanded_wires = {m['run_id']: hashlib.sha256() for m in models}
    request_ids, wire_identities, encoding_identities = set(), set(), set()
    permutation_tokens = defaultdict(dict)
    intended_failures, count = 0, 0
    with (args.output / 'request_token_ledger.jsonl').open('xb') as ledger_file, (args.output / 'overflow_ledger.jsonl').open('xb') as overflow_file:
        for meta, body in requests():
            wire = prep.wire(body)
            if prep.wire(json.loads(wire)) != wire:
                raise ValueError('JSON round-trip changed ordered request')
            record, semantic = to_record(SystemOneRequest.model_validate(json.loads(wire)))
            expected_keys = list(next(iter(body['questions'].values()))['criteria'])
            if len(semantic) != 1 or semantic[0]['keys'] != expected_keys:
                raise ValueError('runtime reordered semantic keys')
            for model_id, accumulator in expanded_wires.items():
                expanded = dict(body, model=model_id)
                expanded_wire = prep.wire(expanded)
                same_record, same_semantic = to_record(SystemOneRequest.model_validate(json.loads(expanded_wire)))
                if same_record != record or same_semantic != semantic:
                    raise ValueError('checkpoint identifier changed model inputs')
                accumulator.update(expanded_wire + b'\n')
            full = encode(tokenizer, record, max_state=10**9, max_branch=10**9,
                          strict=True, option_isolation=False, readout='end')
            pooled = encode(tokenizer, record, max_state=10**9, max_branch=10**9,
                            strict=True, option_isolation=False, readout='mean_span')
            if {k: v for k, v in pooled.items() if k != 'opt_spans'} != full or full['state_truncated']:
                raise ValueError('pointer/pooling input mismatch or unexpected truncation')
            state = full['seg'].count(0)
            branches = [full['seg'].count(k) for k in range(1, len(record['questions']) + 1)]
            packed = len(full['ids'])
            token_sha = prep.digest(prep.wire(full['ids']))
            failures = {}
            for profile_name, profile in PROFILES.items():
                expected_failure = profile_failures(state, branches, packed, profile)
                try:
                    strict = encode(tokenizer, record, strict=True, option_isolation=False, readout='end',
                                    max_state=profile['max_state'], max_branch=profile['max_branch'])
                    if len(strict['ids']) > profile['max_packed']:
                        raise ValueError('packed request exceeds limit')
                except ValueError as error:
                    if not expected_failure:
                        raise
                    failures[profile_name] = dict(reasons=expected_failure, native_error=str(error))
                else:
                    if expected_failure or strict != full:
                        raise ValueError('native strict encoding/length disagrees')
            request_id = prep.digest(prep.wire([meta['condition'], meta['question_id'], meta['candidate_id']]))
            if request_id in request_ids:
                raise ValueError('request source identity collapsed')
            request_ids.add(request_id)
            wire_identities.add(prep.digest(wire))
            encoding_identities.add(token_sha)
            if 'permutation_id' in meta:
                permutation_tokens[(meta['question_id'], meta['candidate_id'], meta['description_id'])][meta['permutation_id']] = token_sha
            entry = dict(**meta, request_id=request_id, wire_sha256=prep.digest(wire),
                record_sha256=prep.digest(prep.wire(record)), input_token_ids_sha256=token_sha,
                semantic_keys=semantic[0]['keys'], pooling_spans_sha256=prep.digest(prep.wire(pooled['opt_spans'])),
                state_tokens=state, branch_only_tokens=branches, state_plus_branch_tokens=[state+b for b in branches],
                packed_tokens=packed, state_truncated=False, profile_failures=failures)
            ledger_file.write(prep.wire(entry) + b'\n')
            if failures:
                overflow_file.write(prep.wire(entry) + b'\n')
            stat = stats[meta['condition']]
            stat['state'].append(state)
            stat['branch'].append(max(branches))
            stat['packed'].append(packed)
            stat['failures'].update(failures.keys())
            count += 1
            if 'bundle_8192' in failures:
                intended_failures += 1
                if intended_failures <= 10:
                    print(json.dumps(dict(stage='INTENDED_SERVICE_OVERFLOW', **entry)), flush=True)
            if count % 5000 == 0 or count == 1584:
                print(json.dumps(dict(stage='tokenizing', completed=count, intended_service_overflows=intended_failures,
                                      elapsed_s=round(time.monotonic()-t0, 1))), flush=True)
    expected_counts = {'gpqa_disjoint5_rerank': 792, 'gpqa_disjoint5_grade': 792, 'piqa_direct': 1838}
    expected_counts.update({'piqa_nli_' + p['id'] + d: 3676 for p in protocol['permutations'] for d in protocol['descriptions']})
    observed = {key: len(stat['packed']) for key, stat in stats.items()}
    if observed != expected_counts or count != 69590 or count * len(models) != protocol['requests_all_eight']:
        raise ValueError('full protocol request counts disagree')
    if len(permutation_tokens) != 1838 * 2 * 3 or any(len(v) != 6 or len(set(v.values())) != 6 for v in permutation_tokens.values()):
        raise ValueError('not all six permutations produce six distinct actual token sequences')
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA unexpectedly initialized')
    tokenizer_files = {p.name: prep.digest(p.read_bytes()) for p in args.tokenizer.iterdir()
                       if p.is_file() and p.name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json')}
    summary = dict(status='PASS' if intended_failures == 0 else 'INPUT_OVERFLOW_REQUIRES_REVIEW',
        protocol_status=protocol['status'], prepared_path=str(args.prepared.resolve()),
        prepared_receipt_sha256=prep.digest((args.prepared/'PREPARE_RECEIPT.json').read_bytes()),
        host=socket.gethostname(), uid=os.getuid(), python_executable=sys.executable,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], cuda_initialized=False,
        model_constructed=False, predictions_generated=False, tokenizer=str(args.tokenizer),
        tokenizer_class=type(native_tokenizer).__name__, tokenizer_files=tokenizer_files,
        delimiter_ids=special_ids, runtime_code_tree=str(args.code_tree), native_source_identities=SOURCE_PINS,
        native_input_builder_sha256={name:prep.digest((args.assets/name).read_bytes()) for name in ('openjev_pointer.py', 'native_eval.py')},
        lengths_include_delimiters_and_native_user_token_rewrite=True,
        profiles=PROFILES, branch_rule='each branch must satisfy state_tokens + branch_only_tokens <= max_branch',
        intended_profile='bundle_8192', intended_service_overflows=intended_failures,
        total_request_templates=count, checkpoint_expanded_request_instances=count*len(models),
        checkpoint_model_wire_stream_sha256={key:value.hexdigest() for key,value in expanded_wires.items()},
        pointer_pooling_identical_tokens=count, unique_template_wires=len(wire_identities),
        unique_token_sequences=len(encoding_identities), source_request_occurrences_retained=count,
        permutation_groups_checked=len(permutation_tokens), each_group_distinct_token_sequences=6,
        gpqa_demos=selected_ids, gpqa_evaluation_questions=198, gpqa_candidate_occurrences_per_stage=792,
        gpqa_normalized_collision_questions=len(duplicates),
        gpqa_exact_duplicate_questions=sum(len({c['exact_text_sha256'] for c in d['candidates']}) < len(d['candidates']) for d in duplicates),
        gpqa_main_demo_pool=249,
        serialized_model_source_field_names=False,
        conditions={name:dict(n=len(s['packed']), checkpoint_expanded_n=len(s['packed'])*8,
                    state_tokens=quantiles(s['state']), branch_only_tokens=quantiles(s['branch']),
                    packed_tokens=quantiles(s['packed']), profile_overflow_requests=dict(s['failures'])) for name,s in stats.items()},
        tokenizer_cache=tokenizer.__call__.cache_info()._asdict(), elapsed_s=round(time.monotonic()-t0, 3),
        remaining='Actual serving/cache isolation still requires worker/runtime qualification; tokenization alone does not certify live caching.')
    prep.write_json(args.output / 'TOKEN_PREFLIGHT_SUMMARY.json', summary)
    prep.write_json(args.output / 'TOKEN_PREFLIGHT_RECEIPT.json', dict(status=summary['status'],
        files={p.name:prep.digest(p.read_bytes()) for p in args.output.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=summary['status'], requests=count, expanded=count*8,
                          intended_service_overflows=intended_failures, output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
