#!/usr/bin/env python3
"""Check all fixed SocialIQA requests with the tokenizer and packer on CPU.

Use bundle_8192 as the serving context profile.
"""
import argparse
from collections import Counter
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))  # siblings stay importable under python -I
import jev_renderer
import socialiqa_requests as sr

PROFILES = {
    'bundle_8192': {'max_state': 8192, 'max_branch': 8192, 'max_packed': 8192},
    'training_strict': {'max_state': 2048, 'max_branch': 2048, 'max_packed': 4096}}
INTENDED = 'bundle_8192'
# The remaining files kev.model needs to be imported unchanged (campaign/PLAN.json source_sha256).
NATIVE_PINS = {**jev_renderer.PINS,
               'kev/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
               'kev/heads.py': 'c31edc946d2616cd8857f8eac917eb5b61525c41e49cf83920df44084c287e46'}
# Same tokenizer files as the accepted fixed-inference preflight (inference/token_report_v1).
TOKENIZER_PINS = {
    'merges.txt': 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d',
    'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
    'tokenizer_config.json': '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8',
    'vocab.json': 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003'}
BASE_CONFIG_SHA256 = 'e80a48542cf4dfbd60f19d3fea5a8a6f89e1c7fa65c992fea572aeeaa906b29c'
BASE_REVISION = '56f0584bb40578a2c33b1b40a08ccd17243ad710'
READOUT_BY_SUFFIX = {'_LL_A': 'end', '_POOL_ALL': 'mean_span', '_POINTER_2EP': 'end', '_POOL_2EP': 'mean_span'}


def model_table(freeze):
    """The sixteen frozen identities with the readout each one was trained with."""
    models = []
    for block in ('e10', 'e11'):
        for name in freeze[block]['models']:
            suffix = [s for s in READOUT_BY_SUFFIX if name.endswith(s)]
            fold = [f for f in ('F2', 'F3') if f'_{f}s' in name]
            if len(suffix) != 1 or len(fold) != 1:
                raise ValueError('unrecognized frozen model identity: ' + name)
            models.append(dict(block=block, run_id=name, fold=fold[0], readout=READOUT_BY_SUFFIX[suffix[0]]))
    counts = Counter((m['block'], m['readout']) for m in models)
    if len({m['run_id'] for m in models}) != 16 or set(counts.values()) != {4} or len(counts) != 4:
        raise ValueError('expected four pointer and four pooling identities in each of E10 and E11')
    return models


def request_image(body):
    """evaluation/population.py identity: only the model ID is dropped; option order is kept as ordered pairs."""
    if set(body) != {'model', 'state', 'questions'}:
        raise ValueError('unexpected fixed text request shape')

    def ordered(value):
        if isinstance(value, dict):
            return [[k, ordered(v)] for k, v in value.items()]
        return [ordered(v) for v in value] if isinstance(value, list) else value
    return sr.wire([['state', body['state']], ['questions', ordered(body['questions'])]])


def counter_digest(counts):
    digest = hashlib.sha256()
    for key, n in sorted(counts.items()):
        digest.update(sr.wire([key, n]) + b'\n')
    return digest.hexdigest()


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
                p95=values[math.ceil(len(values) * .95) - 1], p99=values[math.ceil(len(values) * .99) - 1],
                max=values[-1], mean=sum(values) / len(values), total=sum(values))


def expected_record(protocol, item, identity):
    """What the runtime must show the model, written out from the frozen protocol alone."""
    if identity['interface'] == 'direct':
        keys = list(protocol['candidate_order'])
        instr = protocol['direct_instructions']
        options = [c['id'] + ': ' + c['text'] for c in item['candidates']]
    else:
        text = {c['id']: c['text'] for c in item['candidates']}[identity['candidate_id']]
        keys = list(protocol['nli_order'])
        instr = 'Hypothesis: "' + text + '" How does it relate to the premise?'
        options = [label + ': ' + protocol['nli_criteria'][label] for label in keys]
    return keys, {'state': item['context'] + '\n\n' + item['question'],
                  'questions': [{'instr': instr, 'options': options, 'label': 0}]}


def inspect_requests(protocol, items, models, api, encoders, tokenizer, user_tokens, ledger_file, overflow_file,
            pointer_request=None, progress=None):
    """Encode every request template once and prove each model's copy is the same model input."""
    by_id = {item['item_id']: item for item in items}
    stats = {name: dict(state=[], branch=[], packed=[], failures=Counter()) for name in ('direct', 'nli')}
    streams = {m['run_id']: hashlib.sha256() for m in models}
    identities, wires, sequences = set(), Counter(), set()
    occurrences = {name: (Counter(), Counter()) for name in stats}
    intended_failures = count = 0
    for identity, body in sr.requests(protocol, items):
        item = by_id[identity['item_id']]
        blob = sr.wire(body)
        if sr.wire(json.loads(blob)) != blob:
            raise ValueError('JSON round-trip changed the ordered request')
        record, semantic = api.to_record(api.SystemOneRequest.model_validate(json.loads(blob)))
        keys, expected = expected_record(protocol, item, identity)
        if record != expected or len(semantic) != 1 or semantic[0]['keys'] != keys:
            raise ValueError('runtime record differs from the frozen protocol: ' + repr(identity))
        if identity['interface'] == 'nli' and pointer_request is not None:
            hypothesis = {c['id']: c['text'] for c in item['candidates']}[identity['candidate_id']]
            if sr.wire(pointer_request(sr.MODEL_PLACEHOLDER, (item['premise'], hypothesis))) != blob:
                raise ValueError('NLI wire differs from the original pointer_request serialization')
        for model in models:
            expanded = sr.wire(dict(body, model=model['run_id']))
            if api.to_record(api.SystemOneRequest.model_validate(json.loads(expanded))) != (record, semantic):
                raise ValueError('model identifier changed the model input')
            streams[model['run_id']].update(expanded + b'\n')
        encodings = [encode(tokenizer, record, max_state=10 ** 9, max_branch=10 ** 9, strict=True,
                            option_isolation=False, readout=readout)
                     for encode in encoders for readout in ('end', 'mean_span')]
        full, pooled = encodings[0], encodings[1]
        if any({k: v for k, v in e.items() if k != 'opt_spans'} != full for e in encodings) or \
                any(e['opt_spans'] != pooled['opt_spans'] for e in encodings[1::2]) or full['state_truncated']:
            raise ValueError('pointer/pooling or imported/extracted encodings differ, or the state was truncated')
        isolated = encoders[0](tokenizer, record, max_state=10 ** 9, max_branch=10 ** 9, strict=True,
                               option_isolation=True, readout='end')
        if isolated['ids'] != full['ids']:
            raise ValueError('option isolation changed token identities')
        spans = pooled['opt_spans'][0]
        if [full['ids'][a:b] for a, b in spans] != [user_tokens(tokenizer, o) for o in expected['questions'][0]['options']]:
            raise ValueError('packed option spans are not the frozen options in frozen order')
        state = full['seg'].count(0)
        branches = [full['seg'].count(1)]
        packed = len(full['ids'])
        if state + sum(branches) != packed:
            raise ValueError('segment accounting does not cover the packed request')
        failures = {}
        for name, profile in PROFILES.items():
            expected_failure = profile_failures(state, branches, packed, profile)
            try:
                strict = encoders[0](tokenizer, record, strict=True, option_isolation=False, readout='end',
                                     max_state=profile['max_state'], max_branch=profile['max_branch'])
                if len(strict['ids']) > profile['max_packed']:
                    raise ValueError('packed request exceeds limit')
            except ValueError as error:
                if not expected_failure:
                    raise
                failures[name] = dict(reasons=expected_failure, native_error=str(error))
            else:
                if expected_failure or strict != full:
                    raise ValueError('native strict encoding and counted lengths disagree')
        key = (identity['interface'], identity['item_id'], identity['candidate_id'])
        if key in identities:
            raise ValueError('request source identity collapsed')
        identities.add(key)
        image = request_image(body)
        token_sha = sr.digest(sr.wire(full['ids']))
        wires[sr.digest(blob)] += 1
        sequences.add(token_sha)
        occurrences[identity['interface']][0][sr.digest(image)] += 1
        occurrences[identity['interface']][1][sr.digest(image + b'\n' + str(packed).encode())] += 1
        entry = dict(ordinal=count, **identity, wire_sha256=sr.digest(blob), request_sha256=sr.digest(image),
                     record_sha256=sr.digest(sr.wire(record)), input_token_ids_sha256=token_sha,
                     pooling_spans_sha256=sr.digest(sr.wire(pooled['opt_spans'])), semantic_keys=keys,
                     state_tokens=state, branch_only_tokens=branches,
                     state_plus_branch_tokens=[state + b for b in branches], packed_tokens=packed,
                     state_truncated=False, profile_failures=failures)
        ledger_file.write(sr.wire(entry) + b'\n')
        if failures:
            overflow_file.write(sr.wire(entry) + b'\n')
        stat = stats[identity['interface']]
        stat['state'].append(state)
        stat['branch'].append(branches[0])
        stat['packed'].append(packed)
        stat['failures'].update(failures.keys())
        intended_failures += INTENDED in failures
        count += 1
        if progress and count % 2000 == 0:
            progress(count, intended_failures)
    return dict(count=count, intended_failures=intended_failures, stats=stats, unique_wires=len(wires),
                repeated_wire_occurrences=sum(n - 1 for n in wires.values()), unique_token_sequences=len(sequences),
                streams={k: v.hexdigest() for k, v in streams.items()},
                occurrence_digests={name: dict(request_occurrence_sha256=counter_digest(pair[0]),
                                               request_encoder_tokens_occurrence_sha256=counter_digest(pair[1]))
                                    for name, pair in occurrences.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen', type=Path, required=True)
    ap.add_argument('--study-freeze', type=Path, required=True, help='E10_E11_FREEZE.json (read only)')
    ap.add_argument('--d0-source', type=Path, required=True, help='original openjev_pointer.py')
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
    protocol, items = sr.load_frozen(args.frozen)
    freeze = json.loads(args.study_freeze.read_text(encoding='utf-8'))
    models = model_table(freeze)
    if freeze['new_full_model_scores_observed'] is not False or freeze['e10']['source'] != 'SocialIQA':
        raise ValueError('unexpected parent study freeze')
    if sr.digest(args.d0_source.read_bytes()) != protocol['d0_source_sha256']:
        raise ValueError('original pointer request source differs')
    original, _ = jev_renderer.extract(args.d0_source, ('CRITERIA', 'pointer_request'), {})
    if sr.wire(original['CRITERIA']) != sr.wire({k: protocol['nli_criteria'][k] for k in protocol['nli_order']}):
        raise ValueError('frozen D0 strings or order differ from the original source')
    for relative, expected in NATIVE_PINS.items():
        if jev_renderer.sha256_file(args.code_tree / relative) != expected:
            raise ValueError('native runtime source differs: ' + relative)
    tokenizer_files = {name: jev_renderer.sha256_file(args.tokenizer / name) for name in TOKENIZER_PINS}
    if tokenizer_files != TOKENIZER_PINS or jev_renderer.sha256_file(args.tokenizer / 'config.json') != BASE_CONFIG_SHA256:
        raise ValueError('tokenizer directory is not the pinned NeoHorse-1-4B base')
    print(json.dumps(dict(stage='input_identity_verified', questions=len(items), models=len(models))), flush=True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.code_tree))
    import torch
    from transformers import AutoTokenizer
    import kev.api as api
    import kev.model as native_model
    for module in (api, native_model):
        if not Path(module.__file__).resolve().is_relative_to(args.code_tree.resolve()):
            raise ValueError('a different kev runtime was imported')
    extracted, extracted_identity = jev_renderer.load_encoder(args.code_tree)
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA must remain uninitialized')
    if __import__('pydantic').VERSION != jev_renderer.PYDANTIC_VERSION:
        raise RuntimeError('request validation library is not the training environment version')
    native_tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)

    class CachedTokenizer:
        # Repeated strings are tokenized once by the actual native tokenizer. No estimates.
        @functools.lru_cache(maxsize=100000)
        def __call__(self, text, add_special_tokens=False):
            return native_tokenizer(text, add_special_tokens=add_special_tokens)

        def convert_tokens_to_ids(self, token):
            return native_tokenizer.convert_tokens_to_ids(token)

    tokenizer = CachedTokenizer()
    special_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in native_model.SPECIAL}
    if len(set(special_ids.values())) != 5 or any(v is None or v == native_tokenizer.unk_token_id for v in special_ids.values()):
        raise ValueError('missing delimiter vocabulary')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'request_token_ledger.jsonl').open('xb') as ledger_file, \
            (args.output / 'overflow_ledger.jsonl').open('xb') as overflow_file:
        result = inspect_requests(protocol, items, models, api, [native_model.encode, extracted['encode']], tokenizer,
                         native_model.user_tokens, ledger_file, overflow_file, original['pointer_request'],
                         lambda done, failed: print(json.dumps(dict(
                             stage='tokenizing', completed=done, intended_service_overflows=failed,
                             elapsed_s=round(time.monotonic() - t0, 1))), flush=True))
    observed = {name: len(stat['packed']) for name, stat in result['stats'].items()}
    expected_counts = {'direct': len(items), 'nli': len(items) * protocol['candidates_per_question']}
    if observed != expected_counts or result['count'] != protocol['full_requests_per_model'] or \
            result['count'] * 8 != protocol['E10_requests'] or result['count'] * 8 != protocol['E11_requests']:
        raise ValueError('full protocol request counts disagree')
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA unexpectedly initialized')
    status = 'PASS' if result['intended_failures'] == 0 else 'INPUT_OVERFLOW_REQUIRES_REVIEW'
    summary = dict(
        status=status, protocol_status=protocol['status'], frozen_path=str(args.frozen.resolve()),
        source_freeze_sha256=jev_renderer.sha256_file(args.frozen / 'SOURCE_FREEZE.json'),
        protocol_sha256=jev_renderer.sha256_file(args.frozen / 'PROTOCOL.json'),
        items_sha256=jev_renderer.sha256_file(args.frozen / 'items.jsonl'),
        study_freeze_sha256=jev_renderer.sha256_file(args.study_freeze),
        host=socket.gethostname(), uid=os.getuid(), python_executable=sys.executable,
        python_version=sys.version.split()[0], torch=torch.__version__,
        transformers=__import__('transformers').__version__, tokenizers=__import__('tokenizers').__version__,
        pydantic=__import__('pydantic').VERSION,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], cuda_initialized=False,
        model_constructed=False, predictions_generated=False, gold_read_by_request_builders=False,
        tokenizer=str(args.tokenizer), tokenizer_class=type(native_tokenizer).__name__,
        tokenizer_files=tokenizer_files, base_config_sha256=BASE_CONFIG_SHA256, base_revision_per_campaign_plan=BASE_REVISION,
        delimiter_ids=special_ids, runtime_code_tree=str(args.code_tree.resolve()),
        native_source_identities=NATIVE_PINS, extracted_encoder=extracted_identity,
        encoders_compared='imported kev.model.encode and the pinned-source extraction, end and mean_span readouts',
        d0_source_sha256=protocol['d0_source_sha256'],
        lengths_include_delimiters_and_native_user_token_rewrite=True, profiles=PROFILES,
        branch_rule='each branch must satisfy state_tokens + branch_only_tokens <= max_branch',
        intended_profile=INTENDED, strict=True, truncated_requests=0, skipped_requests=0,
        intended_service_overflows=result['intended_failures'],
        questions=len(items), candidates_per_question=protocol['candidates_per_question'],
        request_templates_per_model=result['count'],
        requests_E10_eight_models=result['count'] * 8, requests_E11_eight_models=result['count'] * 8,
        models=models, model_wire_stream_order='source item order; per item direct, then NLI for A, B, C',
        model_wire_stream_sha256=result['streams'],
        direct=dict(question_id=sr.DIRECT_QID, instructions=protocol['direct_instructions'],
                    criteria_keys=protocol['candidate_order'],
                    rendered_option='"<A|B|C>: <candidate text>" in source A/B/C order'),
        nli=dict(question_id=sr.NLI_QID, criteria_keys=protocol['nli_order'],
                 hypothesis='exact candidate text, no prefix', serialization='original pointer_request, wire-equal'),
        question_ids_are_not_model_inputs=True,
        unique_template_wires=result['unique_wires'], repeated_template_wire_occurrences=result['repeated_wire_occurrences'],
        duplicate_candidate_items=protocol['duplicate_candidate_items'],
        unique_token_sequences=result['unique_token_sequences'],
        pointer_pooling_identical_tokens=result['count'], option_isolation_independent_token_ids=result['count'],
        occurrence_digests=result['occurrence_digests'],
        conditions={name: dict(n=len(s['packed']), state_tokens=quantiles(s['state']),
                               branch_only_tokens=quantiles(s['branch']), packed_tokens=quantiles(s['packed']),
                               profile_overflow_requests=dict(s['failures']))
                    for name, s in result['stats'].items()},
        tokenizer_cache=tokenizer.__call__.cache_info()._asdict(), elapsed_s=round(time.monotonic() - t0, 3),
        remaining='Token qualification only. The scoring worker, served bundles and live cache isolation are not '
                  'qualified here; a worker must reproduce these wires or be preflighted again.')
    (args.output / 'TOKEN_PREFLIGHT_SUMMARY.json').write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    (args.output / 'TOKEN_PREFLIGHT_RECEIPT.json').write_text(json.dumps(dict(
        status=status, files={p.name: jev_renderer.sha256_file(p) for p in sorted(args.output.iterdir()) if p.is_file()}),
        indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=status, requests=result['count'], e10=result['count'] * 8, e11=result['count'] * 8,
                          intended_service_overflows=result['intended_failures'], output=str(args.output))), flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
