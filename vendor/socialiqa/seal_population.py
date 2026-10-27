#!/usr/bin/env python3
"""Validate and seal the SocialIQA evaluation population on CPU.

status reports readiness; seal writes the population and POPULATION_PIN.json.
Sealing requires the final TOKEN_PREFLIGHT_RECEIPT.json and MASK_SUMMARY.json
digests, rebuilt requests that match the token ledger, and encoder equivalence
recorded in ENCODER_WITNESS.json.

Exit codes: 0 sealed/ready; 2 invalid input; 3 required artifacts pending.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # Sibling modules import by name even under PYTHONSAFEPATH or -P.

import e10_contract as c
import e10_models as models

READY, PENDING, INVALID = 'DATA_READY_NOT_SEALED', 'DATA_PENDING_NOT_SEALED', 'DATA_INVALID_NOT_SEALED'
PREFLIGHT_FILES = ('TOKEN_PREFLIGHT_RECEIPT.json', 'TOKEN_PREFLIGHT_SUMMARY.json', 'request_token_ledger.jsonl',
                   'overflow_ledger.jsonl')
MASK_FILES = ('MASK_SUMMARY.json', 'common_eval_mask.jsonl')
WITNESS_SCHEMA = 'jev-e10-encoder-token-witness-v1'
PROFILE = 'bundle_8192'


def missing(preflight, mask, witness):
    """Return missing required artifacts as pending."""
    absent = [str(Path(root) / name) for root, names in ((preflight, PREFLIGHT_FILES), (mask, MASK_FILES))
              for name in names if not (Path(root) / name).is_file()]
    if Path(str(mask).rstrip('/\\') + '.incomplete').exists():
        absent.append(str(mask) + ' is still being written (.incomplete present)')
    if not Path(witness).is_file():
        absent.append(str(witness))
    return absent


def token_qualification(preflight, built, run_ids):
    """The audit's ledger must describe exactly the requests this package builds, all inside the strict 8192 service."""
    preflight = Path(preflight)
    receipt, summary = c.read_json(preflight / PREFLIGHT_FILES[0]), c.read_json(preflight / PREFLIGHT_FILES[1])
    if receipt.get('status') != 'PASS' or set(receipt['files']) != set(PREFLIGHT_FILES[1:]):
        raise ValueError('token preflight receipt is not a complete PASS')
    for name, expected in receipt['files'].items():
        if c.file_sha(preflight / name) != expected:
            raise ValueError('token preflight member changed after its receipt: ' + name)
    table = {model['run_id']: model['readout'] for model in summary['models']}
    if (summary['status'] != 'PASS' or summary['source_freeze_sha256'] != c.SOURCE_FREEZE_SHA256 or
            summary['protocol_sha256'] != c.PROTOCOL_SHA256 or summary['items_sha256'] != c.ITEMS_SHA256 or
            summary['study_freeze_sha256'] != c.STUDY_FREEZE_SHA256 or summary['intended_profile'] != PROFILE or
            summary['profiles'][PROFILE] != dict(max_state=c.LIMIT, max_branch=c.LIMIT, max_packed=c.LIMIT) or
            summary['strict'] is not True or summary['truncated_requests'] != 0 or summary['skipped_requests'] != 0 or
            summary['intended_service_overflows'] != 0 or summary['questions'] != c.QUESTIONS or
            summary['request_templates_per_model'] != c.REQUESTS or summary['predictions_generated'] is not False or
            summary['model_constructed'] is not False or summary['gold_read_by_request_builders'] is not False or
            table != {run_id: model['readout'] for run_id, model in run_ids.items()} or
            set(summary['model_wire_stream_sha256']) != set(run_ids)):
        raise ValueError('token preflight summary does not qualify this frozen study, source and service profile')
    ledger = c.read_jsonl(preflight / 'request_token_ledger.jsonl')
    if len(ledger) != c.REQUESTS:
        raise ValueError('token ledger does not hold 7816 requests')
    rows = []
    for ordinal, ((identity, body), entry) in enumerate(zip(built, ledger)):
        question, = body['questions'].values()
        state, packed, branch = entry['state_tokens'], entry['packed_tokens'], entry['branch_only_tokens']
        if (entry['ordinal'] != ordinal or {key: entry[key] for key in identity} != identity or
                entry['wire_sha256'] != c.digest(c.wire(body)) or entry['request_sha256'] != c.digest(c.request_image(body)) or
                entry['semantic_keys'] != list(question['criteria']) or entry['state_truncated'] is not False or
                PROFILE in entry['profile_failures'] or type(state) is not int or type(packed) is not int or
                len(branch) != 1 or state + branch[0] != packed or not 0 < state < packed <= c.LIMIT):
            raise ValueError('token ledger row differs from the request this package builds: ordinal ' + str(ordinal))
        rows.append(dict(ordinal=ordinal, request_id=c.request_id(identity), **identity, request=body,
                         wire_sha256=entry['wire_sha256'], request_image_sha256=entry['request_sha256'],
                         semantic_keys=entry['semantic_keys'], state_tokens=state, input_tokens=packed,
                         input_token_ids_sha256=entry['input_token_ids_sha256'],
                         pooling_spans_sha256=entry['pooling_spans_sha256']))
    streams = {run_id: c.stream_sha(rows, run_id) for run_id in run_ids}
    if streams != summary['model_wire_stream_sha256']:
        raise ValueError('model wire streams differ from the audited streams')
    if occurrence_digests(rows) != summary['occurrence_digests']:
        raise ValueError('request occurrence digests differ from the audited ones')
    return rows, streams, dict(
        receipt_sha256=c.file_sha(preflight / PREFLIGHT_FILES[0]), summary_sha256=receipt['files'][PREFLIGHT_FILES[1]],
        ledger_sha256=receipt['files']['request_token_ledger.jsonl'], qualified_with_source=summary['native_source_identities'],
        occurrence_digests=summary['occurrence_digests'],
        tokenizer_files=summary['tokenizer_files'], profile=PROFILE, strict=True)


def occurrence_digests(rows):
    """The audit's per-interface multisets of model-free requests, alone and with their packed token count."""
    result = {}
    for interface in c.INTERFACES:
        images, tokens = Counter(), Counter()
        for row in rows:
            if row['interface'] == interface:
                image = c.request_image(row['request'])
                images[c.digest(image)] += 1
                tokens[c.digest(image + b'\n' + str(row['input_tokens']).encode())] += 1
        digests = []
        for counts in (images, tokens):
            h = hashlib.sha256()
            for key, n in sorted(counts.items()):
                h.update(c.wire([key, n]) + b'\n')
            digests.append(h.hexdigest())
        result[interface] = dict(request_occurrence_sha256=digests[0], request_encoder_tokens_occurrence_sha256=digests[1])
    return result


def common_mask(mask, items):
    """One predeclared mask for every model, interface and cohort; detected and unresolved exclusions stay separate."""
    mask = Path(mask)
    summary, rows = c.read_json(mask / MASK_FILES[0]), c.read_jsonl(mask / MASK_FILES[1])
    recorded, counts = summary['files'][MASK_FILES[1]], summary['counts']
    if (summary['schema'] != 'jev-e10-common-eval-mask-v1' or (summary['benchmark'], summary['split']) != ('socialiqa', 'dev') or
            summary['main_denominator'] != c.QUESTIONS or summary['main_denominator_unchanged_by_this_mask'] is not True or
            summary['one_mask_for_all_models'] is not True or summary['training_or_evaluation_data_changed'] is not False or
            summary['identities']['source_freeze_sha256'] != c.SOURCE_FREEZE_SHA256 or
            summary['identities']['frozen_items_sha256'] != c.ITEMS_SHA256 or
            recorded != dict(sha256=c.file_sha(mask / MASK_FILES[1]), rows=c.QUESTIONS) or len(rows) != c.QUESTIONS):
        raise ValueError('mask summary does not bind the frozen source and its complete mask file')
    tally = dict.fromkeys(c.MASK_REASON, 0)
    for item, row in zip(items, rows):
        if (row['item_id'] != item['item_id'] or row['source_ordinal'] != item['source_ordinal'] or
                row['status'] not in c.MASK_REASON or row['in_auxiliary_subset'] is not (row['status'] == c.CLEAR)):
            raise ValueError('mask row differs from the frozen source order: ' + item['item_id'])
        tally[row['status']] += 1
    if tally != {c.CLEAR: counts['auxiliary_subset'], c.DETECTED: counts['excluded_detected'],
                 c.UNRESOLVED: counts['excluded_unresolved']}:
        raise ValueError('mask rows differ from the mask summary counts')
    return rows, tally, dict(summary_sha256=c.file_sha(mask / MASK_FILES[0]), mask_sha256=recorded['sha256'],
                             review=summary['review'], subset_name=summary['subset_name'],
                             not_a_contamination_free_claim=summary['not_a_contamination_free_claim'])


def encoder_witness(path, rows):
    """Both real encoders, E1 and E458/E11, must have produced the audit's token ledger for every request."""
    witness = c.read_json(path)
    trees = witness.get('trees', {})
    expected = c.token_fields_sha(rows)
    if (witness.get('schema') != WITNESS_SCHEMA or witness.get('status') != 'PASS' or witness.get('requests') != c.REQUESTS or
            witness.get('source_freeze_sha256') != c.SOURCE_FREEZE_SHA256 or not {'e1', 'e458'} <= set(trees)):
        raise ValueError('encoder witness is not a complete PASS for the E1 and E458 sources')
    for name, tree in trees.items():
        pins = models.SOURCES[name]
        if tree['source_sha256'] != {key: pins[key] for key in ('kev/api.py', 'kev/model.py')} or tree['token_ledger_sha256'] != expected:
            raise ValueError('encoder witness of the ' + name + ' source differs from the audited token ledger')
    return dict(sha256=c.file_sha(path), trees={name: dict(source_sha256=tree['source_sha256'], final_deployed=tree['final_deployed'])
                                                for name, tree in trees.items()}, token_ledger_sha256=expected)


def inspect(args):
    """-> (status, detail, material). Reads only."""
    absent = missing(args.token_preflight, args.mask, args.encoder_witness)
    if absent:
        return PENDING, dict(missing=absent), None
    if c.file_sha(args.study_freeze) != c.STUDY_FREEZE_SHA256:
        raise ValueError('study freeze is not the frozen E10/E11 decision record')
    freeze, run_ids = c.read_json(args.study_freeze), models.model_map()
    if freeze['new_full_model_scores_observed'] is not False or \
            {block: sorted(freeze[block]['models']) for block in ('e10', 'e11')} != {
                contract['freeze_block']: sorted(r for r, m in run_ids.items() if m['cohort'] == cohort)
                for cohort, contract in models.COHORTS.items()}:
        raise ValueError('model map differs from the frozen identities')
    protocol, items = c.load_frozen(args.frozen)
    rows, streams, tokens = token_qualification(args.token_preflight, list(c.planned_requests(protocol, items)), run_ids)
    mask_rows, tally, mask = common_mask(args.mask, items)
    witness = encoder_witness(args.encoder_witness, rows)
    scoring = [dict(item_id=item['item_id'], source_ordinal=item['source_ordinal'], candidates=list(c.CANDIDATES),
                    candidate_text_sha256={cand['id']: c.digest(cand['text'].encode('utf-8')) for cand in item['candidates']},
                    duplicate_text_groups=c.duplicate_text_groups(item), gold=item['gold'], mask_status=row['status'],
                    in_auxiliary_subset=row['in_auxiliary_subset'], mask_reason=c.MASK_REASON[row['status']],
                    mask_candidate_pairs=row['candidate_pairs']) for item, row in zip(items, mask_rows)]
    detail = dict(token_preflight=tokens, mask=mask, mask_counts=tally, encoder_witness=witness,
                  questions_with_duplicate_candidate_text=sum(bool(row['duplicate_text_groups']) for row in scoring))
    return READY, detail, dict(rows=rows, scoring=scoring, streams=streams)


def seal(args, detail, material):
    for flag, actual in (('--confirm-token-preflight-receipt-sha256', detail['token_preflight']['receipt_sha256']),
                         ('--confirm-mask-summary-sha256', detail['mask']['summary_sha256'])):
        if getattr(args, flag[2:].replace('-', '_')) != actual:
            raise ValueError(flag + ' must be the SHA-256 the audit reported for its final artifact')
    out, pin_path = Path(args.out).resolve(), Path(args.pin)
    if out.exists() or pin_path.exists():
        raise ValueError('a sealed population or pin already exists; a seal is never overwritten')
    out.mkdir(parents=True)
    c.write_jsonl(out / 'requests.jsonl', material['rows'])
    c.write_jsonl(out / 'scoring_population.jsonl', material['scoring'])
    population = dict(
        schema=c.POPULATION_SCHEMA, status=c.SEALED, questions=c.QUESTIONS, requests_per_model=c.REQUESTS,
        interface_requests=dict(direct=c.QUESTIONS, nli=3 * c.QUESTIONS), candidate_order=c.CANDIDATES, nli_order=c.NLI_ORDER,
        nli_score='probabilities[entailment]', tie_rule='first exact maximum in original A/B/C order',
        primary_denominator='full source, 1954 questions per interface', gold_on_wire=False,
        identities=dict(study_freeze_sha256=c.STUDY_FREEZE_SHA256, source_freeze_sha256=c.SOURCE_FREEZE_SHA256,
                        protocol_sha256=c.PROTOCOL_SHA256, items_sha256=c.ITEMS_SHA256,
                        builder_sha256=c.file_sha(HERE / 'e10_contract.py'), **{k: detail[k] for k in ('token_preflight', 'mask', 'encoder_witness')}),
        mask_counts=detail['mask_counts'], ordered_model_wire_sha256=material['streams'],
        model_free_request_stream_sha256=c.digest(b''.join(row['request_image_sha256'].encode() + b'\n' for row in material['rows'])),
        token_ledger_sha256=detail['encoder_witness']['token_ledger_sha256'],
        files={name: c.file_sha(out / name) for name in c.POPULATION_FILES})
    c.write_json(out / 'POPULATION.json', population)
    pin = dict(schema=c.PIN_SCHEMA, population_sha256=c.file_sha(out / 'POPULATION.json'), sealed_population=str(out),
               sealed_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), mask_counts=detail['mask_counts'],
               token_preflight_receipt_sha256=detail['token_preflight']['receipt_sha256'],
               mask_summary_sha256=detail['mask']['summary_sha256'], encoder_witness_sha256=detail['encoder_witness']['sha256'])
    c.write_json(pin_path, pin)
    return pin


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('status', 'seal'))
    for name in ('frozen', 'study-freeze', 'token-preflight', 'mask', 'encoder-witness'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--out', type=Path, help='new sealed population directory (seal)')
    ap.add_argument('--pin', type=Path, default=c.PIN)
    ap.add_argument('--confirm-token-preflight-receipt-sha256')
    ap.add_argument('--confirm-mask-summary-sha256')
    args = ap.parse_args(argv)
    try:
        status, detail, material = inspect(args)
        if status == READY and args.command == 'seal':
            if args.out is None:
                raise ValueError('seal needs --out')
            status, detail = c.SEALED, dict(detail, pin=seal(args, detail, material))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps(dict(status=INVALID, problem=type(exc).__name__ + ': ' + str(exc)), indent=2))
        return 2
    print(json.dumps(dict(status=status, **detail), indent=2))
    return 3 if status == PENDING else 0


if __name__ == '__main__':
    raise SystemExit(main())
