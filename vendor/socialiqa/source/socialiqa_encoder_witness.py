#!/usr/bin/env python3
"""Compare SocialIQA encodings from another source tree with the preflight ledger.

Encode all 7,816 requests with end and mean_span readouts using the same
tokenizer. Compare full encodings and token hashes against the pinned encoder.
"""
import argparse
import functools
import json
import os
from pathlib import Path
import re
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # siblings stay importable under python -I
import jev_renderer
import socialiqa_requests as sr
import socialiqa_token_preflight as preflight

LIMIT = preflight.PROFILES[preflight.INTENDED]


def load_other(label, path):
    functions, text_sha = jev_renderer.extract(path, jev_renderer.ENCODER_NAMES, {'re': re})
    return dict(label=label, path=str(path), sha256=jev_renderer.sha256_file(path), extracted_encoder_sha256=text_sha,
                same_file_as_preflight_pin=jev_renderer.sha256_file(path) == jev_renderer.PINS['kev/model.py']), functions


def compare(protocol, items, api, tokenizer, reference, others, ledger=None):
    """Pack every request with the pinned encoder and with each other encoder; any difference is counted."""
    counts = {label: 0 for label in others}
    ledger_mismatches = requests = 0
    rows = iter(ledger) if ledger is not None else None
    for identity, body in sr.requests(protocol, items):
        blob = sr.wire(body)
        record, _ = api.to_record(api.SystemOneRequest.model_validate(json.loads(blob)))

        def pack(functions):
            packed = [functions['encode'](tokenizer, record, max_state=10 ** 9, max_branch=10 ** 9, strict=True,
                                          option_isolation=False, readout=readout) for readout in ('end', 'mean_span')]
            strict = functions['encode'](tokenizer, record, max_state=LIMIT['max_state'], max_branch=LIMIT['max_branch'],
                                         strict=True, option_isolation=False, readout='mean_span')
            return packed + [strict, [functions['user_tokens'](tokenizer, o) for o in record['questions'][0]['options']]]
        expected = pack(reference)
        if rows is not None:
            entry = next(rows, None)   # a ledger that ends early is a row that was not reproduced
            ledger_mismatches += entry is None or (
                entry['interface'], entry['item_id'], entry['candidate_id'], entry['wire_sha256'],
                entry['input_token_ids_sha256'], entry['pooling_spans_sha256'], entry['packed_tokens']) != (
                identity['interface'], identity['item_id'], identity['candidate_id'], sr.digest(blob),
                sr.digest(sr.wire(expected[0]['ids'])), sr.digest(sr.wire(expected[1]['opt_spans'])), len(expected[0]['ids']))
        for label, functions in others.items():
            counts[label] += pack(functions) != expected or functions['SPECIAL'] != reference['SPECIAL']
        requests += 1
    if rows is not None and next(rows, None) is not None:
        ledger_mismatches += 1
    return dict(requests=requests, encodings_compared_per_source=requests * 3, differing_requests=counts,
                ledger_rows_not_reproduced=ledger_mismatches if ledger is not None else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen', type=Path, required=True)
    ap.add_argument('--code-tree', type=Path, required=True, help='pinned E1 campaign tree used by the preflight')
    ap.add_argument('--preflight', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--model-source', action='append', required=True, metavar='LABEL=PATH')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if socket.gethostname() != os.environ.get('JEV_EVALUATION_HOST', socket.gethostname()) or os.getuid() != int(os.environ.get('JEV_EVALUATION_UID', str(os.getuid()))):
        raise RuntimeError('witness is restricted to WIN3 user')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('launch with CUDA_VISIBLE_DEVICES empty for this CPU-only witness')
    protocol, items = sr.load_frozen(args.frozen)
    receipt = json.loads((args.preflight / 'TOKEN_PREFLIGHT_RECEIPT.json').read_text(encoding='utf-8'))
    for name, expected in receipt['files'].items():
        if jev_renderer.sha256_file(args.preflight / name) != expected:
            raise ValueError('preflight output changed: ' + name)
    if receipt['status'] != 'PASS':
        raise ValueError('preflight did not pass')
    with (args.preflight / 'request_token_ledger.jsonl').open('rb') as stream:
        ledger = [json.loads(line) for line in stream]
    tokenizer_files = {name: jev_renderer.sha256_file(args.tokenizer / name) for name in preflight.TOKENIZER_PINS}
    if tokenizer_files != preflight.TOKENIZER_PINS:
        raise ValueError('tokenizer directory is not the pinned NeoHorse-1-4B base')
    api = jev_renderer.load_api(args.code_tree)
    reference, reference_identity = jev_renderer.load_encoder(args.code_tree)
    sources, others = [], {}
    for spec in args.model_source:
        label, _, path = spec.partition('=')
        identity, others[label] = load_other(label, Path(path))
        sources.append(identity)
    if len(others) != len(args.model_source):
        raise ValueError('model-source labels must be unique')
    from transformers import AutoTokenizer
    native_tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)

    class CachedTokenizer:
        @functools.lru_cache(maxsize=100000)
        def __call__(self, text, add_special_tokens=False):
            return native_tokenizer(text, add_special_tokens=add_special_tokens)

        def convert_tokens_to_ids(self, token):
            return native_tokenizer.convert_tokens_to_ids(token)

    result = compare(protocol, items, api, CachedTokenizer(), reference, others, ledger)
    passed = result['requests'] == protocol['full_requests_per_model'] == len(ledger) and \
        not any(result['differing_requests'].values()) and result['ledger_rows_not_reproduced'] == 0
    summary = dict(
        schema='jev-e10-encoder-witness-v1', status='PASS' if passed else 'FAIL', host=socket.gethostname(),
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], model_constructed=False, predictions_generated=False,
        readouts=['end', 'mean_span'], strict_limits=LIMIT, tokenizer=str(args.tokenizer), tokenizer_files=tokenizer_files,
        preflight_receipt_sha256=jev_renderer.sha256_file(args.preflight / 'TOKEN_PREFLIGHT_RECEIPT.json'),
        preflight_pinned_encoder=reference_identity, compared_sources=sources, **result,
        meaning='a bundle whose kev/model.py has one of the compared SHA-256 values packs every SocialIQA request into '
                'the same tokens, segments, positions and option spans as the preflight ledger records')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=summary['status'], requests=result['requests'],
                          differing_requests=result['differing_requests'],
                          ledger_rows_not_reproduced=result['ledger_rows_not_reproduced'])), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
