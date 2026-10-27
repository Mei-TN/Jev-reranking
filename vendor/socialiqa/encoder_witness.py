#!/usr/bin/env python3
"""Compare SocialIQA request encodings across training source trees on CPU.

For all 7,816 requests, compare token IDs, segments, positions, option indices,
decision indices, and content spans for end and mean_span readouts. Record
each source digest and the token ledger used by seal_population.py.
Request packing uses kev.api.to_record and kev.model.encode with the strict
8,192-token service limits.
"""
import argparse
import ast
import difflib
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e10_contract as c
import e10_models as models

SCHEMA = 'jev-e10-encoder-token-witness-v1'
ENCODER_NAMES = {'SPECIAL', 'MAX_STATE', 'MAX_BRANCH', '_SPECIAL_RE', 'user_tokens', 'OPT_NONE', 'OPT_DECIDE', 'encode'}
READOUTS = ('end', 'mean_span')
TOKENIZER_FILES = ('merges.txt', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json')


def load_api(path, name):
    """The tree's own kev/api.py, executed whole: it needs pydantic only."""
    spec = importlib.util.spec_from_file_location('e10_witness_api_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_encoder(path):
    """encode, user_tokens and their constants, compiled from the original text of kev/model.py without torch."""
    source = Path(path).read_text(encoding='utf-8')
    chosen, found = [], set()
    for node in ast.parse(source).body:
        names = ({node.name} if isinstance(node, ast.FunctionDef) else
                 {leaf.id for target in node.targets for leaf in (target.elts if isinstance(target, ast.Tuple) else [target])
                  if isinstance(leaf, ast.Name)} if isinstance(node, ast.Assign) else set())
        if names & ENCODER_NAMES:
            if not names <= ENCODER_NAMES or names & found:
                raise ValueError('unexpected encoder definition in ' + str(path))
            found |= names
            chosen.append(ast.get_source_segment(source, node))
    if found != ENCODER_NAMES:
        raise ValueError('encoder source lacks ' + ', '.join(sorted(ENCODER_NAMES - found)))
    text, scope = '\n\n'.join(chosen) + '\n', {'re': re}
    exec(compile(text, str(path), 'exec'), scope)
    return scope, text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frozen', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True, help='the pinned NeoHorse-1-4B base directory')
    ap.add_argument('--tree', action='append', required=True, metavar='NAME=PATH', help='e1=..., e458=..., optionally e11=...')
    ap.add_argument('--final', action='append', default=[], metavar='NAME', help='a tree that is the deployed final source')
    ap.add_argument('--output', type=Path, required=True, help='new directory')
    args = ap.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('launch with CUDA_VISIBLE_DEVICES empty for this CPU-only witness')
    if args.output.exists():
        raise ValueError('output must be a new directory')
    trees = dict(item.split('=', 1) for item in args.tree)
    if not {'e1', 'e458'} <= set(trees) or set(trees) - set(models.SOURCES) or set(args.final) - set(trees):
        raise ValueError('trees must name e1 and e458, optionally e11')
    protocol, items = c.load_frozen(args.frozen)
    loaded = {}
    for name, root in trees.items():
        root = Path(root).resolve()
        identity = {key: c.file_sha(root / key) for key in ('kev/api.py', 'kev/model.py')}
        if identity != {key: models.SOURCES[name][key] for key in identity}:
            raise ValueError('tree is not the recorded ' + name + ' source: ' + str(root))
        scope, text = load_encoder(root / 'kev/model.py')
        loaded[name] = dict(root=root, identity=identity, api=load_api(root / 'kev/api.py', name), scope=scope, text=text,
                            streams={readout: hashlib.sha256() for readout in READOUTS})
    import pydantic
    from transformers import AutoTokenizer
    native = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)

    class CachedTokenizer:  # Repeated strings are tokenized once by the actual tokenizer; nothing is estimated.
        @functools.lru_cache(maxsize=200000)
        def __call__(self, text, add_special_tokens=False):
            return native(text, add_special_tokens=add_special_tokens)

        def convert_tokens_to_ids(self, token):
            return native.convert_tokens_to_ids(token)

    tokenizer, ledger = CachedTokenizer(), []
    for ordinal, (identity, body) in enumerate(c.planned_requests(protocol, items)):
        blob = c.wire(body)
        question, = body['questions'].values()
        reference = None
        for name, tree in loaded.items():
            record, semantic = tree['api'].to_record(tree['api'].SystemOneRequest.model_validate(json.loads(blob)))
            if len(semantic) != 1 or semantic[0]['keys'] != list(question['criteria']):
                raise ValueError('runtime semantic keys differ from the requested order: ordinal ' + str(ordinal))
            encodings = {readout: tree['scope']['encode'](tokenizer, record, max_state=c.LIMIT, max_branch=c.LIMIT, strict=True,
                                                          option_isolation=False, readout=readout) for readout in READOUTS}
            end, pooled = encodings['end'], encodings['mean_span']
            if {k: v for k, v in pooled.items() if k != 'opt_spans'} != end or end['state_truncated'] or len(end['ids']) > c.LIMIT:
                raise ValueError('readout changed the packed input, or the strict 8192 service would not accept it: ' + str(ordinal))
            for readout in READOUTS:
                tree['streams'][readout].update(c.wire(encodings[readout]) + b'\n')
            observed = (record, semantic, end, pooled['opt_spans'])
            if reference is None:
                reference = observed
            elif observed != reference:
                raise ValueError('the ' + name + ' source packs a different model input: ordinal ' + str(ordinal))
        record, semantic, end, spans = reference
        ledger.append(dict(ordinal=ordinal, request_id=c.request_id(identity), wire_sha256=c.digest(blob),
                           request_image_sha256=c.digest(c.request_image(body)), semantic_keys=semantic[0]['keys'],
                           input_token_ids_sha256=c.digest(c.wire(end['ids'])), pooling_spans_sha256=c.digest(c.wire(spans)),
                           state_tokens=end['seg'].count(0), input_tokens=len(end['ids'])))
    if len(ledger) != c.REQUESTS:
        raise ValueError('witness does not cover 7816 requests')
    a, b = (loaded[name]['root'].joinpath('kev/model.py').read_text(encoding='utf-8').splitlines() for name in ('e1', 'e458'))
    changed = [line for line in difflib.unified_diff(a, b, 'e1/kev/model.py', 'e458/kev/model.py', lineterm='', n=0)
               if line[:1] in '+-' and line[:3] not in ('+++', '---')]
    added = [line for line in changed if line.startswith('+')]
    if len(changed) != 4 or len(added) != 2 or not all('last_span' in line for line in added):
        raise ValueError('kev/model.py differs between E1 and E458 by more than the two last_span lines')
    args.output.mkdir(parents=True)
    c.write_jsonl(args.output / 'encoder_token_ledger.jsonl', ledger)
    token_sha = c.token_fields_sha(ledger)
    c.write_json(args.output / 'ENCODER_WITNESS.json', dict(
        schema=SCHEMA, status='PASS', requests=len(ledger), source_freeze_sha256=c.SOURCE_FREEZE_SHA256,
        protocol_sha256=c.PROTOCOL_SHA256, items_sha256=c.ITEMS_SHA256, builder_sha256=c.file_sha(HERE / 'e10_contract.py'),
        route='SystemOneRequest.model_validate -> kev.api.to_record -> kev.model.encode(strict, max_state=max_branch=8192, '
              'option_isolation=False), exactly as runtime/serve_bundle.py calls it',
        readouts=list(READOUTS), compared='record, semantic keys, ids, seg, pos, opt, decide_idx, opt_idx, labels, opt_spans',
        trees={name: dict(path=str(tree['root']), source_sha256=tree['identity'], final_deployed=name in args.final,
                          extracted_encoder_sha256=c.digest(tree['text'].encode('utf-8')), token_ledger_sha256=token_sha,
                          encoding_stream_sha256={readout: tree['streams'][readout].hexdigest() for readout in READOUTS})
               for name, tree in loaded.items()},
        all_trees_equal_for_every_request=True, kev_model_py_difference_e1_to_e458=changed,
        whole_file_hashes_shared=False, ledger_sha256=c.file_sha(args.output / 'encoder_token_ledger.jsonl'),
        tokenizer=str(args.tokenizer), tokenizer_files={name: c.file_sha(args.tokenizer / name) for name in TOKENIZER_FILES},
        max_input_tokens=max(row['input_tokens'] for row in ledger), host=socket.gethostname(), python=sys.version.split()[0],
        transformers=__import__('transformers').__version__, tokenizers=__import__('tokenizers').__version__,
        pydantic=pydantic.VERSION, cuda_visible_devices='', model_constructed=False, predictions_generated=False))
    print(json.dumps(dict(status='PASS', requests=len(ledger), token_ledger_sha256=token_sha, output=str(args.output))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
