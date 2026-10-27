"""Descriptive only: how close does any scanned training row come to a SocialIQA premise?

For every evaluation item, the longest run of consecutive tokens of its premise
that also occurs in the model-visible text of any logical training row, under
the study normalization and the scanner tokenizer. This decides nothing and
changes no mask; it bounds how far below the 13-token shingle the nearest text is.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # siblings stay importable under python -I
import audit_scanner
import build_mask
from jev_audit_adapter import canonical
from jev_renderer import sha256_file

FLOOR = 4


def probe(release, registry):
    release, registry = Path(release), Path(registry)
    token_re = build_mask.load_skill().TOKEN_RE
    items = build_mask.read_jsonl(registry / 'items.jsonl')
    premises = [token_re.findall(item['fields']['question']) for item in items]
    owners = {}
    for ordinal, tokens in enumerate(premises):
        for width in range(FLOOR, len(tokens) + 1):
            for start in range(len(tokens) - width + 1):
                owners.setdefault(tuple(tokens[start:start + width]), set()).add(ordinal)
    best = [(0, None)] * len(items)
    seen, rows = set(), 0
    for path in sorted((release / 'messages').glob('*.jsonl')):
        with path.open('rb') as stream:
            for line, raw in enumerate(stream, 1):
                rows += 1
                if raw in seen:
                    continue
                seen.add(raw)
                tokens = token_re.findall(audit_scanner.normalize(json.loads(raw)['messages'][0]['content']))
                for start in range(len(tokens) - FLOOR + 1):
                    width = FLOOR
                    if tuple(tokens[start:start + width]) not in owners:
                        continue
                    while start + width < len(tokens) and tuple(tokens[start:start + width + 1]) in owners:
                        width += 1
                    for ordinal in owners[tuple(tokens[start:start + width])]:
                        if width > best[ordinal][0]:
                            best[ordinal] = (width, dict(split=path.stem, line=line,
                                                         run=' '.join(tokens[start:start + width])))
    lengths = sorted(width for width, _ in best)
    ranked = sorted(range(len(items)), key=lambda i: (-best[i][0], i))[:10]
    return dict(
        schema='jev-e10-proximity-probe-v1', decides_nothing=True,
        definition='longest run of consecutive premise tokens also present in the user text of a logical training row; '
                   f'runs shorter than {FLOOR} tokens are recorded as 0',
        normalization='NFC + whitespace folding, case preserved; scanner TOKEN_RE',
        release_manifest_sha256=sha256_file(release / 'MANIFEST.json'),
        registry_items_sha256=sha256_file(registry / 'items.jsonl'),
        physical_rows_read=rows, logical_rows_compared=len(seen), items=len(items),
        longest_shared_run_tokens=dict(min=lengths[0], median=lengths[len(lengths) // 2], max=lengths[-1]),
        items_by_longest_shared_run=dict(sorted(Counter(lengths).items())),
        items_with_a_shared_run_of_at_least={str(k): sum(1 for n in lengths if n >= k) for k in (6, 8, 10, 13)},
        premise_tokens=dict(min=min(map(len, premises)), max=max(map(len, premises))),
        closest=[dict(item_id=items[i]['item_id'], premise_tokens=len(premises[i]), shared_run_tokens=best[i][0],
                      where=best[i][1]) for i in ranked])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'registry', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.release, args.registry)
    with args.out.open('xb') as stream:
        stream.write(canonical(result) + b'\n')
    print(json.dumps({k: result[k] for k in ('logical_rows_compared', 'longest_shared_run_tokens',
                                             'items_with_a_shared_run_of_at_least')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
