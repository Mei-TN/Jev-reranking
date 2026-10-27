"""SocialIQA request wires fixed by frozen_v1/PROTOCOL.json.

Pure builders: no model call, no score, and neither builder reads gold. The
protocol file is key-sorted for hashing, so the NLI criteria are built from the
explicit `nli_order` list, never from object-key iteration.
"""
import hashlib
import json
from pathlib import Path

# SOURCE_FREEZE.json pins the remaining SocialIQA source files.
SOURCE_FREEZE_SHA256 = 'b373e057cff919ed244c336f5819f90d0ba7a95f9d6230eb7786a8970549a77d'
MODEL_PLACEHOLDER = '__MODEL_ID__'
PLACEHOLDER = '{candidate answer}'
# Question identifiers are response keys only: to_record never renders them.
DIRECT_QID, NLI_QID = 'answer', 'relation'
ENC_ORDER = ['entailment', 'neutral', 'contradiction']


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def load_frozen(frozen):
    """Frozen protocol and the complete item list, after checking every pinned file hash."""
    frozen = Path(frozen)
    if digest((frozen / 'SOURCE_FREEZE.json').read_bytes()) != SOURCE_FREEZE_SHA256:
        raise ValueError('SOURCE_FREEZE.json is not the frozen SocialIQA contract')
    freeze = json.loads((frozen / 'SOURCE_FREEZE.json').read_text(encoding='utf-8'))
    for name, expected in freeze['files'].items():
        if digest((frozen / name).read_bytes()) != expected:
            raise ValueError('frozen SocialIQA asset changed: ' + name)
    protocol = json.loads((frozen / 'PROTOCOL.json').read_text(encoding='utf-8'))
    items = [json.loads(line) for line in (frozen / 'items.jsonl').read_bytes().splitlines()]
    if protocol['status'] != 'FROZEN_BEFORE_MODEL_SCORING' or protocol['gold_on_wire'] is not False:
        raise ValueError('unexpected protocol status')
    if protocol['nli_order'] != ENC_ORDER or set(protocol['nli_criteria']) != set(ENC_ORDER):
        raise ValueError('frozen NLI labels are not the original E/N/C set and order')
    if protocol['nli_instructions'].count(PLACEHOLDER) != 1 or protocol['candidate_order'] != ['A', 'B', 'C']:
        raise ValueError('unexpected frozen request template')
    if len(items) != protocol['full_questions'] or len(items) != freeze['rows']:
        raise ValueError('frozen item count differs')
    for ordinal, item in enumerate(items):
        if item['source_ordinal'] != ordinal or item['item_id'] != f'socialiqa-v1.4-dev:{ordinal:04d}':
            raise ValueError('frozen item order differs')
        if item['premise'] != item['context'] + '\n\n' + item['question']:
            raise ValueError('premise is not context + "\\n\\n" + question')
        if [c['id'] for c in item['candidates']] != protocol['candidate_order']:
            raise ValueError('candidates are not in original A/B/C order')
    return protocol, items


def direct_request(protocol, item, model=MODEL_PLACEHOLDER):
    """One request with the three source candidates, keyed by their A/B/C identities so equal texts stay distinct."""
    return dict(model=model, state=item['premise'], questions={DIRECT_QID: dict(
        type='choice', instructions=protocol['direct_instructions'],
        criteria={c['id']: c['text'] for c in item['candidates']})})


def nli_request(protocol, item, candidate, model=MODEL_PLACEHOLDER):
    return dict(model=model, state=item['premise'], questions={NLI_QID: dict(
        type='choice', instructions=protocol['nli_instructions'].replace(PLACEHOLDER, candidate['text']),
        criteria={label: protocol['nli_criteria'][label] for label in protocol['nli_order']})})


def requests(protocol, items):
    """Every request of one model: (identity, body). Direct first, then NLI per candidate in A/B/C order."""
    for item in items:
        yield dict(interface='direct', item_id=item['item_id'], source_ordinal=item['source_ordinal'],
                   candidate_id=None), direct_request(protocol, item)
        for candidate in item['candidates']:
            yield dict(interface='nli', item_id=item['item_id'], source_ordinal=item['source_ordinal'],
                       candidate_id=candidate['id']), nli_request(protocol, item, candidate)
