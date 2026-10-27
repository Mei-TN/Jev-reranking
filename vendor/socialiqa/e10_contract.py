"""Build and score fixed SocialIQA requests using the standard library.

Request builders use no gold labels. Build NLI criteria from the explicit
nli_order list: the protocol file sorts object keys for hashing.
"""
import hashlib
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
# E10_E11_FREEZE.json and, through SOURCE_FREEZE.json, every frozen SocialIQA file.
STUDY_FREEZE_SHA256 = 'd4add7e3c3bde2dc5443dff9dbc1e1af5119f31b857d4b0803b1c7a9edc42ae5'
SOURCE_FREEZE_SHA256 = 'b373e057cff919ed244c336f5819f90d0ba7a95f9d6230eb7786a8970549a77d'
PROTOCOL_SHA256 = 'c8405101179bfcabab3218b3a02994490cd28812ef7842ec113f3374049b63f5'
ITEMS_SHA256 = '1fe71e5de24780b89cc3d277cbf14e1bcf8c09007708586b4f59d162dd993194'
QUESTIONS, REQUESTS = 1954, 7816
CANDIDATES = ['A', 'B', 'C']
NLI_ORDER = ['entailment', 'neutral', 'contradiction']
DIRECT_QID, NLI_QID = 'answer', 'relation'  # response keys only; the runtime never renders them
MODEL_PLACEHOLDER, PLACEHOLDER = '__MODEL_ID__', '{candidate answer}'
INTERFACES = ('direct', 'nli')
LIMIT = 8192  # the strict service profile the token preflight qualified
# The audit's three mask states; only the first is inside the auxiliary subset.
CLEAR, DETECTED, UNRESOLVED = ('no_training_overlap_detected_in_audited_scope', 'detected_training_overlap',
                               'unresolved_training_overlap_candidate')
MASK_REASON = {CLEAR: 'included_no_training_overlap_detected_in_audited_scope',
               DETECTED: 'excluded_detected_training_overlap',
               UNRESOLVED: 'excluded_unresolved_training_overlap_candidate'}
POPULATION_SCHEMA, PIN_SCHEMA = 'jev-e10-sealed-population-v1', 'jev-e10-population-pin-v1'
SEALED = 'SEALED_TOKEN_QUALIFIED_AND_MASK_BOUND'
PIN = HERE / 'POPULATION_PIN.json'  # absent until the final audit has been read and the population sealed
POPULATION_FILES = ('requests.jsonl', 'scoring_population.jsonl')


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().split(b'\n') if line]


def write_json(path, value):
    """Single use: an earlier record is never overwritten."""
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:  # the same bytes on every platform
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl(path, rows):
    with Path(path).open('xb') as stream:
        for row in rows:
            stream.write(wire(row) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_frozen(frozen):
    """Frozen protocol and all 1954 items in source order, after checking every pinned byte."""
    frozen = Path(frozen)
    if file_sha(frozen / 'SOURCE_FREEZE.json') != SOURCE_FREEZE_SHA256:
        raise ValueError('SOURCE_FREEZE.json is not the frozen SocialIQA contract')
    freeze = read_json(frozen / 'SOURCE_FREEZE.json')
    for name, expected in freeze['files'].items():
        if file_sha(frozen / name) != expected:
            raise ValueError('frozen SocialIQA asset changed: ' + name)
    if freeze['files']['PROTOCOL.json'] != PROTOCOL_SHA256 or freeze['files']['items.jsonl'] != ITEMS_SHA256:
        raise ValueError('frozen protocol or items are not the pinned ones')
    protocol, items = read_json(frozen / 'PROTOCOL.json'), read_jsonl(frozen / 'items.jsonl')
    if (protocol['status'] != 'FROZEN_BEFORE_MODEL_SCORING' or protocol['gold_on_wire'] is not False or
            protocol['nli_order'] != NLI_ORDER or set(protocol['nli_criteria']) != set(NLI_ORDER) or
            protocol['candidate_order'] != CANDIDATES or protocol['nli_instructions'].count(PLACEHOLDER) != 1 or
            protocol['nli_score'] != 'probabilities[entailment]' or protocol['full_questions'] != QUESTIONS or
            protocol['full_requests_per_model'] != REQUESTS or len(items) != QUESTIONS or freeze['rows'] != QUESTIONS):
        raise ValueError('unexpected frozen protocol')
    for ordinal, item in enumerate(items):
        if item['source_ordinal'] != ordinal or item['item_id'] != f'socialiqa-v1.4-dev:{ordinal:04d}':
            raise ValueError('frozen item order differs')
        if item['premise'] != item['context'] + '\n\n' + item['question']:
            raise ValueError('premise is not context + two newlines + question')
        if [c['id'] for c in item['candidates']] != CANDIDATES or item['gold'] not in CANDIDATES:
            raise ValueError('candidates are not the original A/B/C occurrences')
    return protocol, items


def direct_request(protocol, item, model=MODEL_PLACEHOLDER):
    """One choice over the three source candidates, keyed by A/B/C so that equal texts stay distinct."""
    return dict(model=model, state=item['premise'], questions={DIRECT_QID: dict(
        type='choice', instructions=protocol['direct_instructions'],
        criteria={c['id']: c['text'] for c in item['candidates']})})


def nli_request(protocol, item, candidate, model=MODEL_PLACEHOLDER):
    """The original D0 relation prompt for one candidate occurrence, criteria explicitly in E/N/C order."""
    return dict(model=model, state=item['premise'], questions={NLI_QID: dict(
        type='choice', instructions=protocol['nli_instructions'].replace(PLACEHOLDER, candidate['text']),
        criteria={label: protocol['nli_criteria'][label] for label in protocol['nli_order']})})


def request_id(identity):
    return ':'.join(part for part in (identity['interface'], identity['item_id'], identity['candidate_id']) if part)


def planned_requests(protocol, items):
    """Every request of one model as (identity, body): per source item the direct request, then NLI for A, B, C."""
    for item in items:
        yield dict(interface='direct', item_id=item['item_id'], source_ordinal=item['source_ordinal'],
                   candidate_id=None), direct_request(protocol, item)
        for candidate in item['candidates']:
            yield dict(interface='nli', item_id=item['item_id'], source_ordinal=item['source_ordinal'],
                       candidate_id=candidate['id']), nli_request(protocol, item, candidate)


def request_image(body):
    """The audit's model-free request identity: only the model ID is dropped, key order is kept as ordered pairs."""
    if set(body) != {'model', 'state', 'questions'}:
        raise ValueError('unexpected fixed text request shape')

    def ordered(value):
        if isinstance(value, dict):
            return [[k, ordered(v)] for k, v in value.items()]
        return [ordered(v) for v in value] if isinstance(value, list) else value
    return wire([['state', body['state']], ['questions', ordered(body['questions'])]])


def duplicate_text_groups(item):
    """Candidate identities whose texts are exactly equal. They stay separate occurrences and earn no shared credit."""
    groups = {}
    for candidate in item['candidates']:
        groups.setdefault(candidate['text'], []).append(candidate['id'])
    return [ids for ids in groups.values() if len(ids) > 1]


def stream_sha(rows, model):
    """Ordered request stream of one served model identifier."""
    h = hashlib.sha256()
    for row in rows:
        h.update(wire(dict(row['request'], model=model)) + b'\n')
    return h.hexdigest()


def token_fields_sha(rows):
    """Order-sensitive digest of the token witness of every request, for comparing two encoders' ledgers."""
    h = hashlib.sha256()
    for row in rows:
        h.update(wire([row['ordinal'], row['wire_sha256'], row['input_token_ids_sha256'], row['pooling_spans_sha256'],
                       row['state_tokens'], row['input_tokens']]) + b'\n')
    return h.hexdigest()


def population_pin(path=None):
    """The sealed population's identity, or None while the audited data has not been sealed."""
    path = Path(path or PIN)
    if not path.is_file():
        return None
    pin = read_json(path)
    value = pin.get('population_sha256') if isinstance(pin, dict) else None
    if pin.get('schema') != PIN_SCHEMA or not isinstance(value, str) or len(value) != 64 or set(value) - set('0123456789abcdef'):
        raise ValueError('population pin is malformed')
    return pin


def load_population(root, pin):
    """Sealed population -> (manifest, request rows, scoring rows). Every member and every request wire is re-checked."""
    root = Path(root)
    if pin is None:
        raise ValueError('the E10 population is not sealed: final mask and token qualification are pending')
    if not (root / 'POPULATION.json').is_file() or file_sha(root / 'POPULATION.json') != pin['population_sha256']:
        raise ValueError('not the sealed E10 population manifest')
    population = read_json(root / 'POPULATION.json')
    identities = population['identities']
    if (population['schema'] != POPULATION_SCHEMA or population['status'] != SEALED or
            population['questions'] != QUESTIONS or population['requests_per_model'] != REQUESTS or
            identities['study_freeze_sha256'] != STUDY_FREEZE_SHA256 or
            identities['source_freeze_sha256'] != SOURCE_FREEZE_SHA256 or
            identities['protocol_sha256'] != PROTOCOL_SHA256 or identities['items_sha256'] != ITEMS_SHA256 or
            set(population['files']) != set(POPULATION_FILES)):
        raise ValueError('sealed population does not bind the frozen study, source and protocol')
    for name, expected in population['files'].items():
        if file_sha(root / name) != expected:
            raise ValueError('population member changed: ' + name)
    requests, scoring = read_jsonl(root / 'requests.jsonl'), read_jsonl(root / 'scoring_population.jsonl')
    if len(requests) != REQUESTS or len(scoring) != QUESTIONS:
        raise ValueError('sealed population is incomplete')
    seen, counts = set(), dict.fromkeys(INTERFACES, 0)
    for ordinal, row in enumerate(requests):
        body = row['request']
        (qid, question), = body['questions'].items()
        keys = CANDIDATES if row['interface'] == 'direct' else NLI_ORDER
        if (row['ordinal'] != ordinal or row['request_id'] != request_id(row) or row['request_id'] in seen or
                body['model'] != MODEL_PLACEHOLDER or qid != (DIRECT_QID if row['interface'] == 'direct' else NLI_QID) or
                list(question['criteria']) != keys or row['semantic_keys'] != keys or
                digest(wire(body)) != row['wire_sha256'] or digest(request_image(body)) != row['request_image_sha256']):
            raise ValueError('sealed request differs from its token-qualified witness: ordinal ' + str(ordinal))
        if type(row['state_tokens']) is not int or type(row['input_tokens']) is not int or \
                not 0 < row['state_tokens'] < row['input_tokens'] <= LIMIT:
            raise ValueError('sealed request length is not qualified: ordinal ' + str(ordinal))
        seen.add(row['request_id'])
        counts[row['interface']] += 1
    if counts != dict(direct=QUESTIONS, nli=3 * QUESTIONS):
        raise ValueError('sealed population is not 1954 direct and 5862 NLI requests')
    tally = dict.fromkeys(MASK_REASON, 0)
    for ordinal, row in enumerate(scoring):
        block = requests[4 * ordinal:4 * ordinal + 4]
        if (row['source_ordinal'] != ordinal or row['candidates'] != CANDIDATES or row['gold'] not in CANDIDATES or
                [r['item_id'] for r in block] != [row['item_id']] * 4 or
                [r['candidate_id'] for r in block] != [None] + CANDIDATES or
                row['mask_status'] not in MASK_REASON or row['mask_reason'] != MASK_REASON[row['mask_status']] or
                row['in_auxiliary_subset'] is not (row['mask_status'] == CLEAR)):
            raise ValueError('sealed scoring row differs from the source order or the common mask: ' + str(ordinal))
        tally[row['mask_status']] += 1
    if tally != population['mask_counts']:
        raise ValueError('sealed mask counts differ from the scoring rows')
    return population, requests, scoring


def probability_distribution(distribution, keys):
    """Raw served values by semantic key. The service's own rule: each in [0, 1], sum within an absolute 1e-5 of one
    (a float32 softmax sums to one only to about 1e-7). Nothing is normalized or reordered."""
    if not isinstance(distribution, dict) or list(distribution) != list(keys):
        raise ValueError('served semantic keys differ from the requested keys and order')
    values = list(distribution.values())
    if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 1e-5:
        raise ValueError('invalid probability distribution')
    return values


def pick(candidates, scores, gold):
    """First exact maximum in the original source order; full-precision values, no renormalization."""
    if len(scores) != len(candidates) or gold not in candidates or len(set(candidates)) != len(candidates):
        raise ValueError('candidate identity/score coverage differs')
    if any(type(s) not in (int, float) or not math.isfinite(s) for s in scores):
        raise ValueError('invalid decision score')
    maximum = max(scores)
    tied = [cid for cid, score in zip(candidates, scores) if score == maximum]
    return dict(prediction=tied[0], correct=int(tied[0] == gold), exact_tie=len(tied) > 1, maximum_candidates=tied)


def question_rows(requests, events, scoring):
    """One row per interface and source question from the complete ordered responses. Any gap raises:
    a missing response is never an incorrect answer and never a smaller denominator."""
    if len(requests) != REQUESTS or len(events) != REQUESTS or len(scoring) != QUESTIONS:
        raise ValueError('cannot score an incomplete request population')
    scores = {name: {} for name in INTERFACES}
    for expected, event in zip(requests, events):
        if event.get('error') or event['ordinal'] != expected['ordinal'] or event['request_id'] != expected['request_id']:
            raise ValueError('response occurrence does not match the planned request')
        probabilities = event['response']['probabilities']
        qid = DIRECT_QID if expected['interface'] == 'direct' else NLI_QID
        if set(probabilities) != {qid}:
            raise ValueError('response question coverage differs')
        probability_distribution(probabilities[qid], expected['semantic_keys'])
        slot = scores[expected['interface']].setdefault(expected['item_id'], dict(values={}, ordinals=[]))
        if expected['interface'] == 'direct':
            if slot['ordinals']:
                raise ValueError('duplicate direct response')
            slot['values'] = {cid: probabilities[qid][cid] for cid in CANDIDATES}
        else:
            if expected['candidate_id'] in slot['values']:
                raise ValueError('candidate source occurrence duplicated')
            slot['values'][expected['candidate_id']] = probabilities[qid]['entailment']  # by semantic key, never by position
        slot['ordinals'].append(expected['ordinal'])
    rows = []
    for interface in INTERFACES:
        for question in scoring:
            slot = scores[interface].get(question['item_id'])
            if slot is None or set(slot['values']) != set(CANDIDATES):
                raise ValueError('missing source candidate in the final score: ' + question['item_id'])
            values = [slot['values'][cid] for cid in CANDIDATES]
            rows.append(dict(interface=interface, item_id=question['item_id'], source_ordinal=question['source_ordinal'],
                             candidate_order=list(CANDIDATES), candidate_scores=values, gold=question['gold'],
                             **pick(CANDIDATES, values, question['gold']),
                             duplicate_text_groups=question['duplicate_text_groups'], call_ordinals=slot['ordinals'],
                             mask_status=question['mask_status'], in_auxiliary_subset=question['in_auxiliary_subset'],
                             mask_reason=question['mask_reason']))
    return rows


def part(rows):
    correct = sum(row['correct'] for row in rows)
    return dict(n_questions=len(rows), correct=correct, accuracy=correct / len(rows) if rows else None)


def summarize(rows):
    """Full source (primary) and the common auxiliary subset; excluded questions are reported by reason, never dropped."""
    if len(rows) != 2 * QUESTIONS:
        raise ValueError('question rows do not cover both interfaces')
    interfaces = {}
    for interface in INTERFACES:
        own = [row for row in rows if row['interface'] == interface]
        if len(own) != QUESTIONS or len({row['item_id'] for row in own}) != QUESTIONS:
            raise ValueError('fixed question denominator changed: ' + interface)
        interfaces[interface] = dict(
            full_source=part(own), auxiliary_mask=part([row for row in own if row['in_auxiliary_subset']]),
            excluded_detected=part([row for row in own if row['mask_status'] == DETECTED]),
            excluded_unresolved=part([row for row in own if row['mask_status'] == UNRESOLVED]),
            exact_tie_questions=sum(row['exact_tie'] for row in own))

    def difference(scope):
        a, b = interfaces['nli'][scope]['accuracy'], interfaces['direct'][scope]['accuracy']
        return None if a is None or b is None else a - b
    return dict(primary='full_source', interfaces=interfaces,
                nli_minus_direct_accuracy=dict(full_source=difference('full_source'), auxiliary_mask=difference('auxiliary_mask')),
                note='One model. Pooling minus pointer and the interaction need the matched pair of each setting.')
