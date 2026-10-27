"""Stdlib population identities; option/question order and duplicates are retained."""
import hashlib
import json
from collections import Counter

MC_COUNTS = {
    'mmlu_fewshot': (14042, 112336), 'mmlu': (14042, 112336),
    'hellaswag': (10042, 80336), 'gsm8k_mc10': (1319, 26380),
    'arc_easy': (2376, 19002), 'gsm8k_mc4': (1319, 10552),
    'winogrande': (1267, 5068), 'arc_challenge': (1172, 9374),
    'chess': (500, 4000), 'gpqa_fewshot': (198, 1584), 'gpqa': (198, 1584),
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def request_image(body):
    """Normalize only model ID; dictionaries become ordered pairs, never sorted options."""
    if not isinstance(body, dict) or set(body) != {'model', 'state', 'questions'}:
        raise ValueError('unexpected fixed text request shape')
    def ordered(value):
        if isinstance(value, dict):
            return [[k, ordered(v)] for k, v in value.items()]
        if isinstance(value, list):
            return [ordered(v) for v in value]
        return value
    return canonical([['state', body['state']], ['questions', ordered(body['questions'])]])


def request_digest(body):
    return hashlib.sha256(request_image(body)).hexdigest()


def request_token_digest(body, tokens):
    if type(tokens) is not int or tokens <= 0:
        raise ValueError('missing actual positive encoder token count')
    return hashlib.sha256(request_image(body) + b'\n' + str(tokens).encode()).hexdigest()


def counter_digest(values):
    """Sort digest keys only; each key already binds original option order and every duplicate."""
    counts = values if isinstance(values, Counter) else Counter(values)
    h = hashlib.sha256()
    for key, n in sorted(counts.items()):
        if len(key) != 64 or type(n) is not int or n <= 0:
            raise ValueError('invalid request occurrence counter')
        h.update(canonical([key, n]) + b'\n')
    return h.hexdigest()


def dataset_image(rows):
    normalized = []
    for row in rows:
        # Absolute cache paths differ after transfer; all content/revision/order fields remain.
        normalized.append({k: v for k, v in row.items() if k != 'path'})
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()
