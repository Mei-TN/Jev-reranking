"""C14/C15 conservative formatting whitelist, independent of the trainer.

The only payload edits are instruction boundary blank lines and the display of
an existing Choice key/description delimiter. Exact examples for "a: Desc":
brackets produce "[a] Desc" (the colon is removed); newline produces "a:" plus
a newline plus "Desc"; both produce "[a]" plus a newline plus "Desc". Candidate
operations compose marker/separator styles and touch the same delimiter; they
are not edits at disjoint locations. Candidate semantic IDs, order, descriptions,
state and supervision are retained. A caller-supplied strict
context validator is mandatory when enabled. Original candidate augmentation
and its RNG are outside this module; callers must pass an independent RNG.
"""
import copy
from dataclasses import dataclass
import math
import re


WHITELIST_VERSION = 'jev-pure-format-v1'
OPERATIONS = ('instruction_blank_lines', 'choice_key_brackets', 'choice_key_line_separator')
_SAFE_KEY = re.compile(r'[A-Za-z0-9_-]+\Z')


@dataclass(frozen=True)
class SurfaceView:
    record: dict
    coverage: dict


def _choice_parts(question):
    if question.get('qtype') != 'choice':
        return None
    keys, options = question.get('keys'), question.get('options')
    if (not isinstance(keys, list) or not keys
            or any(not isinstance(k, str) or not _SAFE_KEY.fullmatch(k) for k in keys)
            or len(set(keys)) != len(keys)
            or not isinstance(options, list) or len(keys) != len(options)):
        return None
    parts = []
    for key, text in zip(keys, options):
        if not isinstance(text, str):
            return None
        if text == key:
            parts.append((key, None))
        elif text.startswith(key + ': '):
            parts.append((key, text[len(key) + 2:]))
        else:
            return None
    return parts


def _validate(record):
    if (not isinstance(record, dict) or not isinstance(record.get('state'), str)
            or not isinstance(record.get('questions'), list) or not record['questions']):
        raise ValueError('Surface operations require a materialized record')
    for q in record['questions']:
        if (not isinstance(q, dict) or not isinstance(q.get('instr'), str)
                or not isinstance(q.get('options'), list) or not q['options']
                or any(not isinstance(x, str) for x in q['options'])):
            raise ValueError('Materialized instruction and candidate text required')


def apply_surface(record, rng, *, probability, op_count, preserve_candidates, validate_record=None):
    """Sample 1 or 2 distinct eligible formatting operations, with no retries.

    A record with fewer eligible operators uses the available count and reports
    the shortfall. Candidate operators never apply to preserve_candidates
    records. They skip Noul, Score, noncanonical and unsafe Choice key displays.
    On any strict-context rejection the entire proposal is discarded; no
    truncation or replacement sample is permitted. The p=0 path returns the
    original object without reading it, drawing RNG or calling the validator.
    """
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError('Surface probability must be in [0,1]')
    if type(op_count) is not int or op_count not in (1, 2):
        raise ValueError('Surface op_count must be 1 or 2')
    if type(preserve_candidates) is not bool:
        raise ValueError('preserve_candidates must be explicit bool')
    coverage = {'whitelist_version':WHITELIST_VERSION, 'requested_op_count':op_count,
                'preserve_candidates':preserve_candidates, 'eligible_ops':[], 'sampled_ops':[],
                'applied_ops':[], 'status':'disabled', 'rejection':None,
                'changed_instruction_questions':0, 'changed_candidate_questions':0,
                'operator_shortfall':None,
                'truncated':False}
    if probability == 0:
        return SurfaceView(record, coverage)
    _validate(record)
    if validate_record is None:
        raise ValueError('Enabled surface augmentation requires a strict context validator')
    if validate_record(copy.deepcopy(record)) is False:
        raise ValueError('Original record failed strict context validation')
    parts = [_choice_parts(q) for q in record['questions']]
    eligible = [OPERATIONS[0]]
    if not preserve_candidates:
        if any(p is not None for p in parts):
            eligible.append(OPERATIONS[1])
        if any(p is not None and any(desc is not None for _, desc in p) for p in parts):
            eligible.append(OPERATIONS[2])
    coverage['eligible_ops'] = eligible
    coverage['operator_shortfall'] = max(0, op_count - len(eligible))
    if rng.random() >= probability:
        coverage['status'] = 'not_sampled'
        return SurfaceView(record, coverage)
    selected = rng.sample(eligible, min(op_count, len(eligible)))
    coverage['sampled_ops'] = selected
    proposed = copy.deepcopy(record)
    for q, part in zip(proposed['questions'], parts):
        if OPERATIONS[0] in selected:
            q['instr'] = '\n' + q['instr'] + '\n'
            coverage['changed_instruction_questions'] += 1
        if part is not None and not preserve_candidates and any(op in selected for op in OPERATIONS[1:]):
            options = []
            for key, desc in part:
                marker = '[' + key + ']' if OPERATIONS[1] in selected else key
                separator = ('\n' if OPERATIONS[2] in selected else ' ') if OPERATIONS[1] in selected else (':\n' if OPERATIONS[2] in selected else ': ')
                options.append(marker if desc is None else marker + separator + desc)
            if options != q['options']:
                q['options'] = options
                coverage['changed_candidate_questions'] += 1
    try:
        if validate_record(copy.deepcopy(proposed)) is False:
            raise ValueError('Proposed record failed strict context validation')
    except ValueError as error:
        coverage.update(status='context_rejected', rejection=str(error),
                        changed_instruction_questions=0, changed_candidate_questions=0)
        return SurfaceView(record, coverage)
    coverage['applied_ops'] = selected
    coverage['status'] = 'applied' if len(selected) == op_count else 'applied_with_operator_shortfall'
    return SurfaceView(proposed, coverage)
