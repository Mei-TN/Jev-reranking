"""Observe residual-head and baseline probabilities in one forward pass.

Hold the supplied request lock while calling the original model.probs path
and collecting the residual-branch intermediates.
"""
import hashlib
import itertools
import json
import math


def encoded_identity(enc):
    required = ('ids', 'pos', 'seg', 'decide_idx', 'opt_idx')
    if any(k not in enc for k in required):
        raise ValueError('native encoding identity incomplete')
    fields = {k: enc[k] for k in required}
    for k in ('opt_spans', 'option_isolation'):
        if k in enc:
            fields[k] = enc[k]
    raw = json.dumps(fields, sort_keys=True, separators=(',', ':'),
                     allow_nan=False).encode()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'fields': fields,
            'input_tokens': len(enc['ids'])}


def paired_probs(model, enc, request_lock):
    """Return original full probabilities and baseline probabilities together.

    Caller must NOT already hold the non-reentrant lock. The same lock used by
    its service is owned here through buffer installation, forward, conversion,
    equality checks and cleanup. No historical probability is synthesized.
    """
    import torch
    from torch.nn import functional as F
    if model.head_config['kind'] != 'residual_pair_mlp':
        raise ValueError('same-forward branch requires the original H4 head')
    if model.training or model.head.training or model.lm.training:
        raise ValueError('original inference eval mode required')
    if torch.is_autocast_enabled():
        raise ValueError('do not introduce autocast into original inference')
    with request_lock, torch.inference_mode():
        rows = []
        marker = object()
        prior = getattr(model.head, '_diagnostic_observer', marker)
        if prior is not marker and prior is not None:
            raise ValueError('another observer is active on this model')

        def observe(*, q, k, logits, base_logits=None, residual_logits=None):
            if base_logits is None or residual_logits is None:
                raise ValueError('actual pre-addition baseline not captured')
            if logits.dtype != base_logits.dtype or logits.device != base_logits.device:
                raise ValueError('full/base actual device or dtype differs')
            if logits.dtype != torch.float32:
                raise ValueError('original FP32 head inference required')
            # No .float(), CPU reconstruction, in-place change or RNG operation.
            rows.append({'full': logits.detach().clone(),
                         'base': base_logits.detach().clone(),
                         'residual': residual_logits.detach().clone()})

        model.head._diagnostic_observer = observe
        try:
            full = model.probs(enc)  # The original full computation/return path.
            if len(rows) != len(full) or len(rows) != len(enc['decide_idx']):
                raise ValueError('same-forward question occurrence coverage differs')
            base, captures = [], []
            for row, original in zip(rows, full):
                captured_full = F.softmax(row['full'], -1).cpu()
                if original.dtype != captured_full.dtype or not torch.equal(original, captured_full):
                    raise ValueError('current full capture differs from model.probs')
                off = F.softmax(row['base'], -1).cpu()
                if not torch.isfinite(original).all() or not torch.isfinite(off).all():
                    raise ValueError('nonfinite full/base probabilities')
                for p in (original, off):
                    if p.ndim != 1 or (p < 0).any() or (p > 1).any() or abs(float(p.sum())-1) > 1e-5:
                        raise ValueError('invalid full/base probability vector')
                base.append(off)
                captures.append({'dtype': str(row['full'].dtype),
                                 'device': str(row['full'].device),
                                 'full_logits': row['full'].cpu().tolist(),
                                 'pre_addition_base_logits': row['base'].cpu().tolist(),
                                 'residual_logits': row['residual'].cpu().tolist(),
                                 'full_capture_exact': True})
            # Values become independent ordinary lists while the request lock is held.
            return {'full': [p.tolist() for p in full],
                    'branch_zeroed': [p.tolist() for p in base],
                    'captures': captures, 'encoding': encoded_identity(enc),
                    'same_forward': True, 'model_probs_calls': 1,
                    'autocast_added': False, 'historical_full_required': False}
        finally:
            if prior is marker:
                delattr(model.head, '_diagnostic_observer')
            else:
                model.head._diagnostic_observer = prior
            rows.clear()


def paired_decision(tok, model, request, request_lock, *, request_role):
    """Adapter for a new endpoint; only typed rerank requests, never grading.

    Its caller must bind request_role to an immutable rerank-population record,
    not trust an arbitrary HTTP label. Existing service/source files are untouched.
    """
    if request_role != 'rerank':
        raise ValueError('gold grading is not branch reranking evidence')
    if set(request) != {'model', 'state', 'questions'}:
        raise ValueError('unexpected model request fields')
    from kev.api import SystemOneRequest, to_record  # Already-loaded bundle's module.
    rec, metadata = to_record(SystemOneRequest.model_validate(request))
    if any(m['type'] != 'choice' for m in metadata):
        raise ValueError('branch rerank must use native NLI Choice requests')
    enc = model.encode(tok, rec, strict=True, max_state=8192, max_branch=8192)
    if len(enc['ids']) > 8192:
        raise ValueError('packed request exceeds unchanged8192 cap')
    paired = paired_probs(model, enc, request_lock)
    if len(metadata) != len(paired['full']):
        raise ValueError('native response/question association differs')
    result = {'full': {}, 'branch_zeroed': {}, 'capture': paired}
    for i, meta in enumerate(metadata):
        keys = meta['keys']
        if set(keys) != {'entailment', 'neutral', 'contradiction'}:
            raise ValueError('rerank NLI criteria differ')
        for arm in ('full', 'branch_zeroed'):
            values = paired[arm][i]
            if len(keys) != len(values):
                raise ValueError('native candidate key coverage differs')
            result[arm][meta['id']] = dict(zip(keys, values))
    return result


def rerank_effects(candidate_ids, full, base, gold):
    """Pure scorer; gold is not passed to paired_decision or the model."""
    if len(candidate_ids) != len(set(candidate_ids)) or not candidate_ids:
        raise ValueError('native candidate IDs missing or collapsed')
    if set(full) != set(candidate_ids) or set(base) != set(candidate_ids) or gold not in candidate_ids:
        raise ValueError('candidate/gold association differs')
    if not all(math.isfinite(v) for arm in (full, base) for v in arm.values()):
        raise ValueError('nonfinite rerank values')
    def choose(arm):
        maximum = max(arm.values())
        ties = [c for c in candidate_ids if arm[c] == maximum]
        return ties[0], ties
    fc, ft = choose(full); bc, bt = choose(base)
    sign = lambda v: (v > 0) - (v < 0)
    pairs = [{'a': a, 'b': b, 'full_sign': sign(full[a]-full[b]),
              'base_sign': sign(base[a]-base[b])}
             for a, b in itertools.combinations(candidate_ids, 2)]
    return {'full_choice': fc, 'base_choice': bc, 'full_max_ties': ft,
            'base_max_ties': bt, 'choice_changed': fc != bc,
            'top_tie_set_changed': set(ft) != set(bt),
            'full_correct': int(fc == gold), 'base_correct': int(bc == gold),
            'full_uniform_tie_credit': 1/len(ft) if gold in ft else 0,
            'base_uniform_tie_credit': 1/len(bt) if gold in bt else 0,
            'unordered_pair_denominator': len(pairs), 'pairs': pairs,
            'strict_reversals': sum(p['full_sign']*p['base_sign'] == -1 for p in pairs),
            'tie_created': sum(p['full_sign'] != 0 and p['base_sign'] == 0 for p in pairs),
            'tie_broken': sum(p['full_sign'] == 0 and p['base_sign'] != 0 for p in pairs)}
