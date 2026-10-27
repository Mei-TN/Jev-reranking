#!/usr/bin/env python3
"""Freeze validated inference inputs and retain their source digests."""
import argparse
import json
from pathlib import Path
import shutil
import prepare_inputs as prep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('new fixed output directory required')
    previous = json.loads((args.source/'PREPARE_RECEIPT.json').read_text(encoding='utf-8'))
    for name, expected in previous['files'].items():
        if prep.digest((args.source/name).read_bytes()) != expected:
            raise ValueError('proposal identity changed: '+name)
    args.output.mkdir(parents=True, exist_ok=False)
    for path in args.source.iterdir():
        if path.name not in ('PREPARE_RECEIPT.json', 'proposed_protocol.json'):
            shutil.copy2(path, args.output/path.name)
    protocol = json.loads((args.source/'proposed_protocol.json').read_text(encoding='utf-8'))
    protocol.update(status='FIXED_PREINFERENCE_PENDING_TOKEN_AND_RUNTIME_CHECKS',
        version='inference-block-fixed-v1',
        review=dict(url='https://chatgpt.com/c/6ace79b9-9928-83ee-ab5f-0c06765ce4eb',
            response='second complete reply, Worked for 5m21s; root confirmed Response complete',
            adoption_record='revision_20261013/EXPERIMENT_DECISIONS.md, second inference block',
            decision='conditional GO; no additional legacy fewshot arm'),
        decision_head_only=True, preserve_full_precision_scores=True,
        semantic_mapping='probabilities keyed by label; native NLI vector is contradiction/entailment/neutral',
        e9=dict(descriptions='D1/D2 are description perturbations, not asserted strictly equivalent paraphrases',
            reporting='report all six orders separately within each description; paired difference in every condition',
            worst_paired_difference='min_t (H5_t - H0_t); never min_t(H5_t) - min_t(H0_t)',
            cache_identity='checkpoint identity + full ordered request input + description version + fixed protocol',
            runtime_qualification='six permutations must remain distinct actual token sequences and cache identities'),
        e12=dict(comparison='complete protocol revision, not a pure causal demonstration-pool effect',
            alias_fields_on_both_sides=list(prep.QUESTION_FIELDS),
            retain_all_candidate_source_slots=True, evaluation_denominator=198,
            primary='strict source-slot rerank accuracy from argmax entailment over frozen candidate order',
            tie_rule='first candidate in frozen gold-independent candidate order; no source-slot preference',
            auxiliary='text-equivalent accuracy when selected stripped answer text equals stripped gold answer text; separately list all 3 duplicate-text questions',
            wire='source field names Correct Answer/Incorrect Answer and source slots stay in internal ledger only',
            reference_gold='only native grade premise includes the target reference answer; rerank never includes it'))
    prep.write_json(args.output/'proposed_protocol.json', protocol)
    prep.write_json(args.output/'PREPARE_RECEIPT.json', dict(previous,
        status=protocol['status'], source_proposal=str(args.source.resolve()),
        source_receipt_sha256=prep.digest((args.source/'PREPARE_RECEIPT.json').read_bytes()),
        scientific_inputs_changed=False,
        files={p.name:prep.digest(p.read_bytes()) for p in args.output.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=protocol['status'], output=str(args.output.resolve()), requests_all_eight=protocol['requests_all_eight'])))


if __name__ == '__main__':
    main()
