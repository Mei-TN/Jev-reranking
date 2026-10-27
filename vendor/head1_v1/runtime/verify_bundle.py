"""One offline reload verification, retaining the existing 1e-5/argmax criterion."""
import argparse
import json
import math
from pathlib import Path
from .bundle_contract import digest, validate_manifest
from .load_bundle import load_bundle, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    meta = validate_manifest(args.bundle, require_validation=False)
    examples = json.loads((args.bundle / 'verification_examples.json').read_text())
    if not examples: raise ValueError('empty verification set')
    tok, model = load_bundle(args.bundle, args.device, meta['head_config'], require_validation=False)
    max_delta = 0.0
    for example in examples:
        new = predict(tok, model, example['request'])
        if set(new['answers']) != set(example['expected']['answers']):
            raise ValueError('question coverage changed on reload')
        for qid, old in example['expected']['answers'].items():
            actual = new['answers'][qid]
            if set(actual['probabilities']) != set(old['probabilities']):
                raise ValueError('option coverage changed on reload')
            # Every type's distribution argmax, including noul and score.
            if max(old['probabilities'], key=old['probabilities'].get) != max(actual['probabilities'], key=actual['probabilities'].get):
                raise ValueError('argmax changed on reload: ' + str(example['id']))
            for key, value in old['probabilities'].items():
                delta = abs(actual['probabilities'][key] - value)
                if not math.isfinite(delta) or delta > 1e-5:
                    raise ValueError(f'probability reload mismatch: {example["id"]}/{qid}/{key} delta={delta}')
                max_delta = max(max_delta, delta)
    report = dict(status='PASS', requests=len(examples), max_probability_delta=max_delta,
                  argmax_equal=True, offline_reload=True, head_config=meta['head_config'],
                  manifest_sha256=digest(args.bundle / 'model_manifest.json'),
                  scope='merged BF16 export/reload probabilities; not CUDA bitwise training parity')
    # Only the exporter publishes validation_report after also checking that
    # its input checkpoint and source did not change during the full export.
    with (args.bundle / 'reload_verification.json').open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
