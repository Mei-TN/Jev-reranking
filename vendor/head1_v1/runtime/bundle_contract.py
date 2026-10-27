"""Small stdlib-only artifact/client contract; model factory remains authoritative."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

SCHEMA = 'jev-head-ablation-bundle-v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def config_identity(config):
    """Accept only campaign variants, or the exact legacy H0 without RNG anchor."""
    if not isinstance(config, dict):
        raise ValueError('head_config must be an object')
    c = dict(config)
    required = {'version', 'kind', 'dim', 'readout'}
    extra = {'rng_anchor_dim'}
    if c.get('kind') == 'cosine_pointer': extra.add('cosine_scale')
    if c.get('kind') == 'residual_pair_mlp': extra.add('mlp_hidden')
    if set(c) - required - extra or not required <= set(c):
        raise ValueError('unexpected or missing head_config fields')
    if type(c['version']) is not int or c['version'] != 1:
        raise ValueError('unsupported head_config version')
    if type(c['dim']) is not int or c['dim'] not in (64, 256):
        raise ValueError('unsupported head dimension')
    allowed = {('pointer', 256, 'end'), ('pointer', 64, 'end'),
               ('pointer', 256, 'mean_span'), ('tanh_pointer', 256, 'end'),
               ('cosine_pointer', 256, 'end'), ('residual_pair_mlp', 256, 'end')}
    if (c['kind'], c['dim'], c['readout']) not in allowed:
        raise ValueError('not one of the six fixed head variants')
    if 'rng_anchor_dim' in c:
        if type(c['rng_anchor_dim']) is not int or c['rng_anchor_dim'] != 256:
            raise ValueError('campaign RNG anchor must be 256')
    elif (c['kind'], c['dim'], c['readout']) != ('pointer', 256, 'end'):
        raise ValueError('nonlegacy campaign variant requires RNG anchor')
    if c['kind'] == 'cosine_pointer':
        scale = c.get('cosine_scale')
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale != 16:
            raise ValueError('cosine scale must be fixed at 16')
        c['cosine_scale'] = 16.0
    if c['kind'] == 'residual_pair_mlp':
        if type(c.get('mlp_hidden')) is not int or c['mlp_hidden'] != 128:
            raise ValueError('residual MLP hidden dimension must be 128')
    return c


def read_config(path):
    return config_identity(json.loads(Path(path).read_text()))


def safe_member(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or not rel.parts or '..' in rel.parts or '\\' in name:
        raise ValueError('unsafe bundle member path')
    root = Path(root).resolve()
    p = root.joinpath(*rel.parts)
    if p.is_symlink() or any(x.is_symlink() for x in p.parents if x != root and root in x.parents):
        raise ValueError('bundle member must not be a symlink')
    if not p.is_file() or not p.resolve().is_relative_to(root):
        raise ValueError('missing or escaped bundle member: ' + name)
    return p


def validate_manifest(root, expected_config=None, verify_files=True, require_validation=True):
    root = Path(root)
    manifest = json.loads((root / 'model_manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('probability_source') != 'classification_head':
        raise ValueError('wrong classification bundle schema/source')
    c = config_identity(manifest.get('head_config'))
    if expected_config is not None and c != config_identity(expected_config):
        raise ValueError('bundle head_config differs from requested variant')
    if manifest.get('head_kind') != c['kind']:
        raise ValueError('head_kind/config mismatch')
    for key, value in [('backbone_dtype', 'bfloat16'), ('head_dtype', 'float32'), ('merge_dtype', 'float32'), ('temperature', 1.0), ('lora_scale', 1.0)]:
        if manifest.get(key) != value:
            raise ValueError('unsupported bundle value: ' + key)
    if type(manifest.get('option_isolation')) is not bool or type(manifest.get('hybrid')) is not bool:
        raise ValueError('missing model layout flags')
    files = manifest.get('files')
    required = {'classification_head.safetensors', 'head_config.json', 'code/kev/heads.py',
                'code/kev/__init__.py', 'code/kev/model.py', 'code/kev/api.py',
                'code/runtime/__init__.py', 'code/runtime/bundle_contract.py',
                'code/runtime/load_bundle.py'}
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError('missing pinned bundle files')
    for name, sha in files.items():
        if not isinstance(sha, str) or len(sha) != 64 or any(x not in '0123456789abcdef' for x in sha):
            raise ValueError('invalid file digest: ' + name)
        path = safe_member(root, name)
        if verify_files and digest(path) != sha:
            raise ValueError('bundle digest mismatch: ' + name)
    for directory in ('code', 'backbone', 'tokenizer'):
        for path in (root/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                if path.relative_to(root).as_posix() not in files:
                    raise ValueError('unbound bundle content: '+str(path))
    if read_config(root / 'head_config.json') != c:
        raise ValueError('head_config.json/manifest mismatch')
    if require_validation:
        report = json.loads((root / 'validation_report.json').read_text())
        delta = report.get('max_probability_delta')
        if (report.get('status') != 'PASS' or report.get('head_config') != c or
                report.get('manifest_sha256') != digest(root / 'model_manifest.json') or
                type(report.get('requests')) is not int or report['requests'] <= 0 or
                type(delta) not in (int, float) or not math.isfinite(delta) or not 0 <= delta <= 1e-5 or
                report.get('argmax_equal') is not True or
                report.get('source_unchanged') is not True or
                report.get('checkpoint_unchanged') is not True):
            raise ValueError('bundle lacks matching successful export/reload verification')
    return manifest


def validate_response(result, model_id, expected_config, allow_legacy_h0=False):
    expected = config_identity(expected_config)
    if not isinstance(result, dict) or result.get('model') != model_id:
        raise ValueError('unexpected model identity')
    source = result.get('probability_source')
    if source == 'classification_head':
        if config_identity(result.get('head_config')) != expected or result.get('head_kind') != expected['kind']:
            raise ValueError('classification head variant mismatch')
    elif source == 'pointer_head' and allow_legacy_h0:
        legacy = {'version': 1, 'kind': 'pointer', 'dim': 256, 'readout': 'end'}
        if expected != legacy:
            raise ValueError('legacy pointer response only allowed for explicit unanchored H0')
    else:
        raise ValueError('unexpected probability source')
    return result


def validate_probabilities(result, request):
    probabilities = result.get('probabilities')
    if not isinstance(probabilities, dict) or set(probabilities) != set(request['questions']):
        raise ValueError('response question coverage mismatch')
    for qid, question in request['questions'].items():
        keys = (list(question['criteria']) if question['type'] == 'choice' else
                ['false', 'true'] if question['type'] == 'noul' else
                [str(i) for i in range(len(question['criteria']))])
        dist = probabilities[qid]
        if not isinstance(dist, dict) or set(dist) != set(keys):
            raise ValueError('response option coverage mismatch')
        values = list(dist.values())
        if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values) or abs(sum(values)-1) > 1e-5:
            raise ValueError('invalid probability distribution')
    return probabilities


def native_prediction(result, request, model_id, expected_config, kind, allow_legacy_h0=False):
    """Keep the unchanged official parser's exact one-key JSON envelope."""
    validate_response(result, model_id, expected_config, allow_legacy_h0)
    probabilities = validate_probabilities(result, request)
    if kind == 'jevbench':
        probabilities = probabilities['decision']
        if request['questions']['decision']['type'] == 'noul':
            probabilities = {'no':probabilities['false'], 'yes':probabilities['true']}
    elif kind != 'kev':
        raise ValueError('unknown native evaluation kind')
    return json.dumps(dict(probabilities=probabilities), ensure_ascii=False, allow_nan=False)
