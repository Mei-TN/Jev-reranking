"""Bind reproduction scripts to existing runtime environments."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROLES = {'training', 'serving', 'opencompass', 'openjev', 'runweaver'}
GROUP_ROLES = {'train': ('training',), 'export': ('training',), 'serve': ('serving',),
               'score': ('opencompass', 'openjev', 'runweaver')}
REPLACED_KEYS = {'training_python', 'opencompass_python', 'mc_python',
    'training_env/lib/python3.12/site-packages/peft/__init__.py',
    'training_env/lib/python3.12/site-packages/safetensors/__init__.py',
    'training_env/lib/python3.12/site-packages/torch/__init__.py',
    'training_env/lib/python3.12/site-packages/transformers/__init__.py',
    'mc_env/lib/python3.12/site-packages/numpy/__init__.py'}

def require(value, message):
    if not value:
        raise ValueError(message)

def pin(path):
    h = hashlib.sha256(); size = 0
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block); size += len(block)
    return {'bytes': size, 'sha256': h.hexdigest()}

def load():
    m = json.loads((ROOT/'RUNTIME_MANIFEST.json').read_text())
    require(m['schema'] == 'portable-existing-runtime-binding-v2', 'runtime schema differs')
    require(set(m['roles']) == ROLES, 'runtime role inventory differs')
    require(set(m['replaced_assets']) == REPLACED_KEYS, 'runtime replacement inventory differs')
    old = json.loads((ROOT/'ASSET_MANIFEST.json').read_text())
    require(pin(ROOT/'ASSET_MANIFEST.json') == m['original_asset_manifest'], 'scientific/legacy asset manifest changed')
    require(m['replaced_assets'] == {k: old['assets'][k] for k in REPLACED_KEYS}, 'legacy runtime provenance changed')
    for name, role in m['roles'].items():
        root = Path(role['root'])
        require(root.is_absolute() and '..' not in root.parts, 'runtime root must be explicit and absolute')
        files = role['files']
        require(files and len({f['path'] for f in files}) == len(files), 'runtime file inventory is empty or duplicated')
        require(role['python'] in {f['path'] for f in files}, 'interpreter is not byte-bound')
        require(all(Path(f['path']).is_relative_to(root) and Path(f['resolved_path']).is_relative_to(root)
                    for f in files), 'runtime file escapes its declared installation')
        require(all(type(f['bytes']) is int and f['bytes'] >= 0 and len(f['sha256']) == 64 for f in files), 'invalid runtime file pin')
        require(set(role['environment']) <= {'LD_LIBRARY_PATH', 'PYTHONPATH', 'PATH', 'LANG',
                  'SDL_AUDIODRIVER', 'SDL_VIDEODRIVER'}, 'unexpected persistent runtime environment field')
        for field in ('LD_LIBRARY_PATH', 'PYTHONPATH'):
            value = role['environment'].get(field, '')
            require(not value or all(p and Path(p).is_absolute() and Path(p).is_relative_to(root)
                    for p in value.split(os.pathsep)), 'runtime search path escapes installation')
    return m

def role_for_command(name):
    return 'opencompass' if name in ('opencompass', 'jk_audit') else 'openjev'

def python(role):
    return load()['roles'][role]['python']

def verify(groups):
    m = load()
    names = sorted({r for g in groups for r in GROUP_ROLES.get(g, ())})
    count = 0
    for name in names:
        role = m['roles'][name]
        for f in role['files']:
            p = Path(f['path'])
            require(str(p.resolve(strict=True)) == f['resolved_path'], 'runtime link target changed: '+str(p))
            require(pin(p) == {k:f[k] for k in ('bytes','sha256')}, 'runtime identity mismatch: '+str(p))
            count += 1
        for field in ('LD_LIBRARY_PATH','PYTHONPATH'):
            for path in filter(None, role['environment'].get(field,'').split(os.pathsep)):
                require(Path(path).is_dir(), 'runtime search directory absent: '+path)
    return {'roles': names, 'files_checked': count, 'runtime_manifest': pin(ROOT/'RUNTIME_MANIFEST.json'),
            'scope':'Existing interpreter/module/distribution-metadata identities; no import or model execution'}

def compose(role, scientific_env, work, asset):
    """Preserve declared role libraries/overlay and the original scientific cache paths."""
    r = load()['roles'][role]
    env = dict(scientific_env)
    original_sources = env.get('PYTHONPATH', '')
    env.update(r['environment'])
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(filter(None,
        original_sources.split(os.pathsep) + r['environment'].get('PYTHONPATH','').split(os.pathsep))))
    # The exporter resets child PYTHONPATH; expose only the verified role overlay.
    env['JEV_VERIFIED_RUNTIME_OVERLAY'] = r['environment'].get('PYTHONPATH', '')
    state = Path(work)/'runtime-state'/role
    # New process-writable state is confined to this derivative's work directory.
    env.update(HOME=str(state/'home'), TMPDIR=str(state/'tmp'), XDG_CACHE_HOME=str(state/'cache'),
               TRITON_CACHE_DIR=str(state/'triton'), PYTHONNOUSERSITE='1', PYTHONSAFEPATH='1',
               PYTHONDONTWRITEBYTECODE='1', PIP_NO_INDEX='1', PIP_CONFIG_FILE='/dev/null',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', OMP_NUM_THREADS='2')
    if 'HF_HOME' not in env:
        env['HF_HOME'] = str(Path(asset)/'evaluation/hf-cache')
    return env

def process_environment(overrides):
    env = dict(os.environ)
    for key in ('PYTHONHOME','PYTHONPATH','PYTHONSTARTUP','LD_PRELOAD','LD_AUDIT'):
        env.pop(key, None)
    env.update(overrides)
    return env

def prepare_state(env, work):
    """Called only for an explicitly requested run, never by validate or plan."""
    root = (Path(work)/'runtime-state').resolve()
    for key in ('HOME','TMPDIR','XDG_CACHE_HOME','TRITON_CACHE_DIR'):
        p = Path(env[key])
        require(p.resolve().is_relative_to(root), 'writable runtime state escapes derivative work')
        p.mkdir(parents=True, exist_ok=True)
