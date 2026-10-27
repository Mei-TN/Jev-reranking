"""Validate the sixteen one-epoch and two-epoch model identities on CPU.

Bind each model to its training completion, checkpoint tensors, source tree,
and exported bundle. Unfinished training returns PENDING; mismatched artifacts
raise an error. This module uses only the standard library.
"""
import collections
import io
import json
import pickle
import zipfile
from pathlib import Path

from e10_contract import file_sha, read_json

SETTINGS = (('F2', 44), ('F2', 45), ('F3', 44), ('F3', 45))
RECORDS = {'F2': 15456, 'F3': 15442}  # source records of one epoch
# Optimizer updates of the complete horizon: batch 1, accumulation 8, a final flush at the end of every epoch.
COHORTS = {'e10_one_epoch': dict(freeze_block='e10', epochs=1, optimizer_steps={'F2': 1932, 'F3': 1931}),
           'e11_two_epoch': dict(freeze_block='e11', epochs=2, optimizer_steps={'F2': 3864, 'F3': 3862})}
ARMS = {'e10_one_epoch': dict(pointer=('E12_{}s{}_LL_A', 'e1e2', 'e1'), pooling=('E458_{}s{}_POOL_ALL', 'e458', 'e458')),
        'e11_two_epoch': dict(pointer=('E11_{}s{}_POINTER_2EP', 'e11', 'e11'), pooling=('E11_{}s{}_POOL_2EP', 'e11', 'e11'))}
READOUT = dict(pointer='end', pooling='mean_span')
CAMPAIGN_ROOT = dict(e1e2='/path/to/e1e2-campaign', e458='/path/to/e458-campaign',
                     e11='/path/to/e11-campaign')
BASE_REVISION = '56f0584bb40578a2c33b1b40a08ccd17243ad710'
PEAK_LR, LORA_RANK, LORA_ALPHA = 1.25e-05, 64, 128
# Rank-64 LoRA on every target module of NeoHorse-1-4B, and the 256-dimensional q/k head (campaign_e458/PLAN.json).
HEAD_TENSORS, HEAD_NUMEL, ADAPTER_TENSORS, ADAPTER_NUMEL = ('k.bias', 'k.weight', 'q.bias', 'q.weight'), 1311232, 496, 129859584
MODEL_FILES = ('head.pt', 'adapter_model.safetensors', 'adapter_config.json')
EVALUATION_COMPLETE = 'MODEL_EVAL_COMPLETE_PENDING_INDEPENDENT_REVIEW'
# What export and serving execute. Each tree is pinned by its own real bytes: E458 and E11 add the
# last_span readout to kev/model.py, kev/heads.py and runtime/bundle_contract.py, so no shared whole-file hash exists.
_COMMON = {
    'kev/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'kev/api.py': 'c9eebbdb6c625563d33f030299ce0fbe8b50493dd1e92592df9514c30f5720af',
    'kev/evaluate.py': 'c2e03838d8c66334d6f69f8c5353cfc186bb07adb782e2c1aacccf6a1f802076',
    'runtime/__init__.py': '9af52b6bd52fd546e3f54fa3986c3a2c488f999ae9cdfe980732e900d1de0691',
    'runtime/load_bundle.py': '351a5f8e0cd5012f07165e74bbe131fa3514ff2374fe0a818b77c5f6a1e0b1db',
    'runtime/serve_bundle.py': 'df130e5a5e643a9801d25d1016a199b6434d8a30b07743548aa5accd4e4e20b6',
    'runtime/export_bundle.py': '5ba216853940ebd2af48d95c62e45a19105e4430213c3ac66e76339e828f27a1',
    'runtime/verify_bundle.py': '2c8b427cbf99dfc73f4e0ffe1d13d20640483cf5eae2b5bc67be83998ca080e8'}
SOURCES = {
    'e1': dict(_COMMON, **{'kev/model.py': '902427ae67c6cc6d07da4bf0143643cfd1c5d80e58a0f2b3dc1fbb765be0dfc7',
                           'kev/heads.py': 'c31edc946d2616cd8857f8eac917eb5b61525c41e49cf83920df44084c287e46',
                           'runtime/bundle_contract.py': '95fd30e4adcf1a9cc89e33f81abe28ab6a4bbbd057f72b797aca58328a369d11'}),
    'e458': dict(_COMMON, **{'kev/model.py': '07af3dc931e97557506588de46816b30e9c7f653968351025d180755d99068c1',
                             'kev/heads.py': 'b2777b3de16068aee77053fa4cf9907d99f135596a5c004c066c5fddc6407270',
                             'runtime/bundle_contract.py': '7302ae88bb2b9132d2c40264d9864e12a6cae0810f9ff442b0c8e06d2d0d6aaa'})}
# E11 declares changes in kev/train.py, kev/e11_arms.py and kev/e11_reference.py only, so what it exports and serves must
# be the E458 bytes. Its final tree is not deployed yet; a tree that differs here is refused, not adapted to.
SOURCES['e11'] = SOURCES['e458']
EXPORT_ONLY = ('kev/evaluate.py', 'runtime/export_bundle.py', 'runtime/verify_bundle.py')
TRAINED_KEV = ('__init__.py', 'api.py', 'evaluate.py', 'heads.py', 'model.py')


def head_config(readout):
    return dict(version=1, kind='pointer', dim=256, readout=readout, rng_anchor_dim=256)


def model_map():
    """run_id -> the fixed identity. Explicit in MODEL_MAP.json; a test holds both to E10_E11_FREEZE.json."""
    models = {}
    for cohort, contract in COHORTS.items():
        for fold, seed in SETTINGS:
            for structure, (pattern, campaign, source) in ARMS[cohort].items():
                run_id = pattern.format(fold, seed)
                models[run_id] = dict(
                    run_id=run_id, cohort=cohort, fold=fold, seed=seed, structure=structure, readout=READOUT[structure],
                    head_config=head_config(READOUT[structure]), epochs=contract['epochs'],
                    optimizer_steps=contract['optimizer_steps'][fold], records_per_epoch=RECORDS[fold],
                    source_records=contract['epochs'] * RECORDS[fold], campaign=campaign,
                    campaign_plan=CAMPAIGN_ROOT[campaign] + '/PLAN.json', code_tree=CAMPAIGN_ROOT[campaign] + '/source',
                    source_identity=source)
    return models


def cohort_runs(cohort):
    """{(fold, seed): {'pointer': run_id, 'pooling': run_id}} of one cohort; cohorts are never mixed."""
    return {(fold, seed): {structure: pattern.format(fold, seed) for structure, (pattern, _, _) in ARMS[cohort].items()}
            for fold, seed in SETTINGS}


def source_state(code_tree, identity):
    """The tree's export and serving files must be the pinned bytes of the tree that trained this model."""
    code_tree = Path(code_tree)
    actual = {}
    for name, expected in SOURCES[identity].items():
        if not (code_tree / name).is_file() or file_sha(code_tree / name) != expected:
            raise ValueError('code tree is not the qualified ' + identity + ' source: ' + name)
        actual[name] = expected
    return actual


def read_safetensors(path, data=False):
    """Tensor table of a safetensors file from its header; raw bytes only on request."""
    with Path(path).open('rb') as stream:
        size = int.from_bytes(stream.read(8), 'little')
        if not 2 <= size <= 100 << 20:
            raise ValueError('not a safetensors file: ' + str(path))
        header = json.loads(stream.read(size))
        tensors = {}
        for name, entry in header.items():
            if name == '__metadata__':
                continue
            begin, end = entry['data_offsets']
            row = dict(dtype=entry['dtype'], shape=list(entry['shape']))
            if data:
                stream.seek(8 + size + begin)
                row['bytes'] = stream.read(end - begin)
                if len(row['bytes']) != end - begin:
                    raise ValueError('truncated safetensors tensor: ' + name)
            tensors[name] = row
    return tensors


def numel(shape):
    count = 1
    for extent in shape:
        count *= extent
    return count


def read_head_checkpoint(path):
    """(recorded fields, head tensors with raw bytes) of a trainer head.pt, read without torch.

    The pickle may name only an ordered dict, the tensor rebuild function and a storage type. Any other
    global is refused instead of executed, so this is a reader of the saved values and nothing else."""
    def rebuild(storage, offset, size, stride, *unused):
        return dict(storage=storage, offset=offset, shape=list(size), stride=list(stride))

    class Reader(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ('collections', 'OrderedDict'):
                return collections.OrderedDict
            if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
                return rebuild
            if module == 'torch' and name.endswith('Storage'):
                return name
            raise pickle.UnpicklingError('unexpected global in checkpoint: ' + module + '.' + name)

        def persistent_load(self, pid):
            if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != 'storage':
                raise pickle.UnpicklingError('unexpected persistent object in checkpoint')
            return dict(dtype=pid[1], key=str(pid[2]), numel=pid[4])

    with zipfile.ZipFile(path) as archive:
        records = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        if len(records) != 1:
            raise ValueError('not a torch zip checkpoint: ' + str(path))
        prefix = records[0][:-len('data.pkl')]
        if prefix + 'byteorder' in archive.namelist() and archive.read(prefix + 'byteorder').strip() != b'little':
            raise ValueError('checkpoint is not little-endian')
        fields = Reader(io.BytesIO(archive.read(records[0]))).load()
        if not isinstance(fields, dict) or not isinstance(fields.get('head'), dict):
            raise ValueError('checkpoint has no head state')
        tensors = {}
        for name, tensor in fields.pop('head').items():
            shape, count = tensor['shape'], numel(tensor['shape'])
            contiguous = [numel(shape[i + 1:]) for i in range(len(shape))]
            if tensor['storage']['dtype'] != 'FloatStorage' or tensor['stride'] != contiguous:
                raise ValueError('head tensor is not contiguous float32: ' + name)
            raw = archive.read(prefix + 'data/' + tensor['storage']['key'])[4 * tensor['offset']:4 * (tensor['offset'] + count)]
            if len(raw) != 4 * count:
                raise ValueError('truncated head tensor: ' + name)
            tensors[name] = dict(dtype='F32', shape=shape, bytes=raw)
    return fields, tensors


def checkpoint_tensors(out, model):
    """The actual saved head and adapter: structure, element counts and the recorded training end state."""
    fields, head = read_head_checkpoint(out / 'head.pt')
    if (fields.get('head_config') != model['head_config'] or fields.get('head_dim') != 256 or
            fields.get('base_revision') != BASE_REVISION or fields.get('weights_dtype') != 'fp32' or
            fields.get('lora') != LORA_RANK or fields.get('full_finetune') is not False or
            fields.get('option_isolation') is not False or fields.get('special_embeddings') is not False or
            fields.get('stopped_early') is not False or fields.get('optimizer_steps') != model['optimizer_steps'] or
            fields.get('intermediate_checkpoint')):
        raise ValueError('head.pt does not record this complete fixed model')
    if tuple(sorted(head)) != HEAD_TENSORS or sum(numel(t['shape']) for t in head.values()) != HEAD_NUMEL or \
            any(t['shape'][0] != 256 for t in head.values()) or head['q.weight']['shape'] != head['k.weight']['shape']:
        raise ValueError('head.pt does not hold the 256-dimensional q/k head')
    adapter = read_safetensors(out / 'adapter_model.safetensors')
    for name, tensor in adapter.items():
        rank = tensor['shape'][0] if 'lora_A' in name else tensor['shape'][1] if 'lora_B' in name else None
        if tensor['dtype'] != 'F32' or len(tensor['shape']) != 2 or rank != LORA_RANK:
            raise ValueError('adapter tensor is not a rank-64 FP32 LoRA factor: ' + name)
    if len(adapter) != ADAPTER_TENSORS or sum(numel(t['shape']) for t in adapter.values()) != ADAPTER_NUMEL:
        raise ValueError('adapter is not the rank-64 all-target LoRA')
    config = read_json(out / 'adapter_config.json')
    if (config.get('r') != LORA_RANK or config.get('lora_alpha') != LORA_ALPHA or config.get('use_rslora') or
            config.get('use_dora') or config.get('rank_pattern') or config.get('alpha_pattern') or
            config.get('peft_type') != 'LORA'):
        raise ValueError('adapter_config.json is not the plain rank-64 alpha-128 LoRA')
    return head, dict(head={name: dict(shape=t['shape'], sha256=_sha(t['bytes'])) for name, t in sorted(head.items())},
                      head_numel=HEAD_NUMEL, adapter_tensors=len(adapter), adapter_numel=ADAPTER_NUMEL,
                      recorded_optimizer_steps=fields['optimizer_steps'], recorded_head_config=fields['head_config'])


def _sha(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def training_state(model, plan_path=None):
    """('VERIFIED' | 'PENDING', detail). Native completion is re-read from the run's own records every time;
    a recorded status alone is never evidence. A natively failed or different run raises ValueError."""
    plan_path = Path(plan_path or model['campaign_plan'])
    if not plan_path.is_file():
        return 'PENDING', dict(reason='campaign plan is not present', plan=str(plan_path))
    plan = read_json(plan_path)
    runs = [run for run in plan.get('runs', []) if run.get('run_id') == model['run_id']]
    if plan.get('smoke') is not False or len(runs) != 1:
        raise ValueError('campaign plan does not hold exactly this fixed full run')
    run, fold = runs[0], model['fold']
    expected = run['expected']
    if (run['fold'] != fold or run['seed'] != model['seed'] or run['head_config'] != model['head_config'] or
            run['head_peak_lr'] != PEAK_LR or run['adapter_peak_lr'] != PEAK_LR or
            run['lora'].get('rank') != LORA_RANK or run['lora'].get('targets') != 'all' or
            expected['optimizer_steps'] != model['optimizer_steps'] or
            expected['planned_optimizer_steps'] != model['optimizer_steps'] or
            expected['source_records'] != model['source_records'] or expected['status'] != 'complete' or
            plan['base_revision'] != BASE_REVISION or plan['data'][fold]['records'] != model['records_per_epoch'] or
            run['data_sha256'] != plan['data'][fold]['sha256']):
        raise ValueError('campaign plan entry is not this fixed identity, budget and recipe')
    paths = run['paths']
    out, exit_path, result_path = Path(paths['output_dir']), Path(paths['exit_record']), Path(paths['result'])
    if not exit_path.is_file():
        return 'PENDING', dict(reason='training has no native exit record', plan=str(plan_path), checkpoint=str(out))
    native = read_json(exit_path)
    if type(native.get('exit_code')) is not int or native['exit_code'] != 0:
        raise ValueError('training ended with native exit ' + repr(native.get('exit_code')))
    if not result_path.is_file():
        return 'PENDING', dict(reason='training has no result record', plan=str(plan_path), checkpoint=str(out))
    result = read_json(result_path)
    state = result['state']
    if (result['run_id'] != model['run_id'] or result['plan_entry'] != run or state['status'] != 'COMPLETE' or
            type(state['exit_code']) is not int or state['exit_code'] != 0 or set(result['final_model_sha256']) != set(MODEL_FILES)):
        raise ValueError('training result does not record native completion of this plan entry')
    config, metrics = read_json(out / 'training_config.json'), read_json(out / 'training_metrics.json')
    steps, args = model['optimizer_steps'], config['args']
    if (metrics['status'] != 'complete' or metrics['stopped_early'] is not False or
            metrics['optimizer_steps'] != steps or metrics['planned_optimizer_steps'] != steps or
            config['dse']['planned_optimizer_steps'] != steps or metrics['source_records_seen'] != model['source_records']):
        raise ValueError('training did not complete the full %d-epoch horizon: %d updates over %d source records required'
                         % (model['epochs'], steps, model['source_records']))
    saved_head = config['head_config']
    if (args['epochs'] != model['epochs'] or args['seed'] != model['seed'] or args['lr'] != PEAK_LR or
            args['head_lr'] != PEAK_LR or args['lora'] != LORA_RANK or args['lora_targets'] != 'all' or
            args.get('weights_dtype', 'fp32') != 'fp32' or args.get('full_finetune', False) or args.get('option_isolation') or
            (json.loads(saved_head) if isinstance(saved_head, str) else saved_head) != model['head_config'] or
            json.loads(args['head_config']) != model['head_config'] or config['base_revision'] != BASE_REVISION or
            Path(args['base']) != Path(plan['base'])):
        raise ValueError('saved training recipe, head or base differs from this fixed identity')
    trained = {name: config['dse']['source_sha256'].get(name) for name in TRAINED_KEV}
    if trained != {name: SOURCES[model['source_identity']]['kev/' + name] for name in TRAINED_KEV}:
        raise ValueError('checkpoint was not trained with the qualified ' + model['source_identity'] + ' source')
    files = {name: file_sha(out / name) for name in MODEL_FILES}
    if files != result['final_model_sha256']:
        raise ValueError('checkpoint files are not the ones hashed when training completed')
    _, tensors = checkpoint_tensors(out, model)
    return 'VERIFIED', dict(
        plan=str(plan_path), plan_sha256=file_sha(plan_path), campaign=plan['campaign'], checkpoint=str(out),
        base=str(Path(plan['base'])), checkpoint_sha256=files, epochs=args['epochs'], optimizer_steps=metrics['optimizer_steps'],
        source_records_seen=metrics['source_records_seen'], trained_source_sha256=trained, tensors=tensors,
        evidence_sha256={name: file_sha(path) for name, path in (
            ('EXIT.json', exit_path), ('RESULT.json', result_path), ('training_config.json', out / 'training_config.json'),
            ('training_metrics.json', out / 'training_metrics.json'))})


def bundle_state(bundle, model, served_model_id, training, code_tree, contract):
    """Revalidate an exported bundle against this checkpoint's actual bytes and source, not against its path.

    `contract` is the code tree's own stdlib runtime/bundle_contract.py: it rehashes every bound file and requires
    the exporter's passing fresh-process reload verification."""
    bundle, code_tree = Path(bundle), Path(code_tree)
    manifest = contract.validate_manifest(bundle, model['head_config'])
    if (manifest['name'] != served_model_id or manifest['option_isolation'] is not False or
            manifest['base_revision'] != BASE_REVISION or manifest['checkpoint_hashes'] != training['checkpoint_sha256'] or
            Path(manifest['source_checkpoint']) != Path(training['checkpoint'])):
        raise ValueError('bundle does not bind this checkpoint, base revision and served identity')
    pins = SOURCES[model['source_identity']]
    for name, value in manifest['source_hashes'].items():
        if not (code_tree / name).is_file() or file_sha(code_tree / name) != value:
            raise ValueError('bundle was exported from a different source: ' + name)
    for name, value in pins.items():
        if manifest['source_hashes'].get(name) != value:
            raise ValueError('bundle source is not the qualified ' + model['source_identity'] + ' source: ' + name)
        if name not in EXPORT_ONLY and manifest['files'].get('code/' + name) != value:
            raise ValueError('bundle serves a different loader: ' + name)
    trained, _ = checkpoint_tensors(Path(training['checkpoint']), model)
    served = read_safetensors(bundle / 'classification_head.safetensors', data=True)
    if set(served) != set(trained) or any(served[name] != trained[name] for name in trained):
        raise ValueError('served head tensors are not the trained head.pt tensors')
    return dict(path=str(bundle.resolve()), name=manifest['name'], manifest_sha256=file_sha(bundle / 'model_manifest.json'),
                head_sha256=manifest['files']['classification_head.safetensors'], hybrid=manifest['hybrid'],
                bound_files=len(manifest['files']), checkpoint_hashes=manifest['checkpoint_hashes'],
                head_tensors_equal_trained_head=True)


def reuse_state(evaluation, receipt, model, training, code_tree, contract):
    """('VERIFIED' | 'PENDING', detail) for the bundle of the standard full evaluation of this exact checkpoint.

    Usable only after that evaluation finished natively and released its services: its own result, the external
    receipt of its exit, and then the same byte-level revalidation as a fresh export."""
    evaluation = Path(evaluation)
    if not (evaluation / 'RESULT.json').is_file() or not Path(receipt).is_file():
        return 'PENDING', dict(reason='the full evaluation has not finished with an external exit receipt',
                               evaluation=str(evaluation))
    result, plan, external = read_json(evaluation / 'RESULT.json'), read_json(evaluation / 'PLAN.json'), read_json(receipt)
    if (type(external.get('native_exit')) is not int or external['native_exit'] != 0 or
            external.get('result_sha256') != file_sha(evaluation / 'RESULT.json') or
            external.get('evaluation_run_id') != result.get('run_id')):
        raise ValueError('external exit receipt does not bind a successful full evaluation result')
    if (result.get('status') != EVALUATION_COMPLETE or type(result.get('native_exit')) is not int or
            result['native_exit'] != 0 or 'failure' in result or result['cleanup'].get('complete') is not True or
            result['cleanup'].get('residual') != [] or 'error' in result['gpu_post'] or 'export' not in result['completed'] or
            result['plan_sha256'] != file_sha(evaluation / 'PLAN.json')):
        raise ValueError('full evaluation did not complete and release its services')
    if (Path(plan['checkpoint']) != Path(training['checkpoint']) or Path(plan['code_tree']) != Path(code_tree) or
            plan['head_config'] != model['head_config'] or plan['base_revision'] != BASE_REVISION or
            plan['run_id'] != result['run_id'] or plan['run_id'] not in (model['run_id'], model['run_id'] + '_eval')):
        raise ValueError('full evaluation was not of this checkpoint, source and head')
    state = bundle_state(evaluation / 'bundle', model, plan['run_id'], training, code_tree, contract)
    if state['manifest_sha256'] != result.get('bundle_manifest_sha256'):
        raise ValueError('bundle is not the one the full evaluation exported')
    return 'VERIFIED', dict(state, evaluation=str(evaluation), evaluation_result_sha256=file_sha(evaluation / 'RESULT.json'),
                            evaluation_exit_receipt=str(receipt), evaluation_exit_receipt_sha256=file_sha(receipt))
