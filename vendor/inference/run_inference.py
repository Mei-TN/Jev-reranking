#!/usr/bin/env python3
"""Run a fixed inference population for one checkpoint.

Export once, start four cache-isolated serving replicas, and run inference_worker.
--prepare-only checks inputs on CPU without launching processes or writing files.
An incomplete checkpoint returns pending status (exit 3).
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # Sibling modules import by name even under PYTHONSAFEPATH or -P.

import prepare_inputs as prep
from materialize_population import file_sha
from token_preflight import read_json

POPULATION_SHA256 = 'dd9349a0cc7bca145f2b919c1106583382023baba06cc7fc1613a5ac4de1fd51'
REQUESTS = 69590
ASSETS = os.environ.get('JEV_INTERFACE_ASSETS', '/path/to/interface-assets')
PINS = HERE / 'checkpoint_pins_v2.json'
EVALUATION_DIRS = (HERE.parent / 'evaluation',)
# Shared process cleanup and per-replica cache handling.
LIFECYCLE_SHA256 = {
    'run_evaluation.py': '2960be8ea6921784d3842257e6d06dce4061571f9328452a1ff639df9230369c',
    'replica_cache.py': 'fa59602af179378c6cf462e55ac6666ff3157d80e570ca27607a6c584c4e24a8',
    'common.py': '313ff2d1e247536e1cfa49d74eba7507b11bb70bf49e61b0759df90975396fd5'}
# What export and serving execute from the code tree: the head-supplement-v1 loader the token preflight qualified.
LOADER_SHA256 = {
    'kev/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'kev/model.py': '902427ae67c6cc6d07da4bf0143643cfd1c5d80e58a0f2b3dc1fbb765be0dfc7',
    'kev/heads.py': 'c31edc946d2616cd8857f8eac917eb5b61525c41e49cf83920df44084c287e46',
    'kev/api.py': 'c9eebbdb6c625563d33f030299ce0fbe8b50493dd1e92592df9514c30f5720af',
    'kev/evaluate.py': 'c2e03838d8c66334d6f69f8c5353cfc186bb07adb782e2c1aacccf6a1f802076',
    'runtime/__init__.py': '9af52b6bd52fd546e3f54fa3986c3a2c488f999ae9cdfe980732e900d1de0691',
    'runtime/bundle_contract.py': '95fd30e4adcf1a9cc89e33f81abe28ab6a4bbbd057f72b797aca58328a369d11',
    'runtime/load_bundle.py': '351a5f8e0cd5012f07165e74bbe131fa3514ff2374fe0a818b77c5f6a1e0b1db',
    'runtime/serve_bundle.py': 'df130e5a5e643a9801d25d1016a199b6434d8a30b07743548aa5accd4e4e20b6',
    'runtime/export_bundle.py': '5ba216853940ebd2af48d95c62e45a19105e4430213c3ac66e76339e828f27a1',
    'runtime/verify_bundle.py': '2c8b427cbf99dfc73f4e0ffe1d13d20640483cf5eae2b5bc67be83998ca080e8'}
EXPORT_ONLY = ('kev/evaluate.py', 'runtime/export_bundle.py', 'runtime/verify_bundle.py')
CHECKPOINT_FILES = ('head.pt', 'adapter_model.safetensors', 'adapter_config.json',
                    'training_config.json', 'training_metrics.json')
WORKER_SOURCES = ('inference_worker.py', 'inference_scores.py', 'materialize_population.py',
                  'prepare_inputs.py', 'token_preflight.py')
WORKER_OUTPUTS = ('RUN_BINDING.json', 'planned_calls.jsonl', 'completion_calls.jsonl', 'ordered_calls.jsonl',
                  'question_scores.jsonl', 'metrics.json')
LOCK_PREFIX = '/tmp/jev-supplement-eval-gpu-'  # The full evaluation's per-GPU lock: one owner per GPU across both.
LIMITS = dict(max_state=8192, max_branch=8192, max_packed=8192)
ISOLATED = ('HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR')
RESULT_SCHEMA = 'jev-revision-20261013-inference-result-v1'
PREPARED, PENDING = 'CPU_PREPARED_NOT_LAUNCHED', 'CHECKPOINT_PENDING_NOT_LAUNCHED'
INVALID, NOT_LAUNCHED = 'INVALID_NOT_LAUNCHED', 'NOT_LAUNCHED'
COMPLETE = 'INFERENCE_COMPLETE_PENDING_INDEPENDENT_REVIEW'


def lifecycle(directory=None):
    """Import the shared evaluation lifecycle after verifying its source digest."""
    for root in ([Path(directory)] if directory else EVALUATION_DIRS):
        if (root / 'run_evaluation.py').is_file():
            break
    else:
        raise ValueError('reviewed evaluation directory not found; pass --evaluation-dir')
    root = root.resolve()
    for name, expected in LIFECYCLE_SHA256.items():
        if not (root / name).is_file() or file_sha(root / name) != expected:
            raise ValueError('evaluation lifecycle differs from the reviewed bytes: ' + name)
    sys.path.insert(0, str(root))
    import run_evaluation
    if Path(run_evaluation.__file__).resolve().parent != root:
        raise ValueError('a different run_evaluation module is already imported')
    return run_evaluation


def allocation(path, index):
    """The campaign yields a GPU only while its CONTROL.json lists it in exclude_gpus."""
    control = read_json(path)
    excluded = control.get('exclude_gpus') if isinstance(control, dict) else None
    if not isinstance(excluded, list) or not all(type(i) is int for i in excluded):
        raise ValueError('allocation control needs an explicit integer exclude_gpus list')
    if index not in excluded:
        raise ValueError('GPU ' + str(index) + ' is not excluded from campaign training; refusing to use it')
    return dict(path=str(path), sha256=file_sha(path), exclude_gpus=excluded)


def checkpoint_state(checkpoint, run_id, head):
    """'VERIFIED' or 'PENDING' with details; a complete but different checkpoint raises.

    Every byte is compared with the source-side pins, so a partial transfer is pending, never usable.
    """
    pins = read_json(PINS)['runs'][run_id]
    paths = {name: checkpoint.parent / name if name == 'RESULT.json' else checkpoint / Path(name).name for name in pins}
    missing = [name for name, path in paths.items() if not path.is_file()]
    partial = [name for name, path in paths.items() if path.is_file() and path.stat().st_size < pins[name]['bytes']]
    if missing or partial:
        return 'PENDING', dict(missing=missing, incomplete=partial, pins=pins)
    for name, path in paths.items():
        if path.stat().st_size != pins[name]['bytes'] or file_sha(path) != pins[name]['sha256']:
            raise ValueError('checkpoint differs from its source: ' + name)
    result = read_json(paths['RESULT.json'])
    artifacts = {'output/' + name: pin for name, pin in result['checkpoint']['artifacts'].items()}
    if (result['run_id'] != run_id or result['status'] != 'TRAIN_COMPLETE_PENDING_FULL_EVALUATION' or
            type(result['native']) is not int or result['native'] != 0 or
            result['checkpoint']['actual_steps'] != result['spec']['expected_steps'] or
            result['spec']['head_config'] != head or artifacts != {k: v for k, v in pins.items() if k != 'RESULT.json'}):
        raise ValueError('source training RESULT does not record this complete fixed checkpoint')
    return 'VERIFIED', dict(pins=pins, expected_steps=result['spec']['expected_steps'])


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True, help='one of the eight fixed SUPP_WINH4_{H0,H5}_f{2,3}_s{44,45}')
    ap.add_argument('--gpu', required=True, help='one index 0..7 or a complete GPU UUID; never chosen automatically')
    ap.add_argument('--output', required=True, help='new directory; an attempt is never overwritten or resumed')
    ap.add_argument('--allocation-control', help='campaign CONTROL.json; its exclude_gpus must list the GPU (required to launch)')
    ap.add_argument('--prepare-only', action='store_true', help='CPU input validation; print plan, write and start nothing')
    ap.add_argument('--checkpoint', help='default <checkpoint-root>/<run-id>/output, with RESULT.json beside output/')
    ap.add_argument('--checkpoint-root', default=ASSETS + '/checkpoint_assets_v2')
    ap.add_argument('--population', default=ASSETS + '/protocol-preflight-v1/population_v1')
    ap.add_argument('--code-tree', default=str(HERE.parent / 'historical_heads'))
    ap.add_argument('--evaluation-dir', help='directory of the reviewed run_evaluation.py; default: sibling or deployed copy')
    ap.add_argument('--port-base', type=int, help='four consecutive ports; default 32300 + 8 * selected GPU index')
    ap.add_argument('--timeout-seconds', type=int, default=14400, help='budget of this invocation, default 4 hours')
    for key in ('base', 'base-revision', 'serve-python', 'mc-python', 'verification-examples', 'hf-cache'):
        ap.add_argument('--' + key, help='default: the completed evaluation contract')
    return ap


def interpreter_can_own_processes():
    """Provide pidfd support for interpreters built against older kernel headers."""
    return hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal')


def prepare(args, runner):
    if sys.platform == 'linux' and not interpreter_can_own_processes():
        raise ValueError('this interpreter has no os.pidfd_open, so the process lifecycle cannot start: '
                         'run the controller with /usr/bin/python3, not ' + sys.executable)
    population_root = Path(args.population).expanduser().resolve()
    manifest = population_root / 'POPULATION.json'
    if not manifest.is_file() or file_sha(manifest) != POPULATION_SHA256:
        raise ValueError('not the frozen token-qualified population manifest')
    population = read_json(manifest)
    if population['status'] != 'COMPLETE_CPU_TOKEN_QUALIFIED' or population['total_requests'] != REQUESTS:
        raise ValueError('incomplete or unqualified population')
    for name, expected in population['files'].items():
        if file_sha(population_root / name) != expected:
            raise ValueError('population member changed: ' + name)
    models = read_json(population_root / 'checkpoints.json')['models']
    matches = [model for model in models if model['run_id'] == args.run_id]
    if len(models) != 8 or len(matches) != 1:
        raise ValueError('run-id is not one of the eight fixed checkpoint identities')
    if not (re.fullmatch(r'[0-7]', args.gpu) or re.fullmatch(r'GPU-[0-9a-fA-F-]+', args.gpu)):
        raise ValueError('gpu must be one index 0..7 or a complete GPU UUID')
    if args.timeout_seconds <= 0 or (args.port_base is not None and not 1024 <= args.port_base <= 65532):
        raise ValueError('timeout must be positive and port-base must permit four unprivileged ports')
    if not args.prepare_only and not args.allocation_control:
        raise ValueError('launch requires --allocation-control with this GPU in exclude_gpus')
    source = Path(args.code_tree).expanduser().resolve()
    checkpoint = Path(args.checkpoint or Path(args.checkpoint_root) / args.run_id / 'output').expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise ValueError('output already exists; preserve attempts and choose a new path')
    if any(output.is_relative_to(root) for root in (checkpoint.parent, source, population_root)):
        raise ValueError('output must be outside the checkpoint, code tree and population')
    # Do not silently export or serve with a different loader than the one that was qualified.
    for name, expected in LOADER_SHA256.items():
        if not (source / name).is_file() or file_sha(source / name) != expected:
            raise ValueError('code-tree is not the qualified head-supplement-v1 loader: ' + name)
    plan = {key: getattr(args, key) or runner.DEFAULTS[key] for key in
            ('base', 'base_revision', 'serve_python', 'mc_python', 'verification_examples', 'hf_cache')}
    for key in ('base', 'verification_examples', 'hf_cache'):
        plan[key] = str(Path(plan[key]).expanduser().resolve())
    for key in ('serve_python', 'mc_python'):
        # Keep the venv's lexical bin/python: resolving its symlink silently drops the role environment.
        plan[key] = os.path.abspath(os.path.expanduser(plan[key]))
    for key in ('serve_python', 'mc_python', 'verification_examples'):
        if not Path(plan[key]).is_file():
            raise ValueError('missing asset file: ' + key + '=' + plan[key])
    if not (Path(plan['base']) / 'config.json').is_file() or not Path(plan['hf_cache']).is_dir():
        raise ValueError('missing base model or HF cache directory')
    head = runner.contract_at(source).config_identity(matches[0]['head_config'])
    state, detail = checkpoint_state(checkpoint, args.run_id, head)
    if state == 'VERIFIED':
        config, metrics, _ = runner.checkpoint_metadata(checkpoint, source, head)
        recorded = config['dse']['source_sha256']
        if (config.get('base_revision') != plan['base_revision'] or
                Path(config['args']['base']).resolve() != Path(plan['base']).resolve() or
                metrics['optimizer_steps'] != detail['expected_steps']):
            raise ValueError('saved base, base revision or step count does not match this export')
        for name in LOADER_SHA256:
            if name.startswith('kev/') and recorded.get(Path(name).name) != LOADER_SHA256[name]:
                raise ValueError('checkpoint was trained with a different ' + name)
    plan.update(schema='jev-revision-20261013-single-checkpoint-inference-v1', run_id=args.run_id,
                fold=matches[0]['fold'], seed=matches[0]['seed'], method=matches[0]['method'], head_config=head,
                checkpoint=str(checkpoint), checkpoint_state=state, checkpoint_detail=detail,
                code_tree=str(source), loader_sha256=LOADER_SHA256, population=str(population_root),
                population_sha256=POPULATION_SHA256, protocol_version=population['protocol_version'],
                ordered_model_wire_sha256=population['ordered_model_wire_sha256'][args.run_id],
                condition_requests=population['condition_requests'], requests=REQUESTS, replicas=4,
                worker_sources={name: file_sha(HERE / name) for name in WORKER_SOURCES},
                controller_sha256=file_sha(__file__), controller_python=sys.executable,
                evaluation_dir=str(Path(runner.__file__).resolve().parent),
                lifecycle_sha256=LIFECYCLE_SHA256, gpu=args.gpu, output=str(output),
                allocation_control=str(Path(args.allocation_control).expanduser().resolve()) if args.allocation_control else None,
                timeout_seconds=args.timeout_seconds, port_base=args.port_base, serve_overlay=None,
                service_limits=LIMITS, retries=0, loopback_direct=loopback_direct())
    if plan['allocation_control'] and re.fullmatch(r'[0-7]', args.gpu):
        plan['allocation'] = allocation(plan['allocation_control'], int(args.gpu))
    return (PREPARED if state == 'VERIFIED' else PENDING), plan


def loopback_direct():
    """Health checks and every worker request must reach 127.0.0.1 itself, not an inherited HTTP proxy."""
    return 'http' not in urllib.request.getproxies() or bool(urllib.request.proxy_bypass('127.0.0.1'))


def input_identity(plan):
    checkpoint, source = Path(plan['checkpoint']), Path(plan['code_tree'])
    paths = [checkpoint / name for name in CHECKPOINT_FILES] + [checkpoint.parent / 'RESULT.json']
    paths += [p for folder in ('kev', 'runtime') for p in (source / folder).glob('*.py')]
    paths += [Path(plan['population']) / 'POPULATION.json', Path(plan['verification_examples']), PINS]
    paths += [HERE / name for name in WORKER_SOURCES] + [Path(__file__).resolve()]
    return {str(p): file_sha(p) for p in sorted(set(paths))}


def exact(value):
    """Canonical JSON: tells true from 1 and exposes any extra or missing field."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def expected_health(plan, bundle, port, manifest_sha, head_sha):
    head = plan['head_config']
    return dict(model=plan['run_id'], probability_source='classification_head', head_kind=head['kind'], head_config=head,
                checkpoint=str(bundle.resolve()), port=port, gpu=plan['gpu_uuid'], tp=1, dtype='torch.bfloat16',
                head_dtype='torch.float32', merge='premerged_fp32_then_bf16', lora_scale=1, temperature=1,
                bundle_loader='runtime.load_bundle.load_bundle', prefix_cache=False, strict=True,
                supports_vision=False, head_sha256=head_sha, manifest_sha256=manifest_sha, **LIMITS)


def replica_state(proc, expected):
    """The writable state directories the owned service process was actually started with."""
    raw = Path('/proc/' + str(proc.pid) + '/environ').read_bytes().split(b'\0')
    actual = dict(item.decode(errors='replace').split('=', 1) for item in raw if b'=' in item)
    state = {key: actual.get(key) for key in ISOLATED}
    if state != {key: expected[key] for key in ISOLATED} or not all(Path(value).is_dir() for value in state.values()):
        raise ValueError('service process lacks its own isolated state directories')
    return state


def worker_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {p.name: dict(bytes=p.stat().st_size, sha256=file_sha(p)) for p in sorted(directory.iterdir()) if p.is_file()}


def worker_completion(directory, plan, manifest_sha):
    """Bind the worker's own completion record to this plan, bundle and the files it names."""
    directory = Path(directory)
    if (directory / 'FAILED.json').exists() or not (directory / 'COMPLETE.json').is_file():
        raise ValueError('worker did not publish a completion record')
    complete = read_json(directory / 'COMPLETE.json')
    binding = read_json(directory / 'RUN_BINDING.json')
    if (complete['status'] != 'COMPLETE' or complete['run_id'] != plan['run_id'] or complete['requests'] != REQUESTS or
            complete['population_sha256'] != POPULATION_SHA256 or complete['manifest_sha256'] != manifest_sha or
            complete['head_config'] != plan['head_config'] or complete['conditions'] != plan['condition_requests'] or
            complete['questions'] != {'piqa': 1838, 'gpqa_diamond': 198} or
            not set(WORKER_OUTPUTS) <= set(complete['files'])):
        raise ValueError('worker completion does not bind this checkpoint, bundle and population')
    for name, expected in complete['files'].items():
        if file_sha(directory / name) != expected:
            raise ValueError('worker output changed after completion: ' + name)
    if (binding['worker_sha256'] != plan['worker_sources']['inference_worker.py'] or binding['retries'] != 0 or
            binding['manifest_sha256'] != manifest_sha or binding['population_sha256'] != POPULATION_SHA256):
        raise ValueError('worker run binding differs from the reviewed worker and this bundle')
    metrics = read_json(directory / 'metrics.json')
    conditions = metrics['conditions']
    # Denominators only; no score is read to decide anything here.
    if (set(conditions) != set(plan['condition_requests']) or metrics['gpqa_native']['n_questions'] != 198 or
            any(row['n_questions'] != (1838 if name.startswith('piqa_') else 198) for name, row in conditions.items())):
        raise ValueError('worker metrics do not cover the fixed 21 conditions and denominators')
    return dict(status='COMPLETE', requests=REQUESTS, complete_sha256=file_sha(directory / 'COMPLETE.json'),
                metrics_sha256=complete['files']['metrics.json'],
                question_scores_sha256=complete['files']['question_scores.jsonl'],
                ordered_calls_sha256=complete['files']['ordered_calls.jsonl'])


def request_witness(calls, service_logs, endpoints, ordered_wire_sha, total=REQUESTS):
    """Every worker call occurrence must be exactly one logged request/response of the replica it names.

    Occurrences are counted, not deduplicated: four request bodies repeat by design. Acceptance
    depends only on which requests were sent and served, never on what the model answered: the
    number of PIQA NLI groups whose six orders gave different outputs is recorded as description.
    Identical outputs for distinct orders are a valid zero-sensitivity result.
    """
    occurrence = lambda event: prep.digest(prep.wire([event['request'], event['response']]))
    planned, stream_sha, count, groups = [Counter() for _ in endpoints], hashlib.sha256(), 0, {}
    with Path(calls).open(encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            if event['ordinal'] != count or event.get('error'):
                raise ValueError('worker call ledger is not the complete ordered population')
            count += 1
            stream_sha.update(prep.wire(event['request']) + b'\n')
            planned[endpoints.index(event['endpoint'])][occurrence(event)] += 1
            cell = re.fullmatch(r'piqa_nli_(P\d)(D\d)', event['condition'])
            if cell:
                relation = event['response']['probabilities']['relation']
                groups.setdefault((event['question_id'], event['candidate_id'], cell.group(2)), set()).add(
                    tuple(relation[label] for label in prep.LABELS))
    served = []
    for path in service_logs:
        counts = Counter()
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                counts[occurrence(json.loads(line))] += 1
        served.append(counts)
    if count != total or stream_sha.hexdigest() != ordered_wire_sha:
        raise ValueError('worker calls are not the frozen ordered request population of this checkpoint')
    if served != planned:
        raise ValueError('service request logs differ from the worker call occurrences')
    return dict(requests=count, per_replica=[sum(c.values()) for c in served],
                exact_request_response_occurrences_equal=True, ordered_model_wire_sha256=stream_sha.hexdigest(),
                permutation_groups=len(groups),
                groups_with_order_dependent_outputs=sum(len(values) > 1 for values in groups.values()))


def execute(plan, runner):
    durable, sha = runner.durable, runner.sha
    root, lock = Path(plan['output']), None
    try:
        # Nothing exists yet: every refusal here is NOT_LAUNCHED and leaves no output directory.
        gpu = runner.machine_and_gpu(plan['gpu'])
        allocation(plan['allocation_control'], gpu['index'])
        if not plan['loopback_direct']:
            raise ValueError('127.0.0.1 would be sent to an HTTP proxy; NO_PROXY must cover loopback')
        import fcntl  # Execution-only POSIX dependency; prepare is portable and CPU-only.
        lock = open(LOCK_PREFIX + gpu['uuid'] + '.lock', 'a+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('this GPU is locked by another evaluation or inference run') from None
        gpu = runner.machine_and_gpu(gpu['uuid'])
        plan['allocation'] = allocation(plan['allocation_control'], gpu['index'])  # Re-read while holding the lock.
        ports = [(plan['port_base'] or (32300 + 8 * gpu['index'])) + i for i in range(4)]
        runner.bind_ports(ports)
        processes = runner.Processes(root, plan['timeout_seconds'])
        plan.update(gpu_uuid=gpu['uuid'], gpu_index=gpu['index'], ports=ports)
        root.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        if lock is not None:
            lock.close()
        print(json.dumps(dict(status=NOT_LAUNCHED, problem=type(exc).__name__ + ': ' + str(exc)), indent=2))
        return 4
    try:
        durable(root / 'PLAN.json', plan)
        record = dict(schema=RESULT_SCHEMA, run_id=plan['run_id'], status='RUNNING', native_exit=None,
                      started_epoch=time.time(), gpu_pre=gpu, completed=[], plan_sha256=sha(root / 'PLAN.json'),
                      scientific_acceptance=None)
        durable(root / 'ATTEMPT.json', record)
        success = False
        old_signals = {}
        try:
            def interrupted(signum, frame):
                raise RuntimeError('controller interrupted by signal ' + str(signum))
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                old_signals[sig] = signal.signal(sig, interrupted)
            inputs = input_identity(plan)
            durable(root / 'INPUTS.json', inputs)
            # The evaluation contract's environment; it copies os.environ, so NO_PROXY is preserved.
            env = runner.environment(dict(plan, **{key: runner.DEFAULTS[key] for key in
                                                   ('runtime_cache', 'decision_cache', 'oc_root', 'rw_root')}), gpu['uuid'])
            for key in ('RUNWEAVER_OPENJEV_CACHE', 'OPENCOMPASS_DECISION_CACHE', 'PYTHONPATH'):
                env.pop(key)
            head, bundle = root / 'head_config.json', root / 'bundle'
            durable(head, plan['head_config'])
            processes.run('export', [plan['serve_python'], '-B', '-m', 'runtime.export_bundle',
                '--run', plan['checkpoint'], '--out', str(bundle), '--name', plan['run_id'],
                '--base', plan['base'], '--base-revision', plan['base_revision'], '--head-config', str(head),
                '--verification-examples', plan['verification_examples'], '--device', 'cuda'],
                runner.serving_environment(dict(env, PYTHONPATH=plan['code_tree']), plan, root), plan['code_tree'], 1800)
            manifest = runner.contract_at(plan['code_tree']).validate_manifest(bundle, plan['head_config'])
            pins = plan['checkpoint_detail']['pins']
            if (manifest['name'] != plan['run_id'] or manifest['option_isolation'] or
                    manifest['base_revision'] != plan['base_revision'] or manifest['checkpoint_hashes'] !=
                    {name: pins['output/' + name]['sha256'] for name in CHECKPOINT_FILES[:3]}):
                raise ValueError('export does not bind the fixed checkpoint')
            for name, digest in manifest['source_hashes'].items():
                if inputs.get(str(Path(plan['code_tree']) / name)) != digest:
                    raise ValueError('export loader/source binding differs: ' + name)
            for name, digest in LOADER_SHA256.items():
                if name not in EXPORT_ONLY and file_sha(bundle / 'code' / name) != digest:
                    raise ValueError('bundle serves a different loader: ' + name)
            manifest_sha = sha(bundle / 'model_manifest.json')
            head_sha = sha(bundle / 'classification_head.safetensors')
            record.update(bundle_manifest_sha256=manifest_sha, bundle_head_sha256=head_sha, bundle_hybrid=manifest['hybrid'])
            record['completed'].append('export')
            durable(root / 'EXPORT_COMPLETE.json', dict(native_exit=0, bundle_manifest_sha256=manifest_sha,
                                                        bundle_head_sha256=head_sha))
            services = root / 'services'
            services.mkdir()
            endpoints = ['http://127.0.0.1:' + str(p) for p in ports]
            serving, service_envs = [], []
            for i, port in enumerate(ports):
                service_envs.append(runner.serving_environment(dict(env, PYTHONPATH=str(bundle / 'code')), plan, root, i))
                serving.append(processes.start('service' + str(i), [plan['serve_python'], '-B', '-m', 'runtime.serve_bundle',
                    '--bundle', str(bundle), '--model-id', plan['run_id'], '--head-config', str(head),
                    '--port', str(port), '--log-dir', str(services / ('replica' + str(i))), '--device', 'cuda',
                    '--max-state', '8192', '--max-branch', '8192', '--max-packed', '8192'],
                    service_envs[i], bundle, service=True))
            ready_stop = time.monotonic() + processes.remaining(1800)
            health = []
            for i, (proc, port, endpoint) in enumerate(zip(serving, ports, endpoints)):
                while True:
                    processes.assert_services_alive()
                    if time.monotonic() >= ready_stop:
                        raise TimeoutError('serving readiness timed out')
                    try:
                        with urllib.request.urlopen(endpoint + '/health', timeout=min(5, processes.remaining())) as response:
                            state = json.load(response)
                    except (OSError, TimeoutError):
                        time.sleep(0.5)
                        continue
                    if exact(state) != exact(expected_health(plan, bundle, port, manifest_sha, head_sha)):
                        raise ValueError('service health differs from this export and the strict 8192 contract: replica ' + str(i))
                    runner.listener_is_owned(port, proc)
                    health.append(dict(endpoint=endpoint, pid=proc.pid, identity=state,
                                       isolated_state=replica_state(proc, service_envs[i])))
                    break
            if len({value for row in health for value in row['isolated_state'].values()}) != 4 * len(ISOLATED):
                raise ValueError('serving replicas share a writable state directory')
            durable(root / 'SERVICE_READY.json', health)
            record['completed'].append('services')
            worker_env = dict(env, CUDA_VISIBLE_DEVICES='')
            worker_env.pop('PYTHONSAFEPATH', None)  # The worker imports its sibling modules by name.
            processes.run('inference_worker', [plan['mc_python'], '-B', str(HERE / 'inference_worker.py'),
                '--population', plan['population'], '--run-id', plan['run_id'], '--bundle', str(bundle),
                '--endpoints', *endpoints, '--output', str(root / 'worker')], worker_env, root)
            record['worker'] = worker_completion(root / 'worker', plan, manifest_sha)
            record['completed'].append('inference_worker')
            processes.remaining()
            record['served_requests'] = request_witness(root / 'worker/ordered_calls.jsonl',
                [services / ('replica' + str(i)) / 'requests.jsonl' for i in range(4)], endpoints,
                plan['ordered_model_wire_sha256'])
            record['completed'].append('request_witness')
            processes.assert_services_alive()
            for row, proc, port in zip(health, serving, ports):
                runner.listener_is_owned(port, proc)
                with urllib.request.urlopen(row['endpoint'] + '/health', timeout=15) as response:
                    if exact(json.load(response)) != exact(row['identity']):
                        raise ValueError('service identity changed during inference')
            record['replica_triton_cache_files'] = [sum(1 for p in Path(row['isolated_state']['TRITON_CACHE_DIR']).rglob('*')
                                                        if p.is_file()) for row in health]
            if input_identity(plan) != inputs:
                raise ValueError('checkpoint, source, population or worker changed during inference')
            success = True
        except BaseException as exc:
            record['failure'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        finally:
            # Stop only this invocation's process tree, including orphans of wrappers that already exited.
            for sig in old_signals:
                signal.signal(sig, signal.SIG_IGN)
            try:
                record['cleanup'] = processes.cleanup()
            except BaseException as exc:
                record['cleanup'] = dict(complete=False, error=repr(exc))
                success = False
            record['processes'] = processes.records
            try:
                # Whatever the worker left, complete or failed, is preserved and pinned here.
                record['worker_files'] = worker_files(root / 'worker')
                failed = root / 'worker/FAILED.json'
                record['worker_failure'] = read_json(failed) if failed.is_file() else None
            except BaseException as exc:
                record['worker_files_error'] = repr(exc)
                success = False
            try:
                record['gpu_post'] = runner.machine_and_gpu(gpu['uuid'])
            except BaseException as exc:
                record['gpu_post'] = dict(error=repr(exc))
                success = False
            done = success and record['cleanup']['complete']
            record.update(status=COMPLETE if done else 'FAILED', native_exit=0 if done else 1, ended_epoch=time.time())
            durable(root / 'RESULT.json', record)
            for sig, old in old_signals.items():
                signal.signal(sig, old)
        return record['native_exit']
    finally:
        lock.close()


def main(argv=None):
    """Exit 0 complete/prepared, 1 launched and failed (see RESULT.json), 2 invalid inputs,
    3 checkpoint pending, 4 host/GPU/lock/allocation refusal. Exits 2, 3 and 4 create no output directory."""
    args = parser().parse_args(argv)
    try:
        runner = lifecycle(args.evaluation_dir)
        status, plan = prepare(args, runner)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps(dict(status=INVALID, problem=type(exc).__name__ + ': ' + str(exc)), indent=2))
        return 2
    if status != PREPARED or args.prepare_only:
        print(json.dumps(dict(status=status, plan=plan), indent=2))
        return 0 if status == PREPARED else 3
    return execute(plan, runner)


if __name__ == '__main__':
    raise SystemExit(main())
