#!/usr/bin/env python3
"""Evaluate 1,954 direct and 5,862 NLI SocialIQA requests for one fixed model.

Serve one bundle through four cache-isolated replicas. --bundle-mode reuse
validates an existing bundle against the checkpoint and source; fresh exports
a new bundle. --prepare-only checks inputs on CPU without creating output.

Exit codes: 0 complete/prepared; 1 execution failed (see RESULT.json);
2 invalid inputs; 3 pending data, training, or bundle; 4 resource unavailable.
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

import e10_contract as c
import e10_models as models
from e10_worker import OUTPUTS as WORKER_OUTPUTS, expected_health

EVALUATION_DIRS = (HERE.parent / 'evaluation',)
# Shared process cleanup and per-replica cache handling.
LIFECYCLE_SHA256 = {
    'run_evaluation.py': '2960be8ea6921784d3842257e6d06dce4061571f9328452a1ff639df9230369c',
    'replica_cache.py': 'fa59602af179378c6cf462e55ac6666ff3157d80e570ca27607a6c584c4e24a8',
    'common.py': '313ff2d1e247536e1cfa49d74eba7507b11bb70bf49e61b0759df90975396fd5'}
SOURCES = ('run_e10.py', 'e10_worker.py', 'e10_contract.py', 'e10_models.py')
LOCK_PREFIX = '/tmp/jev-supplement-eval-gpu-'  # The full evaluation's per-GPU lock: one owner per GPU across all runners.
ISOLATED = ('HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR')
PLAN_SCHEMA, RESULT_SCHEMA = 'jev-e10-socialiqa-single-model-plan-v1', 'jev-e10-socialiqa-result-v1'
PREPARED, INVALID, NOT_LAUNCHED = 'CPU_PREPARED_NOT_LAUNCHED', 'INVALID_NOT_LAUNCHED', 'NOT_LAUNCHED'
DATA_PENDING, MODEL_PENDING, BUNDLE_PENDING = ('DATA_PENDING_NOT_LAUNCHED', 'MODEL_PENDING_NOT_LAUNCHED',
                                               'BUNDLE_PENDING_NOT_LAUNCHED')
COMPLETE = 'E10_MODEL_COMPLETE_PENDING_INDEPENDENT_REVIEW'
STAGES = dict(fresh=['export', 'bundle', 'services', 'e10_worker', 'request_witness'],
              reuse=['bundle', 'services', 'e10_worker', 'request_witness'])


def lifecycle(directory=None):
    """Import the shared evaluation lifecycle after verifying its source digest."""
    for root in ([Path(directory)] if directory else EVALUATION_DIRS):
        if (root / 'run_evaluation.py').is_file():
            break
    else:
        raise ValueError('reviewed evaluation directory not found; pass --evaluation-dir')
    root = root.resolve()
    for name, expected in LIFECYCLE_SHA256.items():
        if not (root / name).is_file() or c.file_sha(root / name) != expected:
            raise ValueError('evaluation lifecycle differs from the reviewed bytes: ' + name)
    sys.path.insert(0, str(root))
    import run_evaluation
    if Path(run_evaluation.__file__).resolve().parent != root:
        raise ValueError('a different run_evaluation module is already imported')
    return run_evaluation


def allocation(paths, index):
    """Every named campaign yields a GPU only while its CONTROL.json lists it in exclude_gpus."""
    rows = []
    for path in paths:
        control = c.read_json(path)
        excluded = control.get('exclude_gpus') if isinstance(control, dict) else None
        if not isinstance(excluded, list) or not all(type(i) is int for i in excluded):
            raise ValueError('allocation control needs an explicit integer exclude_gpus list: ' + str(path))
        if index not in excluded:
            raise ValueError('GPU ' + str(index) + ' is not excluded from training by ' + str(path) + '; refusing to use it')
        rows.append(dict(path=str(path), sha256=c.file_sha(path), exclude_gpus=excluded))
    return rows


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-id', required=True, help='one of the sixteen fixed E10/E11 identities (MODEL_MAP.json)')
    ap.add_argument('--gpu', required=True, help='one index 0..7 or a complete GPU UUID; never chosen automatically')
    ap.add_argument('--output', required=True, help='new directory; an attempt is never overwritten or resumed')
    ap.add_argument('--bundle-mode', required=True, choices=('reuse', 'fresh'))
    ap.add_argument('--reuse-evaluation', help='output directory of the finished standard full evaluation of this checkpoint')
    ap.add_argument('--reuse-exit-receipt', help="the evaluation queue's EXIT.json of that evaluation (its actual native exit)")
    ap.add_argument('--allocation-control', action='append', default=[],
                    help='campaign CONTROL.json whose exclude_gpus must list the GPU; repeat for every live campaign (required to launch)')
    ap.add_argument('--prepare-only', action='store_true', help='CPU input validation; print plan, write and start nothing')
    ap.add_argument('--population', default=str(HERE / 'sealed_v1'), help='the sealed population; default: beside this file')
    ap.add_argument('--campaign-plan', help='default: the PLAN.json of the campaign that trained this model')
    ap.add_argument('--code-tree', help='default: the source tree of the campaign that trained this model')
    ap.add_argument('--evaluation-dir', help='directory of the reviewed run_evaluation.py; default: sibling or deployed copy')
    ap.add_argument('--port-base', type=int, help='four consecutive ports; default 32500 + 8 * selected GPU index')
    ap.add_argument('--timeout-seconds', type=int, default=10800, help='budget of this invocation, default 3 hours')
    ap.add_argument('--worker-python', default=sys.executable, help='stdlib-only worker interpreter; default: this one')
    for key in ('base', 'serve-python', 'verification-examples', 'hf-cache'):
        ap.add_argument('--' + key, help='default: the completed evaluation contract')
    return ap


def interpreter_can_own_processes():
    """Provide pidfd support for interpreters built against older kernel headers."""
    return hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal')


def loopback_direct():
    """Health checks and every worker request must reach 127.0.0.1 itself, not an inherited HTTP proxy."""
    return 'http' not in urllib.request.getproxies() or bool(urllib.request.proxy_bypass('127.0.0.1'))


def prepare(args, runner):
    if sys.platform == 'linux' and not interpreter_can_own_processes():
        raise ValueError('this interpreter has no os.pidfd_open, so the process lifecycle cannot start: '
                         'run the controller with /usr/bin/python3, not ' + sys.executable)
    model = models.model_map().get(args.run_id)
    if model is None:
        raise ValueError('run-id is not one of the sixteen fixed E10/E11 identities')
    if not (re.fullmatch(r'[0-7]', args.gpu) or re.fullmatch(r'GPU-[0-9a-fA-F-]+', args.gpu)):
        raise ValueError('gpu must be one index 0..7 or a complete GPU UUID')
    if args.timeout_seconds <= 0 or (args.port_base is not None and not 1024 <= args.port_base <= 65532):
        raise ValueError('timeout must be positive and port-base must permit four unprivileged ports')
    if not args.prepare_only and not args.allocation_control:
        raise ValueError('launch requires --allocation-control with this GPU in exclude_gpus')
    reuse = args.bundle_mode == 'reuse'
    if reuse != bool(args.reuse_evaluation) or reuse != bool(args.reuse_exit_receipt):
        raise ValueError('--reuse-evaluation and --reuse-exit-receipt belong to --bundle-mode reuse, and only to it')
    source = Path(args.code_tree or model['code_tree']).expanduser().resolve()
    population_root, output = Path(args.population).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if output.exists():
        raise ValueError('output already exists; preserve attempts and choose a new path')
    plan = {key: getattr(args, key) or runner.DEFAULTS[key] for key in ('base', 'serve_python', 'verification_examples', 'hf_cache')}
    for key in ('base', 'verification_examples', 'hf_cache'):
        plan[key] = str(Path(plan[key]).expanduser().resolve())
    for key, value in (('serve_python', plan['serve_python']), ('worker_python', args.worker_python)):
        # Keep a venv's lexical bin/python: resolving its symlink silently drops the role environment.
        plan[key] = os.path.abspath(os.path.expanduser(value))
        if not Path(plan[key]).is_file():
            raise ValueError('missing interpreter: ' + key + '=' + plan[key])
    gates, pin = {}, c.population_pin()
    # Validate the sealed population and token preflight.
    population = None
    if pin is None:
        gates['data'] = dict(state='PENDING', reason='no POPULATION_PIN.json: the final audit mask and token preflight are not sealed')
    else:
        population, requests, _ = c.load_population(population_root, pin)
        gates['data'] = dict(state='SEALED', population_sha256=pin['population_sha256'], mask_counts=population['mask_counts'])
    # Validate this model's training completion.
    state, training = models.training_state(model, args.campaign_plan)
    gates['training'] = dict(state=state, **({} if state == 'VERIFIED' else training))
    bundle = None
    if state == 'VERIFIED':
        if Path(training['base']).resolve() != Path(plan['base']).resolve():
            raise ValueError('saved base path does not match the export base path')
        models.source_state(source, model['source_identity'])
        if any(output.is_relative_to(root) for root in (Path(training['checkpoint']).parent, source, population_root)):
            raise ValueError('output must be outside the checkpoint, code tree and population')
        if reuse:
            # Validate the bundle against its checkpoint and source.
            state, bundle = models.reuse_state(Path(args.reuse_evaluation).expanduser().resolve(),
                                               Path(args.reuse_exit_receipt).expanduser().resolve(), model, training, source,
                                               runner.contract_at(source))
            gates['bundle'] = dict(state=state, **({} if state == 'VERIFIED' else bundle))
            if output.is_relative_to(Path(args.reuse_evaluation).expanduser().resolve()):
                raise ValueError('output must be outside the reused evaluation')
        else:
            if not Path(plan['verification_examples']).is_file() or not (Path(plan['base']) / 'config.json').is_file() or \
                    not Path(plan['hf_cache']).is_dir():
                raise ValueError('missing export assets: verification examples, base model or HF cache')
            gates['bundle'] = dict(state='FRESH_EXPORT_AT_LAUNCH')
    served = bundle['name'] if reuse and gates.get('bundle', {}).get('state') == 'VERIFIED' else None if reuse else args.run_id
    plan.update(model)
    plan.update(
        schema=PLAN_SCHEMA, checkpoint=training.get('checkpoint'), training=training if gates['training']['state'] == 'VERIFIED' else None,
        campaign_plan=str(Path(args.campaign_plan or model['campaign_plan'])),
        code_tree=str(source), source_sha256=models.SOURCES[model['source_identity']], bundle_mode=args.bundle_mode,
        reused_bundle=bundle if served and reuse else None, served_model_id=served, population=str(population_root),
        population_pin=pin, requests=c.REQUESTS, interface_requests=dict(direct=c.QUESTIONS, nli=3 * c.QUESTIONS),
        qualified_model_wire_sha256=population['ordered_model_wire_sha256'][args.run_id] if population else None,
        served_model_wire_sha256=c.stream_sha(requests, served) if population and served else None, replicas=4,
        package_sha256={name: c.file_sha(HERE / name) for name in SOURCES}, controller_python=sys.executable,
        evaluation_dir=str(Path(runner.__file__).resolve().parent), lifecycle_sha256=LIFECYCLE_SHA256, gpu=args.gpu,
        output=str(output), allocation_control=[str(Path(p).expanduser().resolve()) for p in args.allocation_control],
        timeout_seconds=args.timeout_seconds, port_base=args.port_base, serve_overlay=None,
        service_limits=dict(max_state=c.LIMIT, max_branch=c.LIMIT, max_packed=c.LIMIT), retries=0,
        loopback_direct=loopback_direct(), gates=gates)
    if served == args.run_id and population and plan['served_model_wire_sha256'] != plan['qualified_model_wire_sha256']:
        raise ValueError('rebuilt wire stream differs from the audited stream of this model')
    if plan['allocation_control'] and re.fullmatch(r'[0-7]', args.gpu):
        plan['allocation'] = allocation(plan['allocation_control'], int(args.gpu))
    status = (DATA_PENDING if gates['data']['state'] != 'SEALED' else MODEL_PENDING if gates['training']['state'] != 'VERIFIED'
              else BUNDLE_PENDING if gates['bundle']['state'] == 'PENDING' else PREPARED)
    return status, plan


def input_identity(plan):
    checkpoint, source = Path(plan['checkpoint']), Path(plan['code_tree'])
    paths = [checkpoint / name for name in models.MODEL_FILES + ('training_config.json', 'training_metrics.json')]
    paths += [checkpoint.parent / 'EXIT.json', checkpoint.parent / 'RESULT.json', Path(plan['training']['plan'])]
    paths += [p for folder in ('kev', 'runtime') for p in (source / folder).glob('*.py')]
    paths += [Path(plan['population']) / 'POPULATION.json', c.PIN] + [HERE / name for name in SOURCES]
    if plan['bundle_mode'] == 'reuse':
        reused = plan['reused_bundle']
        paths += [Path(reused['evaluation']) / name for name in ('RESULT.json', 'PLAN.json', 'bundle/model_manifest.json',
                                                                 'bundle/classification_head.safetensors')]
        paths.append(Path(reused['evaluation_exit_receipt']))
    else:
        paths.append(Path(plan['verification_examples']))
    return {str(p): c.file_sha(p) for p in sorted(set(paths))}


def exact(value):
    """Canonical JSON: tells true from 1 and exposes any extra or missing field."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


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
    return {p.name: dict(bytes=p.stat().st_size, sha256=c.file_sha(p)) for p in sorted(directory.iterdir()) if p.is_file()}


def worker_completion(directory, plan, manifest_sha):
    """Bind the worker's own completion record to this plan, bundle and the files it names. No score is read."""
    directory = Path(directory)
    if (directory / 'FAILED.json').exists() or not (directory / 'COMPLETE.json').is_file():
        raise ValueError('worker did not publish a completion record')
    complete, binding = c.read_json(directory / 'COMPLETE.json'), c.read_json(directory / 'RUN_BINDING.json')
    pin = plan['population_pin']['population_sha256']
    if (complete['status'] != 'COMPLETE' or complete['run_id'] != plan['run_id'] or complete['cohort'] != plan['cohort'] or
            complete['served_model_id'] != plan['served_model_id'] or complete['requests'] != c.REQUESTS or
            complete['interface_requests'] != plan['interface_requests'] or complete['questions'] != c.QUESTIONS or
            complete['population_sha256'] != pin or complete['manifest_sha256'] != manifest_sha or
            complete['head_config'] != plan['head_config'] or not set(WORKER_OUTPUTS) <= set(complete['files'])):
        raise ValueError('worker completion does not bind this model, bundle and population')
    for name, expected in complete['files'].items():
        if c.file_sha(directory / name) != expected:
            raise ValueError('worker output changed after completion: ' + name)
    if (binding['worker_sha256'] != plan['package_sha256']['e10_worker.py'] or binding['retries'] != 0 or
            binding['contract_sha256'] != plan['package_sha256']['e10_contract.py'] or
            binding['manifest_sha256'] != manifest_sha or binding['population_sha256'] != pin):
        raise ValueError('worker run binding differs from this package and bundle')
    metrics = c.read_json(directory / 'metrics.json')
    # Denominators only: the full source stays 1954 per interface whatever was answered.
    if set(metrics['interfaces']) != set(c.INTERFACES) or \
            any(row['full_source']['n_questions'] != c.QUESTIONS for row in metrics['interfaces'].values()):
        raise ValueError('worker metrics do not cover both interfaces over the full source')
    return dict(status='COMPLETE', requests=c.REQUESTS, complete_sha256=c.file_sha(directory / 'COMPLETE.json'),
                **{name.split('.')[0] + '_sha256': complete['files'][name] for name in
                   ('metrics.json', 'question_scores.jsonl', 'ordered_calls.jsonl')})


def request_witness(calls, service_logs, endpoints, served_wire_sha, total=c.REQUESTS):
    """Every worker call occurrence must be exactly one logged request/response of the replica it names.

    Occurrences are counted, never deduplicated, and the logs may hold nothing else. Acceptance depends on
    which requests were sent and served, never on what the model answered."""
    occurrence = lambda event: c.digest(c.wire([event['request'], event['response']]))
    planned, stream, count = [Counter() for _ in endpoints], hashlib.sha256(), 0
    interfaces = Counter()
    with Path(calls).open(encoding='utf-8') as lines:
        for line in lines:
            event = json.loads(line)
            if event['ordinal'] != count or event.get('error'):
                raise ValueError('worker call ledger is not the complete ordered population')
            count += 1
            stream.update(c.wire(event['request']) + b'\n')
            planned[endpoints.index(event['endpoint'])][occurrence(event)] += 1
            interfaces[event['interface']] += 1
    served = []
    for path in service_logs:
        counts = Counter()
        with Path(path).open(encoding='utf-8') as lines:
            for line in lines:
                counts[occurrence(json.loads(line))] += 1
        served.append(counts)
    if count != total or stream.hexdigest() != served_wire_sha:
        raise ValueError('worker calls are not the sealed ordered request population of this served model')
    if served != planned:
        raise ValueError('service request logs differ from the worker call occurrences')
    return dict(requests=count, interface_requests=dict(interfaces), per_replica=[sum(n.values()) for n in served],
                exact_request_response_occurrences_equal=True, served_model_wire_sha256=stream.hexdigest())


def execute(plan, runner):
    durable, sha = runner.durable, runner.sha
    root, lock, reuse = Path(plan['output']), None, plan['bundle_mode'] == 'reuse'
    model = models.model_map()[plan['run_id']]
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
        ports = [(plan['port_base'] or (32500 + 8 * gpu['index'])) + i for i in range(4)]
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
        record = dict(schema=RESULT_SCHEMA, run_id=plan['run_id'], cohort=plan['cohort'], bundle_mode=plan['bundle_mode'],
                      status='RUNNING', native_exit=None, started_epoch=time.time(), gpu_pre=gpu, completed=[],
                      plan_sha256=sha(root / 'PLAN.json'), scientific_acceptance=None)
        durable(root / 'ATTEMPT.json', record)
        success, old_signals = False, {}
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
            head, contract = root / 'head_config.json', runner.contract_at(plan['code_tree'])
            durable(head, plan['head_config'])
            if reuse:
                bundle = Path(plan['reused_bundle']['path'])
            else:
                bundle = root / 'bundle'
                processes.run('export', [plan['serve_python'], '-B', '-m', 'runtime.export_bundle',
                    '--run', plan['checkpoint'], '--out', str(bundle), '--name', plan['run_id'],
                    '--base', plan['base'], '--base-revision', models.BASE_REVISION, '--head-config', str(head),
                    '--verification-examples', plan['verification_examples'], '--device', 'cuda'],
                    runner.serving_environment(dict(env, PYTHONPATH=plan['code_tree']), plan, root), plan['code_tree'], 1800)
                record['completed'].append('export')
            # Both modes: every bound bundle file, the checkpoint and source it names, and the trained head tensors.
            state = models.bundle_state(bundle, model, plan['served_model_id'], plan['training'], plan['code_tree'], contract)
            if reuse and exact(state) != exact({key: plan['reused_bundle'][key] for key in state}):
                raise ValueError('reused bundle changed since it was prepared')
            manifest_sha, head_sha = state['manifest_sha256'], state['head_sha256']
            record.update(bundle=state, bundle_manifest_sha256=manifest_sha, bundle_head_sha256=head_sha)
            record['completed'].append('bundle')
            durable(root / 'BUNDLE_VERIFIED.json', state)
            services = root / 'services'
            services.mkdir()
            endpoints = ['http://127.0.0.1:' + str(p) for p in ports]
            serving, service_envs = [], []
            for i, port in enumerate(ports):
                service_envs.append(runner.serving_environment(dict(env, PYTHONPATH=str(bundle / 'code')), plan, root, i))
                serving.append(processes.start('service' + str(i), [plan['serve_python'], '-B', '-m', 'runtime.serve_bundle',
                    '--bundle', str(bundle), '--model-id', plan['served_model_id'], '--head-config', str(head),
                    '--port', str(port), '--log-dir', str(services / ('replica' + str(i))), '--device', 'cuda',
                    '--max-state', str(c.LIMIT), '--max-branch', str(c.LIMIT), '--max-packed', str(c.LIMIT)],
                    service_envs[i], bundle, service=True))
            ready_stop = time.monotonic() + processes.remaining(1800)
            health = []
            for i, (proc, port, endpoint) in enumerate(zip(serving, ports, endpoints)):
                expected = dict(expected_health(model, plan['served_model_id'], bundle, port, manifest_sha, head_sha), gpu=gpu['uuid'])
                while True:
                    processes.assert_services_alive()
                    if time.monotonic() >= ready_stop:
                        raise TimeoutError('serving readiness timed out')
                    try:
                        with urllib.request.urlopen(endpoint + '/health', timeout=min(5, processes.remaining())) as response:
                            identity = json.load(response)
                    except (OSError, TimeoutError):
                        time.sleep(0.5)
                        continue
                    if exact(identity) != exact(expected):
                        raise ValueError('service health differs from this bundle and the strict 8192 contract: replica ' + str(i))
                    runner.listener_is_owned(port, proc)
                    health.append(dict(endpoint=endpoint, pid=proc.pid, identity=identity,
                                       isolated_state=replica_state(proc, service_envs[i])))
                    break
            if len({value for row in health for value in row['isolated_state'].values()}) != 4 * len(ISOLATED):
                raise ValueError('serving replicas share a writable state directory')
            durable(root / 'SERVICE_READY.json', health)
            record['completed'].append('services')
            worker_env = dict(env, CUDA_VISIBLE_DEVICES='')
            worker_env.pop('PYTHONSAFEPATH', None)
            processes.run('e10_worker', [plan['worker_python'], '-B', str(HERE / 'e10_worker.py'),
                '--population', plan['population'], '--run-id', plan['run_id'], '--served-model-id', plan['served_model_id'],
                '--bundle', str(bundle), '--endpoints', *endpoints, '--output', str(root / 'worker')], worker_env, root)
            record['worker'] = worker_completion(root / 'worker', plan, manifest_sha)
            record['completed'].append('e10_worker')
            processes.remaining()
            record['served_requests'] = request_witness(
                root / 'worker/ordered_calls.jsonl', [services / ('replica' + str(i)) / 'requests.jsonl' for i in range(4)],
                endpoints, plan['served_model_wire_sha256'])
            record['completed'].append('request_witness')
            processes.assert_services_alive()
            for row, proc, port in zip(health, serving, ports):
                runner.listener_is_owned(port, proc)
                with urllib.request.urlopen(row['endpoint'] + '/health', timeout=15) as response:
                    if exact(json.load(response)) != exact(row['identity']):
                        raise ValueError('service identity changed during the run')
            record['replica_triton_cache_files'] = [sum(1 for p in Path(row['isolated_state']['TRITON_CACHE_DIR']).rglob('*')
                                                        if p.is_file()) for row in health]
            if input_identity(plan) != inputs or contract.digest(bundle / 'model_manifest.json') != manifest_sha:
                raise ValueError('checkpoint, source, population, bundle or package changed during the run')
            contract.validate_manifest(bundle, plan['head_config'])  # The served bundle is still the verified bytes.
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
                record['worker_failure'] = c.read_json(failed) if failed.is_file() else None
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
