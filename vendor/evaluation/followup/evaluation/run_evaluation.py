#!/usr/bin/env python3
"""Evaluate a NeoHorse checkpoint against a specified training contract.

Export once and serve four replicas for JevBench, Kev, and MC11.
--training-contract selects the required completed training configuration;
without it, require one complete epoch. --prepare-only checks inputs on CPU.
"""
import argparse
from collections import Counter
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

from common import durable, read, sha
from jk_audit import JK_COUNTS
from population import MC_COUNTS, canonical, request_token_digest
from replica_cache import isolate_replica_environment
from scores import collect_scores


HERE = Path(__file__).resolve().parent
WIN3_HOST = os.environ.get('JEV_EVALUATION_HOST', 'configure-site-host')
ASSETS = os.environ.get('JEV_EVALUATION_ASSETS', '/path/to/evaluation-assets')
ENV_ROOT = os.environ.get('JEV_EVALUATION_ENV_ROOT', '/path/to/evaluation-roles')
DEFAULTS = dict(
    base=os.environ.get('JEV_BASE_MODEL', '/path/to/base-model'),
    base_revision='56f0584bb40578a2c33b1b40a08ccd17243ad710',
    serve_python=os.environ.get('JEV_SERVING_PYTHON', '/path/to/serving-role/bin/python'),
    oc_python=ENV_ROOT + '/opencompass/bin/python',
    mc_python=ENV_ROOT + '/openjev-runtime/.venv/bin/python',
    runtime_cache=ENV_ROOT + '/openjev-runtime',
    oc_root=ASSETS + '/code/OpenCompass', rw_root=ASSETS + '/code/RunWeaver',
    selection_root=ASSETS + '/materials/selections',
    decision_cache=ASSETS + '/data/decision-cache', hf_cache=ASSETS + '/data/hf-cache',
    verification_examples=ASSETS + '/models/official-bestcombo-merged-bf16-20260923/verification_examples.json',
    expected_mc_population=str(HERE / 'EXPECTED_MC_POPULATION_V2.json'))
CHECKPOINT_FILES = ('head.pt', 'adapter_model.safetensors', 'adapter_config.json',
                    'training_config.json', 'training_metrics.json')
RUNTIME_FILES = ('__init__.py', 'bundle_contract.py', 'export_bundle.py', 'verify_bundle.py', 'load_bundle.py',
                 'serve_bundle.py', 'run_opencompass.py', 'decision_head_api.py', 'openjev_head.py')
# The only complete trainings a caller may name: epochs, optimizer updates and source-record
# occurrences of F2 (15456 records) and F3 (15442), batch 1, accumulation 8. Every epoch ends with
# a flush of its last accumulation group, so two epochs are twice one epoch (F3: 3862, not 3861).
TRAINING_CONTRACTS = {
    'one_epoch_F2': dict(epochs=1, optimizer_steps=1932, source_records=15456),
    'one_epoch_F3': dict(epochs=1, optimizer_steps=1931, source_records=15442),
    'two_epoch_F2': dict(epochs=2, optimizer_steps=3864, source_records=30912),
    'two_epoch_F3': dict(epochs=2, optimizer_steps=3862, source_records=30884)}


def contract_at(source):
    """Load only the known stdlib artifact contract, never torch/kev or a controller."""
    spec = importlib.util.spec_from_file_location('evaluation_bundle_contract', Path(source) / 'runtime/bundle_contract.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_head(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith('{'):
        return json.loads(value)
    if isinstance(value, str) and value:
        return read(value)
    return None


def checkpoint_metadata(checkpoint, source, explicit_head=None, training_contract=None):
    for name in CHECKPOINT_FILES:
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError('missing/nonempty checkpoint member: ' + name)
    config, metrics = read(checkpoint / 'training_config.json'), read(checkpoint / 'training_metrics.json')
    steps = metrics.get('planned_optimizer_steps')
    if (metrics.get('status') != 'complete' or metrics.get('stopped_early') is not False or
            type(steps) is not int or steps <= 0 or metrics.get('optimizer_steps') != steps or
            config.get('dse', {}).get('planned_optimizer_steps') != steps):
        raise ValueError('checkpoint is not a complete full-horizon training run')
    args = config.get('args', {})
    if training_contract is None:
        if args.get('epochs') != 1:
            raise ValueError('this campaign requires one full epoch and FP32 master LoRA training')
    else:
        # A named contract replaces "one epoch" by that training exactly; it never widens it.
        fixed = TRAINING_CONTRACTS[training_contract]
        records = [metrics.get('source_records_seen'), metrics.get('requested_records'),
                   config.get('dse', {}).get('requested_source_records')]
        if (type(args.get('epochs')) is not int or args['epochs'] != fixed['epochs'] or
                steps != fixed['optimizer_steps'] or
                not all(type(n) is int and n == fixed['source_records'] for n in records)):
            raise ValueError('checkpoint is not the complete ' + training_contract + ' training: ' + str(fixed))
    if args.get('full_finetune', False) or args.get('weights_dtype', 'fp32') != 'fp32':
        raise ValueError('this campaign requires one full epoch and FP32 master LoRA training')
    contract = contract_at(source)
    saved = parse_head(config.get('head_config')) or parse_head(args.get('head_config'))
    if saved is None:
        raise ValueError('training_config must declare head_config explicitly')
    head = contract.config_identity(saved)
    if explicit_head and contract.config_identity(parse_head(explicit_head)) != head:
        raise ValueError('--head-config does not match the actual saved training config')
    return config, metrics, head


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'code-tree', 'run-id', 'gpu', 'output'):
        ap.add_argument('--' + key, required=True)
    ap.add_argument('--head-config', help='JSON file or object; otherwise use saved training_config.head_config')
    ap.add_argument('--prepare-only', action='store_true', help='CPU input validation; print plan, write nothing; no GPU query/import')
    ap.add_argument('--port-base', type=int, help='four consecutive ports; default 32100 + 8 * selected GPU index')
    ap.add_argument('--timeout-seconds', type=int, default=43200, help='new invocation budget, default 12 hours; no historical deadline')
    ap.add_argument('--serve-overlay', help='optional verified serving library overlay; requires exporter support')
    ap.add_argument('--training-contract', choices=sorted(TRAINING_CONTRACTS),
                    help='the fixed complete training this checkpoint must be; default: one full epoch, as reviewed')
    for key, value in DEFAULTS.items():
        ap.add_argument('--' + key.replace('_', '-'), default=value)
    return ap


def prepare(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,150}', args.run_id):
        raise ValueError('run-id must be a safe single path component')
    if not (re.fullmatch(r'[0-7]', args.gpu) or re.fullmatch(r'GPU-[0-9a-fA-F-]+', args.gpu)):
        raise ValueError('gpu must be one index 0..7 or a complete GPU UUID')
    if args.timeout_seconds <= 0:
        raise ValueError('timeout-seconds must be positive')
    if args.port_base is not None and not 1024 <= args.port_base <= 65532:
        raise ValueError('port-base must permit four unprivileged consecutive ports')
    checkpoint, source, output = (Path(getattr(args, key)).expanduser().resolve() for key in ('checkpoint', 'code_tree', 'output'))
    if output.exists():
        raise ValueError('output already exists; preserve attempts and choose a new path')
    if output.is_relative_to(checkpoint) or output.is_relative_to(source):
        raise ValueError('output must be outside checkpoint and code-tree')
    for name in RUNTIME_FILES:
        if not (source / 'runtime' / name).is_file():
            raise ValueError('code-tree lacks required runtime file: ' + name)
    for name in ('__init__.py', 'model.py', 'heads.py', 'api.py', 'evaluate.py'):
        if not (source / 'kev' / name).is_file():
            raise ValueError('code-tree lacks required kev file: ' + name)
    config, metrics, head = checkpoint_metadata(checkpoint, source, args.head_config, args.training_contract)
    plan = {key: getattr(args, key) for key in DEFAULTS}
    for key in DEFAULTS.keys() - {'base_revision'}:
        # A venv's bin/python is commonly a symlink to a system interpreter.
        # Resolving it changes sys.prefix and silently drops the role environment.
        plan[key] = (os.path.abspath(os.path.expanduser(plan[key])) if key.endswith('_python')
                     else str(Path(plan[key]).expanduser().resolve()))
    if config.get('base_revision') != plan['base_revision']:
        raise ValueError('saved base revision does not match export base revision')
    if Path(config['args']['base']).resolve() != Path(plan['base']).resolve():
        raise ValueError('saved base path does not match export base path')
    for key in ('serve_python', 'oc_python', 'mc_python', 'verification_examples', 'expected_mc_population'):
        if not Path(plan[key]).is_file():
            raise ValueError('missing asset file: ' + key + '=' + plan[key])
    for key in ('base', 'oc_root', 'rw_root', 'runtime_cache', 'selection_root', 'decision_cache', 'hf_cache'):
        if not Path(plan[key]).is_dir():
            raise ValueError('missing asset directory: ' + key + '=' + plan[key])
    required_assets = [Path(plan['base']) / 'config.json', Path(plan['oc_root']) / 'run.py',
        Path(plan['rw_root']) / 'scripts/openjev/data_revisions.json',
        Path(plan['rw_root']) / 'src/agentic_evaluate/benchmarks/openjev_adapter.py',
        Path(plan['runtime_cache']) / 'data/gpqa-official/gpqa_diamond.csv']
    for path in required_assets:
        if not path.is_file():
            raise ValueError('incomplete evaluation assets: ' + str(path))
    for task, n in JK_COUNTS.items():
        selection = read(Path(plan['selection_root']) / (task + '.selection.json'))
        if len(selection['selected_ids']) != n or len(set(selection['selected_ids'])) != n:
            raise ValueError('frozen JK selection count differs: ' + task)
    population = read(plan['expected_mc_population'])
    if set(population['tasks']) != set(MC_COUNTS):
        raise ValueError('expected MC reference must contain exactly eleven tasks')
    for task, (questions, calls) in MC_COUNTS.items():
        if population['tasks'][task]['questions'] != questions or population['tasks'][task]['calls'] != calls:
            raise ValueError('expected MC population differs: ' + task)
    # Prevent accidentally selecting the Qwen reference with a different tokenizer.
    if sha(plan['expected_mc_population']) != sha(HERE / 'EXPECTED_MC_POPULATION_V2.json'):
        raise ValueError('MC reference is not the retained NeoHorse population')
    overlay = str(Path(args.serve_overlay).resolve()) if args.serve_overlay else None
    if overlay:
        if not Path(overlay).is_dir() or 'JEV_VERIFIED_RUNTIME_OVERLAY' not in (source / 'runtime/export_bundle.py').read_text():
            raise ValueError('serving overlay missing, or exporter drops overlay during fresh reload verification')
    plan.update(schema='jev-revision-20261013-single-model-eval-v1', checkpoint=str(checkpoint),
                code_tree=str(source), run_id=args.run_id, gpu=args.gpu, output=str(output),
                head_config=head, planned_optimizer_steps=metrics['planned_optimizer_steps'],
                training_contract=(dict(TRAINING_CONTRACTS[args.training_contract], name=args.training_contract)
                                   if args.training_contract else None),
                timeout_seconds=args.timeout_seconds, port_base=args.port_base, serve_overlay=overlay,
                replicas=4, jk_records=sum(JK_COUNTS.values()), mc_calls=sum(n[1] for n in MC_COUNTS.values()))
    return plan


def input_identity(plan):
    paths = [Path(plan['checkpoint']) / name for name in CHECKPOINT_FILES]
    source = Path(plan['code_tree'])
    paths += [p for folder in ('kev', 'runtime') for p in (source / folder).rglob('*.py') if '__pycache__' not in p.parts]
    paths += [Path(plan['selection_root']) / (task + '.selection.json') for task in JK_COUNTS]
    paths += [Path(plan['expected_mc_population']), Path(plan['verification_examples']),
              Path(plan['rw_root']) / 'scripts/openjev/data_revisions.json']
    return {str(p): sha(p) for p in sorted(set(paths))}


def machine_and_gpu(selector):
    if sys.platform != 'linux' or socket.gethostname() != WIN3_HOST or getpass.getuser() != 'user' or os.geteuid() != 1000:
        raise ValueError('execution is restricted to WIN3 / user uid 1000')
    cmd = ['nvidia-smi', '--id=' + selector, '--query-gpu=index,uuid,name,memory.used', '--format=csv,noheader,nounits']
    query = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=20)
    fields = [part.strip() for part in query.stdout.strip().split(',')]
    if len(fields) != 4 or 'H20' not in fields[2] or int(fields[0]) not in range(8):
        raise ValueError('selected device is not one WIN3 H20')
    if int(fields[3]) >= 500:
        raise ValueError('selected GPU is not empty (<500 MiB required)')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                          check=True, capture_output=True, text=True, timeout=20)
    if any(line.split(',')[0].strip() == fields[1] for line in apps.stdout.splitlines()):
        raise ValueError('selected GPU has an existing compute process')
    return dict(index=int(fields[0]), uuid=fields[1], name=fields[2], memory_mib=int(fields[3]),
                host=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                account=getpass.getuser(), sampled_epoch=time.time(), gpu_query=query.stdout, compute_query=apps.stdout)


def bind_ports(ports):
    sockets = []
    try:
        for port in ports:
            listener = socket.socket()
            sockets.append(listener)
            listener.bind(('127.0.0.1', port))
    finally:
        for listener in sockets:
            listener.close()


def listener_is_owned(port, proc):
    """Reject a raced/foreign listener even if its health response happens to match."""
    if proc.poll() is not None:
        raise RuntimeError('owned service exited')
    inodes = set()
    for row in Path('/proc/net/tcp').read_text().splitlines()[1:]:
        fields = row.split()
        if int(fields[1].split(':')[1], 16) == port and fields[3] == '0A':
            inodes.add(fields[9])
    for fd in Path('/proc/' + str(proc.pid) + '/fd').iterdir():
        try:
            link = os.readlink(fd)
        except FileNotFoundError:
            continue
        if link.startswith('socket:[') and link[8:-1] in inodes:
            return
    raise ValueError('port is not owned by the launched service: ' + str(port))


def adopt_orphans():
    """Make this controller the Linux child subreaper of everything it launches.

    Otherwise a descendant whose wrapper exited first is reparented to init, and
    only a reusable session/group number would still tie it to this invocation.
    """
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'cannot become child subreaper')
    os.close(os.pidfd_open(os.getpid()))  # Fail before any launch if exact signalling is unavailable.


def process_row(pid):
    """Kernel identity of one process; start_ticks tells a recycled PID apart."""
    try:
        raw = Path('/proc/' + str(pid) + '/stat').read_text()
    except OSError:
        return None
    tail = raw[raw.rfind(')') + 2:].split()
    return dict(pid=pid, state=tail[0], ppid=int(tail[1]), sid=int(tail[3]),
                threads=int(tail[17]), start_ticks=int(tail[19]))


def is_live(row):
    """A zombie leader whose other threads still run (and may hold the GPU) is alive."""
    return row is not None and row['state'] != 'X' and (row['state'] != 'Z' or row['threads'] > 1)


def command_line(pid):
    try:
        raw = Path('/proc/' + str(pid) + '/cmdline').read_bytes()
    except OSError:
        return None
    return raw.replace(b'\0', b' ').decode(errors='replace').strip()[:240]


def owned_processes():
    """Live processes whose parent chain reaches this controller, keyed by PID.

    Ancestry is the ownership proof. A recorded PID/PGID/SID is only a number that
    the kernel may hand to an unrelated job of the same user, so it never selects
    a target; an orphan of an exited wrapper is found because it was reparented here.
    """
    children = {}
    for name in os.listdir('/proc'):
        if name.isdecimal():
            row = process_row(int(name))
            if is_live(row):
                children.setdefault(row['ppid'], []).append(row)
    owned, frontier = {}, [process_row(os.getpid())]
    while frontier:
        parent = frontier.pop()
        for row in children.get(parent['pid'], ()):
            # A child cannot predate its parent; this rejects a parent PID recycled mid-scan.
            if row['pid'] not in owned and row['start_ticks'] >= parent['start_ticks']:
                owned[row['pid']] = row
                frontier.append(row)
    return owned


def signal_owned(row, owned, sig):
    """Signal one descendant through a pidfd after re-proving its ancestry.

    The pidfd pins the process before the checks, so neither the target nor an
    ancestor PID recycled since the scan can redirect the signal to another job.
    """
    try:
        fd = os.pidfd_open(row['pid'])
    except OSError:
        return False
    try:
        link = row
        while True:
            now = process_row(link['pid'])
            if not is_live(now) or now['start_ticks'] != link['start_ticks']:
                return False
            if now['ppid'] == os.getpid():
                break
            link = owned.get(now['ppid'])
            if link is None:
                return False
        signal.pidfd_send_signal(fd, sig)
        return True
    except OSError:  # Already gone, or not signallable: it then stays in the residual report.
        return False
    finally:
        os.close(fd)


class Processes:
    """Launch and own this controller's process tree; preserve native exit codes."""
    grace = 15  # Seconds allowed after SIGTERM, then again after SIGKILL.

    def __init__(self, root, seconds):
        adopt_orphans()
        if owned_processes():
            # cleanup() stops every descendant; anything already running here is not this evaluation's.
            raise RuntimeError('evaluation controller must be a dedicated process without other children')
        self.root, self.end = root, time.monotonic() + seconds
        self.children, self.records, self.services = [], [], []

    def remaining(self, cap=None):
        left = self.end - time.monotonic()
        if left <= 0:
            raise TimeoutError('this evaluation invocation reached its timeout')
        return min(left, cap) if cap else left

    def assert_services_alive(self):
        for proc in self.services:
            if proc.poll() is not None:
                raise RuntimeError('owned serving replica exited: pid=' + str(proc.pid) + ', native=' + str(proc.returncode))

    def start(self, name, argv, env, cwd, timeout=None, service=False):
        seconds = self.remaining(timeout)
        environment = dict(env, JEV_WORKER_STOP_EPOCH=str(time.time() + seconds),
                           JEV_WORKER_STOP_MONO=str(time.monotonic() + seconds))
        out = (self.root / (name + '.stdout')).open('xb')
        err = (self.root / (name + '.stderr')).open('xb')
        try:
            proc = subprocess.Popen(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=err, start_new_session=True)
        except BaseException:
            out.close(); err.close()
            raise
        record = dict(name=name, argv=list(map(str, argv)), cwd=str(cwd), pid=proc.pid,
                      started_epoch=time.time(), timeout_seconds=seconds, native_exit=None,
                      cuda_visible_devices=environment.get('CUDA_VISIBLE_DEVICES'),
                      pythonpath=environment.get('PYTHONPATH'), service=service)
        self.children.append((proc, out, err, record))
        self.records.append(record)
        if service:
            self.services.append(proc)
        return proc

    def run(self, name, argv, env, cwd, timeout=None):
        proc = self.start(name, argv, env, cwd, timeout)
        record = self.records[-1]
        stop = time.monotonic() + record['timeout_seconds']
        while proc.poll() is None:
            self.assert_services_alive()
            if time.monotonic() >= stop:
                raise TimeoutError(name + ' timed out')
            time.sleep(0.5)
        record.update(native_exit=proc.returncode, ended_epoch=time.time())
        durable(self.root / (name + '.EXIT.json'), record)
        if proc.returncode:
            raise RuntimeError(name + ' failed with native exit ' + str(proc.returncode))

    def cleanup(self):
        """Stop every live descendant of this controller and report native exits.

        A wrapper may already have exited while its descendants live on, so targets
        come from verified ancestry, not from which Popen leaders are still alive.
        Never killpg a recorded PID, and never pkill by name/GPU.
        """
        active = {id(proc) for proc, *_ in self.children if proc.poll() is None}
        sessions = {proc.pid: record['name'] for proc, _, _, record in self.children}
        signalled = {}
        for sig in (signal.SIGTERM, signal.SIGKILL):
            stop = time.monotonic() + self.grace
            while True:
                owned = owned_processes()
                # Newest first, so consumers stop before the services they query.
                for row in sorted(owned.values(), key=lambda r: r['start_ticks'], reverse=True):
                    key = (row['pid'], row['start_ticks'])
                    if key not in signalled:
                        signalled[key] = dict(pid=row['pid'], start_ticks=row['start_ticks'],
                                              session=sessions.get(row['sid']),
                                              command=command_line(row['pid']), signals=[])
                    if sig.name not in signalled[key]['signals'] and signal_owned(row, owned, sig):
                        signalled[key]['signals'].append(sig.name)
                if not owned or time.monotonic() >= stop:
                    break
                time.sleep(0.1)
        residual = owned_processes()
        stopped = []
        for proc, out, err, record in reversed(self.children):
            if proc.returncode is None:
                try:
                    proc.wait(timeout=0 if residual else 5)
                except subprocess.TimeoutExpired:
                    pass
            stopped_here = id(proc) in active
            record.update(native_exit=proc.returncode, ended_epoch=record.get('ended_epoch', time.time()),
                          stopped_by_cleanup=stopped_here)
            for stream in (out, err):
                stream.flush(); os.fsync(stream.fileno()); stream.close()
            stopped.append(dict(name=record['name'], pid=record['pid'], native_exit=proc.returncode,
                                stopped_by_cleanup=stopped_here))
        return dict(complete=not residual and all(proc.returncode is not None for proc, *_ in self.children),
                    processes=stopped, signalled=[entry for entry in signalled.values() if entry['signals']],
                    residual=[dict(pid=row['pid'], start_ticks=row['start_ticks'], state=row['state'],
                                   command=command_line(row['pid'])) for row in residual.values()])


def environment(plan, uuid):
    env = dict(os.environ)
    # Remove inherited Python overrides and per-job writable state from other launches.
    for key in ('PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONPATH', 'JEV_VERIFIED_RUNTIME_OVERLAY',
                'TRITON_CACHE_DIR', 'JEV_WORKER_STOP_EPOCH', 'JEV_WORKER_STOP_MONO'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=uuid, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONHASHSEED='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64',
               HF_HOME=plan['hf_cache'], RUNWEAVER_OPENJEV_CACHE=plan['runtime_cache'],
               OPENCOMPASS_DECISION_CACHE=plan['decision_cache'],
               PYTHONPATH=os.pathsep.join([plan['code_tree'], str(HERE), plan['oc_root'], str(Path(plan['rw_root']) / 'src')]))
    return env


def serving_environment(env, plan, root, replica=None):
    result = dict(env)
    if plan['serve_overlay']:
        result['JEV_VERIFIED_RUNTIME_OVERLAY'] = plan['serve_overlay']
        result['PYTHONPATH'] += os.pathsep + plan['serve_overlay']
    if replica is None:
        state = root / 'runtime-state/export'
        for key, name in (('HOME', 'home'), ('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'cache'), ('TRITON_CACHE_DIR', 'triton')):
            result[key] = str(state / name)
    else:
        result = isolate_replica_environment(result, root, replica)
    for key in ('HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR'):
        Path(result[key]).mkdir(parents=True, exist_ok=True)
    return result


def request_witness(root, processes):
    planned, actual = Counter(), Counter()
    def event_key(event):
        response = event['response']
        return hashlib.sha256(request_token_digest(event['request'], response['input_tokens']).encode()
                              + b'\n' + canonical(response['probabilities'])).hexdigest()
    consumers = [root / 'jk_audit/ordered_predictions.jsonl'] + [root / 'mc11' / task / 'ordered_model_calls.jsonl' for task in MC_COUNTS]
    served = [root / 'services' / ('replica' + str(i)) / 'requests.jsonl' for i in range(4)]
    for paths, counts in ((consumers, planned), (served, actual)):
        for path in paths:
            processes.remaining()
            with path.open() as stream:
                for line in stream:
                    counts[event_key(json.loads(line))] += 1
    if sum(planned.values()) != 389219 or planned != actual:
        raise ValueError('full served probabilities/encoder tokens differ from native consumers')
    return dict(requests=389219, exact_occurrence_probabilities_and_tokens_equal=True)


def evaluate(plan):
    import fcntl  # Execution-only POSIX dependency; prepare is portable and CPU-only.
    gpu = machine_and_gpu(plan['gpu'])
    lock = open('/tmp/jev-supplement-eval-gpu-' + gpu['uuid'] + '.lock', 'a+')
    root = Path(plan['output'])
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = machine_and_gpu(gpu['uuid'])
        ports = [((plan['port_base'] or (32100 + 8 * gpu['index'])) + i) for i in range(4)]
        bind_ports(ports)
        # Before any output exists: an unsupported host must not leave a RUNNING attempt behind.
        processes = Processes(root, plan['timeout_seconds'])
        plan.update(gpu_uuid=gpu['uuid'], gpu_index=gpu['index'], ports=ports)
        root.mkdir(parents=True, exist_ok=False)
        durable(root / 'PLAN.json', plan)
        record = dict(schema='jev-revision-20261013-evaluation-result-v1', run_id=plan['run_id'],
                      status='RUNNING', native_exit=None, started_epoch=time.time(), gpu_pre=gpu,
                      completed=[], plan_sha256=sha(root / 'PLAN.json'), scientific_acceptance=None)
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
            env = environment(plan, gpu['uuid'])
            head, bundle = root / 'head_config.json', root / 'bundle'
            durable(head, plan['head_config'])
            processes.run('export', [plan['serve_python'], '-B', '-m', 'runtime.export_bundle',
                '--run', plan['checkpoint'], '--out', str(bundle), '--name', plan['run_id'],
                '--base', plan['base'], '--base-revision', plan['base_revision'], '--head-config', str(head),
                '--verification-examples', plan['verification_examples'], '--device', 'cuda'],
                serving_environment(env, plan, root), plan['code_tree'], 1800)
            manifest = contract_at(plan['code_tree']).validate_manifest(bundle, plan['head_config'])
            expected = {name: inputs[str(Path(plan['checkpoint']) / name)] for name in CHECKPOINT_FILES[:3]}
            if manifest['checkpoint_hashes'] != expected:
                raise ValueError('export does not bind actual checkpoint')
            for name, digest in manifest['source_hashes'].items():
                if sha(Path(plan['code_tree']) / name) != digest:
                    raise ValueError('export loader/source binding differs')
            record['bundle_manifest_sha256'] = sha(bundle / 'model_manifest.json')
            record['completed'].append('export')
            durable(root / 'EXPORT_COMPLETE.json', dict(native_exit=0, bundle_manifest_sha256=record['bundle_manifest_sha256']))
            services = root / 'services'
            services.mkdir()
            endpoints = ['http://127.0.0.1:' + str(p) for p in ports]
            service_base_env = dict(env, PYTHONPATH=str(bundle / 'code'))
            serving = []
            for i, port in enumerate(ports):
                proc = processes.start('service' + str(i), [plan['serve_python'], '-B', '-m', 'runtime.serve_bundle',
                    '--bundle', str(bundle), '--model-id', plan['run_id'], '--head-config', str(head),
                    '--port', str(port), '--log-dir', str(services / ('replica' + str(i))), '--device', 'cuda'],
                    serving_environment(service_base_env, plan, root, i), bundle, service=True)
                serving.append(proc)
            ready_stop = time.monotonic() + processes.remaining(1800)
            health = []
            for proc, port, endpoint in zip(serving, ports, endpoints):
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
                    if (state['manifest_sha256'] != record['bundle_manifest_sha256'] or
                            state['model'] != plan['run_id'] or state['head_config'] != plan['head_config']):
                        raise ValueError('service health identity differs from actual export')
                    listener_is_owned(port, proc)
                    health.append(dict(endpoint=endpoint, pid=proc.pid, identity=state))
                    break
            durable(root / 'SERVICE_READY.json', health)
            view = root / 'opencompass_source'
            view.mkdir(); (view / 'tmp').mkdir()
            for item in Path(plan['oc_root']).iterdir():
                if item.name not in ('tmp', '.git', '__pycache__'):
                    os.symlink(item, view / item.name, target_is_directory=item.is_dir())
            cpuenv = dict(env, CUDA_VISIBLE_DEVICES='')
            processes.run('opencompass', [plan['oc_python'], '-B', '-m', 'runtime.run_opencompass',
                '--model', plan['run_id'], '--head-config', str(head), '--endpoints', *endpoints,
                '--selection-root', plan['selection_root'], '--oc-root', str(view),
                '--work-dir', str(root / 'opencompass')], cpuenv, plan['code_tree'], 7200)
            # Audit before MC appends more events to the four service request logs.
            processes.run('jk_audit', [plan['oc_python'], '-B', str(HERE / 'jk_audit.py'), '--plan', str(root / 'PLAN.json'),
                '--work', str(root / 'opencompass'), '--services', str(services), '--output', str(root / 'jk_audit')], cpuenv, root, 300)
            record['completed'].append('seven_JK')
            for task in MC_COUNTS:
                processes.run(task, [plan['mc_python'], '-B', str(HERE / 'mc_worker.py'), '--plan', str(root / 'PLAN.json'),
                    '--task', task, '--endpoints', *endpoints, '--output', str(root / 'mc11' / task)], cpuenv, root)
                record['completed'].append(task)
            record['served_probabilities'] = request_witness(root, processes)
            record['scores'] = collect_scores(root, plan)
            durable(root / 'SCORES.json', record['scores'])
            processes.assert_services_alive()
            for port, proc in zip(ports, serving):
                listener_is_owned(port, proc)
            if input_identity(plan) != inputs:
                raise ValueError('checkpoint, source, or evaluation input changed during evaluation')
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
                record['gpu_post'] = machine_and_gpu(gpu['uuid'])
            except BaseException as exc:
                record['gpu_post'] = dict(error=repr(exc))
                success = False
            record.update(status='MODEL_EVAL_COMPLETE_PENDING_INDEPENDENT_REVIEW' if success and record['cleanup']['complete'] else 'FAILED',
                          native_exit=0 if success and record['cleanup']['complete'] else 1, ended_epoch=time.time())
            durable(root / 'RESULT.json', record)
            for sig, old in old_signals.items():
                signal.signal(sig, old)
        return record['native_exit']
    finally:
        lock.close()


def main():
    args = parser().parse_args()
    plan = prepare(args)
    if args.prepare_only:
        print(json.dumps(dict(status='CPU_PREPARED_NOT_LAUNCHED', plan=plan), indent=2))
        return 0
    return evaluate(plan)


if __name__ == '__main__':
    raise SystemExit(main())
