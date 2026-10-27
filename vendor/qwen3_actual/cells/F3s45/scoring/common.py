"""Finite evaluation processes, input identities and durable original receipts."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def model_candidate(actual_native, outcome):
    """A durable producer status cannot override the actual caller-observed exit."""
    return (actual_native == 0 and outcome.get('native') == 0
            and outcome.get('status') == 'MODEL_EVAL_COMPLETE_PENDING_INDEPENDENT_REVIEW'
            and outcome.get('cleanup', {}).get('complete') is True
            and outcome.get('gpu_post', {}).get('gpu', {}).get('native') == 0
            and outcome.get('gpu_post', {}).get('compute_processes', {}).get('native') == 0)


def durable(path, value, exclusive=True):
    path = Path(path)
    temp = path if exclusive else path.with_name(path.name + '.tmp')
    with temp.open('xb' if exclusive else 'wb') as f:
        f.write((json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode())
        f.flush(); os.fsync(f.fileno())
    if not exclusive:
        os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        tail = raw[raw.rfind(')')+2:].split()
        return dict(pid=pid, start_ticks=int(tail[19]),
                    host=socket.gethostname(),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:
        return None


class Budget:
    """A foreign host's monotonic clock is never used on this host."""
    def __init__(self, global_context, seconds=28800, cleanup=120):
        assert global_context['schema'] == 'jev-paper-supplement-actual-context-v1'
        assert global_context['max_seconds'] == 259200
        assert global_context['deadline_epoch'] - global_context['started_epoch'] == 259200
        self.start_epoch, self.start_mono = time.time(), time.monotonic()
        remaining = global_context['deadline_epoch'] - self.start_epoch
        if remaining <= cleanup:
            raise TimeoutError('original stage stop cannot cover cleanup')
        self.seconds = min(seconds, remaining)
        self.end_epoch = self.start_epoch + self.seconds
        self.end_mono = self.start_mono + self.seconds
        self.cleanup = cleanup
        self.foreign_context = global_context

    def remaining(self, producer=True):
        return min(self.end_epoch-time.time(), self.end_mono-time.monotonic()) - (self.cleanup if producer else 0)

    def gate(self, producer=True):
        if self.remaining(producer) <= 0:
            raise TimeoutError('original evaluation/stage stop reached')

    def timeout(self, cap, producer=True):
        self.gate(producer)
        return min(cap, self.remaining(producer))

    def witness(self):
        return dict(started_epoch=self.start_epoch, started_monotonic=self.start_mono,
                    deadline_epoch=self.end_epoch, deadline_monotonic=self.end_mono,
                    max_seconds=self.seconds, reserved_cleanup_seconds=self.cleanup,
                    foreign_monotonic_used=False, original_stage_context=self.foreign_context)


def worker_alarm():
    """Worker bound is supplied from the original parent budget, never restarted."""
    epoch, mono = float(os.environ['JEV_WORKER_STOP_EPOCH']), float(os.environ['JEV_WORKER_STOP_MONO'])
    seconds = min(epoch-time.time(), mono-time.monotonic())
    if seconds <= 0:
        os._exit(124)
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, seconds)


def verify_map(mapping):
    observed = {}
    for path, expected in mapping.items():
        p = Path(path)
        if not p.is_file() or p.stat().st_size != expected['bytes'] or sha(p) != expected['sha256']:
            raise ValueError('pinned input differs: '+path)
        observed[path] = expected
    return observed


def session_members(session_pid):
    owners = []
    for p in Path('/proc').iterdir():
        if p.name.isdecimal():
            try:
                if os.getsid(int(p.name)) == session_pid:
                    owner = identity(int(p.name))
                    if owner:
                        owners.append(owner)
            except ProcessLookupError:
                pass
    return owners


def stop_session(proc, original_owner, budget, previously_owned=()):
    """Record the owned session's exact identities and signal only matching pidfds."""
    # The session is created by our Popen; unrelated process groups are never signalled.
    if (original_owner.get('pid')!=proc.pid or type(original_owner.get('start_ticks')) is not int
            or not original_owner.get('host') or not original_owner.get('boot_id')):
        return dict(complete=False,reason='original_full_owner_unobserved',owners=[],
                    current_root=identity(proc.pid),root_native=proc.poll(),signals_sent=False)
    current_root=identity(proc.pid)
    if current_root is not None and current_root != original_owner:
        return dict(complete=False, reason='root_identity_changed', owners=[])
    observed=session_members(proc.pid)
    current_root_after=identity(proc.pid)
    if current_root_after is not None and current_root_after != original_owner:
        return dict(complete=False,reason='root_identity_changed_during_scan',owners=[])
    if current_root==original_owner and current_root_after==original_owner:
        owners=observed
    else:
        # Current session-ID match after root exit is insufficient provenance.
        owners=[o for o in observed if o in previously_owned]
        unknown=[o for o in observed if o not in previously_owned]
        if unknown:
            return dict(complete=False,reason='unproven_residual_session_ownership',owners=owners,unknown_current_residuals=unknown)
    handles = []
    try:
        for owner in owners:
            budget.gate(False)
            root_now=identity(proc.pid)
            if root_now is not None and root_now!=original_owner:
                return dict(complete=False,reason='root_identity_changed_before_pidfd',owners=owners)
            if identity(owner['pid']) != owner:
                continue
            fd = os.pidfd_open(owner['pid'])
            if identity(owner['pid']) == owner and os.getsid(owner['pid']) == proc.pid:
                handles.append((owner, fd))
            else:
                os.close(fd)
        for owner, fd in handles:
            try:
                signal.pidfd_send_signal(fd, signal.SIGTERM)
            except ProcessLookupError:
                pass
        wait_stop = time.monotonic()+min(4, budget.timeout(4, False))
        while time.monotonic() < wait_stop and any(identity(o['pid']) == o for o, _ in handles):
            proc.poll(); time.sleep(.05)
        for owner, fd in handles:
            if identity(owner['pid']) == owner:
                signal.pidfd_send_signal(fd, signal.SIGKILL)
        try:
            proc.wait(timeout=budget.timeout(4, False))
        except subprocess.TimeoutExpired:
            pass
        states = [dict(owner=o, current=identity(o['pid']), gone=identity(o['pid']) != o)
                  for o in owners]
        residual = session_members(proc.pid)
        return dict(complete=all(x['gone'] for x in states) and not residual,
                    owners=states, residual_current_session=residual, root_native=proc.returncode)
    finally:
        for _, fd in handles:
            os.close(fd)


class Processes:
    def __init__(self, root, budget):
        self.root, self.budget, self.records = Path(root), budget, []
        self.live = []

    def capture(self):
        for proc,owner,_,_,record in self.live:
            if identity(proc.pid)==owner:
                members=session_members(proc.pid)
                if identity(proc.pid)==owner:
                    for member in members:
                        if member not in record['known_owned_members']:
                            record['known_owned_members'].append(member)

    def start(self, name, argv, env, cwd, cap):
        timeout = self.budget.timeout(cap)
        e = dict(env, JEV_WORKER_STOP_EPOCH=str(time.time()+timeout),
                 JEV_WORKER_STOP_MONO=str(time.monotonic()+timeout))
        # Explicit files retain stdout and stderr independently.
        out = (self.root/(name+'.stdout')).open('xb')
        err = (self.root/(name+'.stderr')).open('xb')
        try:
            proc = subprocess.Popen(argv, cwd=cwd, env=e, stdout=out, stderr=err, start_new_session=True)
        except BaseException:
            out.close(); err.close(); raise
        owner = identity(proc.pid)
        if owner is None:
            owner = dict(pid=proc.pid, start_ticks=None, host=socket.gethostname(), boot_id=None)
        record = dict(name=name, argv=argv, cwd=str(cwd), owner=owner,
                      known_owned_members=[owner] if owner['start_ticks'] is not None else [],
                      started_epoch=time.time(), started_monotonic=time.monotonic(), cap_seconds=timeout,
                      worker_stop_epoch=float(e['JEV_WORKER_STOP_EPOCH']),
                      worker_stop_monotonic=float(e['JEV_WORKER_STOP_MONO']))
        self.records.append(record); self.live.append((proc, owner, out, err, record))
        return proc, record

    def wait(self, proc, record):
        timeout = min(record['worker_stop_epoch']-time.time(), record['worker_stop_monotonic']-time.monotonic(), self.budget.remaining())
        if timeout <= 0:
            raise TimeoutError(record['name']+' original cap reached')
        while True:
            self.capture()
            timeout=min(record['worker_stop_epoch']-time.time(),record['worker_stop_monotonic']-time.monotonic(),self.budget.remaining())
            if timeout<=0:raise TimeoutError(record['name']+' original cap reached')
            try:native=proc.wait(timeout=min(.25,timeout));break
            except subprocess.TimeoutExpired:continue
        record.update(native=native, ended_epoch=time.time(), ended_monotonic=time.monotonic())
        if native != 0:
            raise RuntimeError(record['name']+' native '+str(native))
        owner=record['owner']
        if type(owner.get('start_ticks')) is not int or not owner.get('boot_id') or not owner.get('host'):
            raise RuntimeError(record['name']+' original full child identity unobserved; native retained')
        self.budget.gate()
        return native

    def run(self, name, argv, env, cwd, cap):
        proc, record = self.start(name, argv, env, cwd, cap)
        return self.wait(proc, record)

    def cleanup(self):
        results = []
        self.capture()
        for proc, owner, out, err, record in reversed(self.live):
            try:
                cleanup = stop_session(proc, owner, self.budget,record['known_owned_members'])
            except BaseException as exc:
                cleanup = dict(complete=False, error=repr(exc))
            record.update(native=proc.poll(), cleanup=cleanup)
            for f in (out, err):
                self.budget.gate(False)
                f.flush(); os.fsync(f.fileno()); f.close()
            self.budget.gate(False)
            record['stdout_sha256'] = sha(self.root/(record['name']+'.stdout'))
            record['stderr_sha256'] = sha(self.root/(record['name']+'.stderr'))
            self.budget.gate(False)
            results.append(cleanup)
        return dict(complete=all(r['complete'] for r in results), processes=results)
