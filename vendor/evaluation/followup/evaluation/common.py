"""Small stdlib helpers for the retained native scoring workers.

Importing this module reads no plans, launches no processes, and sets no deadlines.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def durable(path, value, exclusive=True):
    path = Path(path)
    target = path if exclusive else path.with_name(path.name + '.tmp')
    with target.open('xb' if exclusive else 'wb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        os.replace(target, path)
    if hasattr(os, 'O_DIRECTORY'):
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def worker_alarm():
    """Use this invocation's parent-supplied bound; never revive an old deadline."""
    epoch = float(os.environ['JEV_WORKER_STOP_EPOCH'])
    mono = float(os.environ['JEV_WORKER_STOP_MONO'])
    seconds = min(epoch - time.time(), mono - time.monotonic())
    if seconds <= 0:
        raise TimeoutError('worker deadline has passed')
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, seconds)
