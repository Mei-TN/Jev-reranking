"""Only relocate writable serving state; no scientific/runtime-library overrides."""
from pathlib import Path


def isolate_replica_environment(environment, work, replica):
    if type(replica) is not int or replica not in range(4):
        raise ValueError('replica must be 0..3')
    state = Path(work) / 'runtime-state' / 'serving' / ('replica' + str(replica))
    env = dict(environment)
    for field, name in (('HOME', 'home'), ('TMPDIR', 'tmp'),
                        ('XDG_CACHE_HOME', 'cache'), ('TRITON_CACHE_DIR', 'triton')):
        env[field] = str(state / name)
    return env
