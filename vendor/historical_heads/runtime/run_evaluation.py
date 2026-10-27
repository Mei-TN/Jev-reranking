"""One isolated sequential JevBench/Kev then MC11 evaluation; no GPU/service ownership."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from .bundle_contract import digest, read_config
from .mc_worker import MC_TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--endpoints', required=True, nargs='+')
    p.add_argument('--head-config', required=True, type=Path)
    p.add_argument('--selection-root', required=True, type=Path)
    p.add_argument('--oc-root', required=True, type=Path)
    p.add_argument('--oc-python', required=True, type=Path)
    p.add_argument('--rw-root', required=True, type=Path)
    p.add_argument('--runtime-cache', required=True, type=Path)
    p.add_argument('--mc-python', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--allow-legacy-h0', action='store_true')
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    expected = read_config(a.head_config)
    implementation = Path(__file__).resolve().parents[1]
    shared = ['--model',a.model,'--head-config',str(a.head_config.resolve()),'--endpoints',*a.endpoints]
    if a.allow_legacy_h0: shared += ['--allow-legacy-h0']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONHASHSEED='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               RUNWEAVER_OPENJEV_CACHE=str(a.runtime_cache.resolve()),
               PYTHONPATH=os.pathsep.join([str(implementation),str(a.oc_root.resolve()),str(a.rw_root.resolve()/'src')]))
    state = dict(model=a.model, head_config=expected, status='running',
                 started_at=time.time(), completed=[], active='opencompass',
                 controller_pid=os.getpid(), host=platform.node(), children=[])
    def save():
        tmp = root/'status.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n'); tmp.replace(root/'status.json')
    def run(name, command):
        state['active']=name; save()
        with (root/(name+'.log')).open('x') as log:
            child = subprocess.Popen(command, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            record = dict(stage=name, pid=child.pid, argv=command, started_at=time.time())
            state['children'].append(record); state['child_pid']=child.pid; save()
            try:
                code = child.wait()
            except BaseException:
                if child.poll() is None: child.terminate()
                code = child.wait()
                record.update(returncode=code, finished_at=time.time(), controller_interrupted=True)
                state.pop('child_pid',None); save()
                raise
            record.update(returncode=code, finished_at=time.time())
        state['last_returncode']=code; state.pop('child_pid',None); save()
        if code: raise subprocess.CalledProcessError(code,command)
    try:
        run('opencompass',[str(a.oc_python.resolve()),'-m','runtime.run_opencompass',*shared,
            '--selection-root',str(a.selection_root.resolve()),'--oc-root',str(a.oc_root.resolve()),
            '--work-dir',str(root/'opencompass')])
        if json.loads((root/'opencompass/completion.json').read_text()).get('complete') is not True:
            raise ValueError('JevBench/Kev did not complete')
        state['completed'].append('opencompass'); save()
        for task in MC_TASKS:
            run(task,[str(a.mc_python.resolve()),'-m','runtime.mc_worker',*shared,
                '--task',task,'--rw-root',str(a.rw_root.resolve()),
                '--runtime-cache',str(a.runtime_cache.resolve()),'--output',str(root/'mc11'/task)])
            summary = json.loads((root/'mc11'/task/'summary.json').read_text())
            if summary.get('complete') is not True or summary.get('head_config') != expected:
                raise ValueError('MC task did not complete with the requested head: '+task)
            state['completed'].append(task); save()
        state.update(status='completed', active=None, finished_at=time.time(),
                     summaries={str(p.relative_to(root)):digest(p) for p in sorted((root/'mc11').glob('*/summary.json'))},
                     opencompass_completion_sha256=digest(root/'opencompass/completion.json'))
        save()
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_at=time.time()); save()
        raise
    print(json.dumps(state), flush=True)


if __name__ == '__main__': main()
