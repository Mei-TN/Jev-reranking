#!/usr/bin/env python3
"""Bind exact prebuilt role inputs at declared locations; no installation or model execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
def pin(p):
    b=Path(p).read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',required=True,type=Path,help='JSON object mapping the five roles to existing installation roots')
    a=p.parse_args()
    target=ROOT/'RUNTIME_MANIFEST.json';receipt=ROOT/'RUNTIME_BINDING_RECEIPT.json'
    if target.exists() or receipt.exists():raise ValueError('binding already exists; use a new derivative directory')
    observations={};context={'phase':'load_roots'}
    def save_receipt(value):
        with receipt.open('x') as f:
            f.write(json.dumps(value,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    try:
        roots=json.loads(a.roots.read_text())
        template=ROOT/'RUNTIME_MANIFEST.template.json'
        text=template.read_text();original=json.loads(text)
        if set(roots)!=set(original['roles']):raise ValueError('provide exactly the five role roots')
        for role,raw in roots.items():
            context={'phase':'resolve_root','role':role,'path':raw}
            root=Path(raw).resolve(strict=True)
            if not root.is_dir():raise ValueError('runtime root is not a directory')
            text=text.replace('{'+role+'_runtime_root}',str(root))
        manifest=json.loads(text)
        for name,role in manifest['roles'].items():
            root=Path(role['root']);records=[];relocated=[]
            for f in role['files']:
                context={'phase':'bind_file','role':name,'path':f['path']}
                path=Path(f['path']);resolved=path.resolve(strict=True)
                if not path.is_relative_to(root) or not resolved.is_relative_to(root):raise ValueError('role file escapes root')
                actual=pin(path)
                # Only installation-location metadata may change at this explicit binding step.
                location_metadata=path.name=='pyvenv.cfg' or (path.name=='RECORD' and path.parent.name.endswith('.dist-info'))
                if not location_metadata and actual!={k:f[k] for k in ('bytes','sha256')}:
                    raise ValueError('fixed interpreter/module/distribution metadata identity differs: '+str(path))
                if path.name=='pyvenv.cfg':
                    cfg=dict(line.split(' = ',1) for line in path.read_text().splitlines() if ' = ' in line)
                    if cfg.get('include-system-site-packages')!='false':raise ValueError('system site packages must be disabled')
                if actual!={k:f[k] for k in ('bytes','sha256')}:
                    relocated.append({'path':str(path),'original_expected':{k:f[k] for k in ('bytes','sha256')},'actual':actual})
                records.append(dict(path=str(path),resolved_path=str(resolved),**actual))
            role['files']=records
            code='import importlib.metadata as m,json,sys; expected=json.loads(sys.argv[1]); actual={k:m.version(k) for k in expected}; assert actual==expected,(actual,expected); print(json.dumps(actual))'
            env={k:v for k,v in os.environ.items() if k not in ('PYTHONHOME','PYTHONPATH','LD_PRELOAD','LD_AUDIT')}
            env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
            context={'phase':'distribution_metadata','role':name}
            q=subprocess.run([role['python'],'-I','-B','-c',code,json.dumps(role['expected_distributions'])],env=env,text=True,capture_output=True,timeout=120)
            observations[name]={'native_exit':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'relocated_metadata':relocated}
            if q.returncode:raise RuntimeError('role metadata check failed')
        manifest['provenance']={'binding_source':'RUNTIME_MANIFEST.template.json','template_identity':pin(template),
            'runtime_identity_scope':'Fixed interpreter/module-entry/distribution METADATA bytes and exact role versions; location-dependent pyvenv.cfg/RECORD are captured explicitly at binding and checked thereafter.',
            'runweaver_usage':'Pinned fork source is imported in-process by mc_worker under the OpenJev interpreter; no RunWeaver model subprocess is added.',
            'gpu_end_to_end_complete':False}
        context={'phase':'publish_binding'}
        with target.open('x') as f:f.write(json.dumps(manifest,indent=2)+'\n')
        package_file=ROOT/'PACKAGE_MANIFEST.json';package=json.loads(package_file.read_text());package['RUNTIME_MANIFEST.json']=pin(target)
        package_file.write_text(json.dumps(package,indent=2)+'\n')
        save_receipt({'status':'RUNTIME_BINDING_CREATED','manifest_identity':pin(target),'observations':observations,
            'scope':'Actual file identities and installed-distribution metadata only; no model/data/service/scoring execution.'})
    except Exception as e:
        def text(value):return value.decode(errors='replace') if isinstance(value,bytes) else value
        save_receipt({'status':'BINDING_FAILED','context':context,'exception_type':type(e).__name__,'error':str(e),
            'native_timeout':isinstance(e,subprocess.TimeoutExpired),'stdout':text(getattr(e,'stdout',None)),
            'stderr':text(getattr(e,'stderr',None)),'observations':observations})
        raise
    print(json.dumps({'status':'RUNTIME_BINDING_CREATED','roles':list(observations),'receipt':str(receipt)}))
if __name__=='__main__':main()
