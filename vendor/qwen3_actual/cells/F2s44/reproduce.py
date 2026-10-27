#!/usr/bin/env python3
"""Relative-path assembly for one fixed pointer/pooling pair; stdlib control plane."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
import importlib.util

ROOT = Path(__file__).resolve().parent
_binding_spec = importlib.util.spec_from_file_location('portable_runtime_binding', ROOT/'runtime_binding.py')
binding = importlib.util.module_from_spec(_binding_spec)
_binding_spec.loader.exec_module(binding)
VARIANTS = ('pointer', 'pooling')
TARGETS = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
TASKS = ('mmlu_fewshot','mmlu','hellaswag','gsm8k_mc10','arc_easy','gsm8k_mc4',
         'winogrande','arc_challenge','chess','gpqa_fewshot','gpqa')


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(value):
    require(isinstance(value, str) and value, 'expected contained relative path')
    p = PurePosixPath(value)
    require(not p.is_absolute()
            and '..' not in p.parts, 'expected contained relative path')
    return Path(value)


def pin(path):
    h = hashlib.sha256(); count = 0
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            count += len(block); h.update(block)
    return {'bytes': count, 'sha256': h.hexdigest()}


def check_file(path, expected):
    got = pin(path)
    require(got == {k: expected[k] for k in ('bytes','sha256')}, 'asset hash/length mismatch: '+str(path))
    return got


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())


def configuration():
    pair = read(ROOT/'configs/pair.json')
    sources = read(ROOT/'SOURCE_MANIFEST.json')
    for name, expected in read(ROOT/'PACKAGE_MANIFEST.json').items():
        check_file(ROOT/relative(name), expected)
    actual = {str(p.relative_to(ROOT)) for d in ('src','scoring') for p in (ROOT/d).rglob('*.py')}
    require(actual == set(sources), 'source closure inventory changed')
    for name, expected in sources.items():
        path = ROOT/relative(name)
        check_file(path, expected)
        compile(path.read_text(), name, 'exec')  # no import, bytecode write, or model load
    require(pair['schema']=='minimal-readout-pair-v1' and pair['fold'] in (2,3) and pair['seed'] in (44,45),
            'fixed first cell changed')
    require((pair['training_records'],pair['optimizer_steps'])=={2:(15456,1932),3:(15442,1931)}[pair['fold']], 'full-fold size changed')
    require(pair['lora_target_modules']==TARGETS, 'seven dense projection targets changed')
    args = {}
    for variant, readout in zip(VARIANTS, ('end','mean_span')):
        cfg = pair['variants'][variant]
        argv = cfg['training_argv']
        require(argv[:4]==['{training_python}','-P','-m','kev.train'] and len(argv[4:])%2==0,
                'unexpected training entry')
        names=argv[4::2]
        require(len(names)==len(set(names)) and all(n.startswith('--') for n in names),
                'duplicate/malformed training argument')
        opts=dict(zip(names,argv[5::2]))
        expected=dict(dim=256,kind='pointer',readout=readout,rng_anchor_dim=256,version=1)
        require(cfg['head_config']==expected==read(ROOT/f'configs/{variant}.head.json')
                ==json.loads(opts['--head_config']), 'head config differs')
        require(opts['--seed']==str(pair['seed']) and opts['--lora_targets']=='all' and opts['--data']=='{training_jsonl}'
                and opts['--out']=='{training_output}' and opts['--base']=='{base}', 'fixed input/configuration differs')
        args[variant]=opts
    differences={k for k in args['pointer'] if args['pointer'][k]!=args['pooling'][k]}
    require(differences=={'--head_config'}, 'paired training arguments differ beyond readout')
    assets=read(ROOT/'ASSET_MANIFEST.json')
    require(assets['base_revision']==pair['base_revision'], 'base revision mismatch')
    for entry in assets['assets'].values():
        relative(entry['path'])
        require(type(entry['bytes']) is int and entry['bytes']>0
                and len(entry['sha256'])==64, 'invalid asset pin')
    binding.load()  # Explicit replacement identities; original asset pins remain unchanged.
    return pair, assets['assets'], len(sources)


def load_site(path):
    site=read(path)
    require(set(site)=={'asset_root','work_root','cuda_visible_devices','ports','checkpoint_mode','phase_timeout_seconds'},
            'unexpected site fields')
    relative(site['asset_root']); relative(site['work_root'])
    require(site['checkpoint_mode'] in ('train','existing'), 'checkpoint_mode must be train or existing')
    require(len(site['ports'])==len(set(site['ports']))==4
            and all(type(v) is int and 1024<=v<=65535 for v in site['ports']), 'four distinct unprivileged ports required')
    require(type(site['phase_timeout_seconds']) is int and 1<=site['phase_timeout_seconds']<=28800,
            'phase timeout must be in 1..28800 seconds')
    require(isinstance(site['cuda_visible_devices'],str) and site['cuda_visible_devices'], 'explicit CUDA device selector required')
    return site


def assets_for(entries, asset_root, groups):
    runtime = binding.verify(groups)
    selected={k:v for k,v in entries.items() if set(v['groups']) & set(groups)
              and k not in binding.REPLACED_KEYS}
    for entry in selected.values():
        check_file(asset_root/relative(entry['path']), entry)
    return len(selected) + runtime['files_checked']


def dependencies(entries, asset_root, extract=False):
    """Require the original full archives, then verify/extract their regular bytes."""
    assets_for(entries,asset_root,['dependencies'])
    target=asset_root/'dependencies/extracted'
    for name in ('opencompass','runweaver'):
        if extract:
            require(not (target/name).exists(), 'dependency destination already exists')
        archive=asset_root/entries[name+'_archive']['path']
        with tarfile.open(archive,'r:') as tar:
            members=tar.getmembers()
            expected_files=set()
            for member in members:
                rel=relative(member.name)
                require(rel.parts[0]==name and (member.isfile() or member.isdir()), 'unsafe archive member')
                dest=target/rel
                if member.isdir():
                    if extract: dest.mkdir(parents=True,exist_ok=True)
                    continue
                expected_files.add(str(rel))
                source=tar.extractfile(member)
                if extract:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with dest.open('xb') as out:
                        for block in iter(lambda:source.read(1<<20),b''):out.write(block)
                        out.flush();os.fsync(out.fileno())
                else:
                    h=hashlib.sha256();n=0
                    for block in iter(lambda:source.read(1<<20),b''):h.update(block);n+=len(block)
                    check_file(dest,{'bytes':n,'sha256':h.hexdigest()})
            actual_files={str(p.relative_to(target)) for p in (target/name).rglob('*') if p.is_file()}
            require(actual_files==expected_files,'dependency file inventory differs from pinned archive')


def recipe(pair, entries, site, variant, absolute=False):
    base=ROOT if absolute else Path('.')
    asset=base/relative(site['asset_root']); work=base/relative(site['work_root'])/variant
    def ap(key):return str(asset/relative(entries[key]['path']))
    head=base/f'configs/{variant}.head.json'
    train=work/'training'; checkpoint=asset/'checkpoints'/variant if site['checkpoint_mode']=='existing' else train
    bundle=work/'bundle'; services=work/'services'; score=work/'scores'
    source=base/'src'; scoring=base/'scoring'; deps=asset/'dependencies/extracted'
    endpoints=['http://127.0.0.1:'+str(p) for p in site['ports']]
    substitutions={'training_python':binding.python('training'),'base':str(asset/'base'),
                   'training_jsonl':ap('training_jsonl'),'training_output':str(train)}
    training=[substitutions[a[1:-1]] if a.startswith('{') and a.endswith('}') and a[1:-1] in substitutions else a
              for a in pair['variants'][variant]['training_argv']]
    model='QWEN3_'+pair['cell']+'_'+variant
    params=dict(schema='readout-reproduction-runtime-parameters-v1',run_id=model,
                code_tree=str(source),rw_root=str(deps/'runweaver'),runtime_cache=str(asset/'evaluation/runtime-cache'),
                expected_mc_population=ap('expected_mc_population'),selection_root=str(asset/'evaluation/selections'),
                head_config=pair['variants'][variant]['head_config'])
    plan=work/'runtime_parameters.json'
    export=[binding.python('training'),'-B','-m','runtime.export_bundle','--run',str(checkpoint),'--out',str(bundle),
            '--name',model,'--base',str(asset/'base'),'--base-revision',pair['base_revision'],
            '--head-config',str(head),'--verification-examples',ap('verification_examples'),'--device','cuda']
    serving=[[binding.python('serving'),'-B','-m','runtime.serve_bundle','--bundle',str(bundle),'--model-id',model,
              '--head-config',str(head),'--port',str(port),'--log-dir',str(services/f'replica{i}'),'--device','cuda']
             for i,port in enumerate(site['ports'])]
    commands=[['opencompass',[binding.python('opencompass'),'-B','-m','runtime.run_opencompass','--model',model,
               '--head-config',str(head),'--endpoints',*endpoints,'--selection-root',params['selection_root'],
               '--oc-root',str(score/'opencompass_source'),'--work-dir',str(score/'opencompass')]],
              ['jk_audit',[binding.python('opencompass'),'-B',str(scoring/'jk_audit.py'),'--plan',str(plan),
               '--work',str(score/'opencompass'),'--services',str(services),'--output',str(score/'jk_audit')]]]
    commands += [[task,[binding.python('openjev'),'-B',str(scoring/'mc_worker.py'),'--plan',str(plan),'--task',task,
                 '--endpoints',*endpoints,'--output',str(score/'mc11'/task)]] for task in TASKS]
    train_env=dict(QWEN_PAIR_INIT_ROOT=str(ROOT.parents[1]/'shared_initial'/pair['cell']), QWEN_PAIR_ID=pair['cell'], QWEN_VARIANT=variant, QWEN_HARDSTOP_EPOCH='1791887400', CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=site['cuda_visible_devices'],
             PYTHONNOUSERSITE='1',PYTHONSAFEPATH='1',PYTHONDONTWRITEBYTECODE='1',
             TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             MKL_NUM_THREADS='2',OMP_NUM_THREADS='2',LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64',
             PYTHONPATH=str(source))
    eval_env=dict(train_env,PYTHONHASHSEED='0',HF_DATASETS_OFFLINE='1',
             HF_HOME=str(asset/'evaluation/hf-cache'),RUNWEAVER_OPENJEV_CACHE=params['runtime_cache'],
             OPENCOMPASS_DECISION_CACHE=str(asset/'evaluation/decision-cache'),
             PYTHONPATH=os.pathsep.join([str(source),str(scoring),str(deps/'opencompass'),str(deps/'runweaver/src')]))
    environments = dict(train=binding.compose('training',train_env,work,asset),
                        export=binding.compose('training',eval_env,work,asset),
                        serve=binding.compose('serving',dict(eval_env,PYTHONPATH=str(bundle/'code')),work,asset),
                        score={name:binding.compose(binding.role_for_command(name),
                                    dict(eval_env,CUDA_VISIBLE_DEVICES=''),work,asset) for name,_ in commands})
    return dict(variant=variant,checkpoint_mode=site['checkpoint_mode'],cwd=str(base),
                runtime_binding_schema='portable-existing-runtime-binding-v2',
                environment_overrides=environments,
                runtime_parameters=params,parameter_path=str(plan),
                train=training,export=export,serve=serving,score=commands)


def child(command, env, receipt_dir, stop_epoch, stop_mono):
    remaining=min(stop_epoch-time.time(),stop_mono-time.monotonic())
    require(remaining>0,'shared phase deadline exhausted')
    receipt_dir.mkdir(parents=True,exist_ok=False)
    overrides=dict(env,JEV_WORKER_STOP_EPOCH=str(stop_epoch),JEV_WORKER_STOP_MONO=str(stop_mono))
    env=binding.process_environment(overrides)
    save(receipt_dir/'command.json',{'argv':command,'environment_overrides':overrides,
                                  'runtime_manifest':binding.pin(ROOT/'RUNTIME_MANIFEST.json')})
    with (receipt_dir/'stdout').open('xb') as out,(receipt_dir/'stderr').open('xb') as err:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
        try:
            code=process.wait(timeout=remaining)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            save(receipt_dir/'exit.json',{'native_exit':process.returncode,'interrupted':True})
            raise
    save(receipt_dir/'exit.json',{'native_exit':code,'interrupted':False})
    require(code==0,'native process failed; inspect its retained log')


def execute(pair, entries, site, variant, stage, replica):
    asset=ROOT/relative(site['asset_root']);work=ROOT/relative(site['work_root'])/variant
    groups=[stage]
    if stage=='export' and site['checkpoint_mode']=='existing':groups+=['checkpoint_'+variant]
    assets_for(entries,asset,groups)  # before output writes or executable launch
    r=recipe(pair,entries,site,variant,absolute=True);env=r['environment_overrides'][stage]
    if stage != 'score':binding.prepare_state(env,work)
    if stage=='train':
        require(site['checkpoint_mode']=='train','existing-checkpoint mode does not run training')
        require(not (work/'training').exists(),'training destination exists')
    if stage=='export':
        require(not (work/'bundle').exists(),'export destination exists')
        if site['checkpoint_mode']=='train':
            receipt=read(work/'receipts/train/exit.json');require(receipt=={'native_exit':0,'interrupted':False},'training exit is not complete')
            config=read(work/'training/training_config.json');metrics=read(work/'training/training_metrics.json')
            require(config['head_config']==pair['variants'][variant]['head_config'],'trained head differs')
            require(metrics['status']=='complete' and metrics['optimizer_steps']==pair['optimizer_steps']
                    and metrics['source_records_seen']==pair['training_records'] and not metrics['stopped_early'],'full training not complete')
    if stage=='serve':
        require(replica in range(4),'serve needs --replica 0..3')
        require(read(work/'receipts/export/exit.json')=={'native_exit':0,'interrupted':False},'new export did not exit successfully')
        require((work/'bundle/model_manifest.json').is_file(),'new export absent')
        os.chdir(ROOT)
        os.execve(r['serve'][replica][0],r['serve'][replica],binding.process_environment(env))
    seconds=min(site['phase_timeout_seconds'],21600 if stage=='train' else 1800 if stage=='export' else 28800)
    stop_epoch=min(time.time()+seconds,1791887370.0);stop_mono=time.monotonic()+max(0,stop_epoch-time.time())
    if stage!='score':
        child(r[stage],env,work/'receipts'/stage,stop_epoch,stop_mono);return
    dependencies(entries,asset)
    require(not (work/'scores').exists(),'score destination exists')
    manifest=read(work/'bundle/model_manifest.json');bundle_sha=pin(work/'bundle/model_manifest.json')['sha256']
    for port in site['ports']:
        with urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/health',timeout=15) as response:health=json.load(response)
        require(health['model']==r['runtime_parameters']['run_id'] and health['manifest_sha256']==bundle_sha
                and health['head_config']==pair['variants'][variant]['head_config'],'service does not bind this export')
    require(manifest['head_config']==pair['variants'][variant]['head_config'],'bundle head differs')
    save(Path(r['parameter_path']),r['runtime_parameters'])
    view=work/'scores/opencompass_source';view.mkdir(parents=True);(view/'tmp').mkdir()
    for entry in (asset/'dependencies/extracted/opencompass').iterdir():
        if entry.name not in ('tmp','.git','__pycache__'):os.symlink(entry,view/entry.name,target_is_directory=entry.is_dir())
    for name,command in r['score']:
        binding.prepare_state(env[name],work)
        child(command,env[name],work/'receipts'/name,stop_epoch,stop_mono)
    print(json.dumps({'native_score_recipe_finished':True,'scientific_acceptance':None,'variant':variant}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['validate','validate-runtime','plan','verify-asset','prepare-dependencies','run'])
    p.add_argument('--site',type=Path,default=ROOT/'configs/site.example.json')
    p.add_argument('--group',choices=['train','export','serve','score','dependencies','checkpoint_pointer','checkpoint_pooling'])
    p.add_argument('--variant',choices=VARIANTS,default='pointer')
    p.add_argument('--stage',choices=['train','export','serve','score'])
    p.add_argument('--replica',type=int)
    p.add_argument('--asset-key');p.add_argument('--path',type=Path)
    a=p.parse_args();pair,entries,source_count=configuration();site=load_site(a.site)
    if a.action=='validate-runtime':
        print(json.dumps({'status':'CPU_RUNTIME_IDENTITIES_PASS',
                          **binding.verify([a.group] if a.group else ['train','export','serve','score']),
                          'execution_performed':False}));return
    if a.action=='validate':
        count=assets_for(entries,ROOT/site['asset_root'],[a.group]) if a.group else 0
        print(json.dumps({'status':'CPU_CONFIG_PASS','source_files':source_count,'asset_files_checked':count,
                          'complete_paired_argv':True,'only_scientific_difference':'readout=end versus mean_span',
                          'projection_targets':TARGETS,'execution_performed':False}));return
    if a.action=='plan':
        print(json.dumps(recipe(pair,entries,site,a.variant),indent=2));return
    if a.action=='verify-asset':
        require(a.asset_key in entries and a.path is not None,'provide valid --asset-key and --path')
        print(json.dumps({'asset_key':a.asset_key,**check_file(a.path,entries[a.asset_key]),'status':'EXACT_ASSET_PASS'}));return
    if a.action=='prepare-dependencies':
        dependencies(entries,ROOT/site['asset_root'],extract=True);print('EXACT_DEPENDENCY_EXTRACTION_PASS');return
    require(a.stage is not None,'run needs --stage')
    execute(pair,entries,site,a.variant,a.stage,a.replica)


if __name__=='__main__':
    main()
