"""Install a new training-role venv from exact archived artifacts, CPU only."""
import hashlib,json,os,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SPEC_FILE=ROOT/'SPEC.json';SPEC=json.loads(SPEC_FILE.read_text())
assert hashlib.sha256(SPEC_FILE.read_bytes()).hexdigest()==(ROOT/'SPEC.sha256').read_text().strip()
assert not (ROOT/'BUILD_STARTED.json').exists(),'Build already started; preserve original outcome'
if hasattr(os,'sched_getaffinity'):
 affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(affinity[:2]))
def save(name,obj):(ROOT/name).write_text(json.dumps(obj,indent=2)+'\n')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def stamp():return {'epoch':time.time(),'monotonic':time.monotonic()}
(ROOT/'logs').mkdir();(ROOT/'receipts').mkdir();(ROOT/'tmp').mkdir();(ROOT/'wheelhouse').mkdir()
save('BUILD_STARTED.json',{'start':stamp(),'root':str(ROOT),'disk':dict(zip(('total','used','free'),shutil.disk_usage(ROOT))),'affinity':sorted(os.sched_getaffinity(0)),'control_sha256':{n:sha(ROOT/n) for n in ('SPEC.json','SPEC.sha256','rebuild_training.py','cpu_import_probe.py')}})
assert shutil.disk_usage(ROOT).free>30*1024**3,'At least 30 GiB available is required for this isolated build'
def environment():
 env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'none','CUDA_DEVICE_ORDER':'PCI_BUS_ID','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PIP_NO_INDEX':'1','PIP_CONFIG_FILE':'/dev/null','PIP_DISABLE_PIP_VERSION_CHECK':'1','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false','TMPDIR':str(ROOT/'tmp'),'PYTHONPATH':str(ROOT/'pillow-overlay'),'HF_HOME':str(ROOT/'cache/hf'),'TRITON_CACHE_DIR':str(ROOT/'cache/triton')}
 paths=[ROOT/'syslibs/usr/lib/x86_64-linux-gnu',ROOT/'syslibs/lib/x86_64-linux-gnu',ROOT/'venv/lib/python3.12/site-packages/torch/lib',ROOT/'pillow-overlay/pillow.libs']
 paths+=sorted((ROOT/'venv/lib/python3.12/site-packages').glob('nvidia/*/lib')) if (ROOT/'venv').exists() else []
 env['LD_LIBRARY_PATH']=':'.join(str(x) for x in paths if x.is_dir());return env
steps=[]
def run(name,argv,timeout=180,allowed=(0,)):
 print('START '+name,flush=True);before=stamp();out=ROOT/'logs'/f'{name}.stdout';err=ROOT/'logs'/f'{name}.stderr';timed_out=False
 with out.open('xb') as stdout,err.open('xb') as stderr:
  process=subprocess.Popen(argv,cwd=ROOT,env=environment(),stdout=stdout,stderr=stderr,start_new_session=True)
  try:code=process.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   timed_out=True;process.terminate()
   try:code=process.wait(timeout=10)
   except subprocess.TimeoutExpired:process.kill();code=process.wait()
 receipt={'name':name,'argv':argv,'start':before,'end':stamp(),'native_exit':code,'timed_out':timed_out,'stdout_bytes':out.stat().st_size,'stdout_sha256':sha(out),'stderr_bytes':err.stat().st_size,'stderr_sha256':sha(err)}
 save('receipts/'+name+'.json',receipt);steps.append(receipt);print('END '+name+' native='+str(code),flush=True)
 if timed_out or code not in allowed:raise RuntimeError(name+' failed; native='+str(code))
 return receipt,out,err
media=SPEC['wheels']+[SPEC['python_archive'],SPEC['pillow_overlay']]+SPEC['isolated_system_libraries']
checked=[];pip=None
try:
 for record in media:
  path=Path(record['source_file']);before=path.stat();digest=sha(path);after=path.stat()
  assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),str(path)
  assert before.st_size==record['bytes'] and digest==record['sha256'],str(path)
  checked.append({'source_file':str(path),'filename':record['filename'],'bytes':before.st_size,'sha256':digest,'inode':before.st_ino,'mtime_ns':before.st_mtime_ns})
 save('ACTUAL_INPUT_MEDIA.json',checked);print('VERIFIED_MEDIA '+str(len(checked)),flush=True)
 for record in SPEC['wheels']:(ROOT/'wheelhouse'/record['filename']).symlink_to(record['source_file'])
 archive=SPEC['python_archive']
 with tarfile.open(archive['source_file']) as tar:tar.extractall(ROOT/'interpreter',filter='data')
 binary=ROOT/'interpreter'/archive['binary_relative'];assert sha(binary)==archive['binary_sha256']
 run('interpreter',[str(binary),'-I','-B','-c','import sys,json;print(json.dumps({"version":sys.version,"executable":sys.executable}))'])
 for record in SPEC['isolated_system_libraries']:run('extract-'+record['package'],['/usr/bin/dpkg-deb','-x',record['source_file'],str(ROOT/'syslibs')],60)
 run('pillow-overlay',[str(binary),'-I','-B','-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--target',str(ROOT/'pillow-overlay'),SPEC['pillow_overlay']['source_file']],120)
 run('venv',[str(binary),'-I','-B','-m','venv',str(ROOT/'venv')],120)
 python=ROOT/'venv/bin/python'
 lock=ROOT/'training.requirements.lock';lock.write_text(''.join('./wheelhouse/'+r['filename']+' --hash=sha256:'+r['sha256']+'\n' for r in SPEC['wheels']))
 run('install',[str(python),'-I','-B','-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','-r',str(lock)],900)
 pip,out,err=run('pip-check',[str(python),'-I','-B','-m','pip','check'],60,allowed=(0,1))
 actual=out.read_text().strip();expected=SPEC['known_pip_conflict_exact_stdout']
 assert pip['native_exit']==1 and actual==expected and err.read_bytes()==b'',('Unexpected dependency check outcome',actual,err.read_text())
 save('DECLARED_METADATA_CONFLICT.json',{'native_exit':1,'actual_stdout':actual,'expected_stdout':expected,'clean_pip_check':False,'pins_modified':False,'interpretation':'Known torch wheel metadata conflict is preserved; subsequent CPU imports do not establish CUDA compatibility.'})
 # -I would ignore the deliberately pinned Pillow overlay; -B plus PYTHONNOUSERSITE and a restricted environment preserve it.
 run('cpu-imports',[str(python),'-B',str(ROOT/'cpu_import_probe.py'),str(SPEC_FILE)],180)
 result={'status':'FRESH_TRAINING_ROLE_CPU_IMPORTS_PASS_WITH_DECLARED_METADATA_CONFLICT','steps':len(steps),'pip_check_native_exit':pip['native_exit'],'clean_pip_check':False,'pins_modified':False,'models_loaded':0,'benchmark_workers_run':0,'GPU_computation':0,'scope':'Installation and dependency imports only; no training, model inference, checkpoint loading, GPU-context initialization or end-to-end evaluation.'}
except BaseException as exc:
 save('FAILURE.json',{'error':repr(exc),'traceback':traceback.format_exc()});result={'status':'FAILED_PRESERVED','error':repr(exc),'steps':len(steps),'pip_check_native_exit':pip['native_exit'] if pip else None}
finally:
 post=[]
 for r in checked:
  st=Path(r['source_file']).stat();post.append({'source_file':r['source_file'],'unchanged_stat':(st.st_ino,st.st_size,st.st_mtime_ns)==(r['inode'],r['bytes'],r['mtime_ns'])})
 save('POST_INPUT_STAT.json',post);result.update(end=stamp(),input_media=len(checked),all_original_media_stat_unchanged=all(x['unchanged_stat'] for x in post),disk_after=dict(zip(('total','used','free'),shutil.disk_usage(ROOT))))
 save('BUILD_RESULT.json',result);print(json.dumps(result),flush=True)
 if result['status']=='FAILED_PRESERVED':raise SystemExit(1)
