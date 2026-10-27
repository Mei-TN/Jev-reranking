"""Imports only: no model construction, tensors, backward, or CUDA kernels."""
import importlib,importlib.metadata,json,os,sys
from pathlib import Path
spec=json.loads(Path(sys.argv[1]).read_text())
assert sys.version_info[:3]==(3,12,13)
assert os.environ['CUDA_VISIBLE_DEVICES']==''
versions={}
for pin in spec['role']['pins']:
 name,expected=pin.split('==');actual=importlib.metadata.version(name)
 assert actual==expected,(name,actual,expected);versions[name]=actual
import torch
assert not torch.cuda.is_initialized()
# A process-local guard records and refuses a PyTorch CUDA initialization request.
# No installed package or source file is modified.
attempts=[]
original_lazy_init=torch.cuda._lazy_init
def forbidden_cuda_init(*args,**kwargs):
 attempts.append('torch.cuda._lazy_init')
 raise RuntimeError('CUDA initialization is forbidden in this CPU import probe')
torch.cuda._lazy_init=forbidden_cuda_init
modules={'torch':str(torch.__file__)}
states=[{'after':'torch','cuda_initialized':torch.cuda.is_initialized()}]
try:
 for name in ['transformers','peft','accelerate','datasets','safetensors','fla','triton','PIL']:
  mod=importlib.import_module(name);modules[name]=str(mod.__file__)
  states.append({'after':name,'cuda_initialized':torch.cuda.is_initialized()})
  assert not torch.cuda.is_initialized()
 assert not attempts
 assert importlib.metadata.version('Pillow')=='11.3.0'
 assert '/pillow-overlay/' in modules['PIL']
 for name,path in modules.items():assert path.startswith(str(Path(sys.prefix))) or (name=='PIL' and '/pillow-overlay/' in path),(name,path)
 result={'status':'CPU_IMPORTS_PASS','python':sys.version,'prefix':sys.prefix,'versions':versions,'Pillow_overlay':'11.3.0','modules':modules,'CUDA_initialization_states':states,'blocked_CUDA_initialization_attempts':attempts,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'torch_cuda_build':torch.version.cuda,'scope':'Package imports and installed-version checks only. No models, tensors, datasets, benchmark workers, inference, training steps, or CUDA contexts were created.'}
 print(json.dumps(result,indent=2))
finally:torch.cuda._lazy_init=original_lazy_init
