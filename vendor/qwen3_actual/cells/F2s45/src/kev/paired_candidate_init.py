"""Bind candidate-local paired step-zero tensors; never a trained checkpoint."""
import fcntl, hashlib, json, os, time
from pathlib import Path

def bind_initial(model, out_dir, seed, revision):
    import torch
    from safetensors.torch import save_file, load_file
    root=Path(os.environ['QWEN_PAIR_INIT_ROOT']);pair=os.environ['QWEN_PAIR_ID'];variant=os.environ['QWEN_VARIANT']
    assert revision=='2c54d5a09e7e92d4f5126b92a5a457448c9593e6'
    assert pair in ('F2s44','F2s45','F3s44','F3s45') and seed==int(pair[-2:])
    assert variant in ('pointer','pooling') and model.lm.config.hidden_size==2560 and model.lm.config.model_type=='qwen3'
    assert model.lora_target_modules==['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
    assert model.head_config['readout']==('end' if variant=='pointer' else 'mean_span')
    assert root.parent.name=='shared_initial' and root.name==pair
    root.mkdir(parents=True,exist_ok=True)
    before_cpu=torch.get_rng_state().clone();before_cuda=[x.clone() for x in torch.cuda.get_rng_state_all()]
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert len(params)==508 and all(p.dtype==torch.float32 for p in params.values())
    state={n:p.detach().cpu().contiguous().clone() for n,p in params.items()}
    common=root/'shared_initial.safetensors';meta_path=root/'INITIAL.json'
    deadline=float(os.environ['QWEN_HARDSTOP_EPOCH'])-60
    with (root/'initial.lock').open('a+b') as lock:
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.time()>=deadline:raise TimeoutError('paired initialization deadline')
                time.sleep(.05)
        if not common.exists():
            assert not meta_path.exists()
            save_file(state,str(root/'shared_initial.tmp'))
            os.replace(root/'shared_initial.tmp',common)
            tensors={n:{'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(t.numpy().tobytes()).hexdigest()} for n,t in state.items()}
            meta={'schema':'qwen3-actual-shared-step-zero-v1','pair':pair,'seed':seed,'base_revision':revision,'hidden_size':2560,'creator_variant':variant,'optimizer_steps':0,'trained_checkpoint':False,'tensor_count':len(state),'tensor_manifest':tensors,'file_sha256':hashlib.sha256(common.read_bytes()).hexdigest()}
            with meta_path.open('x') as f:json.dump(meta,f,indent=2);f.flush();os.fsync(f.fileno())
        meta=json.loads(meta_path.read_text());assert meta['pair']==pair and meta['seed']==seed and meta['base_revision']==revision and meta['optimizer_steps']==0 and meta['trained_checkpoint'] is False
        assert hashlib.sha256(common.read_bytes()).hexdigest()==meta['file_sha256']
        shared=load_file(str(common));assert shared.keys()==state.keys()
        pre_equal=all(torch.equal(state[n],t) for n,t in shared.items())
        with torch.no_grad():
            for n,t in shared.items():
                assert tuple(t.shape)==tuple(params[n].shape) and t.dtype==params[n].dtype
                params[n].copy_(t)
        exact=all(torch.equal(params[n].detach().cpu(),t) for n,t in shared.items());assert exact
    assert torch.equal(before_cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(before_cuda,torch.cuda.get_rng_state_all()))
    receipt={'schema':'qwen3-loaded-shared-step-zero-v1','pair':pair,'variant':variant,'shared_file':str(common),'shared_file_sha256':meta['file_sha256'],'tensor_count':len(shared),'all_loaded_exact':exact,'constructed_before_load_exact':pre_equal,'cpu_and_cuda_rng_unchanged_by_binding':True,'optimizer_steps':0,'base_revision':revision,'head_config':model.head_config}
    with (Path(out_dir)/'shared_initialization.json').open('x') as f:json.dump(receipt,f,indent=2)
