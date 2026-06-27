import torch, glob, os, time, shutil
from safetensors.torch import save_file
snap=glob.glob("/home/woody/dsaa/dsaa115h/hf_cache/hub/models--facebook--esmfold_v1/snapshots/*")[0]
dst="/home/woody/dsaa/dsaa115h/hf_cache/esmfold_local"; os.makedirs(dst, exist_ok=True)
t=time.time(); print("loading bin...", flush=True)
sd=torch.load(os.path.join(snap,"pytorch_model.bin"), map_location="cpu", weights_only=True)
print(f"loaded {len(sd)} tensors in {time.time()-t:.0f}s", flush=True)
sd={k: v.clone().contiguous() for k,v in sd.items() if isinstance(v, torch.Tensor)}
save_file(sd, os.path.join(dst,"model.safetensors"), metadata={"format":"pt"})
for f in ("config.json","special_tokens_map.json","tokenizer_config.json","vocab.txt"):
    shutil.copyfile(os.path.join(snap,f), os.path.join(dst,f))
print("wrote", dst, round(os.path.getsize(os.path.join(dst,'model.safetensors'))/1e9,2),"GB", flush=True)
