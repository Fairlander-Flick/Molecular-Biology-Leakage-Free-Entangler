"""Convert downloaded ESM2-3B (.bin, 2 shards) -> standalone safetensors dir.

transformers 5.12 + torch 2.5 refuses torch.load on sharded .bin, and HF offline
cache won't resolve a manually-added safetensors inside the snapshot. So we build a
clean standalone dir (config + tokenizer + model.safetensors) and point
PPI_ESM2_MODEL at it. Same trick used for ProstT5 (see convert_prostt5.py).
"""
import glob, json, os, shutil, time
import torch
from safetensors.torch import save_file

HF = "/home/woody/dsaa/dsaa115h/hf_cache/hub/models--facebook--esm2_t36_3B_UR50D"
OUT = "/home/woody/dsaa/dsaa115h/hf_cache/esm2_3b_local"

snap = glob.glob(f"{HF}/snapshots/*")[0]
os.makedirs(OUT, exist_ok=True)

# merge both .bin shards into one state dict
sd = {}
for b in sorted(glob.glob(f"{snap}/pytorch_model-*.bin")):
    t = time.time(); print("loading", os.path.basename(b), flush=True)
    part = torch.load(b, map_location="cpu", weights_only=True)
    sd.update({k: v.clone().contiguous() for k, v in part.items()
               if isinstance(v, torch.Tensor)})
    print(f"  +{len(part)} tensors in {time.time()-t:.0f}s", flush=True)

save_file(sd, f"{OUT}/model.safetensors", metadata={"format": "pt"})
print("wrote model.safetensors", round(os.path.getsize(f'{OUT}/model.safetensors')/1e9, 2),
      "GB | tensors:", len(sd), flush=True)

# copy config + tokenizer files (everything except the weights/index)
for f in os.listdir(snap):
    if f.endswith(".bin") or f.endswith(".index.json") or f.startswith("tf_"):
        continue
    src = os.path.join(snap, f)
    if os.path.isfile(src):
        shutil.copy(src, os.path.join(OUT, f))
        print("copied", f, flush=True)
print("DONE ->", OUT, flush=True)
