import torch, subprocess, json, os
info = {
    "hostname": os.uname().nodename,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "device_count": torch.cuda.device_count(),
    "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None,
    "tf32_allowed": torch.backends.cuda.matmul.allow_tf32,
    "gpus": [],
}
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    info["gpus"].append({
        "name": p.name,
        "cc": f"{p.major}.{p.minor}",
        "total_mem_GB": round(p.total_memory/1024**3, 1),
        "sm_count": p.multi_processor_count,
    })
try:
    info["nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi","--query-gpu=name,memory.total,driver_version,compute_cap","--format=csv,noheader"],
        text=True).strip()
except Exception as e:
    info["nvidia_smi"] = str(e)
print(json.dumps(info, indent=2))
