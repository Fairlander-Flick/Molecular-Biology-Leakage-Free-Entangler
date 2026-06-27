import os
os.environ.setdefault("HF_HOME", "/home/woody/dsaa/dsaa115h/hf_cache")
from transformers import AutoTokenizer, EsmForProteinFolding
print("downloading esmfold_v1 ...", flush=True)
AutoTokenizer.from_pretrained("facebook/esmfold_v1")
EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
print("ESMFold cached", flush=True)
