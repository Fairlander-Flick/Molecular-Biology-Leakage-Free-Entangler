"""Dump the 11,018 used proteins to a FASTA for mmseqs search (Track A)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as D

_, id2seq, ordered = D.build_manifest()
out = Path(__file__).resolve().parent / "queries.fasta"
with open(out, "w") as f:
    for pid in ordered:
        f.write(f">{pid}\n{id2seq[pid]}\n")
print(f"wrote {out}: {len(ordered)} proteins")
