#!/bin/bash
# RealPDEBench foil (ICLR 2026) real-world shards + split JSONs: the paper-benchmark protocol (public in/out-dist test splits).
cd /home/mohhakami/Documents/paper/quworld
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("AI4Science-WestlakeU/RealPDEBench", repo_type="dataset", local_dir="data/realpdebench",
                  allow_patterns=["foil/*.json", "foil/mean_std.pt", "foil/hf_dataset/real/*"], max_workers=4)
print("foil real done", flush=True)
PY
echo "REALPDEBENCH_FOIL_REAL_DONE $(date -Is)"
