#!/bin/bash
# RealPDE competition release (public HF mirror, CC BY-NC 4.0): baseline checkpoints first, then the two tarballs.
cd /home/mohhakami/Documents/paper/quworld
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("AI4Science-WestlakeU/RealPDE-Competition-Data", repo_type="dataset", local_dir="data/realpde",
                  allow_patterns=["README.md", "example_data/*", "baseline_checkpoints/*"])
print("checkpoints + example done", flush=True)
snapshot_download("AI4Science-WestlakeU/RealPDE-Competition-Data", repo_type="dataset", local_dir="data/realpde",
                  allow_patterns=["train_real.tar.gz", "train_sim.tar.gz"], max_workers=4)
print("tarballs done", flush=True)
PY
cd data/realpde && for t in train_real train_sim; do [ -d $t ] || tar -xzf $t.tar.gz && echo "unpacked $t"; done
echo "REALPDE_DATA_DONE $(date -Is)"
