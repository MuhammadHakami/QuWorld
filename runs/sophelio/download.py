"""Download the Sophelio fusion-equilibrium-challenge dataset (HF, public, 103.9 GB, 9123 parquet) to data/sophelio/fusion-equilibrium-challenge. Resumable."""
import time
from huggingface_hub import snapshot_download
t0 = time.time(); print("[dl] sophelio start", flush=True)
snapshot_download(repo_id="Sophelio/fusion-equilibrium-challenge", repo_type="dataset",
                  local_dir="/home/mohhakami/Documents/paper/quworld/data/sophelio/fusion-equilibrium-challenge", max_workers=16)
print(f"[dl] sophelio done in {time.time()-t0:.0f}s", flush=True)
