"""Download The Well test+valid splits (and yaml/stats) for rayleigh_benard and planetswe into data/well/<name>/ (mirrors the local MHD_64 layout). Network-only; resumable."""
import sys, time
from huggingface_hub import snapshot_download
ROOT = "/home/mohhakami/Documents/paper/quworld/data/well"
for ds, patterns in [("planetswe", ["data/test/*", "data/valid/*", "*.yaml", "README.md", "data/*.yaml"]),
                     ("rayleigh_benard", ["data/test/*", "data/valid/*", "*.yaml", "README.md"])]:
    t0 = time.time(); print(f"[dl] {ds} start", flush=True)
    p = snapshot_download(repo_id=f"polymathic-ai/{ds}", repo_type="dataset", local_dir=f"{ROOT}/{ds}", allow_patterns=patterns, max_workers=4)
    print(f"[dl] {ds} done in {time.time()-t0:.0f}s -> {p}", flush=True)
