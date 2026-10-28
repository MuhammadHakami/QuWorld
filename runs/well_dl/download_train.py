"""Download The Well TRAIN splits for planetswe and rayleigh_benard (full; ~154 GB + ~294 GB) into data/well/<name>/data/train. Resumable."""
import time
from huggingface_hub import snapshot_download
ROOT = "/home/mohhakami/Documents/paper/quworld/data/well"
for ds in ["planetswe", "rayleigh_benard"]:
    t0 = time.time(); print(f"[dl-train] {ds} start", flush=True)
    snapshot_download(repo_id=f"polymathic-ai/{ds}", repo_type="dataset", local_dir=f"{ROOT}/{ds}", allow_patterns=["data/train/*"], max_workers=4)
    print(f"[dl-train] {ds} done in {time.time()-t0:.0f}s", flush=True)
