"""Download The Well shear_flow (replaces planetswe: the current planetswe data matches none of the released checkpoints) into data/well/shear_flow. Low worker count: unauthenticated HF API is rate-limited (500 req / 5 min)."""
import time
from huggingface_hub import snapshot_download
ROOT = "/home/mohhakami/Documents/paper/quworld/data/well"
for pats in (["data/test/*", "data/valid/*", "*.yaml", "README.md"], ["data/train/*"]):
    t0 = time.time(); print(f"[dl] shear_flow {pats[0]} start", flush=True)
    snapshot_download(repo_id="polymathic-ai/shear_flow", repo_type="dataset", local_dir=f"{ROOT}/shear_flow", allow_patterns=pats, max_workers=2)
    print(f"[dl] shear_flow {pats[0]} done in {time.time()-t0:.0f}s", flush=True)
print("[dl] shear_flow done", flush=True)
