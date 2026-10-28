"""Fetch the TokaMind pretrained checkpoint (best) + DCT3D embedding artifacts from HF (public, no token)."""
import sys, time
from huggingface_hub import snapshot_download
DEST = "data/tokamark/tokamind-base-v2"
for attempt in range(6):
    try:
        p = snapshot_download("UKAEA-IBM-STFC/tokamind-base-v2", local_dir=DEST,
                              allow_patterns=["checkpoints/best/*", "embeddings/*", "README.md", "LICENSE", "*.yaml", "*.json"],
                              max_workers=4)
        print("[tokamind-dl] done:", p); sys.exit(0)
    except Exception as e:  # rate limit / network: retry with backoff
        print(f"[tokamind-dl] attempt {attempt} failed: {e!r}", flush=True); time.sleep(60 * (attempt + 1))
print("[tokamind-dl] FAILED"); sys.exit(1)
