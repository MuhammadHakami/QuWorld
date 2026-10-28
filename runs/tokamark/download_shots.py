"""Download all TokaMark v1 shots (zarr, ~3 MB each, 11,573 shots ~35 GB) from the public STFC S3 bucket into data/tokamark/zarr/<shot>.zarr — resumable, 16 concurrent shots, no listings cache (RSS grew to 3.4 GB with 32 threads + cache) (each zarr is many small objects; sequential was ~30 s/shot)."""
import fsspec, os, time, sys
import logging
for _n in ("asyncio", "fsspec", "s3fs", "botocore", "aiobotocore"):  # Sep 19 23:10: an S3 outage spewed 8.3 GB of "Task exception was never retrieved" tracebacks; fetch() already prints each FAIL
    logging.getLogger(_n).setLevel(logging.CRITICAL)
S3 = "https://s3.echo.stfc.ac.uk"; BUCKET = "mast/tokamark/v1"; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "tokamark", "zarr")
fs = fsspec.filesystem("s3", client_kwargs={"endpoint_url": S3}, anon=True, use_listings_cache=False)
paths = sorted(p for p in fs.ls(BUCKET) if p.endswith(".zarr"))
# priority: the shots run_training.py actually selects for small subsets (their split function, shuffle=True as in config_model.yaml)
from tokamark.data_split import get_train_test_val_shots
from tokamark.tools.path import RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE
head = []
for n in (8, 64, 512):
    tr, te, va = get_train_test_val_shots(max_index=n, shuffle=True, data_splits_file_path=RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE)
    head += [s for s in tr + te + va if s not in head]
prio = [f"{BUCKET}/{sid}.zarr" for sid in head]
prio_set = set(prio)
paths = [p for p in prio if p in set(paths)] + [p for p in paths if p not in prio_set]
print(f"[tokamark-dl] priority shots (split heads): {len(prio)}", flush=True)
print(f"[tokamark-dl] {len(paths)} shots on S3", flush=True)
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
_local = threading.local()
def get_fs():
    if not hasattr(_local, "fs"):
        _local.fs = fsspec.filesystem("s3", client_kwargs={"endpoint_url": S3}, anon=True, use_listings_cache=False)
    return _local.fs
def fetch(p):
    dst = os.path.join(OUT, p.split("/")[-1])
    if os.path.exists(os.path.join(dst, ".complete")):
        return p, "skip"
    try:
        get_fs().get(p, dst, recursive=True); open(os.path.join(dst, ".complete"), "w").close(); return p, "ok"
    except Exception as e:
        return p, f"FAIL {type(e).__name__} {str(e)[:80]}"
t0, n_ok, n_fail, n = time.time(), 0, 0, 0
with ThreadPoolExecutor(max_workers=16) as ex:
    for p, st in ex.map(fetch, paths):
        n += 1
        if st.startswith("FAIL"): n_fail += 1; print(f"[tokamark-dl] {p}: {st}", flush=True)
        else: n_ok += 1
        if n % 200 == 0: print(f"[tokamark-dl] {n}/{len(paths)} ok {n_ok} fail {n_fail} {time.time()-t0:.0f}s", flush=True)
print(f"[tokamark-dl] done: ok {n_ok} fail {n_fail} in {time.time()-t0:.0f}s", flush=True)
