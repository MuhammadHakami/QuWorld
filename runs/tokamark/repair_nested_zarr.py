"""Repair shots that fsspec nested as <id>.zarr/<id>.zarr/ (dst existed from a crashed first run).
The nested copy is the complete download; the outer level holds stale partial groups.
Swap: outer -> data/tokamark/zarr_partial_leftovers/<id>.zarr (kept, reversible), nested -> <id>.zarr.
"""
import os, shutil, sys, json, time
Z = "data/tokamark/zarr"; L = "data/tokamark/zarr_partial_leftovers"; os.makedirs(L, exist_ok=True)
fixed, skipped = [], []
for b in sorted(os.listdir(Z)):
    outer = os.path.join(Z, b); nested = os.path.join(outer, b)
    if not os.path.isdir(nested): continue
    if not (os.path.isfile(os.path.join(nested, ".zgroup")) and os.path.isfile(os.path.join(nested, ".zmetadata"))):
        skipped.append(b); continue
    tmp = os.path.join(Z, b + ".swap"); os.rename(nested, tmp)           # 1. nested out of the outer tree
    os.rename(outer, os.path.join(L, b))                                  # 2. stale outer aside (keeps its .complete)
    os.rename(tmp, outer)                                                 # 3. complete copy into place
    open(os.path.join(outer, ".complete"), "w").close()
    fixed.append(b)
json.dump({"time": time.strftime("%FT%T"), "fixed": fixed, "skipped_incomplete_nested": skipped}, open("runs/tokamark/repair_nested_zarr.json", "w"), indent=1)
print(f"fixed {len(fixed)} nested shots; skipped {len(skipped)} (nested copy incomplete): {skipped[:10]}")
