"""Doc 35 §37b (registered before any TEST score of these arms): rayleigh_benard on the full TEST split, once.

Per TEST window (the_well order, 34,300 windows) and field it records the Well VRMSE of each E-ft member (mr30ryd,
vanilla, mlp30; the gated two-stage fine-tunes of doc 37, loaded exactly as paper/alloc/ft_ensemble.py does) under
single pass, flip2 and tta8 (x-symmetry averaging of common.XSym: reflection with u_x -> -u_x, x-shifts 0/128/256/384),
and of the E-ft ensemble single pass and tta8. Registered arms: mr30ryd single, mr30ryd tta8, E-ft single (must
reproduce the recorded 0.040267 within 1e-5), E-ft tta8. Everything else in the file is descriptive.

Resumable: work is split into shards of trajectories; each launch runs until --budget seconds and exits; a finished
shard is written atomically to runs/strat/rb/test_s37b/shard_<k>.npz. `--summarize` merges the shards (CPU).

    flock runs/strat/gpu.lock env QW_GPU_GB=12 timeout 1800 runs/capped.sh s37b 16 uv run --no-sync python \
        paper/strat/rb/score_test_tta.py [--budget 1500]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / "runs" / "strat" / "rb" / "test_s37b"
DS = "rayleigh_benard"
TRAJ_PER_SHARD = 5
EFT_RECORDED = 0.040267251431941986                     # runs/alloc/ft/rayleigh_benard_Eft.json full_test
NAMES = ["single", "flip2", "tta8"]


def summarize() -> None:
    import torch  # noqa: F401  (only for parity with the GPU path; not needed)
    fs = sorted(OUT.glob("shard_*.npz"))
    parts = [np.load(f) for f in fs]
    vr = np.concatenate([p["vr"] for p in parts])       # (N, arms, C)
    arms = [str(a) for a in parts[0]["arms"]]
    n_total = int(parts[0]["n_total"])
    res = {"n": int(len(vr)), "n_total": n_total, "complete": len(vr) == n_total, "arms": {}}
    for i, a in enumerate(arms):
        v = vr[:, i].mean(1)
        res["arms"][a] = {"mean": float(v.mean()), "median": float(np.median(v)),
                          "per_field": vr[:, i].mean(0).tolist()}
    if res["complete"]:
        e = res["arms"]["ens_single"]["mean"]
        res["eft_check"] = {"recorded": EFT_RECORDED, "rescored": e, "ok": abs(e - EFT_RECORDED) < 1e-5}
    (OUT.parent / "test_s37b.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({a: round(v["mean"], 6) for a, v in res["arms"].items()}), res.get("eft_check"), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=1500)
    ap.add_argument("--bsz", type=int, default=14); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    import torch
    import common as CM                                 # MemAvailable >= 30 GB check + GPU cap before CUDA
    sys.path.insert(0, str(ROOT / "paper" / "alloc"))
    import ft_ensemble as FE
    tags = FE.members(DS)
    assert [t.split("_")[2] for t in tags] == ["mr30ryd", "vanilla", "mlp30"], tags
    loaded = [FE.load(DS, t) for t in tags]
    models, test = [m for m, _ in loaded], loaded[0][1][1]
    assert test.H == 512 and test.W == 128, (test.H, test.W)   # x (periodic, rolled/flipped) is dim 2
    sym = CM.XSym(test.mean, test.std)
    TR = [(f, s) for f in (False, True) for s in (0, 128, 256, 384)]
    SUB = {"single": [0], "flip2": [0, 4], "tta8": list(range(8))}
    arms = [f"{n}_{t.split('_')[2]}" for t in tags for n in NAMES] + ["ens_single", "ens_tta8"]
    OUT.mkdir(parents=True, exist_ok=True)
    n_traj = test.n
    n_total = sum(test.T - 1 - (CM.well2d.N_HIST - 1) for _ in range(n_traj))
    t0 = time.time(); last = 0.0
    torch.cuda.reset_peak_memory_stats()
    for k in range(0, n_traj, TRAJ_PER_SHARD):
        fn = OUT / f"shard_{k // TRAJ_PER_SHARD:03d}.npz"
        if fn.exists():
            continue
        if time.time() - t0 + last > a.budget:                  # stop if the next shard would overrun the slice
            print(f"[s37b] budget reached; resume later ({time.time() - t0:.0f}s)", flush=True); return
        rows, ts = [], time.time()
        pairs = [(j, t) for j in range(k, min(k + TRAJ_PER_SHARD, n_traj)) for t in range(CM.well2d.N_HIST - 1, test.T - 1)]
        with torch.no_grad():
            for x, y in CM.well2d.batched(test, pairs, a.bsz):
                x, y = x.to(CM.DEV), y.to(CM.DEV)
                outs = [[sym.invert(m(sym.apply(x, f, s)), f, s) for f, s in TR] for m in models]
                vr = []
                for o in outs:
                    for n in NAMES:
                        vr.append(CM.vrmse_fields(torch.stack([o[i] for i in SUB[n]]).mean(0), y))
                vr.append(CM.vrmse_fields(torch.stack([o[0] for o in outs]).mean(0), y))
                vr.append(CM.vrmse_fields(torch.stack([torch.stack(o).mean(0) for o in outs]).mean(0), y))
                rows.append(torch.stack(vr, 1).float().cpu().numpy())
        tmp = fn.with_suffix(".tmp.npz")
        np.savez(tmp, vr=np.concatenate(rows), pairs=np.array(pairs), arms=np.array(arms), n_total=n_total)
        os.replace(tmp, fn)
        last = time.time() - ts
        print(f"[s37b] shard {k // TRAJ_PER_SHARD} ({len(pairs)} windows) {time.time() - t0:.0f}s "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    print("S37B_ALL_SHARDS_DONE", flush=True)


if __name__ == "__main__":
    main()
