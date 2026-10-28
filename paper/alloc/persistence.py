"""Persistence baseline (y_hat(t0+1) = x(t0)) on the FULL official TEST split of a Well 2-D row, per window, joined
with every existing per-window trunk CSV (same traj,t0 keys) for paired statistics.

    .venv/bin/python paper/alloc/persistence.py <row>   # -> runs/persistence/<row>_test.csv, <row>_summary.json

MHD_64 (3-D, cache.WellND): per-channel-mean VRMSE over all spatial dims; the trunk CSVs in runs/phase_b_mhd are keyed by
window index, paired here in traj-major, t0-ascending order (checked: same window count).

VRMSE exactly as well2d.vrmse_rows (normalized space; VRMSE is invariant to the per-channel affine normalization).
"""
from __future__ import annotations

import csv
import glob
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "phase_b"))
import gpucap  # noqa: E402
import well2d  # noqa: E402

gpucap.cap_from_env()
DEV = torch.device("cuda")
OUT = ROOT / "runs" / "persistence"


def vrmse_nd(pred, true):
    dims = tuple(range(2, true.dim()))
    mse = (pred - true).square().mean(dims); var = (true - true.mean(dims, keepdim=True)).square().mean(dims)
    return (mse / (var + 1e-7)).sqrt().mean(1)


@torch.no_grad()
def main(row: str):
    if row == "MHD_64":
        sys.path.insert(0, str(HERE)); import cache as CA
        s = CA.WellND("MHD_64", "test"); vr = vrmse_nd
    else:
        s = well2d.Split(row, "test"); vr = well2d.vrmse_rows
    rec = []
    for k in range(s.n):
        t0 = torch.arange(well2d.N_HIST - 1, s.T - 1); v = []
        for i in range(0, len(t0), 16):                                   # chunks: 3-D frames are 7 x 64^3
            tt = t0[i:i + 16]; fr = s.frames(k, list(range(int(tt[0]), int(tt[-1]) + 2))).to(DEV)
            v.append(vr(fr[:-1], fr[1:]).cpu())
        rec.extend((k, int(t), float(x)) for t, x in zip(t0, torch.cat(v)))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{row}_test.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["traj", "t0", "vrmse"]); w.writerows((a, b, f"{c:.6f}") for a, b, c in rec)
    pers = {(a, b): c for a, b, c in rec}; P = np.array([c for *_, c in rec])
    out = {"row": row, "windows": len(rec), "persistence_mean": float(P.mean()), "persistence_median": float(np.median(P)),
           "trunks": {}}
    pat = ROOT / "runs/phase_b_mhd" / "test_*.csv" if row == "MHD_64" else ROOT / "runs/well2d" / row / "**/*test.csv"
    for f in sorted(glob.glob(str(pat), recursive=True)):
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        t = ({(a, b): float(r["vrmse"]) for (a, b, _), r in zip(rec, rows)} if "window" in rows[0] and len(rows) == len(rec)
             else {(int(r["traj"]), int(r["t0"])): float(r["vrmse"]) for r in rows} if "traj" in rows[0] else {})
        keys = [k for k in t if k in pers]
        if len(keys) != len(pers):
            out["trunks"][pathlib.Path(f).name] = {"skipped": f"{len(keys)} of {len(pers)} windows match"}; continue
        T = np.array([t[k] for k in keys]); Pk = np.array([pers[k] for k in keys])
        out["trunks"][str(pathlib.Path(f).relative_to(ROOT))] = {
            "trunk_mean": float(T.mean()), "trunk_median": float(np.median(T)),
            "trunk_beats_persistence_share": float((T < Pk).mean()),
            "oracle_min_mean": float(np.minimum(T, Pk).mean())}
    (OUT / f"{row}_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
