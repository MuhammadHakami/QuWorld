"""Doc 35 §42w diagnostic: shape-head accuracy on out-of-distribution VT shots.
OOD score of a shot = median over its frames of the distance (standardised inputs, all 427 features) to the nearest TRAIN frame
(1/12 TRAIN sample). R2 on a subset uses the variance of ALL VT frames (1 - SSE / (n var_all)) so subsets are comparable.
    .venv/bin/python paper/strat/fusion/ood_heads.py head_boundary_v8_J1_s1 head_boundary_v8_J1RL_s1 ...
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import fx
from vt_eval import SC


def ood_scores(d, shots):
    tr = np.nonzero(d.split == 0)[0][::12]; Xt = np.asarray(d.X[tr], np.float64); mu = Xt.mean(0); sd = Xt.std(0) + 1e-9
    Zt = torch.tensor((Xt - mu) / sd, device="cuda", dtype=torch.float32); out = []
    for s in shots:
        X = np.asarray(d.X[d.shot == s], np.float64); zp = torch.tensor((X - mu) / sd, device="cuda", dtype=torch.float32)
        out.append(float(torch.cdist(zp, Zt).min(1).values.median()))
    return np.array(out)


def main() -> None:
    arms = sys.argv[1:]
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    d = sota.Data(); vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); rows, y = vt["rows"], vt["y"][:, :7].astype(np.float64)
    sh = d.shot[rows]; us = np.unique(sh); f = fx.OUT / "ood_vt_shots.npz"
    if f.exists():
        z = np.load(f); assert (z["shots"] == us).all(); oo = z["ood"]
    else:
        oo = ood_scores(d, us); np.savez(f, shots=us, ood=oo)
    thr = np.quantile(oo, 0.9); top = np.isin(sh, us[oo >= thr]); var = np.nanvar(y, 0)
    print(f"[ood] {len(us)} VT shots; top-10% OOD: {int((oo >= thr).sum())} shots, {top.mean():.3f} of frames (thr {thr:.2f})", flush=True)
    for arm in arms:
        p = GB.heads_gpu(d, rows, arm)[:, :7]; e = (p - y) ** 2; ok = np.isfinite(e)
        r = lambda m: 1 - np.array([np.nansum(e[m, j]) / (ok[m, j].sum() * var[j]) for j in range(7)])
        ra, ro, ri = r(np.ones(len(rows), bool)), r(top), r(~top)
        share = np.array([np.nansum(e[top, j]) / np.nansum(e[:, j]) for j in range(7)])
        print(f"[ood] {arm}: all {ra.mean():.5f} | OOD {ro.mean():.5f} | ID {ri.mean():.5f} | OOD share of SSE {share.mean():.3f} | OOD per scalar "
              + " ".join(f"{n} {v:.4f}" for n, v in zip(SC, ro)), flush=True)
    print("OOD_HEADS_DONE", flush=True)


if __name__ == "__main__":
    main()
