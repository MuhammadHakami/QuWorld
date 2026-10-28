"""§45n: proxy-CV arms read out on out-of-distribution TRAIN shots. The public deficit sits on OOD shots (§42w: our worst public shots
are the most OOD; public's extreme-OOD tail is heavier than VT's), but every proxy screen was judged on the fold mean. OOD score of a
held-out shot = median over its frames of the distance (standardised 427 cache inputs) to the nearest frame of a 1/12 sample of the
TRAIN shots OUTSIDE its fold (what the fold's model was trained on). Per arm: paired per-fold mean-7 delta vs base on the top-q OOD
shots of each fold and on the rest (R2 against the pooled label variance, as cv_report).
    CV_PFX=P6 .venv/bin/python paper/strat/fusion/cv_ood.py base vac subf hub03 hub1 plr1 ema
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

import fx

K = 5
SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def ood_scores(d, base_pfx):
    cache = fx.OUT / f"cv_ood_scores_{base_pfx}.npz"
    if cache.exists():
        z = np.load(cache); return dict(zip(z["shots"].tolist(), z["score"].tolist()))
    trn = np.nonzero(d.split == 0)[0]; samp = trn[::12]
    Xs = np.asarray(d.X[samp], np.float64); mu = Xs.mean(0); sd = Xs.std(0) + 1e-9
    out_s, out_v = [], []
    for k in range(K):
        rows = np.load(fx.OUT / f"head_boundary_v8_{base_pfx}F{k}_s1_oof.npz")["rows_oof"]; held = np.unique(d.shot[rows])
        ref = samp[~np.isin(d.shot[samp], held)]
        Zt = torch.tensor((np.asarray(d.X[ref], np.float64) - mu) / sd, device="cuda", dtype=torch.float32)
        for s in held:
            X = np.asarray(d.X[rows[d.shot[rows] == s]], np.float64)
            zp = torch.tensor((X - mu) / sd, device="cuda", dtype=torch.float32)
            out_s.append(int(s)); out_v.append(float(torch.cdist(zp, Zt).min(1).values.median()))
        del Zt; torch.cuda.empty_cache()
    np.savez(cache, shots=np.array(out_s), score=np.array(out_v))
    return dict(zip(out_s, out_v))


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda")
    pfx = os.environ.get("CV_PFX", "P6"); q = float(os.environ.get("OOD_Q", "0.1"))
    d = sota.Data(); oo = ood_scores(d, pfx + sys.argv[1])
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); rt, Yt = lab["rows_train"], lab["y_train"][:, :7].astype(np.float64); var = np.nanvar(Yt, 0)
    def r2(arm, k, sel):
        z = np.load(fx.OUT / f"head_boundary_v8_{pfx}{arm}F{k}_s1_oof.npz"); i = np.searchsorted(rt, z["rows_oof"])
        Y, P = Yt[i], z["pred_oof"][:, :7].astype(np.float64); m = sel(d.shot[z["rows_oof"]])
        return np.array([1 - np.nanmean((Y[m, j] - P[m, j]) ** 2) / var[j] for j in range(7)])
    base, arms = sys.argv[1], sys.argv[2:]
    thr = {}
    for k in range(K):
        sh = np.unique(d.shot[np.load(fx.OUT / f"head_boundary_v8_{pfx}{base}F{k}_s1_oof.npz")["rows_oof"]])
        v = np.array([oo[int(s)] for s in sh]); thr[k] = set(sh[v >= np.quantile(v, 1 - q)].tolist())
    for part, f in (("OOD", lambda k: (lambda s: np.isin(s, list(thr[k])))), ("rest", lambda k: (lambda s: ~np.isin(s, list(thr[k]))))):
        b = {k: r2(base, k, f(k)) for k in range(K)}
        print(f"[ood] {part:4s} top-{q:.0%} base mean-7 {np.mean([b[k].mean() for k in range(K)]):.5f} | " + " ".join(f"{n} {np.mean([b[k][j] for k in range(K)]):.4f}" for j, n in enumerate(SC)), flush=True)
        for a in arms:
            if not all((fx.OUT / f"head_boundary_v8_{pfx}{a}F{k}_s1_oof.npz").exists() for k in range(K)):
                continue
            dd = np.array([r2(a, k, f(k)) - b[k] for k in range(K)]); m7 = dd.mean(1); se = m7.std(ddof=1) / np.sqrt(K)
            print(f"[ood] {part:4s} {a:7s} mean-7 {m7.mean():+.5f} (se {se:.5f}, t {m7.mean() / se:+.1f}) | " + " ".join(f"{n} {v:+.5f}" for n, v in zip(SC, dd.mean(0))), flush=True)
    print("CV_OOD_DONE", flush=True)


if __name__ == "__main__":
    main()
