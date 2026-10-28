"""Sophelio: the ceiling that the PCA psi decoder imposes (VALID only, no model): project the ground-truth psi onto the
first k PCA components (the trunks' output basis), keep the true q95/betaN, and score with the official scorer on the
NAS's fixed 40-shot VALID subset.

    .venv/bin/python paper/sophelio/pca_ceiling.py [--ks 32 64 128 256]   ->  runs/sophelio_sota/pca_ceiling.json

If the k = 64 ceiling is below the Ch1 leader (0.9932), no trunk that outputs 64 coefficients can reach it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sota  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--ks", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--split", type=int, default=1)
    a = ap.parse_args()
    import train_hybrid as TH
    data = sota.Data(); L = sota.LABELS
    shots = data.shots_of(a.split); pick = [shots[i] for i in np.random.default_rng(1).permutation(len(shots))[:a.n]]
    mean = data.meta["mean"].to(sota.DEV); out = {}
    cache = []
    for s in pick:
        rows = np.nonzero(data.shot == s)[0]
        psi = data.psi_of(rows).reshape(len(rows), -1).astype(np.float32)
        gt = ({"psi": psi.reshape(-1, 65, 65),
               "q95": data.S[rows, L.index("efit_q95")].astype(np.float64), "betaN": data.S[rows, L.index("efit_beta_n")].astype(np.float64)},
              data.meta["names"][s])
        cache.append((gt, torch.tensor(psi, device=sota.DEV)))
    for k in a.ks:
        comps = data.meta["comps"][:k].to(sota.DEV)
        gts, pr = [], []
        for gt, psi in cache:
            rec = (((psi - mean) @ comps.T) @ comps + mean).cpu().numpy()
            gts.append(gt); pr.append({"psirz": rec.reshape(-1, 65, 65), "q95": gt[0]["q95"], "betaN": gt[0]["betaN"]})
        r = TH.official_score(gts, pr, sota.ROOT / "runs" / "sophelio_sota" / "ref_cache")
        out[k] = r
        print(f"[pca-ceiling] k={k}: S {r['S']:.4f} psi {r['r2_psi']:.6f} dlcfs {r['dlcfs']:.4f} cons {r['consistency']:.4f} "
              f"{({n: round(v, 3) for n, v in r['r2_cons_each'].items()})}", flush=True)
    (sota.ROOT / "runs" / "sophelio_sota" / "pca_ceiling.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
