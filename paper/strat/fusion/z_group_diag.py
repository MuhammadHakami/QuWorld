"""§45n diagnostic (label-free on public): is the public Z_axis deficit (error 1.72x the VT level; other scalars 0.76-1.19x) tied to a
shot class whose share differs between public and the labelled data? Classes: magnetics sampling rate (20 kHz / 2 kHz), Ip sign. VT R2
per class from TRAIN-only J1RL s1 + s2 heads on all 743 VT shots.
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python paper/strat/fusion/z_group_diag.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

import fx

SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
RAW = fx.ROOT / "data" / "sophelio" / "fusion-equilibrium-challenge" / "data"


def rate(path):
    t = pd.read_parquet(path, columns=["magnetics_time"]).iloc[0]["magnetics_time"]
    return float(np.median(np.diff(np.asarray(t[:2000], np.float64))))


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    meta = torch.load(fx.ROOT / "data" / "sophelio_proc" / "cache_k256" / "meta.pt", weights_only=False); names = list(meta["names"])
    pub = sorted((RAW / "diii_d_public_test").glob("*.parquet"))
    with mp.Pool(8) as pool:
        rt = np.array(pool.map(rate, [RAW / "diii_d_train" / f"{n}.parquet" for n in names], chunksize=16))
        rp = np.array(pool.map(rate, pub, chunksize=16))
    fast_t, fast_p = rt < 0.2, rp < 0.2
    sg = json.loads((fx.OUT / "ip_sign.json").read_text())
    d = sota.Data(); vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); rows, y = vt["rows"], vt["y"][:, :7].astype(np.float64)
    sh = d.shot[rows]; vt_shots = np.unique(sh)
    print(f"[zg] 20 kHz share: all labelled {fast_t.mean():.3f} | VT shots {fast_t[vt_shots].mean():.3f} | public {fast_p.mean():.3f} (n {len(rp)})", flush=True)
    neg = np.array([sg.get(str(int(s)), 1) < 0 for s in range(len(names))])
    print(f"[zg] reversed-Ip share: VT {neg[vt_shots].mean():.3f}", flush=True)
    p = GB.heads_gpu(d, rows, "head_boundary_v8_J1RL_s1+head_boundary_v8_J1RL_s2")[:, :7].astype(np.float64)
    var = np.nanvar(y, 0)
    def r2(m):
        out = []
        for j in range(7):
            ok = m & np.isfinite(y[:, j]); out.append(1 - np.mean((p[ok, j] - y[ok, j]) ** 2) / var[j])
        return np.array(out)
    for lab, m in (("all", np.ones(len(rows), bool)), ("20kHz", fast_t[sh]), ("2kHz", ~fast_t[sh]), ("Ip+", ~neg[sh]), ("Ip-", neg[sh])):
        r = r2(m); print(f"[zg] VT {lab:6s} frames {m.sum():7d} shots {len(np.unique(sh[m])):4d} (R2 vs pooled VT variance) " + " ".join(f"{n} {v:.5f}" for n, v in zip(SC, r)), flush=True)
    np.savez(fx.OUT / "z_group_diag.npz", rate_train=rt, rate_public=rp)
    print("ZG_DONE", flush=True)


if __name__ == "__main__":
    main()
