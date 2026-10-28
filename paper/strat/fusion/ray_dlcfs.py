"""Doc 35 §41z diagnostic: is D_LCFS lost in the heads or in the head -> psi transfer?
On every VALID + TEST frame (TRAIN-only heads), the official D_LCFS (Hausdorff to the reference LCFS / rgeo, clamp 1) of
  (a) the polygon of the TRUE 64 boundary rays (the 64-ray representation floor),
  (b) the polygon of the head ensemble's rays around the head axis,
and, when runs/strat/fusion/parts_VT-<tag>.pkl exists, (c) the submitted-psi D of that arm (same frames, same aggregation).
    .venv/bin/python paper/strat/fusion/ray_dlcfs.py <'+'-joined heads> [parts tag]   -> runs/strat/fusion/ray_dlcfs.npz
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import torch

import fx
import gpu_pipe as GP

N_ANG, DENSE = 64, 8


def polygon(ax: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """(B, 2) axis, (B, 64) ray radii -> (B, 64 * DENSE, 2) closed polygon, linearly densified between ray tips."""
    th = 2 * np.pi * torch.arange(N_ANG, device=r.device, dtype=r.dtype) / N_ANG
    V = ax[:, None, :] + r[..., None] * torch.stack([torch.cos(th), torch.sin(th)], 1)[None]
    W = torch.roll(V, -1, 1); s = torch.arange(DENSE, device=r.device, dtype=r.dtype) / DENSE
    return (V[:, :, None, :] + s[None, None, :, None] * (W - V)[:, :, None, :]).reshape(len(r), -1, 2)


def hausdorff(P: torch.Tensor, Q: torch.Tensor, rgeo: torch.Tensor) -> torch.Tensor:
    D = torch.cdist(P, Q); d1 = D.min(2).values.amax(1); d2 = D.min(1).values.amax(1)
    return (torch.maximum(d1, d2) / rgeo).clamp(max=1.0)


def shot_mean(v: np.ndarray, shot: np.ndarray, ok: np.ndarray) -> float:
    return float(np.mean([v[(shot == s) & ok].mean() for s in np.unique(shot[ok])]))


def main() -> None:
    heads, tag = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    d = sota.Data(); shots = [int(s) for s in np.unique(d.shot[d.split != 0])]
    rows = [fx.scored_rows(d, s) for s in shots]; allr = np.concatenate(rows); shot = d.shot[allr]
    Cref, rgeo, _, has = GP.refs(d, shots)
    lab = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); pos = {int(r): i for i, r in enumerate(lab["rows"])}
    li = np.array([pos.get(int(r), -1) for r in allr]); Y = np.full((len(allr), 71), np.nan)
    Y[li >= 0] = lab["y"][li[li >= 0]]
    H = GB.heads_gpu(d, allr, heads)
    dev, f = torch.device("cuda"), lambda z: torch.tensor(np.asarray(z), device="cuda", dtype=torch.float32)
    dt, dh = np.full(len(allr), np.nan), np.full(len(allr), np.nan)
    for a in range(0, len(allr), 1024):
        c = slice(a, a + 1024); Q, rg = f(Cref[c]), f(rgeo[c])
        dh[c] = hausdorff(polygon(f(H[c, :2]), f(H[c, 7:])), Q, rg).cpu().numpy()
        dt[c] = hausdorff(polygon(f(Y[c, :2]), f(np.nan_to_num(Y[c, 7:]))), Q, rg).cpu().numpy()
    okt = has & np.isfinite(Y[:, 7:]).all(1) & np.isfinite(dt); okh = has & np.isfinite(dh)
    rep = {"true_rays": shot_mean(dt, shot, okt), "head_rays": shot_mean(dh, shot, okh),
           "head_rays_on_true_ok": shot_mean(dh, shot, okt)}
    out = dict(dt=dt, dh=dh, shot=shot, has=has)
    if tag:
        P = pickle.load(open(fx.OUT / f"parts_VT-{tag}.pkl", "rb")); dp = np.concatenate([P[s]["d"] for s in shots])
        okp = has & np.isfinite(dp); rep[f"psi_{tag}"] = shot_mean(dp, shot, okp); rep[f"psi_{tag}_on_true_ok"] = shot_mean(dp, shot, okt)
        out["dp"] = dp
        for q in (50, 90, 99):
            rep[f"p{q}"] = {k: float(np.percentile(v[okt], q)) for k, v in (("true", dt), ("head", dh), ("psi", dp))}
    np.savez(fx.OUT / "ray_dlcfs.npz", **out)
    print("[ray_dlcfs]", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rep.items()}, flush=True)


if __name__ == "__main__":
    main()
