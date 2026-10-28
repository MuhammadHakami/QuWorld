"""Doc 35 §38v-pre: do the heads' per-shot offsets follow the core Thomson channel geometry?

extract.py resamples the core Thomson profiles along channel order (24 points), but the channel heights differ between
shots (40/42/44/54 channels, top channel Z 0.68-0.93 m): profile point k is a different height in different
configurations. If that drives the per-shot offsets the oracle removes (§38r-pre), the out-of-fold J1 residuals (§38u,
all TRAIN shots) should cluster by configuration. Read-out per scalar: eta^2 = share of the variance of shot-mean
residuals explained by the configuration (channel count x top-Z rounded to 1 cm), and its correlation with top Z.

    .venv/bin/python paper/strat/fusion/ts_geometry_probe.py   -> runs/strat/fusion/ts_geometry_probe.json
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import torch

import fx

RAW = fx.ROOT / "data/sophelio/fusion-equilibrium-challenge/data/diii_d_train"


def core_geometry(name: str) -> tuple[int, float]:
    t = pq.read_table(RAW / f"{name}.parquet", columns=["thomson_chord_name", "thomson_chord_Z"])
    nm = t.column("thomson_chord_name")[0].as_py(); Z = t.column("thomson_chord_Z")[0].as_py()
    zc = [z for n, z in zip(nm, Z) if n.startswith("TS_core")]
    return len(zc), float(max(zc)) if zc else float("nan")


def main() -> None:
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); names7 = [str(x) for x in lab["names"]]
    pos = {r: i for i, r in enumerate(lab["rows_train"])}
    rows, pred = [], []
    for k in range(5):
        z = np.load(fx.OUT / f"head_boundary_v8_J1F{k}_s1_oof.npz"); rows.append(z["rows_oof"]); pred.append(z["pred_oof"][:, :7])
    rows, pred = np.concatenate(rows), np.concatenate(pred)
    y = lab["y_train"][[pos[r] for r in rows], :7]; sd = lab["y_train"][:, :7].std(0)
    shot = np.load(fx.ROOT / "data/sophelio_proc/cache_k256/shot.npy")[rows]
    names = torch.load(fx.ROOT / "data/sophelio_proc/cache_k256/meta.pt", weights_only=False)["names"]
    us, inv = np.unique(shot, return_inverse=True)
    res = np.zeros((len(us), 7)); np.add.at(res, inv, (pred - y) / sd); res /= np.bincount(inv)[:, None]
    geo = [core_geometry(names[s]) for s in us]; n_ch = np.array([g[0] for g in geo]); ztop = np.array([g[1] for g in geo])
    cfg = np.array([f"{n}_{z:.2f}" for n, z in geo]); uc, ci = np.unique(cfg, return_inverse=True)
    rep = dict(shots=len(us), configs={c: int((cfg == c).sum()) for c in uc}, scalars={})
    for j, nmj in enumerate(names7):
        r = res[:, j]; gm = np.bincount(ci, r) / np.bincount(ci)
        eta2 = float(np.sum(np.bincount(ci) * (gm - r.mean()) ** 2) / np.sum((r - r.mean()) ** 2))
        ok = np.isfinite(ztop); rho = float(np.corrcoef(r[ok], ztop[ok])[0, 1])
        rep["scalars"][nmj] = dict(eta2=eta2, corr_ztop=rho, shot_rms=float(np.sqrt((r ** 2).mean())))
        print(f"[tsgeo] {nmj:8s} shot-offset rms {np.sqrt((r ** 2).mean()):.3f} sd | eta2(config) {eta2:.3f} | corr(top Z) {rho:+.3f}", flush=True)
    # null: eta2 with configs shuffled across shots (same group sizes)
    rng = np.random.default_rng(0); null = []
    for _ in range(50):
        cp = rng.permutation(ci)
        null.append(np.mean([np.sum(np.bincount(cp) * (np.bincount(cp, res[:, j]) / np.bincount(cp) - res[:, j].mean()) ** 2)
                             / np.sum((res[:, j] - res[:, j].mean()) ** 2) for j in range(7)]))
    rep["null_eta2_mean"] = float(np.mean(null)); rep["null_eta2_p95"] = float(np.percentile(null, 95))
    print(f"[tsgeo] {len(us)} shots, {len(uc)} configs; shuffled-config eta2 mean {np.mean(null):.3f} (p95 {np.percentile(null, 95):.3f})", flush=True)
    (fx.OUT / "ts_geometry_probe.json").write_text(json.dumps(rep, indent=1))
    print("TSGEO_DONE", flush=True)


if __name__ == "__main__":
    main()
