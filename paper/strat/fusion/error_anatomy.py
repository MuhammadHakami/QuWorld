"""Doc 35 §38x-pre: where does the shape-head error live? (diagnosis on the §38u out-of-fold J1 heads, all TRAIN frames)

For each consistency scalar: (1) calibration - slope of truth on prediction and the R2 a linear recalibration would give;
(2) error share by phase of the shot (normalised time deciles) and by plasma-current fraction (Ip / shot max; ramp-up,
flat-top, ramp-down); (3) error share by regime - quintiles of the shot-mean true value and of shot-mean Ip; (4) shot length.
Shares are of the summed squared standardized error; a bin with share >> its frame share is a weak spot.

    .venv/bin/python paper/strat/fusion/error_anatomy.py   -> runs/strat/fusion/error_anatomy.json
"""
from __future__ import annotations

import json
import re

import numpy as np

import fx


def main() -> None:
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); names = [str(x) for x in lab["names"]]
    pos = {r: i for i, r in enumerate(lab["rows_train"])}
    rows, pred = [], []
    for k in range(5):
        z = np.load(fx.OUT / f"head_boundary_v8_J1F{k}_s1_oof.npz"); rows.append(z["rows_oof"]); pred.append(z["pred_oof"][:, :7])
    rows, pred = np.concatenate(rows), np.concatenate(pred)
    o = np.argsort(rows); rows, pred = rows[o], pred[o]
    y = lab["y_train"][[pos[r] for r in rows], :7]; sd = np.nanstd(lab["y_train"][:, :7], 0)
    shot = np.load(fx.ROOT / "data/sophelio_proc/cache_k256/shot.npy")[rows]
    sota, d = fx.data(); fn = list(d.fn)
    ip_col = [i for i, n in enumerate(fn) if re.fullmatch(r"mag_ip|mag_plasma_current|ip", n)]
    ip_col = ip_col or [i for i, n in enumerate(fn) if "current" in n.lower() and "lag" not in n]
    ip = np.abs(np.asarray(d.X[rows][:, ip_col[0]], np.float64)) if ip_col else None
    us, inv = np.unique(shot, return_inverse=True); n_sh = np.bincount(inv)
    tpos = np.zeros(len(rows)); start = np.r_[0, np.cumsum(n_sh)[:-1]]
    tpos = (np.arange(len(rows)) - start[inv]) / np.maximum(n_sh[inv] - 1, 1)
    rep = dict(frames=len(rows), shots=len(us), ip_feature=fn[ip_col[0]] if ip_col else None, scalars={})
    for j, n in enumerate(names):
        e = (pred[:, j] - y[:, j]) / sd[j]; ok = np.isfinite(e); E2 = np.where(ok, e ** 2, 0); tot = E2.sum()
        A = np.c_[pred[ok, j], np.ones(ok.sum())]; coef, *_ = np.linalg.lstsq(A, y[ok, j], rcond=None)
        r2 = 1 - np.mean((pred[ok, j] - y[ok, j]) ** 2) / np.var(y[ok, j]); r2c = 1 - np.mean((A @ coef - y[ok, j]) ** 2) / np.var(y[ok, j])
        def shares(b, nb):
            return {int(i): [round(float(E2[b == i].sum() / tot), 3), round(float(np.mean(b == i)), 3)] for i in range(nb)}
        phase = np.minimum((tpos * 10).astype(int), 9)
        ipf = None
        if ip is not None:
            ipmax = np.zeros(len(us)); np.maximum.at(ipmax, inv, ip); fr = ip / np.maximum(ipmax[inv], 1e-9)
            ramp = np.where(fr < 0.8, np.where(tpos < 0.5, 0, 2), 1)          # 0 ramp-up, 1 flat-top (>= 80% Ip), 2 ramp-down
            ipf = shares(ramp, 3)
        ys = np.bincount(inv, np.where(np.isfinite(y[:, j]), y[:, j], 0)) / np.maximum(np.bincount(inv, np.isfinite(y[:, j])), 1)
        qy = np.digitize(ys, np.quantile(ys, [0.2, 0.4, 0.6, 0.8]))[inv]
        ql = np.digitize(n_sh, np.quantile(n_sh, [0.2, 0.4, 0.6, 0.8]))[inv]
        rep["scalars"][n] = dict(r2=float(r2), slope=float(coef[0]), r2_recal=float(r2c), phase_deciles=shares(phase, 10),
                                 ramp_flat_down=ipf, true_value_quintiles=shares(qy, 5), shot_length_quintiles=shares(ql, 5))
        print(f"[anatomy] {n:8s} R2 {r2:.4f} slope(truth~pred) {coef[0]:.3f} recal R2 {r2c:.4f}", flush=True)
        print(f"   [err share, frame share] ramp-up/flat/ramp-down {ipf}", flush=True)
        print(f"   phase deciles {[v[0] for v in shares(phase, 10).values()]}", flush=True)
        print(f"   true-value quintiles {shares(qy, 5)}", flush=True)
    (fx.OUT / "error_anatomy.json").write_text(json.dumps(rep, indent=1))
    print(f"ANATOMY_DONE ip feature: {rep['ip_feature']}", flush=True)


if __name__ == "__main__":
    main()
