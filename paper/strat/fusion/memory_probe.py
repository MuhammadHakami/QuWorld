"""Doc 35 §38z-pre: is the end-of-shot error a missing-memory problem? (uses the trained §38u out-of-fold heads; no training)

Hypothesis: li (and the shape scalars that follow it) at ramp-down depend on the current-profile history over the resistive
time (~0.5-3 s), but the heads see at most 200 ms of input history. Test on TRAIN frames in the last 15% of each shot: GBDT
(5-fold by shot) predicting the out-of-fold head residual from (a) instantaneous context [head output, Ip, Ip/Ip_max, t,
elapsed fraction, time to end] vs (b) the same + long-memory features of the plasma current [EWMA of Ip and of dIp/dt with
tau 0.25 / 0.5 / 1 / 2 s, time since Ip peak, integral of Ip, Ip at flat-top]. Read-out: out-of-fold R2 of the residual per
scalar for (a) and (b); memory matters if (b) - (a) >= 0.05 on li.

    .venv/bin/python paper/strat/fusion/memory_probe.py   -> runs/strat/fusion/memory_probe.json
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import fx


def ewma(x: np.ndarray, t: np.ndarray, tau: float) -> np.ndarray:
    out = np.empty_like(x); acc = x[0]
    for i in range(len(x)):
        a = np.exp(-(t[i] - t[i - 1]) / tau) if i else 0.0
        acc = a * acc + (1 - a) * x[i]; out[i] = acc
    return out


def main() -> None:
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); names = [str(x) for x in lab["names"]]
    pos = {r: i for i, r in enumerate(lab["rows_train"])}
    rows, pred = [], []
    for k in range(5):
        z = np.load(fx.OUT / f"head_boundary_v8_J1F{k}_s1_oof.npz"); rows.append(z["rows_oof"]); pred.append(z["pred_oof"][:, :7])
    rows, pred = np.concatenate(rows), np.concatenate(pred); o = np.argsort(rows); rows, pred = rows[o], pred[o]
    y = lab["y_train"][[pos[r] for r in rows], :7]; sd = np.nanstd(lab["y_train"][:, :7], 0)
    sota, d = fx.data(); fn = list(d.fn); X = d.X
    ip = np.abs(np.asarray(X[rows][:, fn.index("mag_plasma_current")], np.float64)); t = np.asarray(X[rows][:, fn.index("t")], np.float64) / 1000.0
    shot = np.load(fx.ROOT / "data/sophelio_proc/cache_k256/shot.npy")[rows]
    us, inv = np.unique(shot, return_inverse=True); starts = np.r_[0, np.cumsum(np.bincount(inv))[:-1]]; ends = np.cumsum(np.bincount(inv))
    inst, mem, keep = [], [], []
    for a, b in zip(starts, ends):
        tt, I = t[a:b], ip[a:b]; n = b - a
        if n < 10:
            continue
        dI = np.gradient(I, tt + 1e-9 * np.arange(n)); imax = I.max(); kmax = int(I.argmax()); frac = np.arange(n) / (n - 1)
        inst.append(np.c_[I, I / max(imax, 1e-9), tt, frac, tt[-1] - tt])
        cols = [ewma(I, tt, tau) for tau in (0.25, 0.5, 1.0, 2.0)] + [ewma(dI, tt, tau) for tau in (0.25, 0.5, 1.0, 2.0)]
        cols += [np.maximum(tt - tt[kmax], 0), np.cumsum(I * np.r_[0, np.diff(tt)]), np.full(n, np.median(I[I > 0.8 * imax]))]
        mem.append(np.stack(cols, 1)); keep.append(np.arange(a, b)[frac >= 0.85])
    inst = np.concatenate(inst); mem = np.concatenate(mem); idx_all = np.concatenate([np.arange(a, b) for a, b in zip(starts, ends) if b - a >= 10])
    pos_in = {r: i for i, r in enumerate(idx_all)}; sel = np.array([pos_in[i] for i in np.concatenate(keep)])
    rep = dict(frames=int(len(sel)), scalars={})
    g = shot[idx_all][sel]
    for j, n in enumerate(names):
        r = ((pred[:, j] - y[:, j]) / sd[j])[idx_all][sel]; ok = np.isfinite(r)
        A = np.c_[pred[idx_all][sel][:, j], inst[sel]][ok]; B = np.c_[A, mem[sel][ok]]; rr, gg = r[ok], g[ok]
        out = {}
        for nm, F in (("instant", A), ("memory", B)):
            p = np.zeros_like(rr)
            for tr, te in GroupKFold(5).split(F, rr, gg):
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                                  early_stopping=True, validation_fraction=0.15, random_state=0)
                p[te] = m.fit(F[tr], rr[tr]).predict(F[te])
            out[nm] = float(1 - np.sum((rr - p) ** 2) / np.sum((rr - rr.mean()) ** 2))
        rep["scalars"][n] = out
        print(f"[memory] {n:8s} end-of-shot residual OOF R2: instant {out['instant']:+.3f} | + memory {out['memory']:+.3f}", flush=True)
    (fx.OUT / "memory_probe.json").write_text(json.dumps(rep, indent=1))
    print("MEMORY_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
