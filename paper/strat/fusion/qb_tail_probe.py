"""Doc 35 §38ab-pre: are the q95 / betaN tail errors learnable? (CPU, GBDT; no training of networks)

§38aa anatomy (HJl9 sample): the worst 1% of frames hold 46% (q95) / 37% (betaN) of the squared error; the top 5% of true
q95 (low Ip: ramp-up / ramp-down) hold 38%. Here, on the 180-shot VALID sample, the deployed q/b prediction (0.5 x 6-seed QH
heads + 0.5 x 7-member entry) gets a residual model: GBDT (5-fold by shot) on [q, betaN predictions, |Ip|, 1/|Ip|, q x |Ip|,
time, fraction of the shot, the J1 heads' kappa / volume / li / R_axis]. Read-out: out-of-fold R2 of the residual and the
r2_q95 / r2_betaN / S change if the OOF correction is applied. A gain >= +0.0001 S registers a calibration stage (fitted
on the whole sample, applied once to TEST).

    .venv/bin/python paper/strat/fusion/qb_tail_probe.py   -> runs/strat/fusion/qb_tail_probe.json
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import fx  # noqa: I001
import strategies as ST


def main() -> None:
    sota, d = fx.data(); shots, _ = ST.sample(); f, comps, mean, H = ST.entry_preds(d, sota)
    iq, ib = H.index("efit_q95"), H.index("efit_beta_n"); L = sota.LABELS; fn = list(d.fn)
    rows = np.concatenate([fx.valid_rows(d, s) for s in shots])
    def head(stem):
        z = np.load(fx.OUT / f"{stem}.npz"); pos = {int(r): i for i, r in enumerate(z["rows_valid"])}
        return z["pred_valid"][[pos[int(r)] for r in rows]].astype(np.float64)
    qh = np.mean([head(f"head_qb_QH_s{s}") for s in range(1, 7)], 0)[:, :2]
    ent = np.concatenate([f(s)[1].mean(0)[:, [iq, ib]] for s in shots])
    pred = 0.5 * qh + 0.5 * ent
    sh = np.mean([head(f"head_boundary_v8_J1_s{s}") for s in range(1, 7)], 0)
    y = np.stack([d.S[rows, L.index("efit_q95")], d.S[rows, L.index("efit_beta_n")]], 1).astype(np.float64)
    X = np.asarray(d.X[rows], np.float64); ip = np.abs(X[:, fn.index("mag_plasma_current")]); t = X[:, fn.index("t")]
    grp = np.concatenate([np.full(len(fx.valid_rows(d, s)), s) for s in shots])
    frac = np.concatenate([np.arange(len(fx.valid_rows(d, s))) / max(len(fx.valid_rows(d, s)) - 1, 1) for s in shots])
    F = np.c_[pred, ip, 1 / np.maximum(ip, 1e-3 * ip.max()), pred[:, 0] * ip, t, frac, sh[:, [0, 2, 5, 6]]]
    rep = {}
    for j, nm in enumerate(("q95", "betaN")):
        r = y[:, j] - pred[:, j]; oof = np.zeros_like(r)
        for tr, te in GroupKFold(5).split(F, r, grp):
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=40,
                                              early_stopping=True, validation_fraction=0.15, random_state=0)
            oof[te] = m.fit(F[tr], r[tr]).predict(F[te])
        v = np.var(y[:, j]); r2_0 = 1 - np.mean(r ** 2) / v; r2_1 = 1 - np.mean((r - oof) ** 2) / v
        rep[nm] = dict(r2_before=float(r2_0), r2_after=float(r2_1), resid_oof_r2=float(1 - np.mean((r - oof) ** 2) / np.var(r)))
        print(f"[qbtail] {nm}: R2 {r2_0:.5f} -> {r2_1:.5f} (residual OOF R2 {rep[nm]['resid_oof_r2']:+.3f})", flush=True)
    dqb = 0.5 * sum(rep[k]["r2_after"] - rep[k]["r2_before"] for k in rep); rep["dS"] = float(0.15 * dqb)
    print(f"[qbtail] implied S change {rep['dS']:+.5f} (pooled-frame R2; the official per-split R2 differs slightly)", flush=True)
    (fx.OUT / "qb_tail_probe.json").write_text(json.dumps(rep, indent=1))
    print("QB_TAIL_DONE", flush=True)


if __name__ == "__main__":
    main()
