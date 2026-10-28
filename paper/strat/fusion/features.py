"""§37 fusion: design matrices for the TRAIN-fit correctors (TRAIN subsample + all VALID; TEST never loaded).

    .venv/bin/python paper/strat/fusion/features.py [--n-train 400]  ->  runs/strat/fusion/feat_{train,valid}.npz

Per frame: entry-ensemble outputs (mean over the 7 members of the 64 PCA coefficients and the 5 scalar heads, plus the
member spread of coefficients 0-15), the entry's own +-1 / +-3-frame neighbours of coefficients 0-15 (temporal context
of the predictions), the re-aligned Ip features (ip_fixed.npy: Ip, Ip(t-200), Ip(t-50), Ip(t+20), dIp/dt), the 21
shipped coil/Ip signals, t and the shot-phase features. Targets: PCA coefficients 0-63 (exact cache coefficients) and
the q95 / betaN labels. TRAIN predictions are in-sample for the members (they were fit on TRAIN): the script reports
TRAIN vs VALID residual RMS per coefficient band so that bias is visible.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import fx


def neighbours(a: np.ndarray, shot: np.ndarray, k: int) -> np.ndarray:
    """a[t+k] within the same shot (edge-clamped)."""
    n = len(a); idx = np.clip(np.arange(n) + k, 0, n - 1)
    same = shot[idx] == shot
    idx = np.where(same, idx, np.arange(n))
    return a[idx]


def build(d, sota, shots, preds_fn, ip, label: str):
    rows = np.concatenate([np.nonzero(d.shot == s)[0] for s in shots])
    shot = d.shot[rows]
    coef, scal, spread = [], [], []
    for s in shots:
        cs, ss = preds_fn(s)                                     # (M, T, 64), (M, T, 5)
        coef.append(cs.mean(0)); scal.append(ss.mean(0)); spread.append(cs[:, :, :16].std(0))
    coef, scal, spread = np.concatenate(coef), np.concatenate(scal), np.concatenate(spread)
    nb = [neighbours(coef[:, :16], shot, k) for k in (-3, -1, 1, 3)]
    fm = d.feature_mask("none", False)                           # 21 shipped signals + t
    X = np.concatenate([coef, scal, spread] + nb + [np.asarray(ip[rows, :5]), np.asarray(d.X[rows][:, fm]),
                        d.edge_feats()[rows]], 1).astype(np.float32)
    names = ([f"c{i}" for i in range(64)] + [f"h_{h}" for h in sota.HEADS] + [f"sd_c{i}" for i in range(16)]
             + [f"nb{k}_c{i}" for k in (-3, -1, 1, 3) for i in range(16)]
             + ["ipfix", "ipfix_m200", "ipfix_m50", "ipfix_p20", "dipdt"] + [n for n, m in zip(d.fn, fm) if m] + ["t_to_end", "frac"])
    L = sota.LABELS
    Y = np.concatenate([np.asarray(d.C[rows, :64]), d.S[rows][:, [L.index("efit_q95"), L.index("efit_beta_n")]]], 1).astype(np.float32)
    np.savez(fx.OUT / f"feat_{label}.npz", X=X, Y=Y, shot=shot, rows=rows, names=np.array(names))
    res = Y[:, :64] - coef
    return {band: float(np.sqrt((res[:, a:b] ** 2).sum(1).mean())) for band, (a, b) in
            {"0-4": (0, 4), "4-16": (4, 16), "16-64": (16, 64)}.items()} | {
        "q95_rmse": float(np.sqrt(np.nanmean((Y[:, 64] - scal[:, sota.HEADS.index("efit_q95")]) ** 2))),
        "betaN_rmse": float(np.sqrt(np.nanmean((Y[:, 65] - scal[:, sota.HEADS.index("efit_beta_n")]) ** 2))), "frames": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n-train", type=int, default=400)
    a = ap.parse_args()
    sota, d = fx.data()
    ip = np.load(fx.OUT / "ip_fixed.npy", mmap_mode="r")
    tr_all = d.shots_of(0)
    tr = sorted(int(s) for s in np.random.default_rng(0).choice(tr_all, a.n_train, replace=False))
    rows_tr = {s: np.nonzero(d.shot == s)[0] for s in tr}
    mem = [fx.member_predict(fx.TRUNKS / m, d, rows_tr) for m in fx.ENTRY]
    rep = {"train": build(d, sota, tr, lambda s: (np.stack([m[s][0] for m in mem]), np.stack([m[s][1] for m in mem])), ip, "train")}
    z = np.load(fx.OUT / "pred_valid.npz"); M = len(fx.ENTRY)
    valid = d.shots_of(1)
    rep["valid"] = build(d, sota, valid, lambda s: (np.stack([z[f"{i}_{s}_c"] for i in range(M)]),
                                                     np.stack([z[f"{i}_{s}_s"] for i in range(M)])), ip, "valid")
    rep["train_shots"] = tr
    (fx.OUT / "features.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "train_shots"}, indent=1))


if __name__ == "__main__":
    main()
