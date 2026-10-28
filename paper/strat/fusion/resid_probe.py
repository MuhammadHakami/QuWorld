"""Fusion §41q: is the J1 heads' consistency-scalar error predictable from long-history inputs? Residual = label - 6-seed J1
mean on the VALID sample rows; HistGradientBoosting on [head prediction, frame inputs subset, (hist features)], 5-fold
GroupKFold by shot; reports the R2 of each scalar before / after adding the OOF residual prediction (CPU, GBDT allowed).

    .venv/bin/python paper/strat/fusion/resid_probe.py [hist|base]
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import fx  # noqa: F401,I001
sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio"))
import sota  # noqa: E402

NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def main() -> None:
    d = sota.Data(); lab = np.load(fx.OUT / "boundary_labels_v8.npz"); va, Y = lab["rows_valid"], lab["y_valid"][:, :7].astype(np.float64)
    P = np.mean([np.load(fx.OUT / f"head_boundary_v8_J1_s{s}.npz")["pred_valid"][:, :7] for s in range(1, 7)], 0).astype(np.float64)
    if os.environ.get("RP_QB") == "1":                                         # q95 / betaN: 6-seed QH heads
        P = np.mean([np.load(fx.OUT / f"head_qb_QH_s{s}.npz")["pred_valid"] for s in range(1, 7)], 0).astype(np.float64)
        Y = np.stack([d.S[va, sota.LABELS.index(h)] for h in ("efit_q95", "efit_beta_n")], 1).astype(np.float64)
        NAMES[:] = ["q95", "betaN"]
    base = [d.fn.index(n) for n in d.fn if "_lag" not in n and n.startswith("mag_")] + [d.fn.index("t")]
    sets = {"base": np.concatenate([P, np.asarray(d.X[va][:, base]), d.edge_feats()[va]], 1)}
    if "hist" in sys.argv[1:] or len(sys.argv) == 1:
        sets["hist"] = np.concatenate([sets["base"], np.asarray(np.load(sota.CACHE / "hist.npy", mmap_mode="r")[va])], 1)
    if "hfull" in sys.argv[1:]:                                                # §41q full-rate history (hist_full.py)
        sets["hfull"] = np.concatenate([sets["base"], np.asarray(np.load(fx.OUT / "H_full_all.npy", mmap_mode="r")[va])], 1)
    g = d.shot[va]
    for nm in sys.argv[1:] or ["base", "hist"]:
        F = sets[nm]; out = []
        for j, sc in enumerate(NAMES):
            ok = np.isfinite(Y[:, j]); r = Y[:, j] - P[:, j]; oof = np.zeros(len(r))
            for tr, te in GroupKFold(5).split(F, r, g):
                tr, te = tr[ok[tr]], te
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0)
                m.fit(F[tr], r[tr]); oof[te] = m.predict(F[te])
            var = np.var(Y[ok, j]); r2a = 1 - np.mean(r[ok] ** 2) / var; r2b = 1 - np.mean((r[ok] - oof[ok]) ** 2) / var
            out.append(f"{sc} {r2a:.4f}->{r2b:.4f}")
        print(f"[resid {nm}] features {F.shape[1]}: " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
