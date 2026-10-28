"""§37 fusion, strategies G-qb and G-coef: CatBoost (CPU, 8 threads) residual correctors on top of the entry ensemble,
fit on the TRAIN subsample only (features.py), applied to every VALID frame. Run in .venv-tdc (catboost 1.2.10).

    .venv-tdc/bin/python paper/strat/fusion/gbdt.py [--no-ip]  ->  runs/strat/fusion/gbdt{_noip}.npz  (+ .json)

G-qb:   q95 and betaN residuals (label - entry head), one RMSE model each.
G-coef: residuals of PCA coefficients 0-15 (cache coefficients - entry mean), one MultiRMSE model (the PCA basis is
        orthonormal, so the summed squared coefficient error is the psi-space MSE of that band).
--no-ip drops the five re-aligned Ip features (ablation: is the gain the Ip information or the stacking itself?).
Early stopping uses a shot-grouped 15 % split of the TRAIN subsample (never VALID).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
from catboost import CatBoostRegressor

OUT = pathlib.Path(__file__).resolve().parents[3] / "runs" / "strat" / "fusion"
IPF = ["ipfix", "ipfix_m200", "ipfix_m50", "ipfix_p20", "dipdt"]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--no-ip", action="store_true"); ap.add_argument("--iters", type=int, default=800)
    a = ap.parse_args()
    tr = np.load(OUT / "feat_train.npz"); va = np.load(OUT / "feat_valid.npz")
    names = [str(n) for n in tr["names"]]
    keep = np.array([not (a.no_ip and n in IPF) for n in names])
    Xtr, Xva = tr["X"][:, keep], va["X"][:, keep]
    Ytr = tr["Y"]; shots = np.unique(tr["shot"])
    hold = set(np.random.default_rng(0).choice(shots, int(0.15 * len(shots)), replace=False).tolist())
    es = np.isin(tr["shot"], list(hold)); fit = ~es
    hq, hb = names.index("h_efit_q95"), names.index("h_efit_beta_n")
    out, info = {}, {}
    t0 = time.time()
    for j, (nm, hcol) in enumerate((("q95", hq), ("betaN", hb))):
        y = Ytr[:, 64 + j] - tr["X"][:, hcol]; ok = np.isfinite(y)
        m = CatBoostRegressor(iterations=a.iters, depth=6, learning_rate=0.08, loss_function="RMSE", thread_count=8,
                              random_seed=0, verbose=0, od_type="Iter", od_wait=60)
        m.fit(Xtr[fit & ok], y[fit & ok], eval_set=(Xtr[es & ok], y[es & ok]))
        out[nm] = m.predict(Xva).astype(np.float32)
        info[nm] = dict(best_iter=int(m.get_best_iteration()), holdout_rmse_before=float(np.sqrt(np.mean(y[es & ok] ** 2))),
                        holdout_rmse_after=float(np.sqrt(np.mean((y[es & ok] - m.predict(Xtr[es & ok])) ** 2))))
        print(f"[gbdt] {nm}: {info[nm]} {time.time() - t0:.0f}s", flush=True)
    R = Ytr[:, :16] - tr["X"][:, :16]
    m = CatBoostRegressor(iterations=a.iters, depth=6, learning_rate=0.08, loss_function="MultiRMSE", thread_count=8,
                          random_seed=0, verbose=0, od_type="Iter", od_wait=60)
    m.fit(Xtr[fit], R[fit], eval_set=(Xtr[es], R[es]))
    out["coef16"] = m.predict(Xva).astype(np.float32)
    pe = m.predict(Xtr[es])
    info["coef16"] = dict(best_iter=int(m.get_best_iteration()), holdout_ss_before=float((R[es] ** 2).sum(1).mean()),
                          holdout_ss_after=float(((R[es] - pe) ** 2).sum(1).mean()))
    fi = m.get_feature_importance(); kn = [n for n, k in zip(names, keep) if k]
    info["coef16"]["top_features"] = [(kn[i], float(fi[i])) for i in np.argsort(fi)[::-1][:15]]
    print(f"[gbdt] coef16: {info['coef16']} {time.time() - t0:.0f}s", flush=True)
    tag = "_noip" if a.no_ip else ""
    np.savez(OUT / f"gbdt{tag}.npz", rows=va["rows"], shot=va["shot"], **out)
    (OUT / f"gbdt{tag}.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
