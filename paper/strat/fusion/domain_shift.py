"""Doc 35 §45k diagnostic (label-free on public): is the public fold covariate-shifted from the labelled data? A CatBoost classifier (CPU,
allowed for GBDT) separates labelled frames (TRAIN every 3rd + VT, the §45a J1RL design matrix) from public frames (gbdt_shape_Xpub), with
3-fold cross-fitting grouped by shot. AUC near 0.5 = no exploitable shift (importance weighting cannot help); a high AUC names the input
blocks that differ, and the out-of-fold propensities give importance weights for a reweighting test.
    .venv-tdc/bin/python paper/strat/fusion/domain_shift.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "paper/strat/fusion")
OUT = __import__("pathlib").Path("runs/strat/fusion")


def main() -> None:
    z = np.load(OUT / "gbdt_shape_data.npz"); rng = np.random.default_rng(0)
    Xl = np.concatenate([z["Xtr"], z["Xvt"]]); rl = np.concatenate([z["rtr"], z["rvt"]])
    Xp = np.load(OUT / "gbdt_shape_Xpub.npy", mmap_mode="r")
    il = rng.choice(len(Xl), 60000, replace=False); ip = np.sort(rng.choice(len(Xp), 60000, replace=False))
    X = np.concatenate([Xl[il], np.asarray(Xp[ip])]).astype(np.float32); y = np.r_[np.zeros(60000), np.ones(60000)]
    # groups = true shots (labelled: cache shot id of the row; public: public shot id) - per-shot constants (hf_ramp0) leak otherwise
    sl = np.load(OUT / "shift_shot_lab.npy"); sp = np.load(OUT / "shift_shot_pub.npy")
    g = np.r_[sl[rl[il]], 10 ** 7 + sp[ip]]
    ug = np.unique(g); fold = {u: i % 3 for i, u in enumerate(rng.permutation(ug))}; f = np.array([fold[u] for u in g])
    p = np.zeros(len(y)); imp = np.zeros(X.shape[1])
    for k in range(3):
        m = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.1, verbose=0, thread_count=16, random_seed=0)
        m.fit(X[f != k], y[f != k]); p[f == k] = m.predict_proba(X[f == k])[:, 1]; imp += m.get_feature_importance()
    auc = roc_auc_score(y, p); top = np.argsort(-imp)[:15]
    print(f"[shift] labelled vs public AUC {auc:.4f} (grouped 3-fold); top features (index: importance) "
          + " ".join(f"{i}:{imp[i] / 3:.1f}" for i in top), flush=True)
    w = p[y == 0] / np.clip(1 - p[y == 0], 1e-3, None); w /= w.mean()
    print(f"[shift] importance weights on labelled frames: p5 {np.percentile(w, 5):.2f} p50 {np.median(w):.2f} p95 {np.percentile(w, 95):.2f} "
          f"max {w.max():.1f}; effective sample fraction {w.sum() ** 2 / (w ** 2).sum() / len(w):.3f}", flush=True)
    json.dump({"auc": auc, "top": top.tolist(), "ess_frac": float(w.sum() ** 2 / (w ** 2).sum() / len(w))}, open(OUT / "domain_shift.json", "w"))


if __name__ == "__main__":
    main()
