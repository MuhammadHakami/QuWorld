"""Doc 35 §45d: gradient-boosted-tree X-point heads (competitor evidence: Anulum blends a tree into its lower X-point R at 0.16).
The deployed X-point heads (XP1 TRAIN-only / XPA all-data, MLP 4096 x 4) read exactly the J1RL input set, so the §45a design matrix
(gbdt_shape_data.npz: every 3rd TRAIN frame, all VT frames, 598 features) is reused with the X-point labels (xp_labels.npz, continuous
saddle of the true psi; lower R_l, Z_l / upper R_u, Z_u, NaN when that X-point is absent). CatBoost GPU (v1 config: depth 8, 3000 it,
lr 0.08) per target on its finite TRAIN rows; VT read-out vs the XP1 s1-3 ensemble with one blend weight per target chosen by 5-fold
shot cross-fitting (out-of-fold), as §45a.
PRE-DECLARED rule (2026-10-16, before any result): adopt iff the out-of-fold blend lowers the lower-X-point squared distance error
(R_l, Z_l jointly, frames with both labels finite) by >= 3% vs the XP1 ensemble with a paired shot-bootstrap 95% CI > 0 -> all-data
trees + an F-build; else closed.
    .venv-tdc/bin/python paper/strat/fusion/gbdt_xp.py fit
    .venv/bin/python paper/strat/fusion/gbdt_xp.py eval
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[3] / "runs" / "strat" / "fusion"
TGT = ["R_l", "Z_l", "R_u", "Z_u"]
XP1 = "head_xp_XP1_s1+head_xp_XP1_s2+head_xp_XP1_s3"


def fit() -> None:
    from catboost import CatBoostRegressor
    z = np.load(OUT / "gbdt_shape_data.npz"); Y = np.load(OUT / "xp_labels.npz")["y"][z["rtr"]].astype(np.float32); out = {}
    for j, nm in enumerate(TGT):
        ok = np.isfinite(Y[:, j]); t = time.time()
        m = CatBoostRegressor(iterations=3000, depth=8, learning_rate=0.08, loss_function="RMSE", task_type="GPU", devices="0", gpu_ram_part=0.2,
                              random_seed=0, verbose=1000, allow_writing_files=False)
        m.fit(z["Xtr"][ok], Y[ok, j]); out[nm] = m.predict(z["Xvt"]).astype(np.float32); m.save_model(str(OUT / f"gbdt_xp_{nm}.cbm"))
        print(f"[gbdt_xp] {nm}: {ok.sum()} rows {time.time() - t:.0f}s", flush=True)
    np.savez(OUT / "gbdt_xp_pred.npz", **out); print("GBDT_XP_FIT_DONE", flush=True)


def evaluate() -> None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"                                      # before fx (which hides the GPU) and gpucap's probe
    import torch
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import fx
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    z = np.load(OUT / "gbdt_shape_data.npz"); rows, sh = z["rvt"], z["shot_vt"]; Y = np.load(OUT / "xp_labels.npz")["y"][rows].astype(np.float64)
    g = np.load(OUT / "gbdt_xp_pred.npz"); d = sota.Data(); H = GB.heads_gpu(d, rows, XP1)[:, :4]
    us = np.random.default_rng(0).permutation(np.unique(sh)); W = np.r_[0:0.55:0.05]; res = {}; B = np.empty_like(H)
    for j, nm in enumerate(TGT):
        y, t, p0 = Y[:, j], g[nm].astype(np.float64), H[:, j]; ok = np.isfinite(y) & np.isfinite(p0) & np.isfinite(t)
        sse = lambda p, m: np.sum((y[m] - p[m]) ** 2)                                                         # noqa: E731
        oof = p0.copy(); ws = []
        for f in range(5):
            te = np.isin(sh, us[f::5]); trm = ok & ~te
            wb = min(W, key=lambda w: sse((1 - w) * p0 + w * t, trm)); oof[te] = ((1 - wb) * p0 + wb * t)[te]; ws.append(float(wb))
        B[:, j] = oof; v = np.var(y[ok])
        res[nm] = {"n": int(ok.sum()), "rmse_head": float(np.sqrt(sse(p0, ok) / ok.sum())), "rmse_tree": float(np.sqrt(sse(t, ok) / ok.sum())),
                   "rmse_oof": float(np.sqrt(sse(oof, ok) / ok.sum())), "r2_head": float(1 - sse(p0, ok) / ok.sum() / v),
                   "r2_oof": float(1 - sse(oof, ok) / ok.sum() / v), "w": ws, "corr_err": float(np.corrcoef((t - y)[ok], (p0 - y)[ok])[0, 1])}
        r = res[nm]; print(f"[gbdt_xp] {nm}: n {r['n']} rmse head {100 * r['rmse_head']:.3f} cm tree {100 * r['rmse_tree']:.3f} OOF blend {100 * r['rmse_oof']:.3f} "
                           f"(R2 {r['r2_head']:.5f} -> {r['r2_oof']:.5f}) w {ws} err corr {r['corr_err']:.3f}", flush=True)
    ok = np.isfinite(Y[:, :2]).all(1) & np.isfinite(H[:, :2]).all(1)
    e0 = ((H[:, :2] - Y[:, :2]) ** 2).sum(1); e1 = ((B[:, :2] - Y[:, :2]) ** 2).sum(1); usv, inv = np.unique(sh[ok], return_inverse=True)
    s0, s1 = np.bincount(inv, e0[ok], len(usv)), np.bincount(inv, e1[ok], len(usv)); rel = 1 - s1.sum() / s0.sum(); rng = np.random.default_rng(0)
    bs = [1 - s1[i].sum() / s0[i].sum() for i in (rng.integers(0, len(usv), len(usv)) for _ in range(1000))]; lo, hi = np.percentile(bs, [2.5, 97.5])
    adopt = bool(rel >= 0.03 and lo > 0); res["lower"] = {"rel_reduction": float(rel), "ci": [float(lo), float(hi)], "adopt": adopt,
                                                          "rms_dist_head_cm": float(100 * np.sqrt(e0[ok].mean())), "rms_dist_oof_cm": float(100 * np.sqrt(e1[ok].mean()))}
    print(f"[gbdt_xp] lower X-point squared distance: head rms {100 * np.sqrt(e0[ok].mean()):.3f} cm -> OOF blend {100 * np.sqrt(e1[ok].mean()):.3f} cm, "
          f"reduction {100 * rel:.2f}% CI [{100 * lo:.2f}, {100 * hi:.2f}]%" + ("  ADOPT" if adopt else "  REJECT"), flush=True)
    (OUT / "gbdt_xp_eval.json").write_text(json.dumps(res, indent=1)); print("GBDT_XP_EVAL_DONE", flush=True)


if __name__ == "__main__":
    {"fit": fit, "eval": evaluate}[sys.argv[1]]()
