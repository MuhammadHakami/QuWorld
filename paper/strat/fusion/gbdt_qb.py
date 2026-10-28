"""Doc 35 §41v: a gradient-boosted-tree family for q95 / betaN (multi-algorithm ensemble; CPU, GBDT allowed).

The q/b members we have are all neural (QH MLP heads, NAS entry trunks, QF trunks; error correlations 0.73-0.82, §41s). Here
CatBoost (depth 8, RMSE) on the QH input set (corrected magnetics block with lags, Thomson summaries; the QH heads' feature mask),
TRAIN frames (every 3rd), one model per target; predictions on all VALID + TEST frames. Read-out: R2 of the tree alone, and of
the deployed blend (0.5 QH x6 + 0.5 entry) mixed with the trees at one weight w, w chosen by 5-fold shot CV over VT (only one
parameter) - out-of-fold R2 vs the deployed blend, paired shot bootstrap.

    .venv/bin/python paper/strat/fusion/gbdt_qb.py dump                 # inputs / labels -> runs/strat/fusion/gbdt_qb_data.npz
    .venv-tdc/bin/python paper/strat/fusion/gbdt_qb.py fit [--all]      # CatBoost (CPU) -> gbdt_qb_pred.npz (--all: + VT rows, final)
    .venv/bin/python paper/strat/fusion/gbdt_qb.py eval                 # blend study on VT
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[3] / "runs" / "strat" / "fusion"
DATA = OUT / "gbdt_qb_data.npz"


def dump() -> None:
    import torch
    import fx
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    import inputs_fix
    d = sota.Data(); inputs_fix.use(d)
    ck = torch.load(OUT / "head_qb_QH_s1.pt", map_location="cpu", weights_only=False); fm, edge = ck["fm"], ck["edge"]
    tr = np.nonzero(d.split == 0)[0][::3]; vt = np.nonzero(d.split != 0)[0]
    lab = [sota.LABELS.index("efit_q95"), sota.LABELS.index("efit_beta_n")]
    np.savez(DATA, Xtr=d.inputs(tr, fm, edge).astype(np.float32), ytr=d.S[tr][:, lab].astype(np.float32), rtr=tr,
             Xvt=d.inputs(vt, fm, edge).astype(np.float32), yvt=d.S[vt][:, lab].astype(np.float32), rvt=vt, shot_vt=d.shot[vt])
    print(f"GBDT_DUMP_DONE train {len(tr)} vt {len(vt)} features {int(np.sum(fm)) + (2 if edge else 0)}", flush=True)


def fit(all_rows: bool) -> None:
    from catboost import CatBoostRegressor
    z = np.load(DATA); X, Y = z["Xtr"], z["ytr"]
    if all_rows:
        X, Y = np.concatenate([X, z["Xvt"][::3]]), np.concatenate([Y, z["yvt"][::3]])
    out = {}
    for j, nm in enumerate(["q95", "betaN"]):
        ok = np.isfinite(Y[:, j]); t = time.time()
        m = CatBoostRegressor(iterations=3000, depth=8, learning_rate=0.08, loss_function="RMSE", thread_count=16, random_seed=0,
                              verbose=500, allow_writing_files=False)
        m.fit(X[ok], Y[ok, j]); out[nm] = m.predict(z["Xvt"]).astype(np.float32)
        m.save_model(str(OUT / f"gbdt_qb_{nm}{'_all' if all_rows else ''}.cbm"))
        print(f"[gbdt] {nm}: {time.time() - t:.0f}s", flush=True)
    np.savez(OUT / f"gbdt_qb_pred{'_all' if all_rows else ''}.npz", **out); print("GBDT_FIT_DONE", flush=True)


def evaluate() -> None:
    import torch
    import fx
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import ens_select as ES
    import gpu_build as GB
    z = np.load(DATA); rows, Y, sh = z["rvt"], z["yvt"].astype(np.float64), z["shot_vt"]; g = np.load(OUT / "gbdt_qb_pred.npz")
    d = sota.Data(); shots = np.unique(sh)
    sel = json.loads((GB.SOTA / "ens_select_mlp.json").read_text())
    ent = [ES.member(str(fx.TRUNKS / m), d, shots) for m in sel["members"]]
    esc = np.concatenate([np.mean([m[s][1] for m in ent], 0) for s in shots])
    qh = GB.heads_gpu(d, rows, "+".join(f"head_qb_QH_s{s}" for s in range(1, 7)))[:, :2]
    H = sota.HEADS; res = {}; us = np.random.default_rng(0).permutation(shots); W = np.r_[0:0.55:0.05]
    for j, nm in enumerate(["q95", "betaN"]):
        y = Y[:, j]; ok = np.isfinite(y); dep = 0.5 * (qh[:, j] + esc[:, H.index(("efit_q95", "efit_beta_n")[j])]); t = g[nm].astype(np.float64)
        r2 = lambda p, m: 1 - np.sum((y[m] - p[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)          # noqa: E731
        curve = {round(float(w), 2): r2((1 - w) * dep + w * t, ok) for w in W}
        oof = np.zeros(len(y))
        for f in range(5):
            te = np.isin(sh, us[f::5]); trm = ok & ~te
            wb = max(W, key=lambda w: r2((1 - w) * dep + w * t, trm)); oof[te] = ((1 - wb) * dep + wb * t)[te]
        res[nm] = {"tree": r2(t, ok), "deployed": r2(dep, ok), "oof_blend": r2(oof, ok), "curve": curve,
                   "corr_err_tree_dep": float(np.corrcoef((t - y)[ok], (dep - y)[ok])[0, 1])}
        print(f"[gbdt] {nm}: tree {res[nm]['tree']:.5f} deployed {res[nm]['deployed']:.5f} OOF blend {res[nm]['oof_blend']:.5f} "
              f"({res[nm]['oof_blend'] - res[nm]['deployed']:+.5f}) err corr {res[nm]['corr_err_tree_dep']:.3f} | curve "
              + " ".join(f"{w}:{v:.5f}" for w, v in curve.items()), flush=True)
    (OUT / "gbdt_qb_eval.json").write_text(json.dumps(res, indent=1)); print("GBDT_EVAL_DONE", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    dump() if mode == "dump" else fit("--all" in sys.argv) if mode == "fit" else evaluate()
