"""Qualitative failure analysis of the final Well models (the registered E-ft selections, doc 37): worst test windows.

    .venv/bin/python paper/alloc/failure_cases.py <dataset> [--k 6] [--testk 2]

For each window of a test sample (full split for TRL2D, stratified otherwise) it computes per-field VRMSE of the
selected model and, for the worst k windows, renders x_t / truth / prediction / error per field. Descriptors per
window and field, to read the pictures against:
  amp   = ||y_hat - x_t|| / ||y - x_t||   (predicted change vs true change; < 1 = under-predicted dynamics)
  corr  = corr(y_hat - x_t, y - x_t)       (is the predicted change pointed the right way)
  shift = best error over integer rolls |dy|,|dx| <= 3 / error at zero shift (< 1 = misplaced structure)
  edge  = share of squared error in the 8 rows next to each y boundary
  hi_k  = share of squared error above half the Nyquist radius
Outputs runs/failure/<dataset>_worst.png and runs/failure/<dataset>.json. Diagnostic only: nothing is fitted.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import finetune_gated as FG
import ft_ensemble as FE

OUT = FG.ROOT / "runs" / "failure"
FIELDS = {"turbulent_radiative_layer_2D": ("density", "pressure", "v_x", "v_y"),
          "rayleigh_benard": ("buoyancy", "pressure", "v_x", "v_y"),
          "shear_flow": ("tracer", "pressure", "v_x", "v_y")}


def field_vrmse(p, y):
    mse = (p - y).square().mean((-2, -1)); var = (y - y.mean((-2, -1), keepdim=True)).square().mean((-2, -1))
    return (mse / (var + 1e-7)).sqrt()                                           # (B, C)


def descriptors(x_t, y, p):
    dt, dp = (y - x_t).flatten(-2), (p - x_t).flatten(-2)
    amp = dp.norm(dim=-1) / (dt.norm(dim=-1) + 1e-12)
    corr = torch.nn.functional.cosine_similarity(dt - dt.mean(-1, keepdim=True), dp - dp.mean(-1, keepdim=True), dim=-1)
    e0 = (p - y).square().mean((-2, -1)); best = e0.clone()
    for sy in range(-3, 4):
        for sx in range(-3, 4):
            best = torch.minimum(best, (p.roll((sy, sx), (-2, -1)) - y).square().mean((-2, -1)))
    err = (p - y).square(); H = y.shape[-2]
    edge = (err[..., :8, :].sum((-2, -1)) + err[..., H - 8:, :].sum((-2, -1))) / (err.sum((-2, -1)) + 1e-12)
    E = torch.fft.fft2(p - y).abs().square(); ky = torch.fft.fftfreq(y.shape[-2], device=y.device).abs()
    kx = torch.fft.fftfreq(y.shape[-1], device=y.device).abs(); R = torch.maximum(ky[:, None] / ky.max(), kx[None] / kx.max())
    hi = (E * (R > 0.5)).sum((-2, -1)) / (E.sum((-2, -1)) + 1e-12)
    return {"amp": amp, "corr": corr, "shift": (best / (e0 + 1e-12)).sqrt(), "edge": edge, "hi_k": hi}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--testk", type=int, default=2); a = ap.parse_args()
    sel = json.loads((FG.OUT / f"{a.dataset}_Eft.json").read_text())
    tags = sel["members"] if sel["selected"] == "E-ft" else [sel["selected"]]
    loaded = [FE.load(a.dataset, t) for t in tags]; models, test = [m for m, _ in loaded], loaded[0][1][1]
    pairs = FG.well2d.all_test_windows(test) if a.dataset == "turbulent_radiative_layer_2D" else FG.strat_pairs(test, a.testk, 3)
    C = test.C; rows, keep = [], []
    for bi, (x, y) in enumerate(FG.well2d.batched(test, pairs, 4)):
        x, y = x.to(FG.DEV), y.to(FG.DEV); p = torch.stack([m(x) for m in models]).mean(0); xt = x[:, 3 * C:]
        v, d = field_vrmse(p, y), descriptors(xt, y, p)
        for i in range(y.shape[0]):
            j = bi * 4 + i
            rows.append({"traj": pairs[j][0], "t0": pairs[j][1], "vrmse": v[i].tolist(), "mean": float(v[i].mean()),
                         "var_y": (y[i] - y[i].mean((-2, -1), keepdim=True)).square().mean((-2, -1)).tolist(),
                         **{k: t[i].tolist() for k, t in d.items()}})
    order = np.argsort([-r["mean"] for r in rows]); worst = [rows[i] for i in order[:a.k]]
    OUT.mkdir(parents=True, exist_ok=True)
    summ = lambda rs, key: np.mean([r[key] for r in rs], 0).round(3).tolist()
    res = {"dataset": a.dataset, "selection": sel["selected"], "n": len(rows), "mean_vrmse": float(np.mean([r["mean"] for r in rows])),
           "fields": FIELDS[a.dataset], "worst": worst,
           "all_mean": {k: summ(rows, k) for k in ("vrmse", "amp", "corr", "shift", "edge", "hi_k")},
           "worst10pct_mean": {k: summ([rows[i] for i in order[:max(1, len(rows) // 10)]], k) for k in ("vrmse", "amp", "corr", "shift", "edge", "hi_k")},
           "t0_of_worst10pct": np.bincount([rows[i]["t0"] for i in order[:max(1, len(rows) // 10)]]).tolist()}
    (OUT / f"{a.dataset}.json").write_text(json.dumps(res, indent=1))
    render(a.dataset, test, models, worst[:4], C)
    print(json.dumps({k: res[k] for k in ("n", "mean_vrmse", "all_mean", "worst10pct_mean")}, indent=1))
    print("worst:", [(w["traj"], w["t0"], round(w["mean"], 3)) for w in worst])


def render(dataset, test, models, worst, C):
    names = FIELDS[dataset]
    tall = test.H > test.W; ratio = min(test.H, test.W) / max(test.H, test.W)
    fig, axes = plt.subplots(len(worst) * C, 4, figsize=(9, 1.25 * len(worst) * C * ratio * 3 + 0.5))
    for wi, w in enumerate(worst):
        x, y = next(iter(FG.well2d.batched(test, [(w["traj"], w["t0"])], 1)))
        x, y = x.to(FG.DEV), y.to(FG.DEV); p = torch.stack([m(x) for m in models]).mean(0)
        xt = x[0, 3 * C:].cpu().numpy(); yy = y[0].cpu().numpy(); pp = p[0].cpu().numpy()
        if tall:                                                                   # display tall grids sideways
            xt, yy, pp = (t.transpose(0, 2, 1) for t in (xt, yy, pp))
        for c in range(C):
            ax = axes[wi * C + c]; lo, hi = np.percentile(yy[c], [1, 99])
            for col, (img, title) in enumerate(((xt[c], "x_t"), (yy[c], "truth"), (pp[c], "prediction"))):
                ax[col].imshow(img, cmap="RdBu_r", vmin=lo, vmax=hi); ax[col].set_xticks([]); ax[col].set_yticks([])
                if wi == 0 and c == 0: ax[col].set_title(title, fontsize=7)
            e = pp[c] - yy[c]; m = np.abs(e).max()
            ax[3].imshow(e, cmap="PuOr", vmin=-m, vmax=m); ax[3].set_xticks([]); ax[3].set_yticks([])
            if wi == 0 and c == 0: ax[3].set_title("error", fontsize=7)
            ax[0].set_ylabel(f"traj {w['traj']} t {w['t0']}\n{names[c]} {w['vrmse'][c]:.2f}", fontsize=5.5)
    fig.tight_layout(pad=0.2); fig.savefig(OUT / f"{dataset}_worst.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
