"""Model outputs on the Well rows (fig_fields_well.pdf): ground truth and absolute error of the gated fine-tuned
classical model, the same model with a matched MLP in the neck, and with the 30-atom register in the neck, on the
median-error TEST window of each row where per-window scores exist, else the middle TEST window (fixed in advance).

    QW_GPU_GB=10 runs/capped.sh fields 24 .venv/bin/python paper/alloc/fig_fields_well.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import finetune_gated as FG  # noqa: E402
from ft_ensemble import load, members  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROWS = (("shear_flow", "shear flow"), ("rayleigh_benard", "convection"), ("turbulent_radiative_layer_2D", "TRL2D"))
NAMES = {"vanilla": "classical", "mlp30": "+ MLP (neck)", "mr30ryd": "+ 30 qubits (neck)"}


@torch.no_grad()
def row_outputs(ds: str) -> dict:
    f = FG.OUT / f"{ds}_Eft_test.csv"
    w = None
    if f.exists():                                                        # median-error TEST window
        v = np.array([float(r.split(",")[1]) for r in f.read_text().splitlines()[1:]]); w = int(np.argsort(v)[len(v) // 2])
    out = {}
    for tag in members(ds):
        variant = json.loads((FG.OUT / f"{tag}.json").read_text())["variant"]
        model, (_, test) = load(ds, tag)
        pairs = FG.well2d.all_test_windows(test)
        w = len(pairs) // 2 if w is None else w                           # else the middle TEST window, fixed in advance
        out["window"] = w; pair = pairs[w]
        x, y = next(iter(FG.well2d.batched(test, [pair], 1)))
        x, y = x.to(FG.DEV), y.to(FG.DEV)
        p = model(x)
        out[variant] = {"pred": p[0].float().cpu().numpy(), "vrmse": float(FG.vrmse(p, y).mean())}
        out["truth"] = y[0].float().cpu().numpy()
        del model; torch.cuda.empty_cache()
    return out


def main() -> None:
    res = {ds: row_outputs(ds) for ds, _ in ROWS}
    mhd = np.load(ROOT / "paper/fig_cache/fig_fields_mhd.npz")             # figs_fields_mhd.py: VALID traj. 0, mid-z density
    res["MHD_64"] = {"window": "valid 0, lead 1", "truth": mhd["gt_1"][None]}
    for v, k in (("vanilla", "Vanilla-TD"), ("mlp30", "MLP-TD"), ("mr30ryd", "Quantum-TD")):
        res["MHD_64"][v] = {"pred": mhd[f"{k}_1"][None], "vrmse": float(mhd[f"vrmse_{k}"][0])}
    for key, f, metric, title in (("1HP", "fig_fields_1hp.npz", "mae", "heat plume (1HP)"),
                                  ("psi", "fig_fields_psi.npz", "nrmse", "fusion equilibrium")):
        c = np.load(ROOT / "paper/fig_cache" / f)                          # dump_fields_1hp.py / dump_fields_psi.py
        res[key] = {"window": "middle test sample", "truth": c["truth"][None], "metric": "MAE degC" if metric == "mae" else "NRMSE"}
        for v in NAMES:
            res[key][v] = {"pred": c[f"{v}_pred"][None], "vrmse": float(c[f"{v}_{metric}"])}
    rows = list(ROWS) + [("MHD_64", "MHD (val.)"), ("1HP", "heat plume"), ("psi", "fusion eq.")]
    plt.rcParams.update({"font.size": 6.5})
    fig, ax = plt.subplots(len(rows), 5, figsize=(7.0, 0.72 * len(rows) + 0.2))
    fig.subplots_adjust(hspace=0.06, wspace=0.04)
    orient = lambda f: f.T if f.shape[0] > f.shape[1] else f
    for i, (ds, title) in enumerate(rows):
        r = res[ds]; gt = r["truth"][0]; lo, hi = np.percentile(gt, [1, 99])
        ax[i, 0].imshow(orient(gt), cmap="RdBu_r", origin="lower", aspect="auto", vmin=lo, vmax=hi)
        ax[i, 0].set_ylabel(title, fontsize=5.8)
        for j, v in enumerate(NAMES):
            ax[i, j + 1].imshow(orient(r[v]["pred"][0]), cmap="RdBu_r", origin="lower", aspect="auto", vmin=lo, vmax=hi)
            ax[i, j + 1].text(0.02, 0.96, f"{r.get('metric', 'VRMSE')} {r[v]['vrmse']:.4f}", transform=ax[i, j + 1].transAxes, fontsize=5.5,
                              va="top", color="w", bbox=dict(fc="k", alpha=0.5, lw=0, pad=1))
        dq = np.abs(r["mr30ryd"]["pred"][0] - gt) - np.abs(r["vanilla"]["pred"][0] - gt)
        m = float(np.percentile(np.abs(dq), 99)) or 1.0
        im = ax[i, 4].imshow(orient(dq), cmap="coolwarm", origin="lower", aspect="auto", vmin=-m, vmax=m)
        if i == 0:
            for j, t in enumerate(["ground truth", "classical", "+ MLP", "+ 30 qubits", "|error|: qubits minus classical"]):
                ax[0, j].set_title(t, fontsize=6)
        print(ds, "window", r["window"], {v: round(r[v]["vrmse"], 5) for v in NAMES if v in r})
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    out = ROOT / "paper/manuscript/iclr2027/fig_fields_well.pdf"
    fig.savefig(out, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
