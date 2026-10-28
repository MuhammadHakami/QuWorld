"""Figure for the allocation-strategy study (33-allocation-strategy-study.md).

(a) captured share vs utilization ceiling U for every (row, strategy) cell at K = 30, register core, mean over seeds;
    the diagonal is the machine-checked bound captured <= U (gain_le_ceiling). Colour = row, marker = strategy.
(b) captured share vs atom count K on the heat plume (the row with the most predictable error): pooled stays flat,
    as pooled_gain_le_dc requires; proxy routing keeps growing with its ceiling.

    .venv/bin/python paper/alloc/figs.py   # -> paper/manuscript/iclr2027/fig_alloc.pdf
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze import NAMES, STRATS, load, mean  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "manuscript" / "iclr2027" / "fig_alloc.pdf"
NAMES = {**NAMES, "pelzer": "heat plume", "MHD_64": "MHD_64"}
MK = {"pooled": "o", "static": "s", "proxy": "^", "learned": "D", "gain": "*", "random": "x"}
FLOOR = 3e-5


def main():
    R = load()
    rows = [b for b in NAMES if any(R[(b, s, "ryd", 30)] for s in STRATS)]
    cols = {b: f"C{i}" for i, b in enumerate(rows)}
    plt.rcParams.update({"font.size": 7.5})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
    for b in rows:
        for s in STRATS:
            v = R[(b, s, "ryd", 30)]
            if not v:
                continue
            U, cap = mean([d["test_U"] for d in v]), mean([d["test_captured"] for d in v])
            ax[0].scatter(U, max(cap, FLOOR), c=cols[b], marker=MK[s], s=18, linewidths=0.8 if s == "random" else 0)
    lim = np.array([1e-4, 1])
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].fill_between(lim, lim, 1, color="0.92", zorder=0)
    ax[0].axhspan(FLOOR / 1.6, FLOOR * 1.6, color="0.85", zorder=0)
    ax[0].text(1.2e-4, FLOOR * 1.9, "captured $\\leq 0$", fontsize=6, color="0.35")
    ax[0].text(1.3e-4, 0.12, "excluded by\n\\texttt{gain\\_le\\_ceiling}" if False else "excluded by the\nLean bound", fontsize=6, color="0.35")
    ax[0].set(xscale="log", yscale="log", xlim=(1e-4, 1), ylim=(FLOOR / 2, 1), xlabel="utilization ceiling $U$",
              ylabel="captured share of error energy")
    h = [ax[0].scatter([], [], c=cols[b], s=14, label=NAMES[b]) for b in rows]
    h += [ax[0].scatter([], [], c="0.3", marker=m, s=14, label=s) for s, m in MK.items()]
    ax[0].legend(handles=h, fontsize=5.3, ncol=2, loc="lower right", frameon=False, handletextpad=0.2, columnspacing=0.6)
    ax[0].set_title("(a) all rows and strategies, 30 atoms (register core)", fontsize=7.5)
    b = "pelzer"
    for s, col in (("pooled", "C7"), ("proxy", "C2")):
        for c, ls in (("ryd", "-"), ("mlp", ":")):
            ks = [K for K in (10, 30, 100, 300) if R[(b, s, c, K)]]
            if len(ks) < 2:
                continue
            ax[1].plot(ks, [mean([d["test_captured"] for d in R[(b, s, c, K)]]) for K in ks], ls, color=col, marker="o",
                       ms=3, lw=1, label=f"{s}, {'register' if c == 'ryd' else 'matched MLP'}")
        ks = [K for K in (10, 30, 100, 300) if R[(b, s, "mlp", K)] or R[(b, s, "ryd", K)]]
        Us = [mean([d["test_U"] for d in (R[(b, s, "ryd", K)] or R[(b, s, "mlp", K)])]) for K in ks]
        ax[1].plot(ks, Us, "--", color=col, lw=0.7, alpha=0.7, label=f"{s}: ceiling $U$")
    ax[1].set(xscale="log", yscale="log", xlabel="atoms $K$", ylabel="share of error energy")
    ax[1].legend(fontsize=5.3, frameon=False)
    ax[1].set_title("(b) qubit count vs utilization (heat plume)", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
