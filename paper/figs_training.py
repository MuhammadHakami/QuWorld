"""Build fig_training.pdf: train/eval curves for TRL2D TD fine-tune and MHD_64 fine-tune.

Panel A: TRL2D round-2 TD fine-tuning (phase_b2), selection metric vrmse8 vs step,
         4 models x 3 seeds (thin per-seed curves + bold seed-mean).
Panel B: MHD_64 fine-tuning (phase_b_mhd), valid vrmse vs step, 3 models, seed 1337.

Usage: uv run paper/figs_training.py
Output: paper/manuscript/fig_training.pdf (+ caption facts printed to stdout).
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
import os
import pathlib
OUT_PDF = pathlib.Path(os.environ["QW_OUT_DIR"]) / "fig_training.pdf" if os.environ.get("QW_OUT_DIR") else REPO / "paper" / "manuscript" / "fig_training.pdf"
VALIDATED = os.environ.get("QW_PALETTE") == "validated"   # dataviz-validated categorical set (ICLR version)

INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

MODELS = {
    "mr30ryd": {"label": "quantum (30q Rydberg)" if VALIDATED else "quantum (mr30ryd)",
                "color": "#2a78d6" if VALIDATED else "#1D4ED8", "ls": "-", "z": 5},
    "mlp30": {"label": "matched MLP", "color": "#eb6834" if VALIDATED else "#B45309", "ls": "-", "z": 4},
    "vanilla": {"label": "vanilla", "color": "#4a3aa7" if VALIDATED else "#4B5563", "ls": "-", "z": 3},
    "noent": {"label": "no-interaction" if VALIDATED else "no-entanglement",
              "color": "#1baf7a" if VALIDATED else "#93C5FD", "ls": "--", "z": 2},
}
SEEDS = (1337, 1338, 1339)

# Full-run val_check lines only; resume lines ("vrmse=... (RESUMED)") lack vrmse8.
RE_TRL2D = re.compile(r"val_check step=(\d+)\s.*?vrmse8=([\d.]+)")
RE_MHD = re.compile(r"val_check step=(\d+)\s+vrmse=([\d.]+)")


def parse_curve(path, regex):
    """Return (steps, values) sorted by step, deduped keeping the LAST occurrence
    (restarted runs re-log earlier steps)."""
    by_step = {}
    for line in path.read_text(errors="replace").splitlines():
        m = regex.search(line)
        if m:
            by_step[int(m.group(1))] = float(m.group(2))
    if not by_step:
        raise ValueError(f"no val_check points parsed from {path}")
    steps = np.array(sorted(by_step), dtype=float)
    vals = np.array([by_step[int(s)] for s in steps])
    return steps, vals


def seed_mean(curves, n_grid=241):
    """Interpolate per-seed curves onto a common grid and average."""
    lo = max(c[0][0] for c in curves)
    hi = min(c[0][-1] for c in curves)
    grid = np.linspace(lo, hi, n_grid)
    stack = np.vstack([np.interp(grid, s, v) for s, v in curves])
    return grid, stack.mean(axis=0)


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.8, length=3, labelsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def kfmt(x, _pos=None):
    return "0" if x == 0 else f"{x / 1000:g}k"


def main():
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "sans-serif",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.4, 2.4))

    # ---- Panel A: TRL2D TD fine-tune, vrmse8 (selection metric), 3 seeds ----
    facts_a = {}
    for name, spec in MODELS.items():
        curves = []
        for seed in SEEDS:
            path = REPO / "runs" / "phase_b2" / f"td_{name}_s{seed}.log"
            s, v = parse_curve(path, RE_TRL2D)
            curves.append((s, v))
            ax_a.plot(s, v, color=spec["color"], ls=spec["ls"], lw=0.55,
                      alpha=0.22, zorder=spec["z"], solid_capstyle="round")
            facts_a.setdefault(name, []).append(
                (seed, float(v.min()), float(s[np.argmin(v)]), float(v[-1])))
        gs, gm = seed_mean(curves)
        ax_a.plot(gs, gm, color=spec["color"], ls=spec["ls"], lw=1.7,
                  zorder=spec["z"] + 10, label=spec["label"], solid_capstyle="round")

    ax_a.set_xlim(0, 10000)
    ax_a.set_ylim(0.55, 0.78)  # a few early per-seed spikes (max 0.84) run off-axis
    ax_a.xaxis.set_major_formatter(plt.FuncFormatter(kfmt))
    ax_a.set_xlabel("fine-tune step")
    ax_a.set_ylabel("rollout valid VRMSE@8")
    ax_a.set_title("TRL2D, TD fine-tune (3 seeds)", color=INK)
    ax_a.legend(frameon=False, fontsize=6.5, loc="upper right",
                handlelength=1.7, borderaxespad=0.1, labelspacing=0.3)
    style_axis(ax_a)

    # ---- Panel B: MHD_64 fine-tune, vrmse, seed 1337, log-y ----
    # Curves overlap almost perfectly, so differentiate by width + dash so all
    # three stay visible: wide gray underneath, dashed amber, thin blue on top.
    panel_b_style = {
        "vanilla": {"lw": 2.6, "ls": "-", "z": 3},
        "mlp30": {"lw": 1.6, "ls": (0, (4, 2.2)), "z": 4},
        "mr30ryd": {"lw": 1.1, "ls": "-", "z": 5},
    }
    facts_b = {}
    for name in ("vanilla", "mlp30", "mr30ryd"):
        spec, st = MODELS[name], panel_b_style[name]
        path = REPO / "runs" / "phase_b_mhd" / f"ft_{name}_s1337.log"
        s, v = parse_curve(path, RE_MHD)
        ax_b.plot(s, v, color=spec["color"], ls=st["ls"], lw=st["lw"],
                  zorder=st["z"] + 10, label=spec["label"], solid_capstyle="round")
        facts_b[name] = (float(v.min()), float(s[np.argmin(v)]), float(v[-1]),
                         float(s[-1]))

    ax_b.set_yscale("log")
    ax_b.set_xlim(0, 20000)
    ax_b.set_ylim(0.168, 0.60)
    ax_b.set_yticks([0.2, 0.3, 0.4, 0.5])
    ax_b.set_yticklabels(["0.2", "0.3", "0.4", "0.5"])
    ax_b.yaxis.set_minor_locator(plt.NullLocator())
    ax_b.xaxis.set_major_formatter(plt.FuncFormatter(kfmt))
    ax_b.set_xlabel("fine-tune step")
    ax_b.set_ylabel("valid VRMSE (log scale)")
    ax_b.set_title("MHD$_{64}$ fine-tune (seed 1337)", color=INK)
    handles, labels = ax_b.get_legend_handles_labels()
    ax_b.legend(handles[::-1], labels[::-1], frameon=False, fontsize=6.5,
                loc="upper right", handlelength=1.7, borderaxespad=0.1,
                labelspacing=0.3)
    style_axis(ax_b)

    fig.tight_layout(pad=0.4, w_pad=1.2)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved {OUT_PDF}")

    # ---- Caption facts ----
    print("\n== Panel A (TRL2D vrmse8) per-seed: (seed, best, best@step, final) ==")
    for name, rows in facts_a.items():
        rows_s = ", ".join(f"s{r[0]}: best {r[1]:.4f}@{int(r[2])} final {r[3]:.4f}"
                           for r in rows)
        bests = [r[1] for r in rows]
        print(f"{name:8s} {rows_s} | mean-of-best {np.mean(bests):.4f}")
    print("\n== Panel B (MHD vrmse): (best, best@step, final, final step) ==")
    for name, (best, bstep, fin, fstep) in facts_b.items():
        print(f"{name:8s} best {best:.4f}@{int(bstep)} final {fin:.4f}@{int(fstep)}")


if __name__ == "__main__":
    main()
