"""Figure 1 of the ICLR manuscript (fig_overview.pdf): the hybrid as an architecture diagram.

    .venv/bin/python paper/alloc/fig_overview.py

Top: the released classical backbone (backbone -> neck -> head), with the standard pooled register in the neck.
Middle: the frequency-aware gate between the backbone and persistence, per window and Fourier octave.
Bottom: the routed co-processor. The residual in band-wise POD modes is routed per input: 30 predictable (analog) modes
to the Rydberg register (per-input 2-D positions, detunings, tethers), the rest (digital) to a classical head; matched
classical cores swap into the register's slot. Right: the training loss. Italics: bounds proved in Lean 4.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/manuscript/iclr2027/fig_overview.pdf"
BLUE, ORANGE, GREEN, PURPLE, GREY = "#dbe8f5", "#fbe3cf", "#d9efe6", "#e6e1f2", "#f2f2f2"


def box(ax, x, y, w, h, text, fc, fs=6.0, ls="-", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec="0.35", lw=0.7,
                                ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, linespacing=1.35)


def arrow(ax, a, b, ls="-", color="0.25", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def note(ax, x, y, text, color="0.3", ha="center", fs=5.4, style="italic"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, style=style)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    # classical backbone
    note(ax, 1, 44.8, "classical model (released checkpoint)", color="#1f4e79", ha="left", fs=6.2, style="normal")
    box(ax, 1, 34, 12, 8, "input history\n$x_{t-k},\\dots,x_t$", GREY)
    box(ax, 16.5, 34, 15, 8, "pretrained backbone\n(CNextU-Net, FNO,\nU-Net, transformer)", BLUE)
    box(ax, 35, 34, 17, 8, "neck: pooled register\n(standard design:\nfeatures averaged first)", BLUE)
    box(ax, 55.5, 34, 8, 8, "head", BLUE)
    for a, b in (((13.2, 38), (16.3, 38)), ((31.7, 38), (34.8, 38)), ((52.2, 38), (55.3, 38))):
        arrow(ax, a, b)
    note(ax, 43.5, 32.3, "reaches only the spatial mean (Prop. 1)")

    # training loss
    box(ax, 67, 30.5, 32.5, 15.5,
        "training loss\n"
        "VRMSE of the output (= mode MSE)\n"
        "+ route supervision (top-$K$ error modes)\n"
        "+ layout: coupled atoms agree, no isolated\n   atom, minimum spacing; + tether cost\n"
        "probabilistic (FGN-style, noise $\\xi_m$ sets the configuration):\n"
        "$\\mathrm{ES}=\\frac{1}{M}\\sum_m\\|\\hat y_m-y\\|-\\frac{1}{M(M-1)}\\sum_{m<m'}\\|\\hat y_m-\\hat y_{m'}\\|$",
        GREY, fs=5.5)

    # frequency-aware gate
    box(ax, 30, 20.5, 38, 8.5,
        "frequency-aware gate (per window $w$, octave $b$)\n"
        "$\\hat y = x_t + \\sum_{b=0}^{4} \\alpha_{w,b}\\, P_b\\,(p - x_t)$",
        PURPLE, fs=6.2)
    arrow(ax, (59.5, 33.8), (59.5, 29.2)); note(ax, 61.2, 31.4, "$p$", style="normal", fs=6.5)
    arrow(ax, (7, 33.8), (29.8, 24.7), ls="--", rad=0.25); note(ax, 9, 29.5, "persistence $x_t$", style="normal")
    note(ax, 49, 18.9, "best gate never worse than persistence or backbone")

    # routed co-processor
    note(ax, 24, 17.2, "routed co-processor: analog modes to qubits, digital modes to a classical head",
         color="#a0522d", ha="left", fs=6.2, style="normal")
    box(ax, 1, 4.5, 16, 9.5, "band-wise POD\nmodes $\\varphi_m$\n(causal features\nof $x_t$, $\\hat y$)", GREY)
    note(ax, 1.2, 2.3, "training target only: $c_m = \\langle y-\\hat y,\\ \\varphi_m\\rangle$", ha="left", fs=5.2)
    arrow(ax, (25.2, 2.3), (38.8, 2.6), ls="--")                      # dashed: supervision, not an inference input
    arrow(ax, (33, 20.3), (12, 14.2), rad=-0.15)
    box(ax, 20.5, 4.5, 15, 9.5, "router (per input)\n30 predictable modes\nto qubits, the rest\nto the classical head", GREY)
    arrow(ax, (17.2, 9.2), (20.3, 9.2))
    box(ax, 39, 8.8, 25, 6.2, "Rydberg register, $3{\\times}10$ atoms\nper input: positions $(x,y)$,\ndetunings $\\Delta_i$, tethers $\\tau_g$",
        ORANGE, fs=5.8)
    box(ax, 39, 1.2, 25, 5.4, "classical head (MLP)\nall other modes", GREEN, fs=5.8)
    arrow(ax, (35.7, 10.5), (38.8, 12)); arrow(ax, (35.7, 8), (38.8, 4))
    box(ax, 67, 9.3, 13, 5.2, "matched cores\n(MLP, GNN,\nmean-field)", "white", fs=5.4, ls="--")
    ax.plot([64.3, 66.8], [11.9, 11.9], color="0.4", lw=0.7, ls=":")
    box(ax, 83, 4.5, 16, 9.5, "output\n$\\hat y + \\sum_m \\Delta c_m\\, \\varphi_m$", GREY, weight="bold")
    arrow(ax, (64.2, 3.9), (82.8, 6.5)); arrow(ax, (80.2, 11.9), (82.8, 11.2))
    arrow(ax, (68.2, 23), (91, 14.2), rad=-0.2); note(ax, 83, 20.6, "$\\hat y$", style="normal", fs=6.5)
    note(ax, 51.5, 0.0, "captured share $\\leq$ reachable share $U$ per octave")

    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)


if __name__ == "__main__":
    main()
