"""Figures for notebook 20 (diagram-first): Fusion entry, Learn2Design optimizer, geothermal hybrid (paper Figure 1 style).
Every function returns a matplotlib Figure; data from notebooks/data/nb20.{npz,json} (nb20_data.py) and paper/fig_cache.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "notebooks" / "data"
CLS, CLS_D = "#dbe8f5", "#2f5d8a"          # classical network
QUB, QUB_D = "#fbe0cc", "#c0581b"          # qubits
PST, PST_D = "#d9efe0", "#2e7d4f"          # post-processing / edit of the output
OPT, OPT_D = "#e6def3", "#5b3f8c"          # optimizer
GRY, GRY_D = "#eeeeee", "#555555"          # data
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 110})


def load():
    return np.load(DATA / "nb20.npz"), json.loads((DATA / "nb20.json").read_text())


def box(ax, x, y, w, h, text, fc=GRY, ec=GRY_D, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, c="#444444", ls="-", lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=c, lw=lw, ls=ls))


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off"); return fig, ax


def legend(ax, x, y, items=(("classical network", CLS, CLS_D), ("qubits (30-atom register)", QUB, QUB_D),
                            ("edit of the output", PST, PST_D), ("optimizer", OPT, OPT_D), ("data", GRY, GRY_D))):
    for k, (t, fc, ec) in enumerate(items):
        ax.add_patch(Rectangle((x + k * 2.1, y), 0.3, 0.25, fc=fc, ec=ec)); ax.text(x + k * 2.1 + 0.4, y + 0.12, t, va="center", fontsize=8)


# ----------------------------------------------------------------------------------------------- overview
def overview():
    fig, ax = canvas(11, 3.6, (0, 11), (0, 3.6))
    cols = [("Fusion Equilibrium\n(competition)", ["coil currents, Ip,\nThomson profiles", "trunk networks\n(3 seeds)", "shape heads\n(6 seeds)", "flux-map edits\n(6 steps)", "flux map psi,\nq95, betaN"],
             [GRY, CLS, CLS, PST, GRY], [GRY_D, CLS_D, CLS_D, PST_D, GRY_D]),
            ("Learn2Design\n(competition)", ["detector topology\n(21 cells)", "donor designs\n(public dataset)", "noisy Adam\n+ restarts", "", "best feasible\ndesign"],
             [GRY, GRY, OPT, None, GRY], [GRY_D, GRY_D, OPT_D, None, GRY_D]),
            ("Geothermal heat plume\n(paper, Figure 1 design)", ["4 input fields\n256 x 16", "U-Net encoder\n(classical)", "30-atom register\nin the bottleneck", "U-Net decoder\n(classical)", "temperature\n256 x 16"],
             [GRY, CLS, QUB, CLS, GRY], [GRY_D, CLS_D, QUB_D, CLS_D, GRY_D])]
    for c, (title, steps, fcs, ecs) in enumerate(cols):
        x0 = 0.3 + c * 3.65; ax.text(x0 + 1.55, 3.35, title, ha="center", va="center", fontsize=10, weight="bold")
        ys = [2.55, 2.0, 1.45, 0.9, 0.35]; prev = None
        for y, s, fc, ec in zip(ys, steps, fcs, ecs):
            if not s:
                continue
            box(ax, x0 + 0.25, y - 0.2, 2.6, 0.42, s, fc, ec)
            if prev is not None:
                arrow(ax, x0 + 1.55, prev - 0.2, x0 + 1.55, y + 0.22)
            prev = y
    return fig


# ----------------------------------------------------------------------------------------------- Fusion
def fusion_pipeline():
    fig, ax = canvas(12, 4.6, (0, 12), (0, 4.6))
    box(ax, 0.1, 3.3, 1.8, 0.9, "inputs per frame\n427 values: coil\ncurrents, Ip, lags,\nThomson profiles", GRY, GRY_D, 8)
    box(ax, 2.4, 3.55, 2.2, 0.75, "trunk MLP x3 seeds\n2048 wide, residual\n(corrected magnetics)", CLS, CLS_D, 8)
    box(ax, 5.0, 3.55, 2.2, 0.75, "psi = mean +\nsum of 64 PCA\ncomponents", CLS, CLS_D, 8)
    box(ax, 2.4, 2.45, 2.2, 0.75, "shape heads x6 seeds\n4096 x 4 MLP\n7 scalars + 64 rays", CLS, CLS_D, 8)
    box(ax, 2.4, 1.35, 2.2, 0.75, "q95 / betaN heads x6\n50 / 50 with the\ntrunk ensemble", CLS, CLS_D, 8)
    arrow(ax, 1.9, 3.75, 2.4, 3.9); arrow(ax, 1.9, 3.65, 2.4, 2.85); arrow(ax, 1.9, 3.55, 2.4, 1.75); arrow(ax, 4.6, 3.92, 5.0, 3.92)
    steps = [("1 axis", "O-point ->\nhead axis"), ("2 boundary", "LCFS ->\n64 head rays"), ("3 kappa", "stretch ->\nhead kappa"),
             ("4 x-point", "top, bottom\n-> head tri"), ("5 joint fit", "6 scalars\nat once (GN)"), ("6 li", "inner profile\n-> head li")]
    for k, (t, s) in enumerate(steps):
        x = 5.0 + k * 1.15; box(ax, x, 2.3, 1.05, 0.95, f"{t}\n\n{s}", PST, PST_D, 7)
        if k:
            arrow(ax, x - 0.1, 2.78, x, 2.78)
    arrow(ax, 6.1, 3.55, 5.55, 3.25); arrow(ax, 4.6, 2.82, 5.0, 2.82, c=CLS_D, ls="--")
    ax.text(7.9, 2.05, "the head targets go to every step (dashed)", ha="center", fontsize=7.5, color=CLS_D)
    box(ax, 9.9, 0.9, 1.95, 1.0, "submission\npsi 65 x 65,\nq95, betaN", GRY, GRY_D, 8.5, True)
    arrow(ax, 11.3, 2.3, 11.0, 1.9); arrow(ax, 4.6, 1.72, 9.9, 1.4)
    ax.text(0.1, 0.55, "Score S = 0.55 psi R2 + 0.15 q95/betaN R2 + 0.10 (1 - D_LCFS) + 0.20 consistency (7 scalars the scorer takes from psi)",
            fontsize=8.5, color="#333333")
    ax.text(0.1, 0.15, "No qubits in this entry: the register stage (doc 35 §38t) gave +0.0000 and was not kept.", fontsize=8.5, color=QUB_D, weight="bold")
    legend(ax, 0.1, 4.35, items=(("classical network", CLS, CLS_D), ("edit of the output", PST, PST_D), ("data", GRY, GRY_D)))
    return fig


def fusion_frame():
    z, m = load(); R, Z = z["R"], z["Z"]; lev = m["level"]; sc = m["scalars"]
    fig, axs = plt.subplots(1, 4, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    for ax, (nm, key, col) in zip(axs, (("truth (EFIT)", "psi_truth", "k"), ("X8 (public entry)", "psi_X8", "#d1495b"), ("HJl9 (new)", "psi_HJl9", "#2e7d4f"))):
        phi = -z[key]; ax.contourf(R, Z, phi, 30, cmap="viridis", alpha=0.85)
        ax.contour(R, Z, -z["psi_truth"], [lev["truth"]], colors="white", linewidths=2.2)
        if key != "psi_truth":
            ax.contour(R, Z, phi, [lev[key.split("_")[1]]], colors=col, linewidths=1.6, linestyles="--")
        s = sc["truth" if key == "psi_truth" else key.split("_")[1]]
        ax.set_title(f"{nm}\nkappa {s['kappa']:.3f}   tri_top {s['tri_top']:.3f}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xlabel("R (m)")
    axs[0].set_ylabel("Z (m)")
    zm = axs[3]                                                          # zoom on the upper X-point, where kappa changes
    for key, col, ls, lw in (("psi_truth", "k", "-", 2.4), ("psi_X8", "#d1495b", "--", 1.8), ("psi_HJl9", "#2e7d4f", "--", 1.8)):
        zm.contour(R, Z, -z[key], [lev[key.split("_")[1]]], colors=col, linestyles=ls, linewidths=lw)
    zm.set_xlim(1.0, 1.75); zm.set_ylim(0.75, 1.35); zm.set_aspect("equal")
    zm.set_xticks(R[(R > 1.0) & (R < 1.75)]); zm.set_yticks(Z[(Z > 0.75) & (Z < 1.35)]); zm.set_xticklabels([]); zm.set_yticklabels([])
    zm.grid(alpha=0.5); zm.set_xlabel("grid columns (dR 2.7 cm)"); zm.set_ylabel("grid rows (dZ 5 cm)")
    zm.set_title("zoom: upper X-point (lines = scorer grid)\nblack truth, red X8, green HJl9", fontsize=9)
    fig.suptitle("One VALID frame (chosen to show the change): white / black = true LCFS, dashed = our LCFS", fontsize=10, weight="bold")
    fig.tight_layout(); return fig


def fusion_ladder():
    _, m = load(); s = m["sample_scores"]; arms = ["X8", "X8axH", "HJ", "HJl9"]
    lab = {"X8": "X8\n(public now)", "X8axH": "+ axis to\nhead axis", "HJ": "+ joint\nfit", "HJl9": "+ li cap\n0.9 (HJl9)"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    v = [s[a]["S"] for a in arms]; a1.bar(range(4), v, color=["#bbbbbb", PST, PST, PST_D], ec="k")
    for i, x in enumerate(v):
        a1.text(i, x + 0.00002, f"{x:.5f}", ha="center", fontsize=8)
    a1.set_ylim(0.9950, 0.9958); a1.set_xticks(range(4)); a1.set_xticklabels([lab[a] for a in arms], fontsize=8); a1.set_title("VALID sample S (official scorer)")
    names = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
    head = dict(R_axis=0.9814, Z_axis=0.9929, kappa=0.9764, tri_top=0.9952, tri_bot=0.9950, volume=0.9924, li=0.9765)
    M = np.array([[s[a]["cons_each"][n] for n in names] for a in arms] + [[head[n] for n in names]])
    im = a2.imshow(M - M[0], cmap="RdYlGn", vmin=-0.006, vmax=0.006, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a2.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center", fontsize=7.5)
    a2.set_xticks(range(7)); a2.set_xticklabels(names, fontsize=8); a2.set_yticks(range(5)); a2.set_yticklabels(arms + ["heads alone"], fontsize=8)
    a2.set_title("consistency R2 per scalar (colour = change vs X8)"); fig.colorbar(im, ax=a2, fraction=0.03)
    fig.tight_layout(); return fig


def fusion_gap():
    terms = ["psi R2", "q95 / betaN", "D_LCFS", "consistency"]; x8 = [0.00005, 0.00031, 0.00027, 0.00110]; hj = [0.00005, 0.00031, 0.00023, 0.00080]
    fig, ax = plt.subplots(figsize=(8, 2.8)); y = np.arange(4)
    ax.barh(y - 0.2, x8, 0.4, color="#bbbbbb", label="X8 (S 0.99459, rank 16)"); ax.barh(y + 0.2, hj, 0.4, color=PST_D, label="HJl9 (S 0.99491, rank 13)")
    for i in range(4):
        ax.text(x8[i] + 0.00001, i - 0.2, f"{x8[i]:.5f}", va="center", fontsize=8); ax.text(hj[i] + 0.00001, i + 0.2, f"{hj[i]:.5f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(terms); ax.invert_yaxis(); ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("S lost to #1 on the public board (#1: S 0.9963)"); ax.set_title("Where the public gap is: 0.0017 (X8) -> 0.0014 (HJl9)")
    fig.tight_layout(); return fig


def fusion_quantisation():
    z, m = load(); a = m["dlcfs_anatomy"]["by_location"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    axs[0].hist(z["rtop_frac"], 20, color=PST_D); axs[0].set_xlabel("position of the top point between two grid columns (0 = on a column)")
    axs[0].set_title("tri: the top point sits near a grid column")
    e = np.linspace(-0.12, 0.12, 13)
    k = [1.433, 1.492, 1.511, 1.532, 1.554, 1.577, 1.644, 1.669, 1.693, 1.717, 1.740, 1.764, 1.788]     # doc 35 §38aa kappa probe, frame 4
    axs[1].plot(e, k, "o-", color=CLS_D); axs[1].axhline(1.623, color=QUB_D, ls="--"); axs[1].text(-0.115, 1.628, "head kappa (in the gap)", color=QUB_D, fontsize=8)
    axs[1].set_xlabel("stretch e (kappa x exp(e))"); axs[1].set_ylabel("kappa the scorer finds"); axs[1].set_title("kappa: a jump of one grid row")
    loc = ["top", "bottom", "inboard", "outboard"]; axs[2].bar(loc, [a[l]["share"] for l in loc], color=[QUB_D, QUB_D, "#bbbbbb", "#bbbbbb"])
    axs[2].set_title("D_LCFS: where the largest distance is"); axs[2].set_ylabel("share of the D_LCFS sum")
    fig.tight_layout(); return fig


# ----------------------------------------------------------------------------------------------- Learn2Design
def l2d_flow():
    fig, ax = canvas(12, 3.4, (0, 12), (0, 3.4))
    st = [("topology\n(21 cells, labels)", GRY, GRY_D), ("transfer: copy values\nof 64 donor designs\nby (component, property)", GRY, GRY_D),
          ("evaluate the pool;\nbest start >= 1.5 ->\nwiden to 512 donors", OPT, OPT_D), ("noisy Adam in\nsigmoid space\nfrom the best start", OPT, OPT_D),
          ("no gain for 500 evals:\nrestart from the best\nlr 0.01 -> 0.003", OPT, OPT_D), ("best feasible\ndesign\n(4 h, H100)", GRY, GRY_D)]
    for k, (t, fc, ec) in enumerate(st):
        x = 0.15 + k * 1.98; box(ax, x, 1.5, 1.75, 1.1, t, fc, ec, 8)
        if k:
            arrow(ax, x - 0.23, 2.05, x, 2.05)
    arrow(ax, 9.9 + 0.87, 1.5, 7.9 + 0.87, 1.5, c=OPT_D, ls="--"); ax.text(8.8, 1.2, "loop until the time ends", ha="center", fontsize=8, color=OPT_D)
    ax.text(0.15, 0.55, "Loss = mean over 50 frequencies of log10(S / S_Voyager), S = noise / signal. Lower is better. Hard limit: 3.5 MW in the arms.",
            fontsize=8.5)
    ax.text(0.15, 0.15, "No qubits in this entry: the register design generator (doc 35 §40g) was worse than random designs.", fontsize=8.5, color=QUB_D, weight="bold")
    legend(ax, 0.15, 3.05, items=(("optimizer", OPT, OPT_D), ("data", GRY, GRY_D)))
    return fig


def l2d_curves():
    z, _ = load()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    for r, c in zip((2, 5, 6), ("#5b3f8c", "#9c7fd0", "#2f5d8a")):
        y = z[f"l2d_v3b_run{r}"]; y = np.where(np.isfinite(y), y, np.nan); a1.plot(np.arange(1, len(y) + 1), y, color=c, label=f"v3b, topology {r}")
    a1.scatter([10000, 40000, 150000], [0.36, 0.08, -0.211], color=QUB_D, zorder=5, label="Round 2 #1 (15 / 60 / 240 min)")
    a1.set_xscale("log"); a1.set_ylim(-0.4, 3); a1.set_xlabel("evaluations"); a1.set_ylabel("best feasible loss")
    a1.axhline(0.504, color="#888888", ls=":", label="best organizer baseline 0.504"); a1.legend(fontsize=7.5); a1.set_title("our search stops near 0.4; #1 goes on for 4 h")
    runs = [1, 3, 4, 5, 6, 8]; v3b = [0.463, 0.424, 0.541, 0.451, 0.227, 0.432]
    a2.bar([str(r) for r in runs], v3b, color=OPT_D, label="ours (v3b)"); a2.axhline(-0.211, color=QUB_D, ls="--", label="#1 mean -0.211 (4 h)")
    a2.axhline(0.36, color="#888888", ls=":", label="#1 mean at 15 min ~0.36"); a2.legend(fontsize=7.5, loc="lower right")
    a2.set_ylim(-0.3, 0.65); a2.set_xlabel("retired Round 2 topology"); a2.set_ylabel("best feasible loss"); a2.set_title("submission 413 at 3,000 evaluations")
    fig.tight_layout(); return fig


# ----------------------------------------------------------------------------------------------- Geothermal (Figure 1 style)
def slab(ax, x, w, h, fc, ec, top, bottom):
    ax.add_patch(Rectangle((x, 2.0 - h / 2), w, h, fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, 2.0 + h / 2 + 0.08, top, ha="center", va="bottom", fontsize=7.5)
    ax.text(x + w / 2, 2.0 - h / 2 - 0.08, bottom, ha="center", va="top", fontsize=7.2)


def geo_arch():
    fig, ax = canvas(12.5, 6.4, (0, 12.5), (-2.6, 4.0))
    ax.add_patch(Rectangle((0.05, 3.55), 12.4, 0.36, fc=CLS_D, ec=CLS_D))
    ax.text(6.25, 3.73, "Classical model: 1HP_NN U-Net (7.8 M parameters)  +  pooled 30-atom register in the bottleneck",
            color="white", ha="center", va="center", weight="bold")
    x0 = 2.0; xs = [x0 + 0.95 * k for k in range(9)]
    spec = [("4 fields", "256x16", 0.9, GRY, GRY_D, 0.32), ("32", "256x16", 1.5, CLS, CLS_D, 0.32), ("64", "128x8", 1.25, CLS, CLS_D, 0.32),
            ("128", "64x4", 1.0, CLS, CLS_D, 0.32), ("256", "32x2", 0.8, "#9cbfe3", CLS_D, 0.42), ("128", "64x4", 1.0, "#cfe6ea", "#3b7f8c", 0.32),
            ("64", "128x8", 1.25, "#cfe6ea", "#3b7f8c", 0.32), ("32", "256x16", 1.5, "#cfe6ea", "#3b7f8c", 0.32), ("1 (temp.)", "256x16", 0.9, "#cfe6ea", "#3b7f8c", 0.32)]
    for k, (t, b, h, fc, ec, w) in enumerate(spec):
        slab(ax, xs[k], w, h, fc, ec, t, b)
        if k:
            arrow(ax, xs[k - 1] + spec[k - 1][5] + 0.02, 2.0, xs[k] - 0.02, 2.0)
    for k in (1, 2, 3):                                                  # skip connections
        ax.annotate("", xy=(xs[8 - k] + 0.16, 2.95), xytext=(xs[k] + 0.16, 2.95),
                    arrowprops=dict(arrowstyle="-|>", ls="--", color="#888888", connectionstyle="arc3,rad=-0.12"))
    bx = xs[4] + 0.21
    ax.text(0.15, 3.25, "encoder | bottleneck | decoder", fontsize=8, color="#333333")
    ax.add_patch(Rectangle((1.05, -0.9), 10.4, 1.35, fc="#fff7f1", ec=QUB_D, lw=1.2, ls="--"))
    ax.text(1.15, 0.28, "pooled register (neck insertion)", color=QUB_D, weight="bold", fontsize=8.5)
    row = [("pool: mean over\n32x2 -> R^256", GRY, GRY_D), ("encode: FC\n256 -> atom controls", CLS, CLS_D),
           ("3 x 10 Rydberg atoms\nRz Rx, exp(-i t V), x4", QUB, QUB_D), ("readout <Z> in R^30; write:\nFC 30 -> 256, x alpha (zero-init)", CLS, CLS_D)]
    for k, (t, fc, ec) in enumerate(row):
        x = 1.25 + 2.55 * k; box(ax, x, -0.75, 2.3, 0.75, t, fc, ec, 7.5)
        if k:
            arrow(ax, x - 0.25, -0.38, x, -0.38)
    arrow(ax, bx - 0.1, 1.58, 1.25 + 1.15, 0.02, c=QUB_D, ls="--"); arrow(ax, 1.25 + 7.65 + 1.15, 0.02, bx + 0.1, 1.58, c=QUB_D, ls="--")
    ax.text(bx + 0.35, 1.0, "broadcast-add\nto all 32x2 positions", fontsize=7.5, color=QUB_D)
    ax.text(0.15, -1.3, "Routed variant (allocation study, frozen U-Net): the residual is split into tiles; a static rule sends 30 of them to the same",
            fontsize=8)
    ax.text(0.15, -1.62, "3 x 10 register; a linear head corrects the rest. Utilization ceiling U = 0.459; captured 0.152; test gain +9.2%.", fontsize=8)
    ax.text(0.15, -2.0, "Register vs a matched MLP in the same slot: -0.18% (1 of 3 seeds better) -> no quantum advantage shown at this size.",
            fontsize=8, color=QUB_D, weight="bold")
    ax.text(0.15, -2.4, "encoder: [conv 5x5, BN, ReLU] x2, max-pool 2;  decoder: convT 2x2/2, concat skip (dashed), conv blocks;  head: conv 1x1",
            fontsize=7.5, color="#333333")
    return fig


def register_detail():
    fig, ax = canvas(10.5, 3.4, (0, 10.5), (0, 3.4))
    for g in range(3):
        cx = 0.9 + g * 1.35
        for a in range(10):
            ang = 2 * np.pi * a / 10; ax.add_patch(Circle((cx + 0.45 * np.cos(ang), 1.9 + 0.45 * np.sin(ang)), 0.08, fc=QUB, ec=QUB_D))
        ax.text(cx, 1.25, f"group {g + 1}\n10 atoms", ha="center", va="top", fontsize=8)
    ax.text(2.25, 2.85, "30 atoms (3 groups x 10)", ha="center", weight="bold")
    seq = [("encode", "Rz Rx per atom,\nangles from\nthe input"), ("interact", "exp(-i tau V),\nRydberg blockade,\ndetuning per input"), ("repeat", "4 layers"),
           ("measure", "<X>, <Y>, <Z>\nper atom")]
    for k, (t, s) in enumerate(seq):
        x = 4.45 + k * 1.5; box(ax, x, 1.3, 1.3, 1.2, f"{t}\n{s}", QUB, QUB_D, 7.2)
        if k:
            arrow(ax, x - 0.2, 1.9, x, 1.9)
    arrow(ax, 3.95, 1.9, 4.45, 1.9)
    ax.text(4.45, 0.6, "detuning delta = b + pi tanh(W f): the input f sets the atom configuration (input-dependent)", fontsize=8)
    return fig


def geo_fields():
    z, m = load(); g = np.load(ROOT / "paper" / "fig_cache" / "fig_fields_1hp_s23.npz"); names = ["pressure gradient", "permeability", "distance to pump (SDF)", "material ID"]
    fig, axs = plt.subplots(1, 8, figsize=(13, 5.2))
    for k in range(4):
        f = z["geo_inputs"][k]; axs[k].imshow(f, cmap="cividis", aspect="auto"); axs[k].set_title(names[k], fontsize=8)
        if np.ptp(f) < 1e-6:
            axs[k].text(0.5, 0.5, "uniform\n(one value\nin this sample)", transform=axs[k].transAxes, ha="center", va="center", color="white", fontsize=8)
    t = g["truth"]; vmin, vmax = np.percentile(t, 1), np.percentile(t, 99.5)
    for ax, (nm, f) in zip(axs[4:7], (("truth", t), (f"classical\nMAE {float(g['vanilla_mae']):.4f}", g["vanilla_pred"]), (f"+ 30 qubits\nMAE {float(g['mr30ryd_mae']):.4f}", g["mr30ryd_pred"]))):
        ax.imshow(f, cmap="inferno", vmin=vmin, vmax=vmax, aspect="auto"); ax.set_title(nm, fontsize=8)
    d = np.abs(g["mr30ryd_pred"] - t) - np.abs(g["vanilla_pred"] - t); lim = np.abs(d).max()
    axs[7].imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto"); axs[7].set_title("error change\n(blue: qubits better)", fontsize=8)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Heat plume, one TEST sample fixed in advance (temperature in deg C; inputs left, outputs right)", fontsize=10, weight="bold")
    fig.tight_layout(); return fig


def geo_results():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    v = [0.0291, 0.0292, 0.0288]; a1.bar(["classical", "+ MLP", "+ 30 qubits"], v, color=[CLS_D, "#999999", QUB_D])
    for i, x in enumerate(v):
        a1.text(i, x + 0.00003, f"{x:.4f}", ha="center", fontsize=8.5)
    a1.set_ylim(0.0284, 0.0294); a1.set_title("pooled register: test error (lower is better)\nqubits +1.3% vs MLP (2 of 3 seeds better)")
    lab = ["DC share\n(pooled ceiling)", "pooled gain", "utilization\nceiling U", "captured", "routed gain"]
    vals = [0.047, 0.0145, 0.459, 0.152, 0.0923]
    a2.bar(lab, vals, color=["#bbbbbb", QUB_D, "#bbbbbb", QUB_D, QUB_D])
    for i, x in enumerate(vals):
        a2.text(i, x + 0.008, f"+{100 * x:.2f}%" if "gain" in lab[i] else f"{x:.3f}", ha="center", fontsize=8)
    a2.set_title("routing raises what 30 qubits can reach (static tiles)"); a2.set_ylim(0, 0.52)
    fig.tight_layout(); return fig
