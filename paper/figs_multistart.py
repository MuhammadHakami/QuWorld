"""Figure: rollout error versus start frame under the sliding-start protocol
(19-multistart-power-plan.md). Left: seed-mean 13:30 VRMSE per arm with the
anchor as a dashed reference. Right: paired difference quantum minus comparator
per start (seed mean, seed min-max band); below zero = quantum better.
Palette validated with the dataviz validator (light mode, all checks pass).
Run: uv run paper/figs_multistart.py  -> paper/manuscript/fig_multistart.{pdf,png}
"""
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs/phase_b2"
SEEDS = ("1337", "1338", "1339")
INK, MUTED, GRID, SURFACE = "#1F2937", "#6B7280", "#E5E7EB", "#FFFFFF"
ARMS = {  # label, validated series color
    "mr30ryd": ("Quantum-TD", "#2a78d6"), "noent": ("No-interaction-TD", "#1baf7a"),
    "vanilla": ("Vanilla-TD", "#4a3aa7"), "mlp30": ("Matched-MLP-TD", "#eb6834")}
SHORT = {"Quantum-TD": "Quantum", "No-interaction-TD": "No-int.", "Vanilla-TD": "Vanilla",
         "Matched-MLP-TD": "MLP", "Anchor (no fine-tune)": "Anchor"}


def load(name):
    return np.load(RUNS / f"multistart_{name}.npz")


def per_start(arm):
    """(seeds, starts) mean over trajectories of the 13:30 window."""
    return np.stack([load(f"{arm}_s{s}")["w1330"].mean(axis=1) for s in SEEDS])


def spread_labels(ys, min_gap):
    """Push label y-positions apart (sorted order preserved)."""
    order = np.argsort(ys)
    pos = np.array(ys, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=7, length=2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({"font.size": 8, "text.color": INK, "axes.labelcolor": INK})
    starts = load("anchor")["starts"]
    anchor = load("anchor")["w1330"].mean(axis=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.75), dpi=200,
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    # --- left: seed-mean VRMSE per start
    handles, ends = [], {}
    h_anchor, = ax1.plot(starts, anchor, ls="--", lw=1.4, color=MUTED, zorder=2,
                         label="Anchor (no fine-tune)")
    ends["Anchor (no fine-tune)"] = (anchor[-1], MUTED)
    for arm, (label, col) in ARMS.items():
        m = per_start(arm).mean(axis=0)
        h, = ax1.plot(starts, m, lw=1.8, color=col, marker="o", ms=3.2,
                      mec=SURFACE, mew=0.6, zorder=3, label=label)
        handles.append(h)
        ends[label] = (m[-1], col)
    labels = list(ends)
    ys = spread_labels([ends[k][0] for k in labels], min_gap=0.016)
    for k, y in zip(labels, ys):
        ax1.annotate(SHORT[k], (starts[-1] + 1.5, y), fontsize=6, color=INK,
                     va="center", annotation_clip=False)
    ax1.set_xlabel("rollout start frame")
    ax1.set_ylabel("VRMSE, steps 13–30 (lower is better)")
    ax1.set_xticks(starts[::2]); ax1.set_xticks(starts, minor=True)
    ax1.set_title("Error versus start (seed mean)", fontsize=8, color=INK, loc="left")
    style(ax1)
    # --- right: paired difference quantum minus comparator
    q = per_start("mr30ryd")
    for arm in ("vanilla", "mlp30", "noent"):
        d = q - per_start(arm)                      # (seeds, starts)
        label, col = ARMS[arm]
        ax2.fill_between(starts, d.min(axis=0), d.max(axis=0), color=col, alpha=0.15, lw=0)
        ax2.plot(starts, d.mean(axis=0), lw=1.8, color=col, marker="o", ms=3.2,
                 mec=SURFACE, mew=0.6, zorder=3, label=f"vs {label.replace('-TD', '')}")
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_xlabel("rollout start frame")
    ax2.set_ylabel("Quantum-TD − comparator")
    ax2.set_xticks(starts[::2]); ax2.set_xticks(starts, minor=True)
    ax2.set_title("Paired difference", fontsize=8, color=INK, loc="left")
    ax2.legend(frameon=False, fontsize=6, loc="lower left")
    style(ax2)
    ax1.legend(handles=handles + [h_anchor], frameon=False, fontsize=6, ncol=3, loc="upper left",
               bbox_to_anchor=(-0.02, -0.28), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(w_pad=3.0)
    fig.subplots_adjust(right=0.98, bottom=0.34)
    out = ROOT / "paper/manuscript/fig_multistart"
    fig.savefig(f"{out}.pdf", facecolor=SURFACE)
    fig.savefig(f"{out}.png", facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main()
