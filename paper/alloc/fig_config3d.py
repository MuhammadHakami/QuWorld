"""Per-input qubit configuration in 3-D (fig_config3d.pdf): atom positions (x, y, z), the Fourier octave of the mode each
atom reads and corrects, and tether strengths between 10-atom groups, for four test windows of one trained routed register.

    QW_OUT=runs/alloc/geom_dump QW_DUMP_GEOM=runs/alloc/geom_dump/<name>.npz .venv/bin/python paper/alloc/adrc.py ...
    .venv/bin/python paper/alloc/fig_config3d.py runs/alloc/geom_dump/<name>.npz [title]
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OCT = ["#d95f02", "#e6ab02", "#1b9e77", "#7570b3", "#666666"]
GRP = ["#1f78b4", "#33a02c", "#e31a1c"]
PAIRS = [(0, 1), (1, 2), (0, 2)]                       # tether index -> (group, group), K = 30 layout


def main(path: str, title: str) -> None:
    d = np.load(path)
    pos, band = d["pos"], d["band"] if "band" in d else np.zeros(d["idx"].shape, int)
    teth = d["tethers"] if "tethers" in d else None
    B, K, _ = pos.shape
    G = K // 10
    tethered = [b for b in range(B) if teth is not None and teth[b].max() > 0.02]
    show = sorted(set([0, B // 2, B - 1] + tethered[:1]))[:4]                  # include a window with a tether on
    plt.rcParams.update({"font.size": 6.5})
    fig = plt.figure(figsize=(7.0, 2.1))
    lo, hi = pos.reshape(-1, 3).min(0), pos.reshape(-1, 3).max(0)
    for j, b in enumerate(show):
        ax = fig.add_subplot(1, len(show), j + 1, projection="3d")
        P = pos[b] + np.repeat(np.arange(G) * 7.0, 10)[:, None] * np.array([1.0, 0.0, 0.0])   # groups side by side
        for g in range(G):
            q = P[10 * g:10 * g + 10]
            ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=12, color=GRP[g], edgecolors="k", linewidths=0.3, depthshade=False)
        if teth is not None:
            cen = [P[10 * g:10 * g + 10].mean(0) for g in range(G)]
            for t, (g1, g2) in enumerate(PAIRS[:teth.shape[1]]):
                if g2 < G and teth[b, t] > 0.02:
                    ax.plot(*zip(cen[g1], cen[g2]), color="k", lw=0.4 + 3 * float(teth[b, t]), alpha=0.7)
        ax.set_xlim(lo[0], hi[0] + 7.0 * (G - 1)); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2] + 1e-3)
        ax.set_xlabel("x", labelpad=-10); ax.set_ylabel("y", labelpad=-10); ax.set_zlabel("z", labelpad=-10)
        ax.tick_params(labelsize=0, length=0); ax.view_init(22, -60)
        tt = "" if teth is None else "\ntethers " + " ".join(f"{v:.1f}" for v in teth[b][:G])
        ax.set_title(f"test window {int(d['rows'][b])}{tt}", fontsize=6)
    hs = [plt.Line2D([], [], marker="o", ls="", color=c, markeredgecolor="k", markersize=4, label=f"atom group {i + 1}")
          for i, c in enumerate(GRP[:G])]
    hs.append(plt.Line2D([], [], color="k", lw=2, label="tether (width = strength)"))
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.04),
               title=f"{title}: per-input atom positions (x, y, z) and tethers; every atom reads a coarse-octave (b0) mode", title_fontsize=6)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.18, wspace=0.0)
    out = ROOT / "paper/manuscript/iclr2027/fig_config3d.pdf"
    fig.savefig(out, bbox_inches="tight"); print("wrote", out)
    moved = np.linalg.norm(pos - pos.mean(0, keepdims=True), axis=-1)
    print("per-input displacement from the mean layout: mean %.3f max %.3f; z range %.3f" %
          (moved.mean(), moved.max(), np.ptp(pos[..., 2])))
    if teth is not None:
        print("tether strengths per input:", np.round(teth[:, :G], 3).tolist())
    print("octaves routed per input:", [np.bincount(band[b].astype(int), minlength=5).tolist() for b in range(B)])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "shear_flow")
