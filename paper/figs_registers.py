"""Qubit-register configuration figure (fig_registers.pdf).

Rows 1-3: the three blockade-decoupled 10-qubit Rydberg registers.
  (a) initial geometry (exact re-derivation: torch.manual_seed(seed) ->
      build_hybrid('mr30ryd'), same call order as finetune_td.py)
  (b) trained geometry, seed 1337 (TD-trained TRL2D winner)
  (c) trained geometry overlaid across seeds 1337/1338/1339
Bottom row: top-3 singular modes of register 1's layer-averaged
input-conditioning map W (delta = pi*tanh(W h)), rendered as per-atom color
on the trained geometry.

Edges: pairs with blockade coupling V = clamp(r, 0.5)^-6 >= V_THR, where
C6 = 1 and distances are in blockade-radius units (V(R_b) = 1); we use
V_THR = 0.5, i.e. r <= 2^(1/6) ~ 1.12 R_b. Thicker/darker = stronger.

Run: cd /home/mohhakami/Documents/paper/quworld && uv run paper/figs_registers.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_PAPER = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER / "phase_b"))
from cnext_hybrid import build_hybrid  # noqa: E402

ROOT = _PAPER.parent
SEEDS = (1337, 1338, 1339)
CKPT = {s: ROOT / "runs" / "phase_b2" / f"mr30ryd_s{s}.pt" for s in SEEDS}
OUT = _PAPER / "manuscript" / "fig_registers.pdf"

INK, MUTED, BLUE = "#1F2937", "#6B7280", "#1D4ED8"
EDGE_LO, EDGE_HI = "#D1D5DB", "#374151"
V_THR = 0.5            # coupling threshold: half the coupling at R_b (V(R_b)=1)
V_MAX = 0.5 ** -6      # clamp ceiling in qlayer._blockade_energy (r >= 0.5)
N_Q, N_LAYERS, D_IN = 10, 4, 672


def reconstruct_init(seed: int) -> list[np.ndarray]:
    """Exact initial positions: same seed + construction order as training."""
    torch.manual_seed(seed)
    model = build_hybrid("mr30ryd")
    regs = model.neck.bottleneck.core.regs
    return [r.pos.detach().numpy().copy() for r in regs]


def load_trained(seed: int) -> dict:
    ck = torch.load(CKPT[seed], map_location="cpu", weights_only=True)
    sd = ck["model_state_dict"]
    pre = "neck.bottleneck.core.regs"
    out = {"val": ck.get("val"), "step": ck.get("step"), "pos": [], "w_det": []}
    for r in range(3):
        out["pos"].append(sd[f"{pre}.{r}.pos"].numpy().copy())
        w = sd[f"{pre}.{r}.cond.weight"].numpy()          # (80, 672)
        out["w_det"].append(w.reshape(N_LAYERS, N_Q, 2, D_IN)[:, :, 0, :])
    return out


def coupling(pos: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    v = np.clip(d, 0.5, None) ** -6.0
    np.fill_diagonal(v, 0.0)
    return v


def edge_list(pos: np.ndarray):
    v = coupling(pos)
    n = pos.shape[0]
    return [(i, j, v[i, j]) for i in range(n) for j in range(i + 1, n)
            if v[i, j] >= V_THR]


def draw_register(ax, pos, edges=True, atom_kw=None, edge_alpha=1.0):
    if edges:
        for i, j, v in edge_list(pos):
            t = np.log(v / V_THR) / np.log(V_MAX / V_THR)   # 0..1
            t = float(np.clip(t, 0.0, 1.0))
            col = tuple((1 - t) * np.array(matplotlib.colors.to_rgb(EDGE_LO)) +
                        t * np.array(matplotlib.colors.to_rgb(EDGE_HI)))
            ax.plot(pos[[i, j], 1], pos[[i, j], 0], "-", lw=0.6 + 1.5 * t,
                    color=col, alpha=edge_alpha, zorder=1,
                    solid_capstyle="round")
    kw = dict(s=26, c=BLUE, edgecolors="white", linewidths=0.5, zorder=3)
    if atom_kw:
        kw.update(atom_kw)
    ax.scatter(pos[:, 1], pos[:, 0], **kw)


def style_panel(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#E5E7EB")
        sp.set_linewidth(0.6)


def stats(name, pos):
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(1).mean()
    c = pos - pos.mean(0)
    rg = np.sqrt((c ** 2).sum(1).mean())
    ev = np.linalg.eigvalsh(np.cov(c.T))[::-1]
    aniso = float(np.sqrt(ev[0] / max(ev[1], 1e-12)))
    ne = len(edge_list(pos))
    print(f"  {name}: edges={ne}  mean-NN={nn:.3f}  Rg={rg:.3f}  "
          f"aniso(sqrt l1/l2)={aniso:.2f}")
    return ne, nn, rg, aniso


def main() -> None:
    print("== reconstructing exact inits (seeded build, CPU) ==")
    init = {s: reconstruct_init(s) for s in SEEDS}
    trained = {s: load_trained(s) for s in SEEDS}
    for s in SEEDS:
        print(f"seed {s}: val={trained[s]['val']:.4f} step={trained[s]['step']}")

    print("\n== per-register geometry stats ==")
    rows = []
    for r in range(3):
        print(f"Register {r + 1}:")
        i_stats = stats("init  s1337", init[1337][r])
        t_stats = stats("train s1337", trained[1337]["pos"][r])
        for s in (1338, 1339):
            stats(f"train s{s}", trained[s]["pos"][r])
        dvec = trained[1337]["pos"][r] - init[1337][r]
        disp = np.linalg.norm(dvec, axis=1)
        c = init[1337][r] - init[1337][r].mean(0)
        rhat = c / np.linalg.norm(c, axis=1, keepdims=True)
        print(f"  displacement s1337: rms={np.sqrt((disp**2).mean()):.3f} "
              f"max={disp.max():.3f} mean-radial={(dvec * rhat).sum(1).mean():+.4f} "
              f"(units of R_b)")
        rows.append((i_stats, t_stats))
        for s in (1338, 1339):
            d2 = np.linalg.norm(trained[s]["pos"][r] - trained[1337]["pos"][r], axis=1)
            print(f"  seed-{s} vs seed-1337 trained: rms dev={np.sqrt((d2**2).mean()):.3f}")

    # --- detuning modes: SVD of register-1 layer-averaged detuning map ---
    print("\n== detuning modes (register 1, seed 1337, layer-averaged W) ==")
    w1 = trained[1337]["w_det"][0].mean(0)                  # (10, 672)
    u, sv, vt = np.linalg.svd(w1, full_matrices=False)
    print(f"  top-6 singular values: {np.round(sv[:6], 4)}")
    print(f"  energy in top-3: {100 * (sv[:3]**2).sum() / (sv**2).sum():.1f}%")
    modes = []
    for j in range(3):
        m = u[:, j] * sv[j]                                  # per-qubit pattern
        if m[np.abs(m).argmax()] < 0:
            m = -m
        modes.append(m)
        print(f"  mode {j + 1}: sign split {int((m > 0).sum())}+/"
              f"{int((m < 0).sum())}-")
    # cross-register alignment of dominant latent direction
    for r in (1, 2):
        _, _, vtr = np.linalg.svd(trained[1337]["w_det"][r].mean(0),
                                  full_matrices=False)
        print(f"  |v1(reg1) . v1(reg{r + 1})| = {abs(vt[0] @ vtr[0]):.3f}")

    # ------------------------------------------------------------- figure ---
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "figure.facecolor": "white",
        "axes.facecolor": "white", "font.family": "sans-serif"})
    fig = plt.figure(figsize=(5.4, 6.3), dpi=200)
    gs = fig.add_gridspec(4, 3, hspace=0.28, wspace=0.10,
                          height_ratios=[1, 1, 1, 1.02],
                          left=0.075, right=0.985, top=0.945, bottom=0.045)

    # shared limits per register row (init + all trained seeds)
    for r in range(3):
        pts = np.concatenate([init[1337][r]] + [trained[s]["pos"][r] for s in SEEDS])
        pad = 0.38
        ylim = (pts[:, 0].min() - pad, pts[:, 0].max() + pad)
        xlim = (pts[:, 1].min() - pad, pts[:, 1].max() + pad)
        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            if c == 0:
                draw_register(ax, init[1337][r])
            elif c == 1:
                # trained geometry + displacement vectors magnified x20
                draw_register(ax, trained[1337]["pos"][r])
                for k in range(N_Q):
                    p0 = trained[1337]["pos"][r][k]
                    dv = 20.0 * (p0 - init[1337][r][k])
                    ax.annotate("", xy=(p0[1] + dv[1], p0[0] + dv[0]),
                                xytext=(p0[1], p0[0]), zorder=4,
                                arrowprops=dict(arrowstyle="-|>", lw=0.7,
                                                color="#B45309",
                                                shrinkA=2.5, shrinkB=0,
                                                mutation_scale=5))
                if r == 0:
                    ax.text(0.97, 0.05, r"shifts $\times$20", ha="right",
                            va="bottom", transform=ax.transAxes,
                            fontsize=6.5, color="#B45309")
            else:
                for s, a in zip(SEEDS, (0.95, 0.55, 0.30)):
                    draw_register(ax, trained[s]["pos"][r], edges=(s == 1337),
                                  atom_kw=dict(alpha=a), edge_alpha=0.5)
            style_panel(ax, xlim, ylim)
            if c == 0:
                ax.set_ylabel(f"Register {r + 1}", fontsize=8, color=INK)
            if r == 0:
                ax.set_title(["(a) initialization",
                              "(b) trained, seed 1337",
                              "(c) trained, 3 seeds"][c], color=INK, pad=3)
            if r == 0 and c == 0:   # scale bar: 1 blockade radius
                y0 = ylim[1] - 0.12
                ax.plot([xlim[0] + 0.15, xlim[0] + 1.15], [y0, y0], "-",
                        lw=1.1, color=MUTED, solid_capstyle="butt")
                ax.text(xlim[0] + 1.28, y0, r"$R_b$", ha="left",
                        va="center", fontsize=7, color=MUTED)

    # bottom row: detuning modes on register-1 trained geometry
    pos1 = trained[1337]["pos"][0]
    pad = 0.38
    ylim = (pos1[:, 0].min() - pad, pos1[:, 0].max() + pad)
    xlim = (pos1[:, 1].min() - pad, pos1[:, 1].max() + pad)
    cmap = plt.get_cmap("RdBu_r")
    for j in range(3):
        ax = fig.add_subplot(gs[3, j])
        draw_register(ax, pos1, edges=True, atom_kw=dict(s=0))
        m = modes[j] / np.abs(modes[j]).max()
        ax.scatter(pos1[:, 1], pos1[:, 0], s=46, c=m, cmap=cmap, vmin=-1,
                   vmax=1, edgecolors=MUTED, linewidths=0.5, zorder=3)
        style_panel(ax, xlim, ylim)
        share = 100.0 * sv[j] ** 2 / (sv ** 2).sum()
        ax.set_title(f"mode {j + 1} ({share:.0f}% var)", color=INK, pad=3)
        if j == 0:
            ax.set_ylabel("detuning modes", fontsize=8, color=INK)
    fig.text(0.53, 0.012,
             "(d) input-conditioned detuning modes "
             r"($\delta=\pi\tanh(Wh)$, register 1, red $+$ / blue $-$)",
             ha="center", fontsize=8, color=INK)

    fig.savefig(OUT, format="pdf")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
