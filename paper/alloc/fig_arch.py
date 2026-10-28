"""Figure 1 of the ICLR manuscript (fig_arch.pdf): layer-level architecture of the hybrid, with tensor sizes.

    .venv/bin/python paper/alloc/fig_arch.py

Design: every module has a coloured title band (name), a one-line role description, and a drawing of the component
(feature maps as slabs, fields as images, fully connected layers as bars, the register as a gate-level circuit, ...).

Sizes are read from the code, not typed from memory:
- Backbone: the released CNextU-net of TRL2D (data/cnext_ckpt/config.json; forward-hook trace, 18,575,344 parameters).
  Stem conv 3x3 16->42; encoder stages of 2 ConvNeXt blocks + LN / conv 2x2 stride 2; neck stage of 1 ConvNeXt block at
  672 x 8 x 24; decoder stages of transposed conv 2x2 stride 2 + skip concat + 2 ConvNeXt blocks; head conv 3x3 42->4.
- Pooled neck register (paper/phase_b/cnext_hybrid.py, mr30ryd): global average pool -> three 10-atom registers, each
  conditioned by Linear 672->80 (4 layers x 10 atoms x 2), trained 2-D positions, Omega and tau per layer -> <Z> (30) ->
  Linear 30->672, gated by a zero-initialised alpha (672) and broadcast over the 8 x 24 positions.
- Gate (paper/alloc/adrc.py Gate): MLP 72->64->20 on log band energies of p - x_t and dx (2 x 20) + 32 global statistics;
  alpha = alpha_fit + 1.5 tanh(.) per field (4) x octave (5).
- Routed co-processor (paper/alloc/adrc.py ADRC, cores.TileRegister): band-wise POD, 4 fields x 128 modes, features
  f_m in R^4 (projections of p - x_t, p, x_t, dx); router MLP 4->32->2 (score, group key), top-30; configuration
  hypernetwork 8->32->3 per atom (dx, dy, dz) and per-input tethers Linear 4->3; register: 3 groups x 10 atoms (2^10
  amplitudes each), detunings delta = b + pi tanh(W f) with W 4->4 (one per layer), 4 layers of R_z(delta) R_x(Omega)
  then the C6 / r^6 interaction phase, readout <X>, <Y>, <Z> (30 x 3), tether feed Linear 3->4 between groups;
  write-back c = s (W_q f + A o), W_q 4->1, A 3->1 (zero-init); classical head Linear 4->1 on every other mode.
"""
from __future__ import annotations

import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/manuscript/iclr2027/fig_arch.pdf"
BLUE, DBLUE, ORANGE, GREEN, PURPLE, GREY = "#dbe8f5", "#9ec1e3", "#fbe3cf", "#d9efe6", "#e6e1f2", "#f2f2f2"
NAVY, RUST, TEAL, PLUM = "#1f4e79", "#a0522d", "#2e7d5b", "#5b4a8a"
# band colours, one per module family
C_STEM, C_ENC, C_NECK, C_DEC, C_HEAD = "#8c8c8c", "#4a7fb5", "#1f4e79", "#3f8f9a", "#2e7d5b"
C_INS, C_GATE, C_GATE2, C_QPU, C_QPU2, C_CLS, C_OUT = "#7a5aa8", "#6a5a9a", "#8a7fbf", "#b5541f", "#d0803f", "#2e7d5b", "#555555"


# ------------------------------------------------------------------------------------------------- primitives
def box(ax, x, y, w, h, text, fc, fs=5.6, ls="-", ec="0.35", ha="center", lw=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.5", fc=fc, ec=ec, lw=lw, ls=ls))
    tx = x + w / 2 if ha == "center" else x + 0.5
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, linespacing=1.3)


def name(ax, x, y, text, color, fs=5.5, ha="center"):
    """Module name above a block (kept for the appendix figure)."""
    ax.text(x, y, text, ha=ha, va="bottom", fontsize=fs, color=color, weight="bold")


def band(ax, x0, x1, y, title, color, h=1.35, fs=4.9, tc="white"):
    """Coloured title band spanning a module."""
    ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, h, boxstyle="round,pad=0.05,rounding_size=0.45", fc=color, ec="none"))
    ax.text((x0 + x1) / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, color=tc, weight="bold")


def desc(ax, x, y, text, fs=4.3, color="0.25", ha="center"):
    """One- or two-line role description (what the module is for)."""
    ax.text(x, y, text, ha=ha, va="top", fontsize=fs, color=color, style="italic", linespacing=1.2)


def arrow(ax, a, b, ls="-", color="0.25", rad=0.0, lw=0.7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0.5, shrinkB=0.5))


def slab(ax, x, yc, c, hw, label, fc=BLUE):
    """A feature map: width ~ log channels, height ~ log spatial size; label = channels / H x W below."""
    w = 0.55 + 0.32 * math.log2(max(c, 8) / 8); h = 1.2 + 0.95 * math.log2(hw[0] / 4)
    ax.add_patch(Rectangle((x, yc - h / 2), w, h, fc=fc, ec="0.3", lw=0.5))
    ax.text(x + w / 2, yc - h / 2 - 0.35, label, ha="center", va="top", fontsize=4.2, linespacing=1.15)
    return x + w, yc


def mlp(ax, x, y, w, h, widths, fc="#8a7fbf", fs=4.2):
    """Fully connected layers: one bar per layer, height ~ log width, fan connections, widths below."""
    n = len(widths); top = max(math.log2(v + 1) for v in widths); cx = [x + w * (i + 0.5) / n for i in range(n)]
    hs = [h * (0.28 + 0.72 * math.log2(v + 1) / top) for v in widths]; yc = y + h / 2 + 0.5
    for i in range(n - 1):
        for fa in (-0.45, 0.0, 0.45):
            for fb in (-0.45, 0.45):
                ax.plot([cx[i] + 0.35, cx[i + 1] - 0.35], [yc + fa * hs[i], yc + fb * hs[i + 1]], color="0.6", lw=0.25)
    for c, hh, v in zip(cx, hs, widths):
        ax.add_patch(Rectangle((c - 0.35, yc - hh / 2), 0.7, hh, fc=fc, ec="0.3", lw=0.4))
        ax.text(c, y + 0.2, str(v), ha="center", va="bottom", fontsize=fs)


def field(ax, x, y, w, h, seed=0, cmap="RdBu_r", label=None):
    """A 2-D field (image) with a frame."""
    r = np.random.default_rng(seed); yy, xx = np.mgrid[0:1:24j, 0:3:72j]
    f = sum(r.normal() * np.sin(2 * np.pi * (r.uniform(0.3, 2) * xx + r.uniform(0.3, 2) * yy) + r.uniform(0, 6)) for _ in range(5))
    ax.imshow(f, extent=(x, x + w, y, y + h), cmap=cmap, aspect="auto", origin="lower", zorder=2)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="0.3", lw=0.5, zorder=3))
    if label:
        ax.text(x + w / 2, y - 0.35, label, ha="center", va="top", fontsize=4.3, linespacing=1.15)


def vec(ax, x, y, h, fc="#bbbbbb", w=0.6):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="0.3", lw=0.4))


def pool(ax, x, yc, label):
    """Global average pool: a feature map funnelled into a vector."""
    ax.add_patch(Rectangle((x, yc - 1.6), 1.6, 3.2, fc=DBLUE, ec="0.3", lw=0.5))
    ax.add_patch(Polygon([(x + 1.7, yc + 1.6), (x + 3.6, yc + 0.3), (x + 3.6, yc - 0.3), (x + 1.7, yc - 1.6)], fc="#e0e0e0", ec="0.4", lw=0.4))
    vec(ax, x + 3.7, yc - 2.2, 4.4)
    ax.text(x + 2.4, yc - 2.6, label, ha="center", va="top", fontsize=4.2, linespacing=1.15)


def circuit(ax, x, y, w, h, fs=4.2, nw=4):
    """One register group as gate layers: |0> wires, [R_z(delta) R_x(Omega)] per atom, the interaction phase over all
    atoms, repeated 4 times, then measurement."""
    wy = [y + h - 0.6 - i * (h - 1.2) / (nw - 1) for i in range(nw)]
    lab = ["1", "2", "$\\vdots$", "10"] if nw == 4 else ["1", "$\\vdots$", "10"]
    for yy, lb in zip(wy, lab):
        if lb != "$\\vdots$":
            ax.plot([x + 1.6, x + w - 0.2], [yy, yy], color="0.35", lw=0.4)
            ax.text(x + 0.1, yy, f"$|0\\rangle_{{{lb}}}$", fontsize=fs, va="center")
        else:
            ax.text(x + 0.9, yy, lb, fontsize=fs, va="center", ha="center")
    gx = x + 3.3
    for yy, lb in zip(wy, lab):
        if lb != "$\\vdots$":
            ax.add_patch(Rectangle((gx, yy - 0.55), 2.6, 1.1, fc="#f6c89f", ec="0.3", lw=0.4))
            ax.text(gx + 1.3, yy, "$R_z\\!R_x$", fontsize=fs - 0.4, ha="center", va="center")
    ix = gx + 3.1
    ax.add_patch(Rectangle((ix, wy[-1] - 0.6), 2.4, wy[0] - wy[-1] + 1.2, fc="#e9a46b", ec="0.3", lw=0.4))
    ax.text(ix + 1.2, (wy[0] + wy[-1]) / 2, "$e^{-i\\tau V}$", fontsize=fs, ha="center", va="center", rotation=90)
    ax.add_patch(Rectangle((gx - 0.4, wy[-1] - 1.0), ix + 2.8 - gx, wy[0] - wy[-1] + 2.0, fc="none", ec=RUST, lw=0.45, ls=(0, (2, 1.2))))
    ax.text(ix + 3.0, wy[0], "$\\times 4$", fontsize=fs + 0.4, color=RUST, va="center", bbox=dict(fc=ORANGE, ec="none", pad=0.3))
    mx = x + w - 2.4
    for yy, lb in zip(wy, lab):
        if lb != "$\\vdots$":
            ax.add_patch(Rectangle((mx, yy - 0.55), 1.8, 1.1, fc="white", ec="0.3", lw=0.4))
            ax.add_patch(Arc((mx + 0.9, yy - 0.3), 1.2, 1.0, theta1=0, theta2=180, lw=0.35))
            ax.plot([mx + 0.9, mx + 1.4], [yy - 0.3, yy + 0.35], color="0.2", lw=0.35)


def octaves(ax, x, y, s):
    """Dyadic l-inf FFT shells: nested squares, one colour per octave b = 0..4 (centre = lowest)."""
    cols = ["#2b4c7e", "#4f78a8", "#7fa3c9", "#b3cbe3", "#e2ecf6"]
    for b in range(4, -1, -1):
        e = s * (2 ** b) / 16
        ax.add_patch(Rectangle((x + (s - e) / 2, y + (s - e) / 2), e, e, fc=cols[b], ec="0.35", lw=0.3))


def modes(ax, x, y, w, h):
    """Band-wise POD modes as small images (coarse to fine)."""
    yy, xx = np.mgrid[0:1:16j, 0:3:48j]; tiles = [(0.5, 0.5), (1, 1), (2, 1), (3, 2), (5, 3), (8, 5)]
    tw, th = w / 3 - 0.3, h / 2 - 0.3
    for i, (kx, ky) in enumerate(tiles):
        f = np.cos(np.pi * kx * xx / 3 * 2) * np.cos(np.pi * ky * yy)
        cx, cy = x + (i % 3) * (tw + 0.3), y + (1 - i // 3) * (th + 0.3)
        ax.imshow(f, extent=(cx, cx + tw, cy, cy + th), cmap="PuOr", aspect="auto", origin="lower", zorder=2)
        ax.add_patch(Rectangle((cx, cy), tw, th, fc="none", ec="0.35", lw=0.35, zorder=3))


def topk(ax, x, y, w, h, k=3, n=10):
    """Per-mode scores with the top-k highlighted (router selection)."""
    r = np.random.default_rng(3); v = r.uniform(0.2, 1.0, n); top = set(np.argsort(-v)[:k])
    bw = w / n
    for i in range(n):
        ax.add_patch(Rectangle((x + i * bw + 0.08, y), bw - 0.16, v[i] * h, fc=C_QPU2 if i in top else "#c8c8c8", ec="0.4", lw=0.25))


def atoms(ax, x, y, w, h):
    """One group of 10 atoms: grid sites (grey) and per-input positions (orange) with blockade radii."""
    r = np.random.default_rng(5)
    for i in range(10):
        gx, gy = x + 0.8 + (i % 5) * (w - 1.6) / 4, y + 0.9 + (i // 5) * (h - 1.8)
        dx, dy = r.uniform(-0.6, 0.6), r.uniform(-0.4, 0.4)
        ax.add_patch(Circle((gx, gy), 0.18, fc="0.7", ec="none"))
        ax.add_patch(Circle((gx + dx, gy + dy), 0.32, fc=C_QPU2, ec="0.3", lw=0.3))
        if i in (2, 7):
            ax.add_patch(Circle((gx + dx, gy + dy), 1.05, fc="none", ec=C_QPU, lw=0.3, ls=(0, (1.5, 1))))


def mixer(ax, x, y, r=1.5):
    ax.add_patch(Circle((x, y), r, fc="white", ec="0.3", lw=0.5))
    ax.text(x, y, "$\\Sigma_b$", ha="center", va="center", fontsize=5.2)


# ------------------------------------------------------------------------------------------------- figure
def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.25))
    ax.set_xlim(0, 100); ax.set_ylim(0, 61); ax.axis("off")

    # ======================= row A: classical model (backbone -> neck -> head) and the pooled neck insertion
    yc = 50.6
    band(ax, 0.5, 67.0, 58.6, "Classical model: CNextU-net, released TRL2D checkpoint (18.6 M parameters)",
         NAVY, h=1.7, fs=5.2)
    field(ax, 0.6, yc - 2.2, 4.4, 4.4, seed=1, label="$x_{t-3..t}$\n16 ch")
    st = [("stem", 42, (128, 384)), ("E1", 84, (64, 192)), ("E2", 168, (32, 96)), ("E3", 336, (16, 48)), ("E4", 672, (8, 24)),
          ("neck", 672, (8, 24)), ("D4", 336, (16, 48)), ("D3", 168, (32, 96)), ("D2", 84, (64, 192)), ("D1", 42, (128, 384)),
          ("head", 4, (128, 384))]
    xs = {"stem": 7.2, "E1": 12.6, "E2": 17.6, "E3": 22.6, "E4": 27.6, "neck": 33.4, "D4": 39.4, "D3": 44.6, "D2": 49.8,
          "D1": 55.0, "head": 60.2}
    fc = {"stem": "#d8d8d8", "neck": DBLUE, "head": GREEN}
    ends = {}
    for k, c, hw in st:
        lab = "4" if k == "head" else f"{c}\n{hw[0]}×{hw[1]}"
        ends[k] = slab(ax, xs[k], yc, c, hw, lab, fc.get(k, BLUE if k[0] == "E" else "#d4ece9"))
    order = [s_[0] for s_ in st]
    arrow(ax, (5.1, yc), (xs["stem"] - 0.1, yc), lw=0.5)
    for a_, b_ in zip(order[:-1], order[1:]):
        arrow(ax, ends[a_], (xs[b_] - 0.05, yc), lw=0.5)
    arrow(ax, ends["head"], (62.3, yc), lw=0.5)
    field(ax, 62.4, yc - 2.2, 4.4, 4.4, seed=2, label="$p$: 4 ch")
    by = 55.6
    band(ax, 6.5, 10.1, by, "stem", C_STEM, fs=4.4)
    band(ax, 11.6, 31.6, by, "encoder (backbone)", C_ENC)
    band(ax, 32.4, 37.0, by, "neck", C_NECK, fs=4.4)
    band(ax, 38.6, 58.8, by, "decoder", C_DEC)
    band(ax, 59.3, 62.3, by, "head", C_HEAD, fs=3.9)
    dy = 44.0
    desc(ax, 8.3, dy, "3×3\nconv", fs=4.1)
    desc(ax, 21.6, dy, "per stage: 2 ConvNeXt blocks,\nthen ↓2 (LN + conv 2×2/2)", fs=4.1)
    desc(ax, 34.7, dy, "1 block;\nlatent", fs=4.1)
    desc(ax, 48.7, dy, "per stage: ↑2 (convT 2×2/2),\nconcat skip, 2 ConvNeXt blocks", fs=4.1)
    desc(ax, 60.6, dy, "3×3\nconv", fs=4.1)

    # pooled neck insertion (inset)
    band(ax, 68.4, 99.5, 58.6, "Neck insertion: pooled register", C_INS, h=1.7, fs=5.1)
    band(ax, 68.4, 74.4, by, "pool", "#9c86c4", fs=4.4); band(ax, 75.1, 81.1, by, "encode", "#9c86c4", fs=4.4)
    band(ax, 81.8, 93.2, by, "registers 3×10", C_QPU, fs=4.4); band(ax, 93.9, 99.5, by, "write", "#9c86c4", fs=4.4)
    pool(ax, 68.8, yc, "672×8×24\n→ $\\mathbb{R}^{672}$")
    mlp(ax, 75.1, yc - 3.6, 6.0, 6.2, [672, 80], fc=C_GATE2)
    circuit(ax, 81.8, yc - 3.2, 11.4, 6.4, fs=3.9, nw=3)
    mlp(ax, 93.9, yc - 3.6, 5.6, 6.2, [30, 672], fc=C_GATE2)
    for a_, b_ in ((73.8, 75.0), (81.1, 81.8), (93.2, 93.9)):
        arrow(ax, (a_, yc + 0.4), (b_, yc + 0.4), lw=0.45)
    desc(ax, 84.0, dy, "$h$ = spatial mean of the neck latent; $\\langle Z\\rangle\\in\\mathbb{R}^{30}$;\n"
         "write ×α (zero-init), broadcast-add: DC mode only (Prop. 1)", fs=4.1)
    ax.plot([ends["neck"][0] - 0.6, ends["neck"][0] - 0.6, 67.7], [yc + 1.8, 54.8, 54.8], color=C_INS, lw=0.5, ls=(0, (2, 1.5)))
    arrow(ax, (67.7, 54.8), (69.4, yc + 1.7), color=C_INS, lw=0.5, ls=(0, (2, 1.5)))

    # ======================= row B: frequency-aware gate (flows right to left, under p)
    gy = 28.8
    band(ax, 72.0, 99.5, 39.0, "Frequency-aware gate (classical)", C_GATE, h=1.7, fs=5.1)
    desc(ax, 85.7, 37.6, "decides, per field and octave, how much of\nthe backbone's change over persistence\n"
         "to keep; the best gate is never worse than\npersistence or the backbone (Lean)", fs=4.2)
    bb = 36.0
    band(ax, 57.6, 70.6, bb, "inputs", C_GATE2, fs=4.5); band(ax, 39.8, 56.2, bb, "octave split", C_GATE2, fs=4.5)
    band(ax, 21.4, 38.4, bb, "gate MLP", C_GATE2, fs=4.5); band(ax, 0.8, 19.8, bb, "mix", C_GATE2, fs=4.5)
    db = 35.4
    desc(ax, 64.1, db, "backbone and persistence", fs=4.1); desc(ax, 48.0, db, "FFT shells of $p-x_t$", fs=4.1)
    desc(ax, 29.9, db, "log band energies → α", fs=4.1); desc(ax, 10.3, db, "keep $\\alpha_b$ of each octave", fs=4.1)
    field(ax, 58.0, gy + 0.6, 5.6, 3.4, seed=2, label="$p$"); field(ax, 64.6, gy + 0.6, 5.6, 3.4, seed=1, label="$x_t$")
    octaves(ax, 40.4, gy - 0.6, 5.4)
    ax.text(46.4, gy + 2.2, "$P_b(p-x_t)$,\n$b$ = 0..4 (centre\n= coarsest), each\n4 × 128×384", fontsize=4.0, ha="left", va="center", linespacing=1.12)
    mlp(ax, 21.6, gy - 0.8, 8.2, 6.2, [72, 64, 20], fc=C_GATE2)
    ax.text(30.4, gy + 2.2, "in: band\nenergies\n(2×20) + 32\nout: α 4×5", fontsize=4.1, va="center", linespacing=1.12)
    mixer(ax, 3.0, gy + 2.2)
    ax.text(4.9, gy + 2.2, "$\\hat y = x_t + \\Sigma_b\\,\\alpha_b P_b(p-x_t)$\n$\\alpha=\\alpha_{\\rm fit}+1.5\\tanh(\\cdot)$",
            fontsize=4.2, va="center", linespacing=1.3)
    for a_, b_ in ((57.5, 56.3), (39.7, 38.5), (21.3, 19.9)):
        arrow(ax, (a_, gy + 2.2), (b_, gy + 2.2), lw=0.5)
    arrow(ax, (64.6, yc - 4.8), (64.6, bb + 1.45), lw=0.5)

    # ======================= row C: routed co-processor (QUOTA)
    band(ax, 12.6, 99.5, 24.9, "Routed co-processor (QUOTA): per input, 30 of 512 residual modes to qubits, the rest to a linear head",
         C_QPU, h=1.7, fs=5.1)
    arrow(ax, (5.0, gy - 0.6), (5.0, 22.95), lw=0.5)
    ax.text(5.7, 26.0, "$\\hat y$", fontsize=5.0, va="center")
    l1, l2 = 21.6, 10.4                                                        # band rows of the qubit / classical lanes
    band(ax, 0.8, 11.6, l1, "POD basis", C_OUT, fs=4.5)
    desc(ax, 6.2, l1 - 0.4, "residual in band-wise\nPOD modes $\\varphi_m$", fs=4.1)
    modes(ax, 1.0, 9.2, 10.4, 7.4)
    ax.text(6.2, 8.8, "4 fields × 128 modes,\nfloor per octave;\n$f_m\\in\\mathbb{R}^4$: proj. of\n$p{-}x_t$, $p$, $x_t$, $\\Delta x$",
            fontsize=4.0, ha="center", va="top", linespacing=1.12)
    band(ax, 12.8, 22.6, l1, "router", C_QPU2, fs=4.5)
    desc(ax, 17.7, l1 - 0.4, "which modes go\nto the qubits", fs=4.1)
    mlp(ax, 13.0, 11.6, 9.4, 6.0, [4, 32, 2], fc=C_QPU2)
    topk(ax, 13.4, 5.4, 8.8, 4.4)
    ax.text(17.7, 4.9, "score per mode;\ntop-30, straight-\nthrough", fontsize=4.0, ha="center", va="top", linespacing=1.12)
    band(ax, 23.8, 34.6, l1, "config net", C_QPU2, fs=4.5)
    desc(ax, 29.2, l1 - 0.4, "atom positions and\ntethers per input", fs=4.1)
    mlp(ax, 23.9, 13.0, 5.2, 4.4, [8, 32, 3], fc=C_QPU2)
    atoms(ax, 29.4, 12.8, 5.2, 4.8)
    band(ax, 35.8, 61.0, l1, "Rydberg register (qubits)", C_QPU, fs=4.5)
    desc(ax, 48.4, l1 - 0.4, "3 groups × 10 atoms; $\\delta=b+\\pi\\tanh(Wf)$, $W$: 4→4\n"
         "matched cores (MLP, GNN, mean-field) swap in here", fs=4.1)
    circuit(ax, 36.0, 12.8, 17.6, 5.2, fs=4.0)
    ax.text(54.2, 15.4, "readout\n$\\langle X,Y,Z\\rangle$\n30 × 3", fontsize=4.0, va="center", linespacing=1.12)
    band(ax, 62.2, 73.6, l1, "write-back", C_QPU2, fs=4.5)
    desc(ax, 67.9, l1 - 0.4, "correction for\neach routed mode", fs=4.1)
    mlp(ax, 62.4, 13.6, 5.0, 4.2, [4, 1], fc=C_QPU2); mlp(ax, 67.8, 13.6, 5.0, 4.2, [3, 1], fc=C_QPU2)
    ax.text(67.9, 13.3, "$\\Delta c_m=s_m(W_q f_m+A\\,o_m)$", fontsize=4.0, ha="center", va="top")
    band(ax, 23.8, 73.6, l2, "classical head (linear)", C_CLS, fs=4.5)
    desc(ax, 48.7, l2 - 0.4, "every other mode (482): $\\Delta c_m = s_m W_{\\rm cls} f_m$, one Linear 4→1 shared across modes",
         fs=4.1)
    mlp(ax, 44.4, 2.6, 8.6, 4.6, [4, 1], fc="#5fa98a")
    band(ax, 76.0, 99.5, l1, "output", C_OUT, fs=4.5)
    desc(ax, 87.8, l1 - 0.4, "backbone + gate + mode corrections", fs=4.1)
    field(ax, 79.0, 9.4, 17.6, 6.6, seed=4, label="$\\hat y+\\Sigma_m\\Delta c_m\\varphi_m$,  4 × 128×384")
    for a_, b_ in (((11.7, 13.0), (12.9, 13.0)), ((22.7, 16.3), (23.8, 16.3)), ((34.7, 16.3), (35.9, 16.3)),
                   ((60.3, 16.3), (62.3, 16.3)), ((73.5, 16.3), (78.9, 13.4))):
        arrow(ax, a_, b_, lw=0.5)
    arrow(ax, (22.3, 7.2), (44.3, 5.0), lw=0.5, rad=0.05); arrow(ax, (53.1, 5.0), (78.9, 11.0), lw=0.5, rad=0.05)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    print(f"[fig_arch] wrote {OUT}")


if __name__ == "__main__":
    main()
