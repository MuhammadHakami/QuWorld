"""Appendix figure (fig_arch_backbones.pdf): the other audit backbones at layer level, with the register's insertion
point. Sizes from paper/alloc/arch_dump.py (runs/alloc/arch_dump.txt) and the released configs:
- FNO, TRL2D arena (.venv-well trace): 19,011,716 params; lifting MLP 16->256->128 (1x1), 4 Fourier layers (128 ch,
  16x16 modes, 1x1 skip, GELU), projection MLP 128->256->4; the register sits before the projection (pooled).
- CNextU-net on rayleigh_benard (512x128), shear_flow (256x512): 18,575,344 params, neck 672 x 32x8 / 16x32;
  MHD_64 (3-D, 28 -> 7 channels, 64^3): 22,315,405 params, neck 672 x 4^3.
- 1HP heat-plume U-Net (1HP_NN): 7,756,609 params; conv 5x5 blocks 32/64/128/256, max-pool, 256 x 32x2 bottleneck
  (insertion), transposed conv + concat decoder, conv 1x1 -> 1 x 256x16.
- Sophelio starter UNetLite: 3,129,121 params; fc 21->512 (insertion, vector latent), fc 512->4096 -> 64 x 8x8,
  convT -> 128 x 16x16 (+ skip fc 21->32768), convT -> 64 x 32x32, convT -> 32 x 66x66, conv 2x2 -> psi 65x65.
- TokaMind (tokamind-base-v2.yaml, checkpoint): per-signal token projections (d 192), CLS + position / id / modality /
  role embeddings, 4 encoder layers (6 heads, FFN 768, GELU), modality-head MLPs 192->192->192, output adapters;
  the register wraps every modality head on h_CLS (vector latent).
- TORAX-Bench 1-D spectral net (paper/torax_bench/train_torax.py): lift 1x1 19->24, 2 spectral blocks (8 modes),
  insertion (pooled over 27 radial points), 2 spectral blocks, projection 24->64->4, residual (delta) output.
- Fusion MLP trunk (paper/sophelio/sota.py Hybrid, entry config): 17,843,013 params; fc 427->2048, 2 residual blocks,
  LN, PCA-64 psi head (+ skip 427->64) and 5 scalar heads; the register variant writes into PCA band [0, 32).

    .venv/bin/python paper/alloc/fig_arch_backbones.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fig_arch import BLUE, DBLUE, GREEN, GREY, NAVY, ORANGE, PLUM, PURPLE, RUST, TEAL, arrow, box, mlp, name, slab  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[2] / "paper/manuscript/iclr2027/fig_arch_backbones.pdf"
FOUR = "#cfe3d9"


def ins(ax, x, y, text="register\n(pooled)"):
    """Insertion point marker."""
    box(ax, x, y, 7.4, 3.6, text, ORANGE, fs=4.3)


def chain(ax, items, y, h=4.6):
    """items: list of (x, w, text, fc); draws boxes and arrows between consecutive ones."""
    for i, (x, w, t, fc) in enumerate(items):
        if fc is not None:
            box(ax, x, y, w, h, t, fc, fs=4.6)
        if i:
            px, pw = items[i - 1][0], items[i - 1][1]
            arrow(ax, (px + pw + 0.1, y + h / 2), (x - 0.1, y + h / 2), lw=0.5)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 87); ax.axis("off")

    # 1. FNO (TRL2D arena)
    y = 80.0
    name(ax, 0.5, y + 5.6, "FNO, TRL2D arena (The Well; 19.0 M parameters) -- spectral operator", NAVY, fs=5.6, ha="left")
    chain(ax, [(0.5, 9.0, "$x_{t-3..t}$\n16$\\times$128$\\times$384", GREY), (11.0, 14.0, "", None),
               (26.5, 22.4, "Fourier layer $\\times$4 (128 ch): FFT,\nkeep 16$\\times$16 modes, $R_k$ 128$\\times$128,\n"
                "iFFT, $+$ 1$\\times$1 skip, GELU", FOUR),
               (50.4, 14.0, "128$\\times$128$\\times$384\n+ register, pooled\nover 49,152 positions", ORANGE), (65.9, 14.0, "", None),
               (81.4, 10.0, "$p$\n4$\\times$128$\\times$384", GREY)], y)
    box(ax, 11.0, y, 14.0, 4.6, "", GREY); mlp(ax, 11.1, y - 0.1, 8.4, 4.6, [16, 256, 128]); ax.text(19.6, y + 2.3, "lifting\n(1$\\times$1)", fontsize=4.4, va="center")
    box(ax, 65.9, y, 14.0, 4.6, "", GREY); mlp(ax, 66.0, y - 0.1, 8.4, 4.6, [128, 256, 4]); ax.text(74.5, y + 2.3, "projection\n(1$\\times$1)", fontsize=4.4, va="center")

    # 2. CNextU-net on the other Well rows
    y = 69.6
    name(ax, 0.5, y + 6.2, "CNextU-net on rayleigh_benard, shear_flow, MHD_64 (released; architecture as Fig. 1)", NAVY, fs=5.6, ha="left")
    box(ax, 0.5, y - 0.4, 99.0, 5.6,
        "rayleigh_benard: 16$\\times$512$\\times$128 $\\to$ stem 42 $\\to$ E1-E4 84/168/336/672 $\\to$ neck 672$\\times$32$\\times$8 (register, pooled) "
        "$\\to$ D4-D1 $\\to$ 4$\\times$512$\\times$128 (18.6 M)\n"
        "shear_flow: 16$\\times$256$\\times$512 $\\to$ neck 672$\\times$16$\\times$32 $\\to$ 4$\\times$256$\\times$512 (18.6 M).  "
        "MHD_64, 3-D convs: 28$\\times$64$^3$ $\\to$ neck 672$\\times$4$^3$ $\\to$ 7$\\times$64$^3$ (22.3 M)",
        BLUE, fs=4.6, ha="left")

    # 3. 1HP heat-plume U-Net
    y = 55.6
    name(ax, 0.5, y + 9.4, "U-Net, 1HP heat plume (1HP_NN; 7.8 M parameters) -- convolutional encoder-decoder", NAVY, fs=5.6, ha="left")
    sts = [(4, (256, 16), GREY, "4"), (32, (256, 16), BLUE, "32"), (64, (128, 8), BLUE, "64"), (128, (64, 4), BLUE, "128"),
           (256, (32, 2), DBLUE, "256"), (128, (64, 4), BLUE, "128"), (64, (128, 8), BLUE, "64"), (32, (256, 16), BLUE, "32"),
           (1, (256, 16), GREEN, "1")]
    xs = [0.8 + 6.6 * i for i in range(len(sts))]; ends = []
    for (c, hw, fc, lb), x in zip(sts, xs):
        ends.append(slab(ax, x, y + 3.2, max(c, 8), hw, f"{lb}\n{hw[0]}$\\times${hw[1]}", fc))
    for i in range(len(sts) - 1):
        arrow(ax, ends[i], (xs[i + 1], y + 3.2), lw=0.5)
    for a_, b_ in ((1, 7), (2, 6), (3, 5)):
        arrow(ax, (ends[a_][0] - 0.3, y + 7.6), (xs[b_] + 0.3, y + 7.6), color="0.6", lw=0.4, ls=(0, (2, 1.5)), rad=-0.08)
    ins(ax, xs[4] - 9.4, y - 7.0, "register on\n256$\\times$32$\\times$2\n(pooled)")
    arrow(ax, (xs[4] - 0.2, y + 1.9), (xs[4] - 2.1, y - 3.5), color=PLUM, lw=0.45, ls=(0, (2, 1.5)))
    ax.text(62.0, y + 3.2, "encoder: [conv 5$\\times$5, BN, ReLU]$\\times$2 per level, max-pool 2;\n"
            "decoder: convT 2$\\times$2/2, concat skip (dashed), conv blocks;\nhead: conv 1$\\times$1 $\\to$ temperature 1$\\times$256$\\times$16",
            fontsize=4.6, va="center", linespacing=1.3)

    # 4. Sophelio starter UNetLite
    y = 40.0
    name(ax, 0.5, y + 6.0, "UNetLite, Sophelio starter (3.1 M parameters) -- MLP into a transposed-convolution decoder", NAVY, fs=5.6, ha="left")
    box(ax, 0.5, y, 6.6, 4.6, "$x\\in\\mathbb{R}^{21}$\ncoils, $I_p$", GREY, fs=4.6)
    box(ax, 8.3, y, 9.0, 4.6, "", GREY); mlp(ax, 8.4, y - 0.1, 5.0, 4.6, [21, 512]); ax.text(13.6, y + 2.3, "fc1\n+ReLU", fontsize=4.4, va="center")
    ins(ax, 18.4, y + 0.5, "register on\n$h\\in\\mathbb{R}^{512}$\n(vector)")
    box(ax, 26.8, y, 11.0, 4.6, "", GREY); mlp(ax, 26.9, y - 0.1, 6.4, 4.6, [512, 4096]); ax.text(33.5, y + 2.3, "fc2 $\\to$\n64$\\times$8$\\times$8", fontsize=4.4, va="center")
    sl = [(128, (16, 16), "128\n16$\\times$16"), (64, (32, 32), "64\n32$\\times$32"), (32, (66, 66), "32\n66$\\times$66"), (8, (65, 65), "1\n65$\\times$65")]
    xs = [40.2, 46.4, 52.6, 58.8]; e = []
    for (c, hw, lb), x in zip(sl, xs):
        e.append(slab(ax, x, y + 2.6, c, hw, lb, GREEN if c == 8 else BLUE))
    for (a_, b_) in ((37.9, 40.1), (e[0][0] + 0.1, 46.3), (e[1][0] + 0.1, 52.5), (e[2][0] + 0.1, 58.7)):
        arrow(ax, (a_, y + 2.6), (b_, y + 2.6), lw=0.5)
    arrow(ax, (7.2, y + 2.3), (8.2, y + 2.3), lw=0.5); arrow(ax, (17.4, y + 2.3), (18.3, y + 2.3), lw=0.5); arrow(ax, (25.9, y + 2.3), (26.7, y + 2.3), lw=0.5)
    ax.text(64.0, y + 2.3, "convT 4$\\times$4/2 + BN + ReLU (three), conv 2$\\times$2 $\\to\\psi$;\nskip: fc 21$\\to$32,768 reshaped to 128$\\times$16$\\times$16, "
            "added\nafter the first convT", fontsize=4.6, va="center", linespacing=1.3)

    # 5. TokaMind
    y = 27.6
    name(ax, 0.5, y + 5.6, "TokaMind, TokaMark (tokamind-base-v2) -- transformer encoder", NAVY, fs=5.6, ha="left")
    chain(ax, [(0.5, 10.0, "signals and\nactuators\n(time windows)", GREY), (11.8, 13.2, "per-signal Linear\ntoken projections\n$d=192$", GREY),
               (26.8, 14.2, "[CLS] + tokens (51 pos.)\n+ position, id,\nmodality, role emb.", GREY),
               (42.6, 15.6, "encoder layer $\\times$4: MHA\n(6 heads) + FFN 192$\\to$768\n$\\to$192, GELU, LN", PURPLE),
               (59.8, 9.6, "$h_{\\rm CLS}\\in\\mathbb{R}^{192}$\n+ register\n(vector)", ORANGE),
               (71.0, 14.4, "modality-head MLPs\n192$\\to$192$\\to$192: time\nseries, profile, video", GREY),
               (87.0, 12.5, "output adapters:\nLinear per\noutput signal", GREY)], y)

    # 6. TORAX-Bench
    y = 15.4
    name(ax, 0.5, y + 5.6, "1-D spectral network, TORAX-Bench (27 radial points) -- 1-D Fourier operator", NAVY, fs=5.6, ha="left")
    chain(ax, [(0.5, 13.0, "$T_e, T_i, n_e, q$: 4$\\times$27\n+ 15 control channels", GREY), (15.3, 9.6, "lift 1$\\times$1\n19$\\to$24", GREY),
               (26.7, 14.4, "spectral block $\\times$2:\nrFFT, 8 modes, $R_k$ 24$\\times$24\n+ 1$\\times$1 skip, GELU", FOUR),
               (43.3, 9.6, "24$\\times$27\n+ register\n(pooled)", ORANGE), (55.1, 14.4, "spectral block $\\times$2\n(as before)", FOUR),
               (71.7, 12.4, "projection 1$\\times$1\n24$\\to$64$\\to$4", GREY), (86.3, 13.2, "$x_{t+1}=x_t+\\Delta$\n4$\\times$27", GREY)], y)

    # 7. Fusion MLP trunk
    y = 1.0
    name(ax, 0.5, y + 9.8, "Residual MLP trunk, Fusion Equilibrium (17.8 M parameters; app. \"Scores relative to published numbers\")",
         NAVY, fs=5.6, ha="left")
    box(ax, 0.5, y + 2.0, 8.2, 5.6, "$x\\in\\mathbb{R}^{427}$\ncoils, $I_p$, lags,\nThomson", GREY, fs=4.6)
    box(ax, 10.0, y + 1.4, 11.0, 6.8, "", GREY); mlp(ax, 10.1, y + 1.0, 7.0, 6.6, [427, 2048]); ax.text(17.3, y + 4.8, "fc\n+GELU", fontsize=4.4, va="center")
    box(ax, 22.4, y + 1.4, 22.4, 6.8, "", PURPLE); mlp(ax, 22.5, y + 1.0, 10.8, 6.6, [2048, 2048, 2048])
    ax.text(33.5, y + 4.8, "residual block $\\times$2:\nLN $\\to$ fc $\\to$ GELU\n$\\to$ dropout $\\to$ fc, $+$", fontsize=4.4, va="center")
    box(ax, 46.2, y + 4.8, 15.4, 3.4, "$\\psi$ head: fc 2048$\\to$64\n(+ skip 427$\\to$64)", GREY, fs=4.5)
    box(ax, 46.2, y + 0.2, 15.4, 3.6, "scalar head: fc 2048$\\to$5\n($q_{95}$, $\\beta_N$, $l_i$, axis)", GREY, fs=4.5)
    box(ax, 62.8, y + 4.8, 13.4, 3.4, "PCA-64 coefficients\n$\\psi=\\bar\\psi+\\Sigma_k c_k u_k$", GREY, fs=4.5)
    box(ax, 62.8, y + 0.2, 13.4, 3.6, "register variant writes\nPCA band [0, 32)", ORANGE, fs=4.4)
    box(ax, 77.4, y + 1.4, 22.1, 6.8, "competition track only: shape heads,\nsame MLP family, 4096$\\times$3 $\\to$ 7 shape\n"
        "scalars + 64 LCFS ray radii; flux\nedits toward them (not in main tables)", "white", fs=4.4, ls=(0, (2, 1.5)))
    for a_, b_ in (((8.8, y + 4.8), (9.9, y + 4.8)), ((21.1, y + 4.8), (22.3, y + 4.8)), ((44.9, y + 6.5), (46.1, y + 6.5)),
                   ((44.9, y + 2.0), (46.1, y + 2.0)), ((61.7, y + 6.5), (62.7, y + 6.5))):
        arrow(ax, a_, b_, lw=0.5)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"[fig_arch_backbones] wrote {OUT}")


if __name__ == "__main__":
    main()
