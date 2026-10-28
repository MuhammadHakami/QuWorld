"""Diagrams, tables and plots for notebooks/19_models_and_interpretation.ipynb (built by paper/build_interpret_notebook.py)."""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "phase_b"))
import analyze  # noqa: E402

NAMES = {"turbulent_radiative_layer_2D": "TRL2D", "rayleigh_benard": "rayleigh_benard", "shear_flow": "shear_flow",
         "MHD_64": "MHD_64", "realpde_unet": "RealPDE (U-Net)", "realpde_fno": "RealPDE (FNO)", "pelzer": "Pelzer heat plume"}
C_BACK, C_NECK, C_HEAD, C_Q, C_IO = "#dbe9f6", "#fde7c8", "#e3f1dc", "#f7d6e0", "#eeeeee"


# ---------------------------------------------------------------------------------------------------------- diagrams

def _box(ax, x, y, w, h, text, color, fs=7.5, bold=False, ec="0.25"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
            wrap=True)


def _arrow(ax, x0, y0, x1, y1, style="-|>", color="0.2", lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10, color=color, lw=lw,
                                 linestyle=ls))


def architecture(figsize=(15, 7.2)):
    """Backbone / neck / head of a QUOTA hybrid (CNextU-net shown; other trunks map onto the same three slots), with the
    two quantum insertion points: the pooled register in the neck and the routed co-processor after the head."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 15); ax.set_ylim(0.1, 7.3); ax.axis("off")
    # lane labels
    for y, t in ((6.85, "CLASSICAL TRUNK\n(pretrained; the 'digital' part)"), (2.95, "ROUTED QUANTUM CO-PROCESSOR\n(the 'analog' part the trunk gets wrong)")):
        ax.text(0.1, y, t, fontsize=8.5, weight="bold", color="0.3", va="center")
    # input
    _box(ax, 0.1, 4.7, 1.5, 1.2, "input history\n$x_{t-3..t}$\n(4 frames x C fields)", C_IO)
    # backbone
    ax.add_patch(FancyBboxPatch((1.9, 4.2), 3.6, 2.2, boxstyle="round,pad=0.02,rounding_size=0.1", fc="none", ec="#4a7ab5", lw=1.6, ls="--"))
    ax.text(3.7, 6.55, "BACKBONE (encoder)", fontsize=9, weight="bold", color="#4a7ab5", ha="center")
    _box(ax, 2.0, 4.8, 0.8, 1.0, "in_proj\nconv", C_BACK, fs=7)
    for i, (w, lab) in enumerate([(0.55, "42"), (0.55, "84"), (0.55, "168"), (0.55, "336")]):
        _box(ax, 2.95 + i * 0.63, 5.05 - i * 0.12, w, 0.7 + 0.0 * i, f"stage\n{lab}ch\n↓2", C_BACK, fs=6)
    ax.text(3.7, 4.35, "ConvNeXt blocks, 4 stages, stride 2 each", fontsize=6.5, ha="center", color="0.3")
    # neck
    ax.add_patch(FancyBboxPatch((5.8, 3.9), 3.1, 2.8, boxstyle="round,pad=0.02,rounding_size=0.1", fc="none", ec="#c77c1e", lw=1.6, ls="--"))
    ax.text(7.35, 6.85, "NECK (bottleneck, 672 ch at 1/16 res.)", fontsize=9, weight="bold", color="#c77c1e", ha="center")
    _box(ax, 5.95, 5.35, 1.3, 0.9, "neck stage\nConvNeXt\n672 ch", C_NECK, fs=7)
    _box(ax, 7.45, 5.35, 1.3, 0.9, "mean-pool\n$h=\\frac{1}{N}\\sum_s z_s$", C_Q, fs=7)
    _box(ax, 7.45, 4.1, 1.3, 1.0, "3 x 10-atom\nRydberg register\n(or matched MLP)", C_Q, fs=7, bold=True)
    _box(ax, 5.95, 4.1, 1.3, 1.0, "gated broadcast\n$z+\\alpha\\odot U c(h)$\n$\\alpha=0$ at init", C_Q, fs=6.5)
    _arrow(ax, 7.25, 5.8, 7.45, 5.8); _arrow(ax, 8.1, 5.35, 8.1, 5.1); _arrow(ax, 7.45, 4.6, 7.25, 4.6)
    ax.text(7.35, 3.98, "audit insertion: reads/writes only the DC mode (Lean pool_blind)", fontsize=6, ha="center", color="#a0522d")
    # head
    ax.add_patch(FancyBboxPatch((9.2, 4.2), 3.3, 2.2, boxstyle="round,pad=0.02,rounding_size=0.1", fc="none", ec="#4f8f3a", lw=1.6, ls="--"))
    ax.text(10.85, 6.55, "HEAD (decoder + output)", fontsize=9, weight="bold", color="#4f8f3a", ha="center")
    for i, lab in enumerate(["336", "168", "84", "42"]):
        _box(ax, 9.3 + i * 0.63, 4.69 + i * 0.12, 0.55, 0.7, f"stage\n{lab}ch\n↑2", C_HEAD, fs=6)
    _box(ax, 11.85, 4.8, 0.55, 1.0, "out\nproj\n1x1", C_HEAD, fs=6.5)
    ax.text(10.85, 4.35, "U-Net skips from the encoder", fontsize=6.5, ha="center", color="0.3")
    _box(ax, 12.9, 4.8, 1.9, 1.0, "trunk prediction $p$\n(next frame)", C_IO)
    for x0, x1 in ((1.6, 2.0), (5.5, 5.95), (8.9, 9.3), (12.45, 12.9)):
        _arrow(ax, x0, 5.3, x1, 5.3)
    # skip arcs
    ax.add_patch(FancyArrowPatch((3.9, 6.3), (10.2, 6.3), connectionstyle="arc3,rad=-0.12", arrowstyle="-|>",
                                 mutation_scale=8, color="0.55", lw=0.8, ls=":"))
    # co-processor lane
    y = 1.1
    _box(ax, 0.1, y, 1.9, 1.35, "residual basis\northonormal Haar tiles\nof $p-x_t,\\ p,\\ x_t,\\ \\Delta x$", C_IO, fs=6.8)
    _box(ax, 2.35, y, 2.2, 1.35, "ALLOCATION (per input)\npooled | static | proxy\nlearned | gain | random\n-> K tiles (K=30)", C_Q, fs=6.8, bold=False)
    _box(ax, 4.9, y, 2.4, 1.35, "GEOMETRY from allocation\natom $i$ at tile $i$:\n$x,y$ <- position, $z$ <- scale\ngroups of 10 (blockade-separated)", C_Q, fs=6.5)
    _box(ax, 7.65, y, 2.4, 1.35, "REGISTER (x K/10)\n$\\delta_{\\ell i}=b_{\\ell i}+\\pi\\tanh(w_\\ell^\\top f_i)$\n$[R_z(\\delta)R_x(\\Omega)]\\,e^{-i\\tau\\sum V_{ij}n_in_j}$, L=4", C_Q, fs=6.4, bold=False)
    _box(ax, 10.4, y, 2.1, 1.35, "WRITE-BACK HEAD\n$o_i=\\langle Z,X,Y\\rangle_i$\n$c_i=s_i(W_{\\kappa}f_i+A\\,o_i)$\n$A=0$ at init", C_HEAD, fs=6.5)
    _box(ax, 12.85, y, 1.95, 1.35, "final prediction\n$y=p+\\mathcal{H}^{-1}(c)$\n(inverse Haar on\nthe K tiles only)", C_IO, fs=6.8)
    for x0, x1 in ((2.0, 2.35), (4.55, 4.9), (7.3, 7.65), (10.05, 10.4), (12.5, 12.85)):
        _arrow(ax, x0, y + 0.67, x1, y + 0.67)
    _arrow(ax, 13.85, 4.8, 13.85, 2.45, color="#4f8f3a")
    _arrow(ax, 13.2, 4.8, 1.0, 2.45, color="0.5", ls="--")
    ax.text(7.0, 3.3, "trunk output $p$ (and inputs) feed the residual basis; the atoms work only on the K allocated tiles",
            fontsize=7, color="0.35", ha="center")
    ax.text(7.5, 0.35, "Matched classical control: identical boxes, register replaced by a parameter-matched MLP "
            "(DeepSets over each group of 10).   Lean: captured <= U (gain_le_ceiling); pooled gain <= DC energy (pooled_gain_le_dc).",
            fontsize=7, ha="center", color="0.3")
    fig.tight_layout()
    return fig


def register_closeup(pos, V, det, readout=None, title="one trained 10-atom register on a real test input"):
    """pos (10, 3) atom positions, V (10, 10) couplings, det (10,) layer-1 detunings, readout (10, 3) optional."""
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    vmax = np.percentile(V[np.triu_indices(10, 1)], 90)
    for i in range(10):
        for j in range(i + 1, 10):
            if V[i, j] > 0.02 * vmax:
                ax.plot(*zip(pos[i], pos[j]), color="C0", lw=0.3 + 2.5 * min(V[i, j] / vmax, 1), alpha=0.6)
    sc = ax.scatter(*pos.T, c=det, cmap="coolwarm", s=90, edgecolors="k", vmin=-np.pi, vmax=np.pi)
    ax.set_xlabel("x / blockade"); ax.set_ylabel("y / blockade"); ax.set_zlabel("scale plane")
    ax.set_title("atoms at allocated tiles\n(line width ~ $V_{ij}$, colour = detuning)", fontsize=8)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="detuning (rad)")
    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.imshow(np.log10(V + np.eye(10) * V.max() + 1e-9), cmap="viridis")
    ax2.set_title("interaction matrix $\\log_{10} V_{ij}$", fontsize=8); fig.colorbar(im, ax=ax2, shrink=0.8)
    ax3 = fig.add_subplot(1, 3, 3)
    if readout is not None:
        for k, (lab, col) in enumerate(zip("ZXY", ("C0", "C1", "C2"))):
            ax3.bar(np.arange(10) + (k - 1) * 0.27, readout[:, k], 0.27, color=col, label=f"<{lab}>")
        ax3.legend(fontsize=7); ax3.set_ylim(-1.05, 1.05); ax3.set_xlabel("atom")
    ax3.set_title("readout per atom", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- backbone/neck/head

def count(m) -> int:
    return sum(p.numel() for p in m.parameters())


def cnext_parts(model) -> dict:
    enc = sum(count(s) for s in model.encoder) + count(model.in_proj)
    dec = sum(count(s) for s in model.decoder) + count(model.out_proj)
    return {"backbone (in_proj + encoder)": enc, "neck (bottleneck stage)": count(model.neck), "head (decoder + out_proj)": dec}


BACKBONES = [
    # benchmark, backbone, neck (where the register goes), head, source of the released weights
    ("TRL2D (The Well)", "CNextU-net: in_proj + 4 ConvNeXt encoder stages (42->336 ch, /16)", "bottleneck stage 672 ch at 8x24; pooled register (audit) or none (frozen study)", "4 decoder stages with skips + 1x1 out_proj -> 4 fields; routed co-processor on the residual", "polymathic-ai UNetConvNext (released)"),
    ("rayleigh_benard", "CNextU-net (same family), 512x128", "bottleneck stage; pooled register", "decoder + out_proj; routed co-processor", "released"),
    ("shear_flow", "CNextU-net, 256x512", "bottleneck stage; pooled register", "decoder + out_proj; routed co-processor", "released (does not reproduce its published number)"),
    ("MHD_64", "3-D CNextU-net, 64^3, 7 fields", "bottleneck stage; pooled register", "decoder + out_proj; routed co-processor (3-D Haar)", "released (does not reproduce)"),
    ("RealPDEBench airfoil", "RealPDEBench U-Net / FNO (20->20 frames, 64x128)", "none (trunk frozen)", "low-mode Fourier corrector (register or MLP core) or routed Haar co-processor", "RealPDEBench released checkpoints"),
    ("Pelzer 1HP heat plume", "1HP_NN U-Net (depth 3, 32 feat.)", "deepest encoder (256 ch at 32x2); pooled register", "decoders + conv head -> temperature; routed co-processor", "trained by us with the 1HP_NN recipe"),
    ("Sophelio Ch1 (DIII-D)", "residual MLP over magnetics (14 lags) + Thomson features", "latent coupling: register on the 1-D latent, or routed coupling writing PCA components k>=8", "PCA-64 flux decoder + scalar heads (q95, betaN, li, R, Z)", "NAS-selected trunks, 7-member ensemble"),
    ("TokaMark 1-1 (MAST)", "TokaMind-Base-v2 transformer (CLS latent)", "register wraps every modality head input (vector latent, no pooling)", "TokaMind modality heads", "UKAEA/IBM TokaMind-Base-v2 (released)"),
    ("TORAX-Bench", "1-D spectral net: lift + 2 spectral blocks (27-cell rho grid)", "mid slot (Bottleneck); pooled register", "2 spectral blocks + projection; delta prediction", "trained by us"),
    ("TDC ADMET (22 endpoints)", "CheMeleon D-MPNN foundation model (+ MapLight / GIN features)", "register after mean aggregation (molecule embedding)", "FFN property head; VALID-selected stack with MapLight(+GNN) CatBoost", "CheMeleon released weights"),
]


def backbone_table() -> pd.DataFrame:
    return pd.DataFrame(BACKBONES, columns=["benchmark", "backbone", "neck (quantum insertion)", "head", "weights"])


# ------------------------------------------------------------------------------------------------- study results

def row_table(bench: str, K: int = 30) -> pd.DataFrame:
    R = analyze.load()
    rows = []
    for s in analyze.STRATS + ("oracle",):
        rec = {"strategy": s}
        for c in ("lin", "mlp", "ryd"):
            v = R[(bench, s, c, K)]
            if v:
                rec["U"] = analyze.mean([d["test_U"] for d in v])
                rec[f"{c} gain %"] = 100 * analyze.mean([d["test_rel_gain"] for d in v])
                rec[f"{c} captured"] = analyze.mean([d["test_captured"] for d in v])
        if len(rec) > 1:
            rec["efficiency ryd"] = rec.get("ryd captured", np.nan) / rec["U"] if rec.get("U") else np.nan
            rows.append(rec)
    return pd.DataFrame(rows).set_index("strategy")


def controls(bench: str) -> pd.DataFrame:
    R = analyze.load(); bs = analyze.best(bench); out = []
    for other in ("mlp", "lin", "gutz", "ryd2d", "mlpX4"):
        pq = analyze.paired(R, bench, bs, "ryd", other)
        if pq:
            out.append({"register vs": other, "strategy": bs, "rel. gain %": 100 * pq["gain"], "wins": f"{pq['wins']}/{pq['n']}", "p": pq["p"]})
    pq = analyze.paired(R, bench, bs, "hyb", "mlp2")
    if pq:
        out.append({"register vs": "hyb vs mlp2 (division of labour)", "strategy": bs, "rel. gain %": 100 * pq["gain"],
                    "wins": f"{pq['wins']}/{pq['n']}", "p": pq["p"]})
    return pd.DataFrame(out)


def ksweep(bench: str, ax=None):
    R = analyze.load()
    ax = ax or plt.subplots(figsize=(5, 3.4))[1]
    for s, col in (("pooled", "C7"), ("proxy", "C2")):
        for c, ls in (("ryd", "-"), ("mlp", ":")):
            ks = [K for K in (10, 30, 100, 300) if R[(bench, s, c, K)]]
            if ks:
                ax.plot(ks, [max(1e-5, analyze.mean([d["test_captured"] for d in R[(bench, s, c, K)]])) for K in ks], ls,
                        color=col, marker="o", ms=3, label=f"{s} {'register' if c == 'ryd' else 'MLP'}: captured")
        ks = [K for K in (10, 30, 100, 300) if R[(bench, s, "mlp", K)]]
        if ks:
            ax.plot(ks, [analyze.mean([d["test_U"] for d in R[(bench, s, "mlp", K)]]) for K in ks], "--", color=col, lw=0.8,
                    label=f"{s}: ceiling U")
    ax.set(xscale="log", yscale="log", xlabel="atoms K", ylabel="share of error energy", title=f"{NAMES[bench]}: qubits vs utilization")
    ax.legend(fontsize=6)
    return ax


def allocation_plot(alloc: dict, bench: str):
    shape = alloc["shape"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    mk = {"oracle": ("*", "C3"), "static": ("s", "C0"), "proxy": ("^", "C2"), "random": ("x", "0.5")}
    for rule, (m, c) in mk.items():
        p = alloc[rule]["pos"]
        ax.scatter(p[:, 1], p[:, 0], marker=m, c=c, s=30 + 12 * alloc[rule]["level"], label=rule, alpha=0.8)
    H, W = (shape[-2], shape[-1]) if len(shape) >= 2 else (1, shape[0])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal" if W / max(H, 1) < 8 else "auto")
    ax.set_title(f"{NAMES[bench]}: where 30 atoms go on one test window (marker size ~ scale)", fontsize=8)
    ax.legend(fontsize=6, ncol=4, loc="upper right")
    ax2 = axes[1]
    L = len(alloc["oracle"]["levels_all"])
    for k, (rule, (m, c)) in enumerate(mk.items()):
        h = alloc[rule]["levels_all"]; ax2.bar(np.arange(L) + (k - 1.5) * 0.2, h / h.sum(), 0.2, color=c, label=rule)
    ax2.set_xticks(range(L)); ax2.set_xticklabels([f"{2 ** (j + 1)}px" for j in range(L)], fontsize=6)
    ax2.set_title("scale of the allocated tiles (all test windows)", fontsize=8); ax2.legend(fontsize=6)
    fig.tight_layout()
    return fig


def scatter_corr(x, y, bench, ax=None):
    ax = ax or plt.subplots(figsize=(3.6, 3.4))[1]
    lim = np.percentile(np.abs(x), 99)
    ax.hexbin(np.clip(x, -lim, lim), np.clip(y, -lim, lim), gridsize=50, bins="log", cmap="Blues")
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
    r = np.corrcoef(x, y)[0, 1] if np.std(y) > 0 else float("nan")
    ax.set(xlabel="true residual / tile scale", ylabel="register correction / tile scale", title=f"{NAMES[bench]}: r = {r:.2f}")
    return ax, r


def sota_table() -> pd.DataFrame:
    """Current best published numbers (verified Sep 27 2026: PDE-FM arXiv:2511.21861 T3, Flowers arXiv:2603.04430 T1,
    EddyFormer NeurIPS 2025 T3); The Well's original baselines in parentheses as historical anchors."""
    rows = [
        ("TRL2D one-step VRMSE", "Flowers 0.1930 (4->1; The Well CNextU-net 0.1956, released re-eval 0.1934)", "0.1909 (gated 3-backbone ensemble, full test)", "-1.1% (ensemble, more compute)", "none: routed arms tie at 0.1922"),
        ("TRL2D rollout 13:30", "CNextU-net 1.01", "0.9528 (TD fine-tune + register)", "-6% (better)", "register-specific: MLP 1.032, trunk 1.030"),
        ("RealPDEBench foil rel. L2", "released U-Net 0.00948", "0.00931 (low-mode corrector + ensemble)", "-1.8% (better)", "register corrector 0.00939 = MLP"),
        ("rayleigh_benard VRMSE", "PDE-FM 0.0415 (PhysiX 0.1470; The Well TFNO 0.6566)", "0.0403 (gated 3-backbone ensemble, full test)", "-3% (parity: ensemble vs single model)", "routed register 0.0426 (strat.) vs MLP 0.0428"),
        ("Fusion Equilibrium Ch1 S", "0.9932", "0.9910 (single TEST scoring)", "-0.0022 (below)", "register-coupled ensemble 0.9905"),
        ("TokaMark 1-1 NRMSE", "TokaMind-Scratch 0.140", "0.1400 (registered ensemble; single Scratch run 0.1379)", "matched", "register 0.1501 vs MLP 0.1504"),
        ("TDC ADMET (21 scored)", "per-endpoint leaders", "classical stack #1 on 7, top-3 on 15-16", "partial", "register stack #1 on 6; register < MLP on 13/22"),
        ("MHD_64 VRMSE", "EddyFormer 0.1160 (Flowers 0.1165; The Well CNextU-net claim 0.1633)", "0.1709 (gated, divergence-free, full test)", "+47% (behind)", "tie"),
        ("shear_flow VRMSE", "PDE-FM 0.0345 (PhysiX 0.0700; The Well CNextU-net 0.8080)", "0.0207 (gate + routed co-processor, 30 qubits, full test)", "-40% (better; classical part alone 0.0236)", "register 0.0207 vs MLP 0.0213"),
        ("Pelzer 1HP MAE", "0.046 C (unverifiable, dropped)", "0.0291 C", "n/a", "tie"),
        ("TORAX-Bench VRMSE@8", "own benchmark", "0.0141 (MLP adapter)", "n/a", "register worst (0.0168)"),
    ]
    return pd.DataFrame(rows, columns=["benchmark", "published best", "our best", "vs SoTA", "register's role"])


def audit_rows() -> pd.DataFrame:
    d = json.loads((ROOT / "runs" / "judge" / "scorecard.json").read_text())
    out = []
    for r in d["rows"]:
        m = r["means"]; qm = r.get("q_vs_mlp") or {}
        out.append({"row": r["name"], "tier": r["tier"], "trunk": m.get("trunk"), "matched MLP": m.get("mlp"), "register": m.get("q"),
                    "register vs MLP %": qm.get("rel_pct"), "p": qm.get("p")})
    return pd.DataFrame(out)


def probe_frame() -> pd.DataFrame:
    R = defaultdict(list)
    for f in (ROOT / "runs" / "alloc" / "bias_probe").glob("*.json"):
        d = json.loads(f.read_text())
        R[(d["target"] if d["target"] != "analog" else f"analog w={d['omega']:g}", d["core"])].append(d["test_r2"])
    df = pd.Series({k: np.mean(v) for k, v in R.items()}).unstack()
    order = [c for c in ("ryd", "gutz", "mlp", "mlp+", "mlpX4", "rydlin", "mlpsin") if c in df.columns]
    return df[order].round(3)


def forest(ax=None):
    R = analyze.load()
    ax = ax or plt.subplots(figsize=(5, 3.2))[1]
    for i, b in enumerate(NAMES):
        bs = analyze.best(b); pq = analyze.paired(R, b, bs, "ryd", "mlp")
        if not pq:
            continue
        g = np.array([1 - d["test"] / m["test"] for d in R[(b, bs, "ryd", 30)] for m in R[(b, bs, "mlp", 30)] if d["seed"] == m["seed"]]) * 100
        ax.errorbar(g.mean(), i, xerr=g.std(ddof=1) if len(g) > 1 else 0, fmt="o", color="C3")
        ax.scatter(g, [i] * len(g), s=8, color="0.5")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(range(len(NAMES))); ax.set_yticklabels(list(NAMES.values()), fontsize=7)
    ax.set_xlabel("register vs matched MLP, test metric gain % (per seed; + = register better)")
    ax.set_title("the core does not matter at this scale", fontsize=8)
    return ax
