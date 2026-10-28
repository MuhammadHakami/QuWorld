"""Doc 35 §41z: Fusion error atlas - figures from saved predictions / labels / logged results (no training), written as PNGs and
one self-contained HTML page (figures embedded). Reference model for the anatomy = HJl9's TRAIN-only components (J1 x6 shape heads,
QH x6 + entry q/b) evaluated on all VALID + TEST frames (743 shots, out-of-sample), whose public scores are known.

    .venv/bin/python paper/strat/fusion/viz_report.py   -> runs/strat/fusion/report/{*.png, fusion_atlas.html}
"""
from __future__ import annotations

import base64
import copy
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import fx  # noqa: E402

REP = fx.OUT / "report"; REP.mkdir(exist_ok=True)
SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
ALL9 = SC + ["q95", "betaN"]
INK, MUTED, ACC, WARN, GOOD, GRID = "#1d2733", "#6b7785", "#2f6db5", "#c4472d", "#2e8b57", "#e3e7ec"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110, "savefig.bbox": "tight", "legend.frameon": False})
FIGS = []


def save(fig, name, caption):
    p = REP / f"{name}.png"; fig.savefig(p); plt.close(fig); FIGS.append((name, caption)); print(f"[atlas] {name}", flush=True)


def load():
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    import inputs_fix
    import submit_public as SP
    d = sota.Data(); vt = np.nonzero(d.split != 0)[0]; lab = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); assert np.array_equal(lab["rows"], vt)
    z = np.load(fx.OUT / "ood_pred.npz")
    Y = np.concatenate([lab["y"][:, :7], d.S[vt][:, [sota.LABELS.index("efit_q95"), sota.LABELS.index("efit_beta_n")]]], 1).astype(np.float64)
    P = np.concatenate([z["j_all"][vt], 0.5 * (z["q_all"][vt] + z["e_vt"])], 1).astype(np.float64)
    sh = d.shot[vt]; st = np.r_[0, np.nonzero(np.diff(sh))[0] + 1, len(sh)]; last = np.zeros(len(sh), int); first = np.zeros(len(sh), int)
    for a, b in zip(st[:-1], st[1:]):
        last[a:b] = np.arange(b - a)[::-1]; first[a:b] = np.arange(b - a)
    pos = d.edge_feats()[vt, 1]
    dm = inputs_fix.use(copy.copy(d)); pub = SP.PublicData(d.meta); pm = inputs_fix.use(copy.copy(pub), public=True)
    return dict(d=d, sota=sota, vt=vt, Y=Y, P=P, sh=sh, last=last, first=first, pos=pos, dm=dm, pm=pm, pub=pub, z=z)


def r2(y, p):
    ok = np.isfinite(y) & np.isfinite(p); return 1 - np.mean((y[ok] - p[ok]) ** 2) / np.var(y[ok])


# ---------------------------------------------------------------------------------------------------------------- figures
def fig_scoreboard(pubd):
    top1 = {"psi": 0.9999, "qb": 0.9984, "dlcfs": 0.0068, "cons": 0.9881}
    f2 = pubd["F2"]; gaps = {"psi (0.55 x R2)": 0.55 * (top1["psi"] - f2["psi"]), "q95 / betaN (0.15 x R2)": 0.15 * (top1["qb"] - (f2["q95"] + f2["betaN"]) / 2),
                             "LCFS distance (0.10 x D)": 0.10 * (f2["dlcfs"] - top1["dlcfs"]), "consistency (0.20 x mean R2)": 0.20 * (top1["cons"] - f2["cons"])}
    top5 = {"R_axis": (0.9832, 0.9862), "Z_axis": (0.9916, 0.9946), "kappa": (0.979, 0.983), "tri_top": (0.9936, 0.9958), "tri_bot": (0.9936, 0.9958),
            "volume": (0.9926, 0.9933), "li": (0.969, 0.973), "q95": (0.99815, 0.99867), "betaN": (0.99775, 0.99814)}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.45})
    k = list(gaps); v = [gaps[x] * 1e4 for x in k]; ax[0].barh(k[::-1], v[::-1], color=[WARN if x > 2.5 else ACC for x in v[::-1]])
    for i, x in enumerate(v[::-1]):
        ax[0].text(x + 0.1, i, f"{x:.1f}", va="center", color=INK)
    ax[0].set_xlabel("composite points behind #1 (x 1e-4 of S)"); ax[0].set_title("gap to #1 by term (13e-4 total)", loc="left")
    names = ALL9; y = np.arange(len(names))
    for i, n in enumerate(names):
        lo, hi = top5[n]; ax[1].plot([1 - hi, 1 - lo], [i, i], color=GOOD, lw=6, alpha=0.35, solid_capstyle="butt")
        ax[1].plot(1 - pubd["HJl9"][n], i, "o", mfc="white", mec=MUTED, ms=7); ax[1].plot(1 - pubd["F2"][n], i, "o", color=ACC, ms=7)
    ax[1].set_xscale("log"); ax[1].set_yticks(y); ax[1].set_yticklabels(names); ax[1].invert_yaxis()
    ax[1].set_xlabel("public error 1 - R2 (log; left is better)"); ax[1].set_title("public error: HJl9 (hollow), F2 (blue), top-5 (green)", loc="left")
    tk = [0.0015, 0.002, 0.003, 0.005, 0.01, 0.02, 0.04]; ax[1].set_xticks(tk); ax[1].set_xticklabels([f"{t:g}" for t in tk]); ax[1].minorticks_off()
    save(fig, "scoreboard", "Composite gap to #1 by term, and every scored scalar against the top-5 band. Every term is behind; q/b and LCFS distance carry as much as consistency.")


def fig_trajectory(pubd):
    tags = ["HJl9", "F1a", "F1b", "F2"]; fig, ax = plt.subplots(figsize=(11, 3.6))
    for n in ALL9:
        e = np.array([1 - pubd[t][n] for t in tags]); ax.plot(tags, e / e[0], "o-", lw=1.6, label=n, color=WARN if n == "li" else None)
    ax.axhline(1, color=MUTED, lw=0.8); ax.set_ylabel("public error relative to HJl9"); ax.legend(ncol=9, fontsize=8, loc="lower left")
    ax.set_title("submission history (Oct 4 -> Oct 11): li, kappa, R_axis fell 5-15%; q95 / betaN flat", loc="left")
    save(fig, "trajectory", "Public error per scalar across our submissions, normalised to HJl9 (all-data refit + Thomson on fixed coordinates + full-rate history).")


def fig_ood(o):
    g = o["gap"]; fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, n in enumerate(ALL9):
        v, p = g[n]["vt"] if n != "betaN" else g[n]["vt_trimmed"], g[n]["public"]
        ax[0].plot([1 - v, 1 - p], [i, i], color=GRID, lw=3); ax[0].plot(1 - v, i, "o", color=GOOD, ms=7); ax[0].plot(1 - p, i, "o", color=WARN, ms=7)
    ax[0].set_xscale("log"); ax[0].set_yticks(range(9)); ax[0].set_yticklabels(ALL9); ax[0].invert_yaxis()
    ax[0].set_xlabel("1 - R2 (log)"); ax[0].set_title("same TRAIN-only model: in-distribution VT (green) vs public (red)", loc="left")
    m = [g[n]["mse_ratio_pub_vt"] for n in ALL9]; ax[1].barh(ALL9[::-1], m[::-1], color=[WARN if x > 1.3 else MUTED for x in m[::-1]])
    ax[1].axvline(1, color=INK, lw=0.8); ax[1].set_xlabel("public MSE / VT MSE (label spread calibrated)")
    ax[1].set_title(f"error inflation on public; input shift AUC {o['auc_public_vs_TRAIN']:.3f} (none)", loc="left")
    save(fig, "ood", "Generalisation gap: the public fold is not shifted in its inputs (shot-level domain classifier AUC 0.507, importance weighting predicts no gap), "
                     "yet errors are 1.25-1.9x larger on Z_axis, tri, li, volume, q95.")


def fig_phase(D):
    Y, P, pos = D["Y"], D["P"], D["pos"]; bins = np.linspace(0, 1, 21); c = 0.5 * (bins[1:] + bins[:-1]); fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for k, (grp, a) in enumerate(((SC, ax[0]), (["q95", "betaN"], ax[1]))):
        for n in grp:
            j = ALL9.index(n); ok = np.isfinite(Y[:, j]); v = np.var(Y[ok, j]); e2 = (Y[:, j] - P[:, j]) ** 2
            m = [np.nanmean(np.where(ok, e2, np.nan)[(pos >= a_) & (pos < b_)]) / v for a_, b_ in zip(bins[:-1], bins[1:])]
            a.plot(c, m, lw=1.8, label=n, color=WARN if n in ("li", "betaN") else None)
        a.set_yscale("log"); a.set_xlabel("position in shot (0 = first EFIT frame, 1 = last)"); a.set_ylabel("MSE / label variance"); a.legend(fontsize=8, ncol=2)
    ax[0].set_title("shape scalars: error explodes in the last 10% of the shot", loc="left"); ax[1].set_title("q95 / betaN", loc="left")
    save(fig, "phase", "Normalised error vs position in the shot (VT, 743 shots). The ramp-down end dominates every scalar; li and betaN worst.")


def fig_tails(D):
    Y, P, last = D["Y"], D["P"], D["last"]; fig, ax = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.2, 1]})
    cols = ["last frame", "last 2-3", "last 4-10", "first 10", "rest"]; masks = [last == 0, (last >= 1) & (last <= 2), (last >= 3) & (last <= 9), D["first"] < 10, None]
    masks[4] = ~(masks[0] | masks[1] | masks[2] | masks[3]); M = np.zeros((10, 5))
    for i, n in enumerate(ALL9):
        e2 = np.nan_to_num((Y[:, i] - P[:, i]) ** 2); M[i] = [e2[m].sum() / e2.sum() for m in masks]
    M[9] = [m.mean() for m in masks]; im = ax[0].imshow(M, cmap="Oranges", vmin=0, vmax=0.7, aspect="auto")
    for i in range(10):
        for j in range(5):
            ax[0].text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=8, color=INK)
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels(cols); ax[0].set_yticks(range(10)); ax[0].set_yticklabels(ALL9 + ["share of frames"]); ax[0].grid(False)
    ax[0].set_title("share of squared error by where the frame sits", loc="left")
    for i, n in enumerate(ALL9):
        e2 = np.sort(np.nan_to_num((Y[:, i] - P[:, i]) ** 2))[::-1]; cum = np.cumsum(e2) / e2.sum(); x = np.arange(1, len(e2) + 1) / len(e2)
        ax[1].plot(x, cum, lw=1.6, label=n, color=WARN if n in ("li", "betaN") else None)
    ax[1].set_xscale("log"); ax[1].set_xlim(1e-5, 1); ax[1].set_xlabel("fraction of worst frames"); ax[1].set_ylabel("share of total squared error")
    ax[1].legend(fontsize=7, ncol=3); ax[1].set_title("tail dominance (one VT frame = 32% of betaN SSE)", loc="left")
    save(fig, "tails", "Where the squared error lives: 4.5% of frames (the last 10 per shot) hold 45% of li and 42% of betaN error; the worst 1% of frames hold 25-60%.")


def fig_scatter(D):
    Y, P, last = D["Y"], D["P"], D["last"]; fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    for a, n in zip(ax, ["li", "Z_axis", "tri_top", "betaN"]):
        j = ALL9.index(n); ok = np.isfinite(Y[:, j]); e = ok & (last <= 2)
        a.hexbin(Y[ok, j], P[ok, j], gridsize=70, bins="log", cmap="Blues", mincnt=1, linewidths=0)
        a.scatter(Y[e, j], P[e, j], s=3, color=WARN, alpha=0.5, label="last 3 frames")
        lo, hi = np.percentile(Y[ok, j], [0.05, 99.95]); a.plot([lo, hi], [lo, hi], color=INK, lw=0.8); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_title(f"{n}: R2 {r2(Y[:, j], P[:, j]):.4f}", loc="left"); a.set_xlabel("EFIT label"); a.grid(False)
    ax[0].set_ylabel("prediction"); ax[0].legend(fontsize=8, loc="upper left")
    save(fig, "scatter", "Prediction vs label (VT). Red = the last three frames of each shot: the off-diagonal mass is mostly there.")


def fig_pca(D):
    d, dm, pm, Y, P = D["d"], D["dm"], D["pm"], D["Y"], D["P"]; rng = np.random.default_rng(0); fm = np.ones(len(d.fn), bool)
    tr = rng.choice(np.nonzero(d.split == 0)[0], 60000, replace=False); Xt = dm.inputs(tr, fm).astype(np.float64); mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    Z = (Xt - mu) / sd; U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False); ev = S ** 2 / np.sum(S ** 2)
    proj = lambda X: ((np.asarray(X, np.float64) - mu) / sd - Z.mean(0)) @ Vt[:2].T                              # noqa: E731
    pr = rng.choice(len(D["pub"].X), 40000, replace=False); Pp = proj(pm.inputs(pr, fm)); Ptr = proj(Xt)
    vsel = rng.choice(len(D["vt"]), 50000, replace=False); Pv = proj(dm.inputs(D["vt"][vsel], fm))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"wspace": 0.4})
    ax[0].hexbin(Ptr[:, 0], Ptr[:, 1], gridsize=80, bins="log", cmap="Greys", mincnt=1, linewidths=0)
    H, xe, ye = np.histogram2d(Pp[:, 0], Pp[:, 1], bins=60); ax[0].contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), np.log1p(H.T), levels=6, colors=WARN, linewidths=0.8)
    ax[0].set_title(f"inputs PCA: TRAIN (grey) vs public (red contours)\nPC1 {ev[0]:.0%}, PC2 {ev[1]:.0%} of variance", loc="left")
    for a, n in zip(ax[1:], ["li", "Z_axis"]):
        j = ALL9.index(n); e2 = ((Y[vsel, j] - P[vsel, j]) ** 2) / np.nanvar(Y[:, j]); o = np.argsort(np.nan_to_num(e2))
        sc = a.scatter(Pv[o, 0], Pv[o, 1], c=np.log10(np.nan_to_num(e2[o]) + 1e-6), s=2, cmap="magma_r", vmin=-5, vmax=-0.5)
        plt.colorbar(sc, ax=a, label="log10 normalised squared error"); a.set_title(f"VT frames coloured by {n} error (worst on top)", loc="left")
    for a in ax:
        a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.grid(False)
    save(fig, "pca", "Input space (427 corrected inputs, PCA fitted on TRAIN): the public fold sits on the TRAIN density with no region of its own; high-error frames are spread through the occupied region, with a dense high-li-error band at PC1 0-15, PC2 near -5.")


def fig_regime(D):
    d, dm, Y, P, vt, pos = D["d"], D["dm"], D["Y"], D["P"], D["vt"], D["pos"]; X = dm.inputs(vt, np.ones(len(d.fn), bool))
    ip = np.abs(X[:, d.fn.index("mag_plasma_current")]); ne = X[:, d.fn.index("thomson_core_ne_mean")]
    feats = [("|Ip| (corrected)", ip), ("predicted li", P[:, 6]), ("core ne mean", ne), ("predicted kappa", P[:, 2])]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    for a, (lab, v) in zip(ax, feats):
        q = np.quantile(v, np.linspace(0, 1, 16)); c = 0.5 * (q[1:] + q[:-1])
        for n in ["li", "Z_axis", "volume", "q95", "betaN"]:
            j = ALL9.index(n); ok = np.isfinite(Y[:, j]); e2 = np.where(ok, (Y[:, j] - P[:, j]) ** 2, np.nan) / np.nanvar(Y[:, j])
            a.plot(c, [np.nanmean(e2[(v >= a_) & (v <= b_)]) for a_, b_ in zip(q[:-1], q[1:])], lw=1.6, label=n)
        a.set_yscale("log"); a.set_xlabel(lab + " (quantile bins)")
    ax[0].set_ylabel("MSE / label variance"); ax[0].legend(fontsize=8)
    ax[0].set_title("error vs operating regime", loc="left")
    save(fig, "regime", "Error by operating regime (VT): low plasma current, predicted li above 1.5, kappa below 1.7 and low density carry the error; high-current flat-top states are 5-40x more accurate.")


def fig_transfer():
    z = np.load(fx.OUT / "transfer_F2.npz"); Dv, H = z["derived"], z["heads"]; fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].scatter(H[:, 6], Dv[:, 6], s=2, alpha=0.4, color=ACC); lim = [0.4, 5]; ax[0].plot(lim, lim, color=INK, lw=0.8)
    ax[0].set_xlabel("head li"); ax[0].set_ylabel("li of the submitted psi"); ax[0].set_title("li: a few hundred frames fall short of the head", loc="left")
    e = Dv[:, 3] - H[:, 3]; ax[1].hist(e[np.isfinite(e)], bins=200, range=(-0.08, 0.08), color=ACC); ax[1].set_yscale("log")
    ax[1].set_xlabel("tri_top: submitted - head"); ax[1].set_title("tri: wide residual on most shots", loc="left")
    tv = [np.nanmean((Dv[:, j] - H[:, j]) ** 2) / np.nanvar(H[:, j]) for j in range(7)]
    he = [1 - x for x in [0.98372, 0.99506, 0.97345, 0.99496, 0.99313, 0.99145, 0.97601]]
    y = np.arange(7); ax[2].barh(y - 0.2, he, 0.4, color=MUTED, label="head error (VT, 1 - R2)"); ax[2].barh(y + 0.2, tv, 0.4, color=WARN, label="transfer loss (public)")
    ax[2].set_yticks(y); ax[2].set_yticklabels(SC); ax[2].invert_yaxis(); ax[2].set_xscale("log"); ax[2].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, -0.38), ncol=2); ax[2].set_title("post-processing loss vs head error", loc="left")
    save(fig, "transfer", "Does the submitted psi carry the heads' predictions? On the public fold the triangularity and li transfer losses are 56-63% / 20% of the heads' own error: an inference-only lever (§41y running).")


def fig_ensemble():
    b = json.loads((fx.OUT / "blend_study.json").read_text()); fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, n in zip(ax, ["li", "q95"]):
        m = b[n]["members"]; C = np.array(b[n]["err_corr"]); im = a.imshow(C, vmin=0.4, vmax=1, cmap="viridis"); a.grid(False)
        for i in range(len(m)):
            for j in range(len(m)):
                a.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if C[i, j] < 0.8 else INK)
        a.set_xticks(range(len(m))); a.set_xticklabels(m, rotation=40); a.set_yticks(range(len(m))); a.set_yticklabels(m)
        a.set_title(f"{n}: member error correlation", loc="left")
    plt.colorbar(im, ax=ax, fraction=0.025)
    save(fig, "ensemble", "Why re-weighting saturates: shape-head members (MLP, register PI-KAN, thz variant) share 0.90-0.97 of their error; only the weaker NAS entry is diverse.")


def fig_levers():
    L = [("all data + thz + history (F2, public)", 0.00112 / 1, None, "consistency, public"),
         ("thz + history vs base heads (VT)", 0.00053, None, "mean-7 R2"),
         ("register heads in ensemble (VT)", 0.000072, (0.000022, 0.000128), "mean-7 R2"),
         ("stacking weights (VT, OOF)", -0.00003, None, "mean-7 R2"),
         ("thp in-sample psi (VT)", -0.00097, (-0.00156, -0.00044), "mean-7 R2"),
         ("future context fut (CV, 5 folds)", -0.00024, (-0.00064, 0.00016), "mean-7 R2"),
         ("q/b + fut (VT)", -0.00022, (-0.000445, 0.000045), "q/b R2"),
         ("CatBoost q/b blend (VT, OOF)", 0.000035, None, "q/b R2"),
         ("6 -> 8 seeds / psi seeds (VT sample)", 0.0, None, "S")]
    fig, ax = plt.subplots(figsize=(11, 3.9)); y = np.arange(len(L))
    for i, (n, v, ci, u) in enumerate(L):
        c = GOOD if v > 0.0001 else (WARN if v < -0.0001 else MUTED); ax.plot(v * 1e4, i, "o", color=c, ms=8)
        if ci:
            ax.plot([ci[0] * 1e4, ci[1] * 1e4], [i, i], color=c, lw=2)
        ax.text(11.8, i, u, va="center", fontsize=8, color=MUTED)
    ax.axvline(0, color=INK, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([x[0] for x in L]); ax.invert_yaxis(); ax.set_xlim(-12, 14)
    ax.set_xlabel("effect (x 1e-4 R2; 95% CI where paired)"); ax.set_title("what moved the model (Oct 10-11)", loc="left")
    save(fig, "levers", "Measured effects of every lever tested in the last two days. Only data and input information moved the model; re-weighting and the extra input blocks did not.")


def fig_learning():
    fig, ax = plt.subplots(figsize=(6.5, 3.6)); x = np.array([8 / 9, 1.0]); y1 = np.array([0.98557, 0.98654]); y2 = np.array([0.98600, 0.98661])
    ax.plot(x * 6298, y1, "o-", color=ACC, label="seed 1"); ax.plot(x * 6298, y2, "s-", color=ACC, alpha=0.6, label="seed 2")
    b = np.polyfit(np.log([5598, 6298]), [0.5 * (y1[0] + y2[0]), 0.5 * (y1[1] + y2[1])], 1); xs = np.array([5598, 7041, 7915])
    ax.plot(xs, np.polyval(b, np.log(xs)), "--", color=MUTED, label="log-linear extrapolation")
    ax.axvline(7041, color=GOOD, lw=0.8); ax.text(7060, 0.9856, "all labelled\nshots", fontsize=8, color=GOOD)
    ax.axvline(7915, color=WARN, lw=0.8); ax.text(7930, 0.9856, "+ public\npseudo-labels", fontsize=8, color=WARN)
    ax.set_xlabel("training shots"); ax.set_ylabel("head mean-7 R2 (VALID sample)"); ax.legend(fontsize=8); ax.set_title("data-limited heads", loc="left")
    save(fig, "learning", "Learning curve of the shape heads: 11% less data costs about 0.0008 R2. Every extra labelled-like shot still pays.")


# ---------------------------------------------------------------------------------------------------------------- page
def page(pubd):
    def img(n):
        return "data:image/png;base64," + base64.b64encode((REP / f"{n}.png").read_bytes()).decode()
    cap = dict(FIGS); f2 = pubd["F2"]
    sec = lambda n, h: f'<section class="fig" id="{n}"><h2>{h}</h2><img src="{img(n)}" alt="{h}"><p>{cap[n]}</p></section>'   # noqa: E731
    novelty = [
        ("Shape from direct heads, then psi edited to match", "7 scalar + 64-ray heads; GPU twin of the official extraction steers the flux map (axis shift, boundary / kappa / X-point warps, joint Gauss-Newton, li remap)", "Public competitor: psi network only, geometry from a surrogate loss", "consistency 0.9855 vs 0.9819 (competitor, own dev)"),
        ("Hybrid quantum register heads", "PI-KAN head with a simulated 30-atom Rydberg layer (input-dependent), 2/14 of the shape ensemble", "none known", "removing them costs 7e-5 R2 (CI excludes 0)"),
        ("Orientation-canonical psi trunks", "Ip-sign canonical current inputs: -34..-51% raw psi error", "none known", "psi R2 0.99984 public"),
        ("All-data refit with scorer-derived labels", "boundary labels from the official extraction on all 7,041 shots", "competitor keeps 351 audit shots out", "F1a / F2 public +0.0005"),
        ("Evaluation discipline", "shot-level paired bootstrap on 743 held-out shots, stratified 5-fold CV, registered rules", "", "rejected 6 of 8 levers in 2 days"),
    ]
    excel = ["consistency (0.9855) above every documented public method; R_axis at the top-5 level (0.9846 vs 0.9832-0.9862)",
             "fast, scorer-faithful inference stack (GPU twin, 193k frames in ~40 min) and a rehearsed Final-phase path"]
    weak = [f"q95 / betaN {f2['q95']:.4f} / {f2['betaN']:.4f} vs top-5 0.998 / 0.998: 3.6e-4 of S",
            f"LCFS distance {f2['dlcfs']:.4f} vs 0.0068: 3.4e-4 of S (X-point tips, grid quantisation)",
            "public error 1.25-1.9x the in-distribution error on Z_axis, tri, li, volume, q95 without any input shift",
            "post-processing does not carry the heads' tri (and extreme li) into psi",
            "end-of-shot frames: 4.5% of frames, 40-45% of li / betaN error; current-trajectory features did not help"]
    nxt = [
        ("1", "Close the head -> psi transfer loss (li cap, tri warp iterations; then a final exact tri / li projection)", "consistency (tri, li)", "+1 to +3e-4", "hours, inference only", "running (§41y)", ""),
        ("2", "Self-training: pseudo-label the public (and on 19 Oct the private) inputs with the full ensemble, refit heads", "all terms", "+0.5 to +1e-4", "~5 GPU-h", "learning curve", "[4], [5]"),
        ("3", "A better psi base model (physics-constrained: Grad-Shafranov residual / flux-gradient loss, surrogate functional loss)", "LCFS distance, psi, consistency", "up to +3e-4", "days, risky", "top-5 psi 2-3x better", "[1], [2], [3]"),
        ("4", "q95 / betaN: members that are diverse AND accurate (temporal-convolution heads over the shot; distillation of the entry)", "q/b", "+0.5 to +1.5e-4", "1-2 days", "member correlation 0.73-0.82", "[6], [7], [8]"),
        ("5", "Robustness to hard shots: shot-level reweighting toward high-error regimes (low Ip, extreme li)", "Z_axis, li, volume", "+0.5e-4", "hours", "regime + OOD panels", "[9]"),
        ("6", "End-of-shot frames: deprioritise; inputs do not determine the EFIT ramp-down labels (fut CV failed)", "li, betaN", "~0", "-", "phase + tails", "[10]"),
    ]
    refs = ["[1] S. Joung et al., Deep neural network Grad-Shafranov solver constrained with measured magnetic signals, Nucl. Fusion 60, 016034 (2020).",
            "[2] J. T. Wai, M. D. Boyer, E. Kolemen, Neural net modeling of equilibria in NSTX-U, Nucl. Fusion 62, 086042 (2022).",
            "[3] L. L. Lao et al., Application of machine learning and artificial intelligence to extend EFIT equilibrium reconstruction, Plasma Phys. Control. Fusion 64, 074001 (2022).",
            "[4] D.-H. Lee, Pseudo-Label: the simple and efficient semi-supervised learning method for deep neural networks, ICML Workshop (2013).",
            "[5] Q. Xie, M.-T. Luong, E. Hovy, Q. V. Le, Self-training with Noisy Student improves ImageNet classification, CVPR (2020).",
            "[6] A. Krogh, J. Vedelsby, Neural network ensembles, cross validation, and active learning, NIPS 7 (1995).",
            "[7] B. Lakshminarayanan, A. Pritzel, C. Blundell, Simple and scalable predictive uncertainty estimation using deep ensembles, NeurIPS (2017).",
            "[8] S. Bai, J. Z. Kolter, V. Koltun, An empirical evaluation of generic convolutional and recurrent networks for sequence modeling, arXiv:1803.01271 (2018).",
            "[9] H. Shimodaira, Improving predictive inference under covariate shift by weighting the log-likelihood function, J. Stat. Plan. Inference 90, 227-244 (2000).",
            "[10] J. Kates-Harbeck, A. Svyatkovskiy, W. Tang, Predicting disruptive instabilities in controlled fusion plasmas through deep learning, Nature 568, 526-531 (2019)."]
    css = """
:root{--bg:#f5f6f3;--fg:#1d2733;--muted:#5d6874;--rule:#d6dbd3;--accent:#2f6db5;--warn:#c4472d;--good:#2e8b57;--paper:#ffffff;
--display:"Fraunces",Georgia,serif;--body:"Source Sans 3","Segoe UI",system-ui,sans-serif;--mono:"JetBrains Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141a20;--fg:#e4e9ee;--muted:#9aa6b2;--rule:#2b343d;--accent:#7fb0ea;--warn:#f08a70;--good:#6cc596;--paper:#f7f8f6;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141a20;--fg:#e4e9ee;--muted:#9aa6b2;--rule:#2b343d;--accent:#7fb0ea;--warn:#f08a70;--good:#6cc596;--paper:#f7f8f6;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:15px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:28px 60px;display:flex;flex-direction:column;gap:28px}
h1{font-family:var(--display);font-weight:600;font-size:2.1rem;margin:0;text-wrap:balance}
h2{font-family:var(--display);font-weight:600;font-size:1.25rem;margin:0 0 10px;text-wrap:balance}
.sub{color:var(--muted);margin:6px 0 0;max-width:70ch}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.kpi{border-top:3px solid var(--accent);padding-top:8px}.kpi b{display:block;font-family:var(--mono);font-size:1.35rem;font-variant-numeric:tabular-nums}
.kpi span{color:var(--muted);font-size:.85rem;letter-spacing:.02em}.kpi.bad{border-color:var(--warn)}
.fig img{background:var(--paper);border-radius:6px;padding:8px;width:100%;height:auto;display:block}
.fig p{color:var(--muted);margin:8px 0 0;max-width:95ch;font-size:.92rem}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem;min-width:720px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule);vertical-align:top}th{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px}.cols>div{min-width:0}
ul{margin:0;padding-left:20px}li{margin:4px 0}.ok li::marker{color:var(--good)}.no li::marker{color:var(--warn)}
.refs{font-size:.82rem;color:var(--muted)}.refs p{margin:3px 0}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:.85rem}nav a{color:var(--accent);text-decoration:none}nav a:focus-visible{outline:2px solid var(--accent)}
"""
    kp = [("public S (F2)", "0.99539", ""), ("#1", "0.9967", ""), ("gap to #1", "0.0013", "bad"), ("rank", "16 / 123", ""),
          ("q95 / betaN", f"{(f2['q95'] + f2['betaN']) / 2:.4f}", "bad"), ("LCFS distance", f"{f2['dlcfs']:.4f}", "bad"), ("consistency", f"{f2['cons']:.4f}", "")]
    html = ['<title>Fusion Error Atlas</title>',
            '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=JetBrains+Mono:wght@500&family=Source+Sans+3:wght@400;600&display=swap">',
            f"<style>{css}</style>", '<div class="wrap">',
            '<header><h1>Fusion Error Atlas</h1><p class="sub">DIII-D equilibrium challenge, development phase, 11 Oct 2026. Anatomy uses our TRAIN-only components on all 743 '
            'held-out VALID + TEST shots (out-of-sample); public numbers are Codabench detailed results.</p></header>',
            '<div class="kpis">' + "".join(f'<div class="kpi {c}"><b>{v}</b><span>{k}</span></div>' for k, v, c in kp) + "</div>",
            '<nav>' + "".join(f'<a href="#{n}">{n}</a>' for n, _ in FIGS) + '<a href="#approach">approach</a><a href="#next">next</a></nav>']
    titles = {"scoreboard": "Where the gap is", "trajectory": "What our submissions changed", "ood": "Generalisation: VT vs public", "phase": "Error across the shot",
              "tails": "Tails and shot ends", "scatter": "Prediction vs label", "pca": "Input space (PCA)", "regime": "Operating regimes",
              "transfer": "Head -> psi transfer", "ensemble": "Ensemble diversity", "levers": "Measured levers", "learning": "Data scaling"}
    html += [sec(n, titles[n]) for n, _ in FIGS]
    html.append('<section id="approach"><h2>Our approach vs others</h2><div class="tbl"><table><tr><th>element</th><th>ours</th><th>others</th><th>evidence</th></tr>'
                + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{e}</td></tr>" for a, b, c, e in novelty) + "</table></div>"
                '<div class="cols"><div><h2>Strong</h2><ul class="ok">' + "".join(f"<li>{x}</li>" for x in excel) + '</ul></div><div><h2>Weak</h2><ul class="no">'
                + "".join(f"<li>{x}</li>" for x in weak) + "</ul></div></div></section>")
    html.append('<section id="next"><h2>Where to head for first place</h2><div class="tbl"><table><tr><th>#</th><th>lever</th><th>term</th><th>expected S</th><th>cost</th><th>evidence</th><th>refs</th></tr>'
                + "".join(f'<tr><td class="num">{a}</td><td>{b}</td><td>{c}</td><td class="num">{d}</td><td>{e}</td><td>{f}</td><td>{g}</td></tr>' for a, b, c, d, e, f, g in nxt)
                + '</table></div><div class="refs">' + "".join(f"<p>{r}</p>" for r in refs) + "</div></section></div>")
    out = REP / "fusion_atlas.html"; out.write_text("\n".join(html)); print(f"ATLAS_DONE {out} {out.stat().st_size / 1e6:.1f} MB", flush=True)


def main() -> None:
    pubd = json.loads((REP / "public_detail.json").read_text()); o = json.loads((fx.OUT / "ood_analysis.json").read_text())
    for k in ("HJl9", "F1a", "F1b", "F2"):
        pubd[k]["cons"] = pubd[k]["cons"]
    fig_scoreboard(pubd); fig_trajectory(pubd); fig_ood(o)
    D = load(); fig_phase(D); fig_tails(D); fig_scatter(D); fig_pca(D); fig_regime(D)
    fig_transfer(); fig_ensemble(); fig_levers(); fig_learning(); page(pubd)


if __name__ == "__main__":
    main()
