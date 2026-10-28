"""Figure 2 for every benchmark (doc 35 §23 models): ground truth | classical | + matched MLP | + 30 qubits (3-D, tethered)
| change in absolute error from classical to qubits (blue: lower). One sample per benchmark, fixed in advance; each panel
carries that sample's error. Rows whose data are not ready are skipped.

    .venv/bin/python paper/alloc/fig_outputs_all.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = ROOT / "paper/fig_cache"
ARMS = ("classical", "mlp", "qubits", "qubits50")
TITLES = ("ground truth", "classical", "+ MLP", "+ 30 qubits", "+ 50 qubits", "|error|: qubits minus classical")


def nrmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)) / (np.std(t) + 1e-12))


def from_rows(name):
    f = CACHE / "rows_s23" / f"{name}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


def from_legacy(file, metric, label, keys=("vanilla", "mlp30", "mr30ryd"), mkey=None):
    f = CACHE / file
    if not f.exists():
        return None
    d = np.load(f)
    r = {"label": label, "metric": metric, "kind": "field", "truth": d["truth"]}
    for a, k in zip(ARMS, keys):
        r[f"pred_{a}"] = d[f"{k}_pred"]; r[f"m_{a}"] = float(d[f"{k}_{mkey}"])
    return r


def from_runs(folder, pattern, label, metric="NRMSE", kind="field"):
    fs = {a: ROOT / folder / pattern.format(v=v) for a, v in zip(ARMS, ("none", "q30-geo=3d-mlp", "q30-geo=3d"))}
    if not all(f.exists() for f in fs.values()):
        return None
    r = {"label": label, "metric": metric, "kind": kind}
    for a, f in fs.items():
        d = np.load(f); r["truth"] = d["truth"]; r[f"pred_{a}"] = d["pred"]
        r[f"m_{a}"] = float(d["vrmse8"]) if metric.startswith("VRMSE") and "vrmse8" in d.files else nrmse(d["pred"], d["truth"])
    return r


def rows():
    return [r for r in (
        from_rows("shear_flow_routed") or from_rows("shear_flow"), from_rows("rayleigh_benard_routed") or from_rows("rayleigh_benard"),
        from_rows("turbulent_radiative_layer_2D_routed") or from_rows("turbulent_radiative_layer_2D"), from_rows("trl2d_rollout"),
        from_rows("MHD_64_routed") or from_rows("MHD_64"),
        from_legacy("fig_fields_1hp_s23.npz", "MAE degC", "heat plume", mkey="mae"),
        from_rows("fusion_sota") or from_legacy("fig_fields_psi_s23.npz", "NRMSE", "fusion eq.", mkey="nrmse"),   # §29b trunk first
        from_runs("runs/torax_s23", "{v}_k4_s1337_fig.npz", "TORAX (T_e)", kind="spacetime"),
        from_rows("tokamark"), from_rows("realpde"),
        from_rows("routefinder"), from_rows("tdc")) if r is not None]


def orient(f):
    return f.T if f.ndim == 2 and f.shape[0] > 2 * f.shape[1] else f


def badge(ax, r, a):
    ax.text(0.02, 0.96, f"{r['metric']} {r[f'm_{a}']:.4f}", transform=ax.transAxes, fontsize=4.6, va="top", color="w",
            bbox=dict(fc="k", alpha=0.5, lw=0, pad=0.8))


def main() -> None:
    R = rows()
    plt.rcParams.update({"font.size": 6.0})
    fig, ax = plt.subplots(len(R), 6, figsize=(7.0, 0.36 * len(R) + 0.25), squeeze=False)
    fig.subplots_adjust(hspace=0.05, wspace=0.03)
    for i, r in enumerate(R):
        gt = np.asarray(r["truth"], dtype=float)
        have50 = "pred_qubits50" in r; qa = "qubits50" if have50 else "qubits"
        ax[i, 0].set_ylabel(r["label"] if len(r["label"]) <= 12 else r["label"].replace(" (", "\n(", 1), fontsize=5.0,
                            rotation=0, ha="right", va="center")
        if r.get("kind") == "scatter":                                  # per-sample scalar targets (TDC): pred vs truth
            lo, hi = np.nanpercentile(gt, [1, 99]); o = np.argsort(gt)
            ax[i, 0].plot(np.arange(len(gt)), gt[o], ".", ms=0.8, color="0.2"); ax[i, 0].set_ylim(lo, hi)
            for j, a in enumerate(ARMS):
                if f"pred_{a}" in r:
                    ax[i, j + 1].plot(gt, np.asarray(r[f"pred_{a}"], float), ".", ms=0.8, color="C0"); ax[i, j + 1].plot([lo, hi], [lo, hi], "k-", lw=0.4)
                    ax[i, j + 1].set_xlim(lo, hi); ax[i, j + 1].set_ylim(lo, hi); badge(ax[i, j + 1], r, a)
            dq = np.abs(np.asarray(r[f"pred_{qa}"], float) - gt) - np.abs(np.asarray(r["pred_classical"], float) - gt)
            ax[i, 5].scatter(gt, dq, s=0.8, c=np.sign(dq), cmap="coolwarm", vmin=-1, vmax=1); ax[i, 5].axhline(0, color="k", lw=0.4)
        else:
            lo, hi = np.nanpercentile(gt, [1, 99]); kw = dict(cmap="RdBu_r", origin="lower", aspect="auto", vmin=lo, vmax=hi)
            ax[i, 0].imshow(orient(gt), **kw)
            for j, a in enumerate(ARMS):
                if f"pred_{a}" in r:
                    ax[i, j + 1].imshow(orient(np.asarray(r[f"pred_{a}"], dtype=float)), **kw); badge(ax[i, j + 1], r, a)
            dq = np.abs(np.asarray(r[f"pred_{qa}"], float) - gt) - np.abs(np.asarray(r["pred_classical"], float) - gt)
            m = float(np.nanpercentile(np.abs(dq), 99)) or 1.0
            ax[i, 5].imshow(orient(dq), cmap="coolwarm", origin="lower", aspect="auto", vmin=-m, vmax=m)
        if not have50:
            ax[i, 4].set_facecolor("0.93"); ax[i, 4].text(0.5, 0.5, "not run", transform=ax[i, 4].transAxes, ha="center",
                                                          va="center", fontsize=5.0, color="0.45")
        print(f"{r['label']:14s} " + "  ".join(f"{a} {r[f'm_{a}']:.4f}" for a in ARMS if f"m_{a}" in r))
    for j, t in enumerate(TITLES):
        ax[0, j].set_title(t, fontsize=5.6)
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    out = ROOT / "paper/manuscript/iclr2027/fig_outputs_all.pdf"
    fig.savefig(out, bbox_inches="tight"); print("wrote", out, "rows", len(R))


if __name__ == "__main__":
    main()
