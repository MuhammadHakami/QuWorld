"""Figure for the evaluation-error analysis (docs 36-37): fig_gate.pdf in the ICLR manuscript directory.

    .venv/bin/python paper/alloc/adrc_summary.py && .venv/bin/python paper/alloc/fig_gate.py

(a) VRMSE per Well row, from the published leader down the stack: persistence, trunk, band blend, per-window gate, and
    the routed co-processor with the register / GNN / MLP (log scale; seeds as dots).
(b) Where the co-processor's capture lands per Fourier band after the gate (route-supervised register, seed 1):
    captured by qubit-routed modes, by the classical head, and the part of the band left uncaptured; the qubit-slot
    share per band as a line.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/manuscript/iclr2027/fig_gate.pdf"
PUB = {"shear_flow": 0.0345, "rayleigh_benard": 0.0415,   # PDE-FM (arXiv:2511.21861, Table 3)
       "turbulent_radiative_layer_2D": 0.1930, "MHD_64": 0.1160}   # Flowers (arXiv:2603.04430, 4->1); EddyFormer (NeurIPS 2025)
LABEL = {"shear_flow": "shear_flow", "rayleigh_benard": "rayleigh_benard$^\\dagger$", "turbulent_radiative_layer_2D": "turbulence",
         "MHD_64": "MHD_64"}
ROWS = [("shear_flow", "shear_flow__gate_mr30ryd_s1337"), ("rayleigh_benard", "rayleigh_benard__gatest_mr30ryd_s1337"),
        ("turbulent_radiative_layer_2D", "turbulent_radiative_layer_2D__gate_mr30ryd_s1337"), ("MHD_64", "MHD_64__gate_vanilla_s1337")]
STEPS = [("published", "#9e9e9e"), ("persistence", "#bdbdbd"), ("backbone", "#6baed6"), ("blend", "#3182bd"), ("gate", "#08519c"),
         ("+register", "#d95f02"), ("+GNN", "#7570b3"), ("+MLP", "#1b9e77"), ("gated fine-tune", "#e7298a")]


def summary():
    S = json.loads((ROOT / "runs/alloc/adrc_summary.json").read_text())
    return ({r["row"]: r for r in S if r["gate"] and not r.get("route_sup")},
            {r["row"]: r for r in S if r["gate"] and r.get("route_sup") and not r.get("qubit_bands")})


def panel_a(ax):
    plain, rsup = summary()
    rows = [(b, k) for b, k in ROWS if k in plain]
    w = 0.1
    for j, (base, key) in enumerate(rows):
        r = plain[key]; rs = rsup.get(key, {"arms": {}})
        arm = lambda a: rs["arms"].get(a) or r["arms"].get(a)
        vals = [PUB[base], r["persistence_test"], r["trunk0_test"], r["anchor_test"],
                (r["arms"].get("none") or {}).get("mean"),
                *[(arm(a) or {}).get("mean") for a in ("adrc_ryd", "adrc_gnn", "adrc_mlp")]]
        ft = ROOT / "runs/alloc/ft" / f"{base}_Eft.json"                     # registered E-ft pick, full TEST
        vals.append(json.loads(ft.read_text()).get("full_test") if ft.exists() else None)
        seeds = [None] * 5 + [(arm(a) or {}).get("test") for a in ("adrc_ryd", "adrc_gnn", "adrc_mlp")] + [None]
        for i, ((name, col), v, sd) in enumerate(zip(STEPS, vals, seeds)):
            if v is None:
                continue
            x = j + (i - 4) * w
            ax.bar(x, v, w * 0.92, color=col, label=name if j == 0 else None, zorder=2)
            if sd:
                ax.scatter([x] * len(sd), sd, s=2, color="k", zorder=3)
    ax.set_yscale("log"); ax.set_xticks(range(len(rows)), [LABEL[b].replace("rayleigh_benard", "convection").replace("shear_flow", "shear flow").replace("MHD_64", "MHD") for b, _ in rows], fontsize=6, rotation=25, ha="right")
    ax.set_ylim(0.012, 1.0); ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("test VRMSE (log)"); ax.grid(axis="y", which="both", lw=0.3, alpha=0.5, zorder=0)
    ax.legend(ncol=5, fontsize=5.2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=0.9, columnspacing=0.6)
    ax.text(0.01, 0.02, "(a)", transform=ax.transAxes, fontsize=8, weight="bold")


def panel_b(ax):
    keys = [("shear_flow", "shear_flow__gate_mr30ryd_s1337"), ("rayleigh_benard", "rayleigh_benard__gatest_mr30ryd_s1337")]
    x0, ticks, labels = 0, [], []
    for base, key in keys:
        f = ROOT / "runs/alloc/adrc_results" / f"{key}_adrc_ryd_gate_rs_s1.json"
        if not f.exists():
            continue
        b = json.loads(f.read_text())["bands_test"]; nb = len(b["full"]); xs = x0 + np.arange(nb)
        tot = float(np.sum(b["full"]))                                   # normalize: share of the error left after the gate
        q, c = np.clip(b["captured_qubit"], 0, None) / tot, np.clip(b["captured_classical"], 0, None) / tot
        rest = np.clip(np.array(b["full"]) / tot - q - c, 0, None)
        ax.bar(xs, q, 0.8, color="#d95f02", label="captured by qubits" if x0 == 0 else None)
        ax.bar(xs, c, 0.8, bottom=q, color="#1b9e77", label="captured classically" if x0 == 0 else None)
        ax.bar(xs, rest, 0.8, bottom=q + c, color="#d9d9d9", label="left (after gate)" if x0 == 0 else None)
        ax2 = ax.twinx() if x0 == 0 else ax2
        ax2.plot(xs, 100 * np.array(b["qubit_slot_share"]), "k.-", lw=0.8, ms=3, label="qubit slots (%)" if x0 == 0 else None)
        ticks += list(xs); labels += [f"b{i}" for i in range(nb)]
        ax.text(x0 + nb / 2 - 0.5, -0.1, LABEL[base].replace("rayleigh_benard", "convection").replace("shear_flow", "shear flow"), ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=6)
        x0 += nb + 1
    ax.set_xticks(ticks, labels, fontsize=5); ax.set_ylabel("share of error after the gate", fontsize=6.5)
    ax.set_xlabel("Fourier octave (b0 coarsest \u2192 b4 finest)", fontsize=6, labelpad=8)
    ax2.set_ylabel("qubit slots (%)"); ax2.set_ylim(0, 105); ax.set_ylim(0, 1.0)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5.6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.0)
    ax.text(0.97, 0.9, "(b)", transform=ax.transAxes, fontsize=8, weight="bold", ha="right")


def panel_c(ax):
    """Per-window test VRMSE: trunk vs persistence on rayleigh_benard (full test, 34,300 windows)."""
    import csv
    load = lambda f: {(int(r["traj"]), int(r["t0"])): float(r["vrmse"]) for r in csv.DictReader(open(f))}
    P = load(ROOT / "runs/persistence/rayleigh_benard_test.csv")
    T = load(ROOT / "runs/well2d/rayleigh_benard/cnext/mr30ryd_s1337_test.csv")
    k = [x for x in T if x in P]; x = np.array([P[i] for i in k]); y = np.array([T[i] for i in k])
    early = np.array([i[1] < 10 for i in k])
    ax.scatter(x[~early], y[~early], s=0.3, color="#6baed6", alpha=0.3, rasterized=True, label="later windows")
    ax.scatter(x[early], y[early], s=0.8, color="#d62728", alpha=0.6, rasterized=True, label="$t_0<10$ (onset)")
    ax.plot([1e-3, 1e2], [1e-3, 1e2], "k-", lw=0.5); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-2, 1.0); ax.set_ylim(1e-3, 30); ax.set_title("convection, per test window", fontsize=6.5); ax.set_xlabel("persistence VRMSE"); ax.set_ylabel("backbone VRMSE")
    ax.legend(fontsize=5.6, frameon=False, loc="upper left", markerscale=6)
    ax.text(0.97, 0.03, "(c)", transform=ax.transAxes, fontsize=8, weight="bold", ha="right")


def main():
    plt.rcParams.update({"font.size": 7.5})
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [1.35, 1.05, 0.85]})
    panel_a(ax[0]); panel_b(ax[1]); panel_c(ax[2])
    fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight"); print("wrote", OUT)


if __name__ == "__main__":
    main()
