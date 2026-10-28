"""How the qubit budget fills the error spectrum (doc 35 §13): for each budget K, the per-octave error left after the
gate split into what the qubit-routed modes captured, what the classical head captured, and what is left (register
arm, seed mean; bands_test of each run). Writes paper/manuscript/iclr2027/fig_spectral_fill.pdf.
    .venv/bin/python paper/alloc/fig_spectral_fill.py            # vanilla trunk (doc 35 §19), the paper's figure
"""
import json, pathlib, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]; RES = ROOT / "runs/alloc/adrc_results"
ROWS = [("shear_flow__gate_vanilla_s1337", "shear flow"), ("rayleigh_benard__gatest_vanilla_s1337", "convection (rayleigh_benard$^\\dagger$)"),
        ("turbulent_radiative_layer_2D__gate_vanilla_s1337", "turbulence (TRL2D)"), ("MHD_64__gate_vanilla_s1337", "MHD (MHD_64)")]


def main():
    plt.rcParams.update({"font.size": 6.5}); fig, ax = plt.subplots(2, 2, figsize=(7.0, 3.5)); ax = ax.flatten()
    for j, (key, title) in enumerate(ROWS):
        x0, ticks, labels = 0, [], []
        for k in (5, 10, 20, 30, 50):
            sfx = f"_k{k}" if k != 30 else ""
            if not list(RES.glob(f"{key}_adrc_ryd_gate_rs{sfx}_dyn_s[0-9].json")):
                continue
            rr = [json.loads(f.read_text())["bands_test"] for f in sorted(RES.glob(f"{key}_adrc_ryd_gate_rs{sfx}_dyn_s[0-9].json"))]
            full = np.mean([r["full"] for r in rr], 0); tot = full.sum(); nb = len(full); xs = x0 + np.arange(nb)
            q = np.clip(np.mean([r["captured_qubit"] for r in rr], 0), 0, None) / tot
            c = np.clip(np.mean([r["captured_classical"] for r in rr], 0), 0, None) / tot
            left = np.clip(full / tot - q - c, 0, None)
            ax[j].bar(xs, q, 0.85, color="#d95f02", label="qubit-routed modes" if k == 5 and j == 0 else None)
            ax[j].bar(xs, c, 0.85, bottom=q, color="#1b9e77", label="classical head" if k == 5 and j == 0 else None)
            ax[j].bar(xs, left, 0.85, bottom=q + c, color="#d9d9d9", label="left" if k == 5 and j == 0 else None)
            ax[j].text(x0 + nb / 2 - 0.5, 1.02, f"$K={k}$", ha="center", va="bottom", transform=ax[j].get_xaxis_transform(), fontsize=6.5)
            ticks += list(xs); labels += [f"$b_{i}$" for i in range(nb)]; x0 += nb + 1
        ax[j].set_xticks(ticks, labels, fontsize=5); ax[j].set_title(title, fontsize=7.5, pad=12)
        if j % 2 == 0:
            ax[j].set_ylabel("share of error after the gate"); 
    ax[0].legend(fontsize=5.5, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3); out = ROOT / "paper/manuscript/iclr2027/fig_spectral_fill.pdf"; fig.savefig(out, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
