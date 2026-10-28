"""Appendix figure fig_spectral.pdf: what is left after the gate, per Fourier octave (runs/failure/*_spectral.json).

    .venv/bin/python paper/alloc/spectral_diag.py <row>   (x3)  &&  .venv/bin/python paper/alloc/fig_spectral.py

Top: error energy over true-change energy (solid) and its Wiener floor 1 - coherence (dashed), per field.
Bottom: lag-one pole of each octave's change, truth (solid) against the final model (dashed); -1/2 marks frame-to-frame
white content (x = s + e with e i.i.d. gives corr(D_{t+1}, D_t) = -1/2).
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/manuscript/iclr2027/fig_spectral.pdf"
ROWS = [("turbulent_radiative_layer_2D", "TRL2D", ("density", "pressure", "$v_x$", "$v_y$")),
        ("rayleigh_benard", "rayleigh_benard", ("buoyancy", "pressure", "$v_x$", "$v_y$")),
        ("shear_flow", "shear_flow", ("tracer", "pressure", "$v_x$", "$v_y$"))]
COL = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a")


def main():
    plt.rcParams.update({"font.size": 7})
    fig, ax = plt.subplots(2, 3, figsize=(7.0, 3.3), sharex=True)
    for j, (key, title, fields) in enumerate(ROWS):
        d = json.loads((ROOT / "runs/failure" / f"{key}_spectral.json").read_text()); b = np.arange(len(d["share"][0]))
        for c, (name, col) in enumerate(zip(fields, COL)):
            ax[0, j].plot(b, d["err_over_D"][c], "-o", color=col, ms=2.5, lw=1, label=name)
            ax[0, j].plot(b, d["wiener_floor"][c], "--", color=col, lw=0.8)
            ax[1, j].plot(b, d["pole_true"][c], "-o", color=col, ms=2.5, lw=1)
            ax[1, j].plot(b, d["pole_model"][c], "--", color=col, lw=0.8)
        ax[0, j].set_title(title, fontsize=7.5); ax[0, j].set_ylim(0, 1.25); ax[1, j].set_ylim(-0.6, 0.8)
        ax[1, j].axhline(-0.5, color="0.6", lw=0.5, ls=":"); ax[1, j].axhline(0, color="0.8", lw=0.4)
        ax[1, j].set_xticks(b, [f"$b_{i}$" for i in b]); ax[0, j].legend(fontsize=5.5, frameon=False, loc="upper left")
    ax[0, 0].set_ylabel("error / true-change energy\n(dashed: Wiener floor)"); ax[1, 0].set_ylabel("lag-one pole of the change\n(solid truth, dashed model)")
    fig.tight_layout(pad=0.3); fig.savefig(OUT, bbox_inches="tight"); print("wrote", OUT)


if __name__ == "__main__":
    main()
