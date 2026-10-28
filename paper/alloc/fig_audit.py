"""Audit forest plot (fig_audit.pdf): register vs matched MLP and vs trunk per primary row, seed-paired relative gain
(%, + = register better) with 95% t-intervals from runs/judge/scorecard.json; intervals clipped at +/-40%.

    .venv/bin/python paper/alloc/fig_audit.py
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
d = json.loads((ROOT / "runs/judge/scorecard.json").read_text())
rows = [r for r in d["rows"] if r["tier"] == "primary"]
plt.rcParams.update({"font.size": 7})
fig, ax = plt.subplots(figsize=(3.4, 2.5))
L = 40
for i, r in enumerate(rows[::-1]):
    for off, key, col, lab in ((0.14, "q_vs_mlp", "#d95f02", "vs matched classical MLP"), (-0.14, "q_vs_trunk", "#3182bd", "vs pretrained backbone alone")):
        c = r.get(key)
        if not c:
            continue
        h = stats.t.ppf(0.975, max(c["n"] - 1, 1)) * c["se_rel"]
        lo, hi = max(c["rel_pct"] - h, -L), min(c["rel_pct"] + h, L)
        ax.plot([lo, hi], [i + off] * 2, color=col, lw=1.0)
        ax.plot(c["rel_pct"], i + off, "o", color=col, ms=3, label=lab if i == 0 else None)
        for x, m in ((c["rel_pct"] - h, "<"), (c["rel_pct"] + h, ">")):
            if abs(x) > L:
                ax.plot(L if m == ">" else -L, i + off, m, color=col, ms=3)
ax.axvline(0, color="k", lw=0.6)
NICE = {"TRL2D": "Turbulence (TRL2D)", "MHD_64": "MHD, 3-D (MHD_64)", "rayleigh_benard": "Convection (RB)",
        "shear_flow": "Shear flow", "TORAX": "Tokamak transport (TORAX)", "Pelzer": "Geothermal plume (1HP)",
        "Sophelio": "Fusion equilibrium", "TokaMark": "Tokamak signals (TokaMark)"}
names = [next((v for k, v in NICE.items() if r["name"].startswith(k)), r["name"]) for r in rows[::-1]]
ax.set_yticks(range(len(rows)), names, fontsize=6.5); ax.set_xlim(-L - 3, L + 3)
ax.set_xlabel("error reduction by the register (%)\n(right of 0 = register better; seed-paired 95% CI)", fontsize=6.5)
ax.legend(fontsize=5.8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
fig.tight_layout(); fig.savefig(ROOT / "paper/manuscript/iclr2027/fig_audit.pdf", bbox_inches="tight"); print("ok")
