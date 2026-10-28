"""Where the model fails, per benchmark, on TEST (frozen-trunk rows of the allocation study).

    .venv/bin/python paper/alloc/failure.py      # -> runs/alloc/failure.json, runs/alloc/failure.md

For each row and the register co-processor at its VALID-best strategy (seed-1 interpretation model):
  error budget (share of the trunk's weighted TEST error energy):
    captured                    removed by the co-processor
    predictability failure      on allocated tiles but not removed          = U - captured
    allocation-rule failure     reachable by an oracle with the same K     = U_oracle - U
    budget failure              beyond the K most-erroneous tiles           = 1 - U_oracle
  where the remaining error lives: by physical field (channel share of the metric), by trajectory time (window start),
  and by window (concentration: share of the error in the worst 10% of windows; does the gain go where the error is?).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import interp  # noqa: E402

ROOT = HERE.parents[1]
FIELDS = {"turbulent_radiative_layer_2D": ["density", "pressure", "vx", "vy"],
          "rayleigh_benard": ["buoyancy", "pressure", "vx", "vy"], "shear_flow": ["tracer", "pressure", "vx", "vy"],
          "MHD_64": ["density", "Bx", "By", "Bz", "vx", "vy", "vz"], "realpde_unet": ["u", "v"], "realpde_fno": ["u", "v"],
          "pelzer": ["temperature"]}


@torch.no_grad()
def analyse(bench: str) -> dict:
    run, m, st, d = interp.load(bench, "ryd")
    sp = run.sp["test"]
    cap, U, gain_c = [], [], []
    for i in range(0, sp.N, 256):
        rows = torch.arange(i, min(i + 256, sp.N), device="cuda")
        c, u, g = run.gains(m, sp, rows, st, 30)
        cap.append(c); U.append(u); gain_c.append(g)
    cap, U, gain_c = torch.cat(cap), torch.cat(U), torch.cat(gain_c)
    w = sp.etot                                                           # weighted error energy per window
    tot = w.sum()
    agg = lambda x: float((x * w).sum() / tot)                            # energy-weighted aggregate of per-window shares
    Uo = sp.ceil["oracle@30"].double()
    budget = {"captured": agg(cap), "predictability_failure": agg(U - cap), "allocation_rule_failure": agg((Uo - U).clamp_min(0)),
              "budget_failure": agg(1 - Uo)}
    # by field: share of the metric-weighted error per channel, before and after
    Ec = sp.E / sp.D; Ea = (sp.E - gain_c) / sp.D
    names = FIELDS.get(bench, [f"ch{i}" for i in range(sp.C)])
    by_field = {n: {"share_before": float(Ec[:, j].sum() / Ec.sum()), "removed_frac": float(1 - Ea[:, j].sum() / Ec[:, j].sum())}
                for j, n in enumerate(names)}
    # by window
    v0 = sp.metric_of(sp.E).cpu().numpy(); v1 = sp.metric_of(sp.E - gain_c).cpu().numpy()
    worst = v0 >= np.quantile(v0, 0.9)
    by_window = {"metric_before": float(v0.mean()), "metric_after": float(v1.mean()), "median_before": float(np.median(v0)),
                 "median_after": float(np.median(v1)), "worst10pct_share_of_metric": float(v0[worst].sum() / v0.sum()),
                 "worst10pct_rel_gain": float(1 - v1[worst].mean() / v0[worst].mean()),
                 "rest_rel_gain": float(1 - v1[~worst].mean() / v0[~worst].mean()),
                 "frac_windows_worse": float((v1 > v0 + 1e-12).mean()),
                 "corr_error_gain": float(np.corrcoef(v0, v0 - v1)[0, 1]) if np.std(v0 - v1) > 0 else 0.0}
    # by trajectory time (window start), where windows carry (traj, t0)
    wins = sp.meta.get("windows")
    by_time = None
    if wins and isinstance(wins[0], (list, tuple)):
        t0 = np.array([t for _, t in wins]); qs = np.quantile(t0, [0, 0.1, 0.5, 1.0])
        bins = [("early (first 10%)", t0 <= qs[1]), ("middle", (t0 > qs[1]) & (t0 <= qs[2])), ("late", t0 > qs[2])]
        by_time = {k: {"n": int(msk.sum()), "metric_before": float(v0[msk].mean()), "rel_gain": float(1 - v1[msk].mean() / v0[msk].mean())}
                   for k, msk in bins if msk.any()}
    return {"strategy": st, "budget": budget, "by_field": by_field, "by_window": by_window, "by_time": by_time,
            "U_dc": float(sp.ceil["dc"].double().mul(w).sum() / tot)}


def main():
    out = {b: analyse(b) for b in interp.ROWS}
    (ROOT / "runs" / "alloc" / "failure.json").write_text(json.dumps(out, indent=1))
    lines = ["| row | strategy | captured | predictability failure | allocation-rule failure | budget failure (beyond K=30) | worst 10% windows: share of metric / gain | rest gain | early-time gain |",
             "|---|---|---|---|---|---|---|---|---|"]
    for b, r in out.items():
        bu, bw = r["budget"], r["by_window"]
        early = (r["by_time"] or {}).get("early (first 10%)", {}).get("rel_gain")
        lines.append(f"| {b} | {r['strategy']} | {bu['captured']:.3f} | {bu['predictability_failure']:.3f} | {bu['allocation_rule_failure']:.3f} | "
                     f"{bu['budget_failure']:.3f} | {bw['worst10pct_share_of_metric']:.2f} / {100 * bw['worst10pct_rel_gain']:+.1f}% | "
                     f"{100 * bw['rest_rel_gain']:+.1f}% | {'' if early is None else f'{100 * early:+.1f}%'} |")
    lines.append("\nBy field (share of the metric-weighted error before; fraction removed):\n")
    for b, r in out.items():
        lines.append(f"- {b}: " + ", ".join(f"{n} {v['share_before']:.2f} ({100 * v['removed_frac']:+.1f}%)" for n, v in r["by_field"].items()))
    (ROOT / "runs" / "alloc" / "failure.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
