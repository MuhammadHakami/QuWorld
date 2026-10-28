"""Summaries for the allocation-strategy study (33-allocation-strategy-study.md), implementing its decision rules.

    .venv/bin/python paper/alloc/analyze.py best <bench>     # VALID-best tile strategy of the ryd core (for phase C)
    .venv/bin/python paper/alloc/analyze.py table            # -> runs/alloc/summary.md, summary.json
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "runs" / "alloc" / "results"
TILE = ("static", "proxy", "learned", "gain", "random")
STRATS = ("pooled",) + TILE
NAMES = {"turbulent_radiative_layer_2D": "TRL2D", "rayleigh_benard": "rayleigh_benard", "shear_flow": "shear_flow",
         "planetswe": "planetswe", "realpde_unet": "RealPDE (U-Net)", "realpde_fno": "RealPDE (FNO)"}


def load():
    R = defaultdict(list)
    for f in RES.glob("*/*.json"):
        d = json.loads(f.read_text())
        if f.stem != f"{d['strategy']}_{d['core']}_K{d['K']}_s{d['seed']}":
            continue                                             # tagged probes / tuning runs are not study cells
        R[(d["bench"], d["strategy"], d["core"], d["K"])].append(d)
    return R


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def best(bench: str) -> str:
    R = load()
    cand = {s: mean([d["valid"] for d in R[(bench, s, "ryd", 30)]]) for s in TILE if R[(bench, s, "ryd", 30)]}
    return min(cand, key=cand.get)


def dl_pool(effects, variances):
    """DerSimonian-Laird random-effects pooled effect and 95% CI."""
    e, v = np.asarray(effects), np.maximum(np.asarray(variances), 1e-10)
    w = 1 / v; fe = (w * e).sum() / w.sum()
    Q = (w * (e - fe) ** 2).sum(); df = len(e) - 1
    tau2 = max(0.0, (Q - df) / (w.sum() - (w ** 2).sum() / w.sum())) if df > 0 else 0.0
    ws = 1 / (v + tau2); re = (ws * e).sum() / ws.sum(); se = math.sqrt(1 / ws.sum())
    I2 = max(0.0, (Q - df) / Q) if Q > 0 and df > 0 else 0.0
    return re, re - 1.96 * se, re + 1.96 * se, I2


def paired(R, bench, strat, qa, qb, K=30):
    """Relative gain of core qa over core qb (1 - m_a / m_b), paired by seed."""
    A = {d["seed"]: d["test"] for d in R[(bench, strat, qa, K)]}; B = {d["seed"]: d["test"] for d in R[(bench, strat, qb, K)]}
    s = sorted(set(A) & set(B))
    if not s:
        return None
    g = np.array([1 - A[k] / B[k] for k in s])
    p = stats.ttest_1samp(g, 0).pvalue if len(g) > 1 and g.std() > 0 else float("nan")
    return {"gain": g.mean(), "var": g.var(ddof=1) / len(g) if len(g) > 1 else 1e-6, "wins": int((g > 0).sum()), "n": len(g), "p": p}


def table():
    R = load()
    benches = sorted({k[0] for k in R}, key=lambda b: list(NAMES).index(b) if b in NAMES else 99)
    out, js = [], {"cells": [], "hq": {}, "hu": {}}
    out.append("# Allocation-strategy study: frozen trunk + routed co-processor (K = 30 atoms unless stated)\n")
    out.append("Test metric relative gain vs the frozen trunk (mean over seeds); captured = share of the trunk's weighted "
               "error energy removed; U = utilization ceiling of the allocation (share of error energy on the allocated "
               "components; captured <= U by `residual_after_band_removal`).\n")
    for b in benches:
        trunk = mean([d["trunk_test"] for k, v in R.items() if k[0] == b for d in v])
        out.append(f"\n## {NAMES.get(b, b)} (trunk TEST {trunk:.5f})\n")
        out.append("| strategy | U | lin gain / captured | mlp gain / captured | ryd gain / captured | ryd vs mlp |")
        out.append("|---|---|---|---|---|---|")
        for s in STRATS + ("oracle",):
            row, U = [], None
            for c in ("lin", "mlp", "ryd"):
                v = R[(b, s, c, 30)]
                if not v:
                    row.append("—"); continue
                U = mean([d["test_U"] for d in v])
                g, cap = mean([d["test_rel_gain"] for d in v]), mean([d["test_captured"] for d in v])
                row.append(f"{100 * g:+.2f}% / {cap:.4f} (n={len(v)})")
                js["cells"].append({"bench": b, "strategy": s, "core": c, "U": U, "gain": g, "captured": cap, "n": len(v)})
            pq = paired(R, b, s, "ryd", "mlp")
            pr = f"{100 * pq['gain']:+.2f}% ({pq['wins']}/{pq['n']}, p={pq['p']:.2f})" if pq else "—"
            out.append(f"| {s}{' (diagnostic)' if s == 'oracle' else ''} | {U if U is None else f'{U:.4f}'} | " + " | ".join(row) + f" | {pr} |")
        ks = [(K, s, c, mean([d["test_captured"] for d in R[(b, s, c, K)]]), mean([d["test_U"] for d in R[(b, s, c, K)]]))
              for s in ("pooled", "proxy") for c in ("mlp", "ryd") for K in (10, 30, 100, 300) if R[(b, s, c, K)]]
        if ks:
            out.append("\nK sweep (captured / U):\n")
            out.append("| strategy, core | K=10 | K=30 | K=100 | K=300 |"); out.append("|---|---|---|---|---|")
            for s in ("pooled", "proxy"):
                for c in ("mlp", "ryd"):
                    cells = {K: f"{cap:.4f} / {U:.4f}" for K, s2, c2, cap, U in ks if s2 == s and c2 == c}
                    out.append(f"| {s}, {c} | " + " | ".join(cells.get(K, "—") for K in (10, 30, 100, 300)) + " |")
        ctl = [(c, R[(b, best(b), c, 30)]) for c in ("gutz", "ryd2d", "mlpX4")] if any(R[(b, s, "ryd", 30)] for s in TILE) else []
        if any(v for _, v in ctl):
            bs = best(b)
            out.append(f"\nControls at the VALID-best tile strategy of the register ({bs}):\n")
            for c, v in ctl:
                if v:
                    pq = paired(R, b, bs, "ryd", c)
                    out.append(f"- ryd vs {c}: {100 * pq['gain']:+.2f}% ({pq['wins']}/{pq['n']}, p={pq['p']:.2f})")
            pq = paired(R, b, bs, "hyb", "mlp2")
            if pq:
                out.append(f"- division of labor, hyb (register + MLP) vs mlp2 (two MLPs): {100 * pq['gain']:+.2f}% "
                           f"({pq['wins']}/{pq['n']}, p={pq['p']:.2f})")

    # H_U: utilization governs gain -----------------------------------------------------------------------------------
    out.append("\n# H_U: utilization governs the gain\n")
    for c in ("lin", "mlp", "ryd"):
        cells = [x for x in js["cells"] if x["core"] == c and x["strategy"] in STRATS]
        if len(cells) < 4:
            continue
        rho = stats.spearmanr([x["U"] for x in cells], [x["captured"] for x in cells]).statistic
        bl = sorted({x["bench"] for x in cells}); rng = np.random.default_rng(0); boots = []
        for _ in range(2000):
            pick = rng.choice(bl, len(bl)); cc = [x for bb in pick for x in cells if x["bench"] == bb]
            if len({x["U"] for x in cc}) > 2:
                boots.append(stats.spearmanr([x["U"] for x in cc], [x["captured"] for x in cc]).statistic)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2
        js["hu"][c] = {"rho": rho, "ci": [lo, hi], "cells": len(cells)}
        out.append(f"- {c}: Spearman(U, captured) over {len(cells)} (bench x strategy) cells = {rho:.2f} "
                   f"[bench-bootstrap 95% CI {lo:.2f}, {hi:.2f}]")
    for c in ("mlp", "ryd"):
        wins = []
        for b in benches:
            pooled = mean([d["test"] for d in R[(b, "pooled", c, 30)]])
            routed = min([mean([d["test"] for d in R[(b, s, c, 30)]]) for s in ("proxy", "learned", "gain") if R[(b, s, c, 30)]] or [float("nan")])
            gain_r = mean([d["test"] for d in R[(b, "gain", c, 30)]]); energy_r = min(mean([d["test"] for d in R[(b, s, c, 30)]]) for s in ("proxy", "learned"))
            wins.append((b, routed <= pooled, gain_r <= energy_r))
        out.append(f"- {c}: causal routing <= pooled on {sum(w[1] for w in wins)}/{len(wins)} benchmarks (P2); "
                   f"gain routing <= energy routing on {sum(w[2] for w in wins)}/{len(wins)} (P4)")

    # H_Q: quantum core at each bench's VALID-best strategy ----------------------------------------------------------
    out.append("\n# H_Q: register vs matched MLP at each benchmark's VALID-best tile strategy\n")
    for other in ("mlp", "lin"):
        eff, var, rows = [], [], []
        for b in benches:
            if not any(R[(b, s, "ryd", 30)] for s in TILE):
                continue
            bs = best(b); pq = paired(R, b, bs, "ryd", other)
            if pq:
                eff.append(pq["gain"]); var.append(pq["var"]); rows.append((b, bs, pq))
        if not eff:
            continue
        better, worse = sum(e > 0 for e in eff), sum(e < 0 for e in eff)
        sp = stats.binomtest(better, better + worse).pvalue if better + worse else float("nan")
        re, lo, hi, I2 = dl_pool(eff, var)
        js["hq"][other] = {"better": better, "worse": worse, "sign_p": sp, "re": re, "ci": [lo, hi], "I2": I2}
        out.append(f"- ryd vs {other}: better/worse {better}/{worse} (sign p={sp:.2f}); random-effects "
                   f"{100 * re:+.3f}% [{100 * lo:+.3f}, {100 * hi:+.3f}], I2={100 * I2:.0f}%")
        for b, bs, pq in rows:
            out.append(f"  - {NAMES.get(b, b)} ({bs}): {100 * pq['gain']:+.3f}% ({pq['wins']}/{pq['n']})")
    (ROOT / "runs" / "alloc" / "summary.md").write_text("\n".join(out) + "\n")
    (ROOT / "runs" / "alloc" / "summary.json").write_text(json.dumps(js, indent=1, default=float))
    print("\n".join(out))


def latex():
    """Paper table rows: pooled (DC) ceiling and gain; the register's VALID-best tile strategy with its ceiling U,
    captured share and test gain; register vs matched MLP at that strategy (seed-paired)."""
    R = load()
    names = {**NAMES, "pelzer": "Heat plume (1HP\\_NN U-Net)", "MHD_64": "MHD\\_64 (CNextU-net)$^\\dagger$",
             "turbulent_radiative_layer_2D": "TRL2D (CNextU-net)", "rayleigh_benard": "rayleigh\\_benard (CNextU-net)",
             "shear_flow": "shear\\_flow (CNextU-net)$^\\dagger$", "realpde_unet": "RealPDEBench (U-Net)",
             "realpde_fno": "RealPDEBench (FNO)"}
    order = ["turbulent_radiative_layer_2D", "rayleigh_benard", "shear_flow", "MHD_64", "realpde_unet", "realpde_fno", "pelzer"]
    for b in order:
        if not any(R[(b, s, "ryd", 30)] for s in TILE):
            continue
        cell = lambda s: (mean([d["test_U"] for d in R[(b, s, "ryd", 30)]]), mean([d["test_captured"] for d in R[(b, s, "ryd", 30)]]),
                          mean([d["test_rel_gain"] for d in R[(b, s, "ryd", 30)]]))
        Ud, cd, gd = cell("pooled"); bs = best(b); Ub, cb, gb = cell(bs)
        pq = paired(R, b, bs, "ryd", "mlp")
        q = f"${100 * pq['gain']:+.2f}\\%$ ({pq['wins']}/{pq['n']})" if pq else "---"
        print(f"{names[b]} & {Ud:.3f} & ${100 * gd:+.2f}\\%$ & {bs} & {Ub:.3f} & {max(cb, 0):.3f} & ${100 * gb:+.2f}\\%$ & {q} \\\\")


if __name__ == "__main__":
    if sys.argv[1] == "best":
        print(best(sys.argv[2]))
    elif sys.argv[1] == "latex":
        latex()
    else:
        table()
