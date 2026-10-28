"""§37 routing analysis: per-variant (stratum) and pooled paired deltas with bootstrap 95% intervals.

Reference = the EAL-MD row's model and protocol decode: register q30-geo=3d, dihedral x8, multi-start greedy, bf16
(scores/q30-geo=3d__g8_bf16.json). Delta per instance = 100 * (cost_S - cost_ref) / cost_ref (relative cost change in %,
negative = better; ~ the change in gap-to-reference in pp divided by 1 + gap). Extra-compute strategies are also scored
as the union with the reference decode (per-instance min), which is what spending the extra compute on top of the
protocol decode gives. Pooled = equal-weight mean over the 16 variants (as the benchmark averages variants); the pooled
interval resamples instances within every stratum (stratified paired bootstrap, 10,000 draws, seed 0).

    .venv-vrp/bin/python paper/strat/routing/analyze.py   # -> runs/strat/routing/strategies.json
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
S = ROOT / "runs" / "strat" / "routing"
REF = ("q30-geo=3d", "g8_bf16")
B = 10_000
ATTR = {"O": "open_route", "TW": "time_windows", "L": "duration_limit", "B": "backhaul"}
EXTRA = {"g32sym": 4.0, "s8x4": 4.0, "s8x4t05": 4.0}          # decode passes relative to the reference


def load(arm: str, strat: str, offval: bool = False):
    p = S / "scores" / f"{arm}__{strat}{'__offval' if offval else ''}.json"
    return json.loads(p.read_text()) if p.exists() else None


def boot(deltas: dict[str, np.ndarray], rng) -> dict:
    """Per-variant mean and CI, plus the equal-weight pooled mean with a stratified paired bootstrap."""
    out, pooled = {}, np.zeros(B)
    for v, d in deltas.items():
        idx = rng.integers(0, len(d), (B, len(d)))
        bm = d[idx].mean(1); pooled += bm / len(deltas)
        out[v] = dict(mean=float(d.mean()), ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                      frac_better=float((d < -1e-9).mean()), frac_worse=float((d > 1e-9).mean()))
    m = float(np.mean([d.mean() for d in deltas.values()]))
    out["pooled"] = dict(mean=m, ci95=[float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5))])
    return out


def marginals(per_variant: dict, strata: dict) -> dict:
    """Equal-weight mean delta over the variants with / without each attribute (descriptive, no CI)."""
    res = {}
    for short, key in ATTR.items():
        w = [per_variant[v]["mean"] for v in per_variant if v != "pooled" and strata[v][key]]
        wo = [per_variant[v]["mean"] for v in per_variant if v != "pooled" and not strata[v][key]]
        res[short] = dict(with_=float(np.mean(w)), without=float(np.mean(wo)), n_with=len(w), n_without=len(wo))
    return res


def main() -> None:
    meta = json.loads((S / "strata.json").read_text())
    strata = {s["variant"]: s["attributes"] for s in meta["strata"]}
    ref = load(*REF)
    assert ref is not None, "reference scores missing"
    rc = {v: np.array(ref["variants"][v]["cost"]) for v in strata}
    rng = np.random.default_rng(0)
    comps = {}

    def add(name, costs, kind, note):
        d = {v: 100 * (costs[v] - rc[v]) / rc[v] for v in strata}
        r = boot(d, rng); comps[name] = dict(kind=kind, note=note, per_variant=r, marginals=marginals(r, strata))

    arm = REF[0]
    for st in ("g8_fp32", "g8_fp16"):
        x = load(arm, st)
        if x:
            add(f"register {st}", {v: np.array(x["variants"][v]["cost"]) for v in strata}, "protocol-faithful",
                "same decode, same compute, different inference precision")
    for st, passes in EXTRA.items():
        x = load(arm, st)
        if x:
            c = {v: np.array(x["variants"][v]["cost"]) for v in strata}
            add(f"register {st} alone", c, "extra test-time compute", f"{passes:g}x decode passes, replaces the x8 greedy decode")
            add(f"register {st} + protocol decode", {v: np.minimum(c[v], rc[v]) for v in strata}, "extra test-time compute",
                f"{passes + 1:g}x decode passes (union with the x8 greedy decode)")
    for other in ("vanilla", "q30-geo=3d-mlp"):
        for st in ("g8_bf16", "g8_fp32"):
            x = load(other, st)
            if x:
                add(f"{other} {st} (arm comparison)", {v: np.array(x["variants"][v]["cost"]) for v in strata},
                    "arm comparison", "a different fine-tuned arm (seed 1), same decode")
    arms = [load(a, "g8_bf16") for a in ("q30-geo=3d", "vanilla", "q30-geo=3d-mlp")]
    if all(arms):
        add("ensemble min(register, vanilla, MLP) g8_bf16",
            {v: np.min([np.array(x["variants"][v]["cost"]) for x in arms], 0) for v in strata},
            "extra test-time compute", "3 models x the x8 greedy decode (3x compute, 3x parameters)")
        add("ensemble min(register, MLP) g8_bf16",
            {v: np.min([np.array(x["variants"][v]["cost"]) for x in (arms[0], arms[2])], 0) for v in strata},
            "extra test-time compute", "2 models x the x8 greedy decode")
    for st in ("s8x4t05",):
        xs = [load(a, st) for a in ("vanilla", "q30-geo=3d-mlp")]
        for a, x in zip(("vanilla", "q30-geo=3d-mlp"), xs):
            base = load(a, "g8_bf16")
            if x and base:
                c = {v: np.minimum(np.array(x["variants"][v]["cost"]), np.array(base["variants"][v]["cost"])) for v in strata}
                add(f"{a} {st} + protocol decode (vs register g8_bf16)", c, "extra test-time compute", "arm's own union")
    # per-augmentation-copy analysis of the reference decode: how much does each extra dihedral copy buy?
    aug_curve = {}
    for k in (1, 2, 4, 8):
        c = {v: np.array(ref["variants"][v]["per_aug"])[:, :k].min(1) for v in strata}
        aug_curve[k] = float(np.mean([100 * (c[v] - rc[v]).mean() / rc[v].mean() for v in strata]))
    sym = load(arm, "g32sym")
    if sym:
        for k in (8, 16, 32):
            c = {v: np.minimum(np.array(sym["variants"][v]["per_aug"])[:, :k].min(1), rc[v]) for v in strata}
            aug_curve[f"dihedral8+sym{k}"] = float(np.mean([100 * (c[v] - rc[v]).mean() / rc[v].mean() for v in strata]))
    offval = load(arm, "g8_bf16", offval=True)
    res = dict(reference=f"{REF[0]} {REF[1]} (EAL-MD register row, protocol decode)", bootstrap=B, comparisons=comps,
               aug_curve_rel_cost_pct=aug_curve,
               reference_mean_cost={v: float(rc[v].mean()) for v in strata},
               official_val_harness_check=None if offval is None else dict(
                   note="RouteFinder's released val files (seed 4321, 128/variant), gap vs their PyVRP HGS 20 s costs",
                   gap_mean_over_variants=float(np.mean([offval["variants"][v]["gap_mean"] for v in strata if v in offval["variants"]])),
                   n_variants=sum(v in offval["variants"] for v in strata),
                   per_variant={v: offval["variants"][v]["gap_mean"] for v in strata if v in offval["variants"]}))
    (S / "strategies.json").write_text(json.dumps(res, indent=1))
    for name, c in comps.items():
        p = c["per_variant"]["pooled"]
        print(f"{name:62s} pooled {p['mean']:+.3f}% [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]")
    print("aug curve", aug_curve)
    if offval:
        print("official-val harness gap", res["official_val_harness_check"]["gap_mean_over_variants"])


if __name__ == "__main__":
    main()
