"""Quantitative judge for the benchmark campaign.

    .venv/bin/python paper/judge/judge.py        -> runs/judge/scorecard.{json,md}

Per row and comparison (Q vs trunk, Q vs MLP, MLP vs trunk), seed-paired differences d are sign-corrected so
d > 0 means the first arm is better. Reported: relative gain, wins k/n, paired t and exact sign-test p, the
paired effect size dz, Glass's delta (gain in units of the baseline's seed SD), and the minimum detectable
effect (MDE) of a two-sided paired t-test at alpha 0.05 given the observed spread — the honest statement of
what n seeds can resolve. Across PRIMARY rows (one per benchmark, fixed a priori): a sign test on the direction
of the mean effect and a DerSimonian–Laird random-effects pool of the relative gains.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sources as S  # noqa: E402

OUT = S.ROOT / "runs" / "judge"
PAIRS = (("q", "trunk"), ("q", "mlp"), ("mlp", "trunk"))
TIE_REL = 0.5  # |relative gain| below this (%) is called a tie regardless of p


def compare(a: dict, b: dict, lower: bool) -> dict | None:
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    x, y = np.array([a[s] for s in seeds]), np.array([b[s] for s in seeds])
    d = (y - x) if lower else (x - y)                      # > 0: a better than b
    n, mb = len(d), abs(float(y.mean()))
    r = dict(n=n, mean_a=float(x.mean()), mean_b=float(y.mean()), mean_d=float(d.mean()),
             rel_pct=float(d.mean() / mb * 100), wins=int((d > 0).sum()),
             sign_p=float(stats.binomtest(int((d > 0).sum()), int((d != 0).sum()) or 1).pvalue) if n else None)
    if n >= 2:
        sd = float(d.std(ddof=1)); sdb = float(y.std(ddof=1))
        tcrit = float(stats.t.ppf(0.975, n - 1))
        r.update(sd_d=sd, se_rel=float(sd / math.sqrt(n) / mb * 100),
                 t=float(d.mean() / (sd / math.sqrt(n))) if sd > 0 else math.inf,
                 p=float(stats.ttest_rel(x, y).pvalue) if sd > 0 else 0.0,
                 dz=float(d.mean() / sd) if sd > 0 else math.inf,
                 glass=float(d.mean() / sdb) if sdb > 0 else math.inf,
                 mde_rel_pct=float(tcrit * sd / math.sqrt(n) / mb * 100))
    r["verdict"] = verdict(r)
    return r


def verdict(r: dict) -> str:
    if abs(r["rel_pct"]) < TIE_REL:
        return "tie"
    better = r["rel_pct"] > 0
    sig = r.get("p") is not None and r["p"] < 0.05
    return ("better" if better else "worse") + (" (p<.05)" if sig else " (n.s.)")


def dersimonian_laird(effects: list[float], ses: list[float]) -> dict:
    y, v = np.array(effects), np.array(ses) ** 2
    w = 1 / v; fe = float((w * y).sum() / w.sum())
    q = float((w * (y - fe) ** 2).sum()); k = len(y)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if k > 1 else 0.0
    wr = 1 / (v + tau2); re = float((wr * y).sum() / wr.sum()); se = float(math.sqrt(1 / wr.sum()))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return dict(k=k, fixed=fe, random=re, ci=(re - 1.96 * se, re + 1.96 * se), tau2=tau2, I2=i2,
                p=float(2 * (1 - stats.norm.cdf(abs(re / se)))))


def _loo(cs: list[dict], names: list[str]) -> dict:
    """Random-effects pool with each row left out: exposes a pool carried by one low-variance row."""
    out = {}
    for i, name in enumerate(names):
        rest = [(c["rel_pct"], c["se_rel"]) for j, c in enumerate(cs) if j != i and c.get("se_rel")]
        if len(rest) > 1:
            pl = dersimonian_laird(*zip(*rest)); out[name] = (pl["random"], pl["ci"][0], pl["ci"][1])
    return out


def sota_standing(mean_q: float, ref: float | None, lower: bool) -> float | None:
    """Relative position of the Q arm vs the published number, % (> 0 = better than the reference)."""
    if ref is None:
        return None
    return (ref - mean_q) / ref * 100 if lower else (mean_q - ref) / ref * 100


def judge() -> dict:
    rows = []
    for name, fn, lower, ref, ref_name, tier, note in S.ROWS:
        arms = fn()
        if not all(arms.get(k) for k in ("trunk", "mlp", "q")):
            continue                                 # row has no complete data yet (e.g. replication pending)
        row = dict(name=name, tier=tier, lower_is_better=lower, reference=ref, reference_name=ref_name, note=note,
                   means={k: (float(np.mean(list(v.values()))) if v else None) for k, v in arms.items()},
                   sds={k: (float(np.std(list(v.values()), ddof=1)) if len(v) > 1 else None) for k, v in arms.items()})
        if name.startswith("Sophelio"):              # hpo() stores -S; report S itself
            row["means"] = {k: (abs(m) if m is not None else None) for k, m in row["means"].items()}
            arms = {k: {s: abs(x) for s, x in v.items()} for k, v in arms.items()}
        if "tuned" in name:
            row["protocol"] = S.hpo_protocol({"TORAX": "torax", "Pelzer": "pelzer", "Sophelio": "sophelio"}[name.split("-")[0].split(" ")[0]])
        for a, b in PAIRS:
            if arms.get(a) and arms.get(b):
                row[f"{a}_vs_{b}"] = compare(arms[a], arms[b], lower)
        best = min if lower else max
        row["best_arm"] = best((k for k, m in row["means"].items() if m is not None), key=lambda k: row["means"][k])
        row["q_vs_reference_pct"] = sota_standing(row["means"]["q"], ref, lower)
        row["trunk_vs_reference_pct"] = sota_standing(row["means"]["trunk"], ref, lower)
        rows.append(row)
    return dict(rows=rows, meta=_meta(rows, "primary"), meta_replication=_meta(rows, "replication"))


def _meta(rows: list[dict], tier: str) -> dict:
    meta = {}
    prim = [r for r in rows if r["tier"] == tier]
    if len(prim) < 2:
        return meta
    for a, b in PAIRS:
        key = f"{a}_vs_{b}"
        cs = [r[key] for r in prim if r.get(key)]
        pos = sum(c["rel_pct"] > TIE_REL for c in cs); neg = sum(c["rel_pct"] < -TIE_REL for c in cs)
        pooled = [(c["rel_pct"], c["se_rel"]) for c in cs if c.get("se_rel")]
        rel = np.array([c["rel_pct"] for c in cs])
        # rows as the units: robust to a row whose near-zero variance (identical arms) would dominate an IV pool
        unw = (float(rel.mean()), *(float(x) for x in stats.t.interval(0.95, len(rel) - 1, loc=rel.mean(),
                                                                          scale=stats.sem(rel)))) if len(rel) > 2 else None
        meta[key] = dict(rows=len(cs), a_better_rows=pos, a_worse_rows=neg,
                         sign_p=float(stats.binomtest(pos, pos + neg).pvalue) if pos + neg else 1.0, unweighted=unw,
                         significant_better=sum(c["verdict"] == "better (p<.05)" for c in cs),
                         significant_worse=sum(c["verdict"] == "worse (p<.05)" for c in cs),
                         ties=sum(c["verdict"] == "tie" for c in cs),
                         median_rel_pct=float(np.median([c["rel_pct"] for c in cs])),
                         median_mde_rel_pct=float(np.median([c["mde_rel_pct"] for c in cs if "mde_rel_pct" in c])),
                         pool=dersimonian_laird(*zip(*pooled)) if len(pooled) > 1 else None,
                         leave_one_out=_loo(cs, [r["name"] for r in prim if r.get(key)]))
    return meta


def fmt(c: dict | None) -> str:
    if not c:
        return "—"
    p = f"p={c['p']:.2f}" if "p" in c else "n=1"
    return f"{c['rel_pct']:+.1f}% ({c['wins']}/{c['n']}, {p})"


def markdown(res: dict) -> str:
    L = ["| Row | tier | trunk | MLP | Q | Q vs trunk | Q vs MLP | MLP vs trunk | MDE (Q vs trunk) | Q vs ref |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        m, sd = r["means"], r["sds"]
        cell = lambda k: f"{m[k]:.4f}" + (f" ± {sd[k]:.4f}" if sd.get(k) else "") if m.get(k) is not None else "—"
        qt = r.get("q_vs_trunk") or {}
        ref = (f"{r['q_vs_reference_pct']:+.1f}% (trunk {r['trunk_vs_reference_pct']:+.1f}%)"
               if r["q_vs_reference_pct"] is not None else "—")
        name = r["name"] + (f" [{r['protocol']}]" if r.get("protocol", "").startswith("MIXED") else "")
        L.append(f"| {name} | {r['tier']} | {cell('trunk')} | {cell('mlp')} | {cell('q')} | {fmt(r.get('q_vs_trunk'))} | "
                 f"{fmt(r.get('q_vs_mlp'))} | {fmt(r.get('mlp_vs_trunk'))} | "
                 f"{('±%.1f%%' % qt['mde_rel_pct']) if 'mde_rel_pct' in qt else '—'} | {ref} |")
    for title, key in (("primary rows", "meta"), ("PRE-REGISTERED replication (new units only)", "meta_replication")):
        if res.get(key):
            L += _meta_table(res[key], title)
    return "\n".join(L)


def _meta_table(meta: dict, title: str) -> list[str]:
    L = ["", f"| Across {title} | better / worse / tie (rows) | sign-test p (ties excl.) | sig. better | sig. worse | median gain | unweighted mean over rows [95% t-CI] | random-effects pooled gain [95% CI], I² |",
         "|---|---|---|---|---|---|---|---|"]
    for k, v in meta.items():
        pl = v["pool"]
        pool = f"{pl['random']:+.2f}% [{pl['ci'][0]:+.2f}, {pl['ci'][1]:+.2f}], I²={pl['I2']:.0%}" if pl else "—"
        u = v.get("unweighted"); unw = f"{u[0]:+.2f}% [{u[1]:+.2f}, {u[2]:+.2f}]" if u else "—"
        L.append(f"| {k.replace('_vs_', ' vs ')} | {v['a_better_rows']} / {v['a_worse_rows']} / {v['ties']} | {v['sign_p']:.2f} | "
                 f"{v['significant_better']} | {v['significant_worse']} | {v['median_rel_pct']:+.2f}% | {unw} | {pool} |")
    L += ["", f"Leave-one-out pool, {title} (row left out → pooled gain [95% CI]):", ""]
    for k, v in meta.items():
        loo = ", ".join(f"−{n.split(' ')[0]}: {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]" for n, (m, lo, hi) in v["leave_one_out"].items())
        L.append(f"- **{k.replace('_vs_', ' vs ')}**: {loo}")
    return L


if __name__ == "__main__":
    res = judge()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scorecard.json").write_text(json.dumps(res, indent=1, default=float))
    md = markdown(res); (OUT / "scorecard.md").write_text(md + "\n"); print(md)
