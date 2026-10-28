"""Doc 35 §17 summary: noise-conditioned configuration (FGN-style) vs the deterministic hybrid, register vs matched MLP.

    .venv/bin/python paper/alloc/fgn_summary.py        # writes runs/alloc/fgn_summary.json and prints a table

Per row, arm and seed: the TEST fair energy score (M = 16) of the probabilistic hybrid, its ensemble-mean VRMSE and
spread-skill ratio, and the deterministic hybrid's VRMSE (§13 runs, same K = 30 configuration and seed), which is the
energy score of a point forecast. Register vs MLP is paired by seed (§6 standard: 3 seeds, sign-consistent).
"""
from __future__ import annotations

import json
import pathlib
import statistics as st

from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "runs/alloc/adrc_results"
ROWS = [("turbulent_radiative_layer_2D__gate_vanilla_s1337", "TRL2D"), ("shear_flow__gate_vanilla_s1337", "shear_flow"),
        ("rayleigh_benard__gatest_vanilla_s1337", "rayleigh_benard"), ("MHD_64__gate_vanilla_s1337", "MHD_64")]   # doc 35 §17 amendment
ARMS = ("adrc_ryd", "adrc_mlp")


def load(key, arm, seed, fgn, sfx=None):
    """sfx overrides the suffix: '_fgnt' (§17 (d), VALID-tuned spread scale) or '_dressr' (§17 (e), the primary control)."""
    f = RES / f"{key}_{arm}_gate_rs_dyn{sfx if sfx is not None else ('_fgn' if fgn else '')}_s{seed}.json"
    return json.loads(f.read_text()) if f.exists() else None


def controls(key, arm):
    """Seed means of the §17 (d) and (e) TEST energy scores, with their seeds; None where no run exists yet."""
    t = [(s, load(key, arm, s, False, "_fgnt")) for s in (1, 2, 3)]; t = [(s, d["tau"]) for s, d in t if d]
    e = [(s, load(key, arm, s, False, "_dressr")) for s in (1, 2, 3)]; e = [(s, d["dress"]["all"]) for s, d in e if d]
    return {"tau": {"seeds": [s for s, _ in t], "es": [d["prob_test"]["es"] for _, d in t], "tau": [d["tau"] for _, d in t]} if t else None,
            "dress_rel": {"seeds": [s for s, _ in e], "es": [d["test_es"] for _, d in e], "scale": [d["scale"] for _, d in e]} if e else None}


def main():
    out = {}
    print(f"{'row':16s} {'arm':9s} {'det VRMSE':>10s} {'ES (M=16)':>10s} {'ES/det':>7s} {'ens-mean':>9s} {'mean/det':>8s} {'SSR':>5s}  n")
    for key, name in ROWS:
        out[name] = {}
        for arm in ARMS:
            pairs = [(s, load(key, arm, s, False), load(key, arm, s, True)) for s in (1, 2, 3)]
            pairs = [(s, d, p) for s, d, p in pairs if d and p]
            if not pairs:
                continue
            det = [d["test"] for _, d, _ in pairs]; es = [p["prob_test"]["es"] for _, _, p in pairs]
            em = [p["prob_test"]["vrmse_mean"] for _, _, p in pairs]; ssr = [p["prob_test"]["ssr"] for _, _, p in pairs]
            out[name][arm] = {"seeds": [s for s, _, _ in pairs], "det": det, "es": es, "ens_mean": em, "ssr": ssr,
                              "es_over_det": [e / d for e, d in zip(es, det)], "mean_over_det": [m / d for m, d in zip(em, det)],
                              **controls(key, arm)}
            print(f"{name:16s} {arm:9s} {st.mean(det):10.5f} {st.mean(es):10.5f} {st.mean(es) / st.mean(det):7.3f} "
                  f"{st.mean(em):9.5f} {st.mean(em) / st.mean(det):8.3f} {st.mean(ssr):5.2f}  {len(pairs)}")
        r, m = out[name].get("adrc_ryd"), out[name].get("adrc_mlp")
        if r and m:
            common = sorted(set(r["seeds"]) & set(m["seeds"]))
            er = [r["es"][r["seeds"].index(s)] for s in common]; emlp = [m["es"][m["seeds"].index(s)] for s in common]
            wins = sum(a < b for a, b in zip(er, emlp))
            p = stats.ttest_rel(er, emlp).pvalue if len(common) >= 2 else None
            out[name]["reg_vs_mlp_es"] = {"seeds": common, "rel": 1 - st.mean(er) / st.mean(emlp), "wins": wins, "p": p}
            if r.get("tau") and m.get("tau"):
                c2 = sorted(set(r["tau"]["seeds"]) & set(m["tau"]["seeds"]))
                tr_ = [r["tau"]["es"][r["tau"]["seeds"].index(s)] for s in c2]; tm = [m["tau"]["es"][m["tau"]["seeds"].index(s)] for s in c2]
                out[name]["reg_vs_mlp_es_tau"] = {"seeds": c2, "rel": 1 - st.mean(tr_) / st.mean(tm), "wins": sum(a < b for a, b in zip(tr_, tm)),
                                                  "p": stats.ttest_rel(tr_, tm).pvalue if len(c2) >= 2 else None}
            print(f"{'':16s} register vs MLP on ES: {100 * (1 - st.mean(er) / st.mean(emlp)):+.2f}% ({wins}/{len(common)} seeds"
                  + (f", paired p={p:.3f})" if p is not None else ")"))
    (ROOT / "runs/alloc/fgn_summary.json").write_text(json.dumps(out, indent=1))
    write_table(out)


def write_table(out):
    """LaTeX table for the paper (app:fgn): seed means; ES relative to the deterministic VRMSE of the same arm and seeds."""
    names = {"TRL2D": "TRL2D", "shear_flow": "shear\\_flow", "rayleigh_benard": "rayleigh\\_benard$^\\dagger$", "MHD_64": "MHD\\_64"}
    lines = ["\\begin{tabular}{llcccccccc}", "\\toprule",
             "Row & core & seeds & deterministic VRMSE & energy score & ES / det. & ensemble-mean VRMSE & spread--skill"
             " & ES, tuned $\\tau$ & dressed (e) \\\\", "\\midrule"]
    cell = lambda c: f"{st.mean(c['es']):.4f} ({len(c['seeds'])})" if c else "---"
    for name, arms in out.items():
        first = True
        for arm, lab in (("adrc_ryd", "register"), ("adrc_mlp", "MLP")):
            r = arms.get(arm)
            if not r:
                continue
            det, es, em = st.mean(r["det"]), st.mean(r["es"]), st.mean(r["ens_mean"])
            lines.append(f"{names[name] if first else ''} & {lab} & {len(r['seeds'])} & {det:.4f} & {es:.4f} & "
                         f"{es / det:.3f} & {em:.4f} & {st.mean(r['ssr']):.2f} & {cell(r.get('tau'))} & {cell(r.get('dress_rel'))} \\\\"); first = False
        rv = arms.get("reg_vs_mlp_es")
        if rv and len(rv["seeds"]) >= 2:
            rt = arms.get("reg_vs_mlp_es_tau"); tail = ""
            if rt and len(rt["seeds"]) >= 2:
                tail = f"; tuned $\\tau$: {100 * rt['rel']:+.1f}\\% ({rt['wins']}/{len(rt['seeds'])}, $p={rt['p']:.2f}$)"
            lines.append(f"& \\multicolumn{{9}}{{l}}{{\\footnotesize register vs MLP on the energy score: {100 * rv['rel']:+.1f}\\% "
                         f"({rv['wins']}/{len(rv['seeds'])} seeds, paired $p={rv['p']:.2f}$){tail}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/manuscript/iclr2027/fgn_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
