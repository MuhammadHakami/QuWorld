"""Doc 38: four-seed TD replication summary (primary: VALID-selected checkpoints; secondary: final-step weights).

    .venv/bin/python paper/phase_b/td_replication_summary.py   # -> runs/phase_b2/td_replication.json, iclr2027/td_repl_table.tex

Per arm and seed: official-start and all-start 13:30 VRMSE from runs/phase_b2/multistart_<name>.npz logs. Per seed and
comparison (register vs vanilla, register vs MLP): the pre-specified primary test of paired_stats_multistart.compare
(one-sided per-trajectory Wilcoxon on start-averaged 13:30 error), Holm-corrected across the four seeds.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/phase_b2"
sys.path.insert(0, str(ROOT / "paper/phase_b"))
from paired_stats_multistart import compare, holm  # noqa: E402

SEEDS = (1337, 1338, 1339, 1340)
ARMS = (("mr30ryd", "register"), ("mlp30", "matched MLP"), ("vanilla", "vanilla continued"))
ANCHOR_OFFICIAL, ANCHOR_ALL = 1.0132, 0.9707


def metric(name: str, key: str) -> float | None:
    f = RUNS / f"multistart_{name}.log"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        if line.startswith(key + ":"):
            return float(line.split()[1])
    return None


def block(suffix: str) -> dict:
    out = {"arms": {}, "tests": {}}
    for v, _ in ARMS:
        off = [metric(f"{v}{suffix}_s{s}", "start0_13_30") for s in SEEDS]
        alls = [metric(f"{v}{suffix}_s{s}", "allstart_13_30") for s in SEEDS]
        out["arms"][v] = {"official": off, "allstart": alls}
    for base in ("vanilla", "mlp30"):
        rows = []
        for s in SEEDS:
            o, b = f"mr30ryd{suffix}_s{s}", f"{base}{suffix}_s{s}"
            if (RUNS / f"multistart_{o}.npz").exists() and (RUNS / f"multistart_{b}.npz").exists():
                rows.append((s, compare(RUNS, o, b)["primary_1330"]))
        if len(rows) == len(SEEDS):
            adj, rej = holm([r["p_less"] for _, r in rows])
            out["tests"][base] = [{"seed": s, "mean_diff": r["mean_diff"], "p": r["p_less"], "holm": a, "sig": bool(j)}
                                  for (s, r), a, j in zip(rows, adj, rej)]
    return out


def fmt(xs):
    xs = [x for x in xs if x is not None]
    return f"{st.mean(xs):.4f} $\\pm$ {st.stdev(xs):.4f}" if len(xs) > 1 else "---"


def main():
    res = {"primary": block(""), "secondary_final": block("final")}
    (RUNS / "td_replication.json").write_text(json.dumps(res, indent=1))
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Checkpoint & arm & official start & all 9 starts & reg.\\ better (Holm, 4 seeds) & seed 1340 \\\\", "\\midrule"]
    for key, lab in (("primary", "VALID-selected"), ("secondary_final", "final step")):
        b = res[key]; first = True
        for v, name in ARMS:
            a = b["arms"][v]
            t = b["tests"].get(v)
            wins = (f"{sum(r['sig'] for r in t)}/4" if t else "---") if v != "mr30ryd" else "---"
            s1340 = a["official"][-1]
            lines.append(f"{lab if first else ''} & {name} & {fmt(a['official'])} & {fmt(a['allstart'])} & {wins} & "
                         f"{s1340:.4f} \\\\" if s1340 is not None else f"{lab if first else ''} & {name} & --- & --- & --- & --- \\\\")
            first = False
        if key == "primary":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/manuscript/iclr2027/td_repl_table.tex").write_text("\n".join(lines) + "\n")
    for key in res:
        print(key, {v: [round(x, 4) if x else None for x in res[key]["arms"][v]["official"]] for v, _ in ARMS})
        for base, t in res[key]["tests"].items():
            print("  reg vs", base, [(r["seed"], round(r["mean_diff"], 4), round(r["holm"], 4), r["sig"]) for r in t])


if __name__ == "__main__":
    main()
