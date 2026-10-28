"""TokaMark 1-1: shot-level paired comparison of the 3-seed ensembles (vanilla / mlp30 / mr30ryd).

    .venv/bin/python paper/tokamark/paired_ens.py  ->  runs/tokamark/results/paired_ens3.json

Unit = test shot (1,020): per-shot task score = mean over the 15 EFIT features of TokaMind's per-shot NRMSE_mean, so the
mean over shots equals the task NRMSE_mean. Paired t, Wilcoxon signed-rank and a shot bootstrap CI of the relative gain.
This conditions on the trained models (it tests "this ensemble vs that one on these shots"); the seed-level variation
is the separate 3-vs-3 comparison in the judge.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "runs" / "tokamark" / "results"
ARMS = ("vanilla", "mlp30", "mr30ryd")


def per_shot(arm: str) -> pd.Series:
    f = RES / f"tokamind_task_1-1_{arm}_ens3_eval" / "metrics" / "task_1-1" / "shots_metrics.csv"
    df = pd.read_csv(f)
    return df.pivot_table(index="shot_id", columns="feature_name", values="NRMSE_mean").mean(axis=1)


def compare(a: pd.Series, b: pd.Series, rng) -> dict:
    """Gain of a over b (lower NRMSE is better): positive = a better."""
    a, b = a.align(b, join="inner")
    d = (b - a).to_numpy()
    boot = [(b.to_numpy()[i].mean() - a.to_numpy()[i].mean()) / b.to_numpy()[i].mean()
            for i in (rng.integers(0, len(d), len(d)) for _ in range(5000))]
    return dict(n_shots=len(d), mean_a=float(a.mean()), mean_b=float(b.mean()), rel_gain=float(d.mean() / b.mean()),
                rel_gain_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                t_p=float(stats.ttest_rel(b, a).pvalue), wilcoxon_p=float(stats.wilcoxon(d).pvalue))


def main() -> None:
    rng = np.random.default_rng(0)
    s = {a: per_shot(a) for a in ARMS}
    out = {"task_nrmse": {a: float(v.mean()) for a, v in s.items()},
           "q_vs_mlp": compare(s["mr30ryd"], s["mlp30"], rng), "q_vs_trunk": compare(s["mr30ryd"], s["vanilla"], rng),
           "mlp_vs_trunk": compare(s["mlp30"], s["vanilla"], rng)}
    (RES / "paired_ens3.json").write_text(json.dumps(out, indent=1))
    print("task NRMSE (mean over shots):", {k: round(v, 5) for k, v in out["task_nrmse"].items()})
    for k in ("q_vs_mlp", "q_vs_trunk", "mlp_vs_trunk"):
        c = out[k]
        print(f"{k:13s} gain {100*c['rel_gain']:+.2f}% [{100*c['rel_gain_ci95'][0]:+.2f}, {100*c['rel_gain_ci95'][1]:+.2f}]  "
              f"wins {c['wins']}/{c['n_shots']}  t p={c['t_p']:.3g}  wilcoxon p={c['wilcoxon_p']:.3g}")


if __name__ == "__main__":
    main()
