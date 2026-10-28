"""Paired comparison of Physics-IQ arms over the 198 scenario-view rows.

Runs inside baseline/physics-iq's uv env (needs the `physiq` package,
pandas, scipy):
  cd baseline/physics-iq && uv run --no-sync python .../physiq_compare.py \
      --arms vanilla quantum --csvs <vanilla>.csv <quantum>.csv --out cmp.json

Paired unit = (scenario, view): `clip(IQTable.compute_scores_scenario_by_view())`
(calculate_iq_score_stable.py:266-275, 396-434) — its column
`final_arithmetic` averages to the verified score `final_score_view`, so the
row-wise difference between two arms is the paired decomposition of the
verified-score gap. Per-scenario rows (66, `compute_scores_scenario`) are
reported as a secondary unit. Tests: paired t, Wilcoxon signed-rank,
sign counts, 95% CI of the mean difference.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from physiq.calculate_iq_score_stable import IQTable, clip

COMPONENTS = ("spatial_iou_score", "weighted_spatial_iou_score", "spatiotemporal_iou_score", "mse_score")


def per_view_table(csv_path: str) -> pd.DataFrame:
    table = IQTable.from_csv(csv_path)
    rows = clip(table.compute_scores_scenario_by_view())
    n_views = len(table.views)
    rows = rows.copy()
    rows.insert(0, "view", np.tile(np.asarray(table.views), len(table.df)))
    rows.insert(0, "scenario", np.repeat(table.df["scenario"].to_numpy(), n_views))
    return rows.sort_values(["scenario", "view"]).reset_index(drop=True)


def per_scenario_table(csv_path: str) -> pd.DataFrame:
    table = IQTable.from_csv(csv_path)
    rows = clip(table.compute_scores_scenario()).copy()
    rows.insert(0, "scenario", table.df["scenario"].to_numpy())
    return rows.sort_values("scenario").reset_index(drop=True)


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Statistics of d = a - b (a minus b; positive favours `a`)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    half = stats.t.ppf(0.975, n - 1) * se if n > 1 else float("nan")
    nonzero = d[d != 0]
    wilcoxon_p = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 10 else float("nan")
    ttest_p = float(stats.ttest_rel(a, b).pvalue) if n > 1 else float("nan")
    return {"n": n, "mean_diff": mean, "ci95": [mean - half, mean + half], "se": se,
            "wilcoxon_p": wilcoxon_p, "ttest_rel_p": ttest_p,
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_tie": int((d == 0).sum()),
            "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b))}


def compare_pair(name_a: str, name_b: str, views: dict, scen: dict) -> dict:
    va, vb = views[name_a], views[name_b]
    if not (va[["scenario", "view"]].equals(vb[["scenario", "view"]])):
        raise SystemExit(f"row keys differ between {name_a} and {name_b}")
    out = {"a": name_a, "b": name_b,
           "per_view_final": paired_stats(va["final_arithmetic"].to_numpy(), vb["final_arithmetic"].to_numpy()),
           "per_view_components": {c: paired_stats(va[c].to_numpy(), vb[c].to_numpy()) for c in COMPONENTS if c in va},
           "per_scenario_final": paired_stats(scen[name_a]["final_arithmetic"].to_numpy(),
                                              scen[name_b]["final_arithmetic"].to_numpy())}
    s = out["per_view_final"]
    out["summary"] = (f"{name_a}-{name_b} verified-score rows (n={s['n']}): mean diff {s['mean_diff']:+.4f} "
                      f"[95% CI {s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}], Wilcoxon p={s['wilcoxon_p']:.3g}, "
                      f"paired-t p={s['ttest_rel_p']:.3g}, {s['n_pos']} rows better / {s['n_neg']} worse / "
                      f"{s['n_tie']} tied; arm means {s['mean_a']:.4f} vs {s['mean_b']:.4f}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--csvs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if len(a.arms) != len(a.csvs) or len(a.arms) < 2:
        raise SystemExit("--arms and --csvs must have the same length (>= 2)")
    views = {arm: per_view_table(csv) for arm, csv in zip(a.arms, a.csvs)}
    scen = {arm: per_scenario_table(csv) for arm, csv in zip(a.arms, a.csvs)}
    pairs = [compare_pair(x, y, views, scen) for x, y in itertools.combinations(a.arms, 2)]
    result = {"arms": a.arms, "csvs": a.csvs, "pairs": pairs,
              "summary": " | ".join(pr["summary"] for pr in pairs)}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(result["summary"])


if __name__ == "__main__":
    main()
