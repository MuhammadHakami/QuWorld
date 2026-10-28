"""Paired significance test between two per-window VRMSE CSVs.

Usage: uv run paper/phase_b/paired_stats.py <ours.csv> <baseline.csv>
Wilcoxon signed-rank (two-sided + one-sided 'ours < baseline'), rank-biserial
effect size, win rate, means/medians. This is the pre-registered H1d test:
ours = hybrid fine-tune, baseline = official checkpoint reproduction
(runs/eval_official_cnext.csv, mean 0.1934).
"""

import csv
import sys

import numpy as np
from scipy.stats import wilcoxon


def load(path: str) -> np.ndarray:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["vrmse"]) for r in rows])


def main() -> None:
    ours, base = load(sys.argv[1]), load(sys.argv[2])
    assert len(ours) == len(base), f"window mismatch {len(ours)} vs {len(base)}"
    d = ours - base
    w2 = wilcoxon(ours, base)
    w1 = wilcoxon(ours, base, alternative="less")
    n_nonzero = np.count_nonzero(d)
    rank_biserial = 1 - 2 * w2.statistic / (n_nonzero * (n_nonzero + 1) / 2)
    print(f"windows:        {len(ours)}")
    print(f"ours   mean/median: {ours.mean():.4f} / {np.median(ours):.4f}")
    print(f"base   mean/median: {base.mean():.4f} / {np.median(base):.4f}")
    print(f"mean diff:      {d.mean():+.4f}  (negative = ours better)")
    print(f"win rate:       {(d < 0).mean():.1%} of windows")
    print(f"wilcoxon two-sided p: {w2.pvalue:.2e}")
    print(f"wilcoxon one-sided (ours<base) p: {w1.pvalue:.2e}")
    print(f"rank-biserial effect: {rank_biserial:+.3f}")
    verdict = "PASS" if (ours.mean() < base.mean() and w1.pvalue < 0.05) else "no pass"
    print(f"H1d verdict (mean lower AND p<0.05): {verdict}")


if __name__ == "__main__":
    main()
