"""Pre-specified statistics for the sliding-start rollout protocol
(19-multistart-power-plan.md), on runs/phase_b2/multistart_<name>.npz files.

Usage: uv run paper/phase_b/paired_stats_multistart.py <ours> <base> [seed ...]
  With seeds, names expand to <ours>_s<seed> / <base>_s<seed> (the anchor has no
  seed) and the primary one-sided p-values are Holm-corrected across seeds.
Primary: per-trajectory paired Wilcoxon on start-averaged window VRMSE.
Secondary: per-(trajectory, start) paired Wilcoxon (correlated within trajectory).
Also: cluster bootstrap over trajectories for the mean paired difference.
"""

import pathlib
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "runs/phase_b2"


def holm(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Holm step-down: adjusted p-values (monotone) and rejections at alpha."""
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1 or len(p) == 0 or np.any((p < 0) | (p > 1)):
        raise ValueError(f"p-values must be a non-empty 1-D array in [0, 1], got {pvals}")
    m, order = len(p), np.argsort(p)
    adjusted, running = [0.0] * m, 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted, [a <= alpha for a in adjusted]


def paired_wilcoxon(ours: np.ndarray, base: np.ndarray) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank on 1-D arrays; 'less' = ours better (lower)."""
    ours, base = np.asarray(ours, float).ravel(), np.asarray(base, float).ravel()
    if ours.shape != base.shape:
        raise ValueError(f"paired arrays differ in shape: {ours.shape} vs {base.shape}")
    d = ours - base
    if np.all(d == 0):
        p_two, p_less = 1.0, 1.0
    else:
        p_two = float(wilcoxon(ours, base).pvalue)
        p_less = float(wilcoxon(ours, base, alternative="less").pvalue)
    return {"n": int(d.size), "mean_diff": float(d.mean()),
            "win_frac": float((d < 0).mean()), "p_two": p_two, "p_less": p_less}


def primary_test(ours: np.ndarray, base: np.ndarray) -> Dict[str, float]:
    """Per-trajectory test on start-averaged values: arrays (starts, trajectories)."""
    return paired_wilcoxon(np.asarray(ours).mean(axis=0), np.asarray(base).mean(axis=0))


def secondary_test(ours: np.ndarray, base: np.ndarray) -> Dict[str, float]:
    """Per-(trajectory, start) test; units are correlated within a trajectory."""
    return paired_wilcoxon(np.asarray(ours).ravel(), np.asarray(base).ravel())


def cluster_bootstrap(ours: np.ndarray, base: np.ndarray, n_boot: int = 10_000,
                      seed: int = 0) -> Tuple[float, float, float]:
    """Mean paired difference with a 95% CI from resampling trajectories."""
    d = (np.asarray(ours, float) - np.asarray(base, float)).mean(axis=0)   # (n,)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _load(runs: pathlib.Path, name: str):
    path = runs / f"multistart_{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run eval_rollout_multistart.py first")
    return np.load(path)


def compare(runs: pathlib.Path, ours: str, base: str) -> Dict[str, Dict[str, float]]:
    a, b = _load(runs, ours), _load(runs, base)
    if a["w1330"].shape != b["w1330"].shape:
        raise ValueError(f"{ours} {a['w1330'].shape} vs {base} {b['w1330'].shape}: "
                         "different starts or trajectories")
    md, lo, hi = cluster_bootstrap(a["w1330"], b["w1330"])
    return {"primary_1330": primary_test(a["w1330"], b["w1330"]),
            "secondary_1330": secondary_test(a["w1330"], b["w1330"]),
            "bootstrap_1330": {"mean_diff": md, "ci_low": lo, "ci_high": hi},
            "primary_612": primary_test(a["w612"], b["w612"]),
            "secondary_612": secondary_test(a["w612"], b["w612"])}


def _names(prefix: str, seed: str) -> str:
    return prefix if prefix == "anchor" else f"{prefix}_s{seed}"


def main(argv: List[str]) -> None:
    if len(argv) < 3:
        raise SystemExit(__doc__)
    ours, base, seeds = argv[1], argv[2], argv[3:] or [""]
    rows = []
    for seed in seeds:
        o, b = (_names(ours, seed), _names(base, seed)) if seed else (ours, base)
        r = compare(RUNS, o, b)
        rows.append((o, b, r))
    adjusted, reject = holm([r["primary_1330"]["p_less"] for _, _, r in rows])
    print(f"{'ours':>16} {'base':>16} | primary(13:30) n mean_diff win p_less holm | "
          f"secondary n win p_less | bootstrap CI95")
    for (o, b, r), pa, rj in zip(rows, adjusted, reject):
        p, s, c = r["primary_1330"], r["secondary_1330"], r["bootstrap_1330"]
        print(f"{o:>16} {b:>16} | {p['n']:2d} {p['mean_diff']:+.4f} {p['win_frac']:.2f} "
              f"{p['p_less']:.4f} {pa:.4f}{'*' if rj else ' '} | {s['n']:3d} {s['win_frac']:.2f} "
              f"{s['p_less']:.2e} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]")


if __name__ == "__main__":
    main(sys.argv)
