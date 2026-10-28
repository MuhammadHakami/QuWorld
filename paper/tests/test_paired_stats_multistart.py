"""Pre-specified statistics for the sliding-start protocol
(19-multistart-power-plan.md): Holm correction, primary per-trajectory test on
start-averaged VRMSE, secondary per-(trajectory, start) test, cluster bootstrap."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "phase_b"))
import paired_stats_multistart as ps  # noqa: E402

S, N = 3, 4  # starts, trajectories


def _synthetic(shift: float, noise: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = 1.0 + 0.1 * rng.standard_normal((S, N))
    ours = base + shift + noise * rng.standard_normal((S, N))
    return ours, base


def test_holm_rejects_sequentially_and_reports_adjusted_pvalues():
    adjusted, reject = ps.holm([0.01, 0.04, 0.03], alpha=0.05)
    assert reject == [True, False, False]
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_primary_pairs_trajectories_on_start_averaged_values():
    ours, base = _synthetic(shift=-0.1)
    r = ps.primary_test(ours, base)
    assert r["n"] == N
    assert r["mean_diff"] == pytest.approx(-0.1)
    assert r["win_frac"] == 1.0
    assert r["p_less"] < r["p_two"] <= 1.0


def test_secondary_uses_every_start_trajectory_pair():
    ours, base = _synthetic(shift=-0.1)
    r = ps.secondary_test(ours, base)
    assert r["n"] == S * N and r["win_frac"] == 1.0


def test_cluster_bootstrap_resamples_trajectories():
    ours, base = _synthetic(shift=-0.1)            # exact shift: CI is degenerate
    mean_diff, lo, hi = ps.cluster_bootstrap(ours, base, n_boot=200, seed=0)
    assert mean_diff == pytest.approx(-0.1) and lo == pytest.approx(-0.1) and hi == pytest.approx(-0.1)
    ours_n, base_n = _synthetic(shift=-0.1, noise=0.05)
    _, lo, hi = ps.cluster_bootstrap(ours_n, base_n, n_boot=500, seed=0)
    assert lo <= -0.1 <= hi


def test_compare_reads_multistart_npz_files(tmp_path):
    ours, base = _synthetic(shift=-0.1)
    for name, w in (("q", ours), ("c", base)):
        np.savez(tmp_path / f"multistart_{name}.npz", w1330=w, w612=w + 0.5,
                 per_step=np.zeros((S, 30, N)), starts=np.arange(S))
    r = ps.compare(tmp_path, "q", "c")
    assert set(r) >= {"primary_1330", "secondary_1330", "bootstrap_1330", "primary_612"}
    assert r["primary_1330"]["mean_diff"] == pytest.approx(-0.1)


def test_compare_rejects_shape_mismatch(tmp_path):
    np.savez(tmp_path / "multistart_a.npz", w1330=np.zeros((S, N)), w612=np.zeros((S, N)))
    np.savez(tmp_path / "multistart_b.npz", w1330=np.zeros((S + 1, N)), w612=np.zeros((S + 1, N)))
    with pytest.raises(ValueError):
        ps.compare(tmp_path, "a", "b")
