"""Sliding-start rollout protocol helpers (ICLR power extension, pre-registered
in 19-multistart-power-plan.md): start indices and The Well window means."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "phase_b"))
import rollout_windows as rw  # noqa: E402


def test_start_indices_cover_trajectory_without_overrun():
    starts = rw.start_indices(n_frames=101, n_hist=4, max_t=30, stride=8)
    assert starts == [0, 8, 16, 24, 32, 40, 48, 56, 64]
    assert starts[-1] + 4 + 30 <= 101 < starts[-1] + 8 + 4 + 30


def test_start_indices_begin_with_the_official_protocol():
    assert rw.start_indices(101, 4, 30, 8)[0] == 0


def test_start_indices_reject_trajectories_too_short_for_one_rollout():
    with pytest.raises(ValueError):
        rw.start_indices(n_frames=33, n_hist=4, max_t=30, stride=8)


def test_window_means_use_published_well_windows():
    v = np.zeros((2, 30, 3))          # (starts, rollout steps, trajectories)
    v[:, 5:12] = 1.0                  # steps 6..12
    v[:, 12:30] = 2.0                 # steps 13..30
    w612, w1330 = rw.window_means(v)
    assert w612.shape == (2, 3) and w1330.shape == (2, 3)
    assert np.all(w612 == 1.0) and np.all(w1330 == 2.0)


def test_window_means_reject_wrong_step_count():
    with pytest.raises(ValueError):
        rw.window_means(np.zeros((2, 29, 3)))
