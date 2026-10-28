"""Sliding-start rollout protocol helpers (pre-registered 2026-09-06,
19-multistart-power-plan.md). Pure numpy for the window logic; the TRL2D test
loader mirrors eval_rollout_pertraj.py exactly (lazy imports keep tests light).
"""

import pathlib
from typing import List, Tuple

import numpy as np

MAX_T = 30
WINDOW_612 = slice(5, 12)     # rollout steps 6..12   (row 0 = step 1)
WINDOW_1330 = slice(12, 30)   # rollout steps 13..30


def start_indices(n_frames: int, n_hist: int, max_t: int, stride: int) -> List[int]:
    """Start frames s = 0, stride, 2*stride, ... with s + n_hist + max_t <= n_frames,
    so every rollout has its full history and horizon inside the trajectory."""
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    last = n_frames - n_hist - max_t
    if last < 0:
        raise ValueError(f"{n_frames} frames cannot hold {n_hist} history + {max_t} horizon")
    return list(range(0, last + 1, stride))


def window_means(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """v: (..., MAX_T, n) per-step VRMSE, step axis second-to-last.
    Returns The Well's 6:12 and 13:30 window means, shape (..., n)."""
    if v.ndim < 2 or v.shape[-2] != MAX_T:
        raise ValueError(f"expected step axis of length {MAX_T}, got shape {v.shape}")
    return v[..., WINDOW_612, :].mean(axis=-2), v[..., WINDOW_1330, :].mean(axis=-2)


def load_test_trajectories(root: pathlib.Path):
    """(n, T, 4, H, W) z-scored TRL2D test trajectories, identical to
    eval_rollout_pertraj.py (channels density, pressure, vx, vy)."""
    import glob

    import h5py
    import torch
    import yaml
    s = yaml.safe_load(open(root / "data/trl2d/stats.yaml"))
    mean = torch.tensor([s["mean"]["density"], s["mean"]["pressure"],
                         *s["mean"]["velocity"]]).view(1, 1, 4, 1, 1)
    std = torch.tensor([s["std"]["density"], s["std"]["pressure"],
                        *s["std"]["velocity"]]).view(1, 1, 4, 1, 1)
    trajs = []
    for f in sorted(glob.glob(str(root / "data/trl2d/data/test/*.hdf5"))):
        with h5py.File(f) as h:
            den = torch.from_numpy(h["t0_fields/density"][:])
            pre = torch.from_numpy(h["t0_fields/pressure"][:])
            vel = torch.from_numpy(h["t1_fields/velocity"][:])
        trajs.append(torch.stack([den, pre, vel[..., 0], vel[..., 1]], dim=2))
    if not trajs:
        raise FileNotFoundError(f"no test trajectories under {root / 'data/trl2d/data/test'}")
    return torch.nan_to_num((torch.cat(trajs).float() - mean) / std)
