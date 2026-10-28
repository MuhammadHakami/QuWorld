"""Tiny-data factory for The Well turbulent_radiative_layer_2D subset.

Stacks (density, pressure, vx, vy) -> (traj, T, 4, H, W), avg-pools spatial x2
to 64x192, z-scores with train statistics, caches to data/cache.pt.
"""

import glob
import pathlib

import h5py
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"
CACHE = ROOT / "cache.pt"


def _load_split(pattern: str) -> torch.Tensor:
    trajs = []
    for f in sorted(glob.glob(pattern)):
        with h5py.File(f) as h:
            den = torch.from_numpy(h["t0_fields/density"][:])
            pre = torch.from_numpy(h["t0_fields/pressure"][:])
            vel = torch.from_numpy(h["t1_fields/velocity"][:])
        x = torch.stack([den, pre, vel[..., 0], vel[..., 1]], dim=2)  # (n,T,4,H,W)
        trajs.append(x)
    x = torch.cat(trajs)
    n, t, c, hh, ww = x.shape
    x = F.avg_pool2d(x.reshape(n * t, c, hh, ww), 2).reshape(n, t, c, hh // 2, ww // 2)
    return x


def load(rebuild: bool = False):
    if CACHE.exists() and not rebuild:
        d = torch.load(CACHE, weights_only=True)
        return d["train"], d["valid"], d["mean"], d["std"]
    train = _load_split(str(ROOT / "trl2d/data/train/*.hdf5"))
    valid = _load_split(str(ROOT / "trl2d/data/valid/*.hdf5"))
    mean = train.mean(dim=(0, 1, 3, 4), keepdim=True)
    std = train.std(dim=(0, 1, 3, 4), keepdim=True).clamp(min=1e-6)
    train = (train - mean) / std
    valid = (valid - mean) / std
    torch.save({"train": train, "valid": valid, "mean": mean, "std": std}, CACHE)
    return train, valid, mean, std


def vrmse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Variance-scaled RMSE per sample+channel, averaged (The Well style).
    pred/true: (B, C, H, W)."""
    mse = (pred - true).square().mean(dim=(2, 3))
    var = (true - true.mean(dim=(2, 3), keepdim=True)).square().mean(dim=(2, 3))
    return (mse / (var + 1e-8)).sqrt().mean()
