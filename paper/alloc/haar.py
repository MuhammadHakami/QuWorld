"""Orthonormal multi-level Haar transform over any set of axes, plus a flat tile index (33-allocation-strategy-study.md).

A tile is one coefficient position (level, orientation, spatial position) with all channels kept together. Orientation
m in 1..2^d-1 is the detail sub-band bitmask (bit a = high-pass along transformed axis a); m = 0 is the coarsest
approximation band, which is included as tiles so low-frequency error is allocatable too. Orthonormality makes every
per-tile energy add up exactly to the field energy (Lean: band_energy_split, residual_after_band_removal).
"""
from __future__ import annotations

import math

import torch

R2 = 1.0 / math.sqrt(2.0)


def _split(x: torch.Tensor, ax: int):
    e, o = x.narrow(ax, 0, x.shape[ax]).unflatten(ax, (x.shape[ax] // 2, 2)).unbind(ax + 1)
    return (e + o) * R2, (e - o) * R2


def _merge(lo: torch.Tensor, hi: torch.Tensor, ax: int) -> torch.Tensor:
    return torch.stack([(lo + hi) * R2, (lo - hi) * R2], dim=ax + 1).flatten(ax, ax + 1)


def haar(x: torch.Tensor, axes: tuple[int, ...], levels: int):
    """x -> (approx, [ {m: band} per level, fine -> coarse ]). Every axis length must be divisible by 2**levels."""
    details, a = [], x
    for _ in range(levels):
        bands = {0: a}
        for bit, ax in enumerate(axes):
            nxt = {}
            for m, t in bands.items():
                lo, hi = _split(t, ax)
                nxt[m], nxt[m | (1 << bit)] = lo, hi
            bands = nxt
        a = bands.pop(0)
        details.append(bands)
    return a, details


def ihaar(a: torch.Tensor, details, axes: tuple[int, ...]) -> torch.Tensor:
    """Exact inverse of haar()."""
    for bands in reversed(details):
        cur = dict(bands); cur[0] = a
        for bit in reversed(range(len(axes))):
            ax, nxt = axes[bit], {}
            for m in [m for m in cur if not m & (1 << bit)]:
                nxt[m] = _merge(cur[m], cur[m | (1 << bit)], ax)
            cur = nxt
        a = cur[0]
    return a


class TileBank:
    """Flat tile index for fields shaped (B, C, *S) transformed over all of S (d = len(S) <= 3).

    level (N,), orient (N,), pos (N, 3) tile centre in pixel units (unused axes 0). Order: level 0 (finest) details by
    orientation, ..., coarsest details, then the approximation band."""

    def __init__(self, shape: tuple[int, ...], levels: int):
        d = len(shape)
        assert 1 <= d <= 3 and all(s % (2 ** levels) == 0 for s in shape), (shape, levels)
        self.shape, self.levels, self.d = tuple(shape), levels, d
        lv, ori, pos, self.band_shapes = [], [], [], []
        for j in range(levels + 1):                                   # j = levels: the approximation band
            step = 2 ** (min(j, levels - 1) + 1)
            bshape = tuple(s // step for s in shape)
            grid = torch.stack(torch.meshgrid(*[(torch.arange(b) + 0.5) * step for b in bshape], indexing="ij"), -1)
            grid = grid.reshape(-1, d)
            grid = torch.cat([grid, torch.zeros(len(grid), 3 - d)], 1)
            for m in ([0] if j == levels else range(1, 2 ** d)):
                lv.append(torch.full((len(grid),), min(j, levels - 1))); ori.append(torch.full((len(grid),), m))
                pos.append(grid); self.band_shapes.append((j, m, bshape))
        self.level, self.orient, self.pos = torch.cat(lv), torch.cat(ori), torch.cat(pos).float()
        self.n = int(self.level.numel())
        self.n_orient = 2 ** d

    def flatten(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, *S) -> (B, N, C) Haar coefficients in bank order."""
        axes = tuple(range(2, 2 + self.d))
        a, details = haar(x, axes, self.levels)
        parts = []
        for j, m, _ in self.band_shapes:
            t = a if j == self.levels else details[j][m]
            parts.append(t.flatten(2).transpose(1, 2))
        return torch.cat(parts, 1)

    def unflatten(self, flat: torch.Tensor) -> torch.Tensor:
        """(B, N, C) -> (B, C, *S) field (inverse transform)."""
        B, _, C = flat.shape
        axes = tuple(range(2, 2 + self.d))
        details, a, i = [dict() for _ in range(self.levels)], None, 0
        for j, m, bshape in self.band_shapes:
            k = math.prod(bshape)
            t = flat[:, i:i + k].transpose(1, 2).reshape(B, C, *bshape); i += k
            if j == self.levels:
                a = t
            else:
                details[j][m] = t
        return ihaar(a, details, axes)
