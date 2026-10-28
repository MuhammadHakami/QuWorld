"""Continual adaptation stream for the bake-off: A -> B -> C.

  A: TRL2D tcool=0.03  — data/cache.pt train trajs 0-7,  valid traj 0
  B: TRL2D tcool=3.16  — data/cache.pt train trajs 16-23, valid traj 2
  C: TORAX-Bench v1    — data/torax_bench_v1/discharge_*.npz, splits per
                         paper/torax_bench/splits.json (train ids for the
                         stream, valid ids for evaluation)

The stream is a deterministic ORDERED sequence of N (traj, t) samples per
segment, drawn once with a fixed seed (STREAM_SEED + segment index) that is
independent of the run seed — every strategy and every seed sees the identical
stream (budget parity is in eval-units, not data).

TORAX -> (C, H, W) mapping (documented per the brief):
  1. channels = [T_e, T_i, n_e, q]; q lives on the 26-point face grid and is
     edge-padded (repeat last face value) to 27 so all four channels share one
     radial axis. Actuators are NOT included (profile channels only).
  2. per-channel z-score, statistics computed over the loaded TRAIN discharges
     (time x radial), applied to valid too.
  3. radial axis (27) -> W by 1D linear interpolation (align_corners=True).
  4. the radial profile is replicated across all H rows (stored as an
     expanded (…, C, 1, W) view — zero memory cost), yielding frames with the
     exact TRL2D tensor interface (C=4, H, W). Rows are identical, so VRMSE
     over (H, W) equals radial VRMSE over W.
"""

import dataclasses
import json

import numpy as np
import torch
import torch.nn.functional as F

from _bootstrap import PAPER_DIR, REPO_DIR, add_paths

add_paths()


STREAM_SEED = 20260816
K_MAX = 4  # max pushforward depth any strategy may request
SEGMENTS = ("A", "B", "C")
_TRL2D = {"A": dict(train_trajs=list(range(0, 8)), val_traj=0,
                    label="tcool=0.03"),
          "B": dict(train_trajs=list(range(16, 24)), val_traj=2,
                    label="tcool=3.16")}
CACHE_PT = REPO_DIR / "data" / "cache.pt"
TORAX_DIR = REPO_DIR / "data" / "torax_bench_v1"
SPLITS_JSON = PAPER_DIR / "torax_bench" / "splits.json"


@dataclasses.dataclass(frozen=True)
class StreamConfig:
    n_samples: int = 512
    batch: int = 16
    pool: int = 1        # spatial avg-pool factor for TRL2D frames
    height: int = 64     # frame H (TORAX replication height; TRL2D check)
    width: int = 192     # frame W (TORAX radial interpolation width)
    torax_train_files: int = 24
    torax_val_files: int = 4


FULL_STREAM = StreamConfig()
SMOKE_STREAM = StreamConfig(n_samples=32, batch=8, pool=4, height=16, width=48,
                            torax_train_files=4, torax_val_files=1)


class Segment:
    """Ordered, cycling adaptation stream over one regime + its val data."""

    def __init__(self, name: str, label: str, train: torch.Tensor,
                 val: torch.Tensor, n_samples: int):
        if train.dim() != 5 or val.dim() != 4:
            raise ValueError("train must be (Ntraj,T,C,H,W), val (T,C,H,W)")
        self.name, self.label = name, label
        self.train, self.val = train, val
        n_traj, t = train.shape[0], train.shape[1]
        if t <= K_MAX + 1:
            raise ValueError(f"segment {name}: too few frames ({t})")
        g = torch.Generator().manual_seed(STREAM_SEED + SEGMENTS.index(name))
        self.ni = torch.randint(0, n_traj, (n_samples,), generator=g)
        self.ti = torch.randint(0, t - 1 - K_MAX, (n_samples,), generator=g)
        self.n_samples = n_samples

    def batches(self, batch: int, k: int = 1):
        """Yield (x_t, [x_{t+1}..x_{t+k}]) cycling the ordered stream."""
        if not 1 <= k <= K_MAX:
            raise ValueError(f"k must be in [1, {K_MAX}]")
        i = 0
        while True:
            idx = torch.arange(i, i + batch) % self.n_samples
            i = int((i + batch) % self.n_samples)
            ni, ti = self.ni[idx], self.ti[idx]
            x = self.train[ni, ti]
            ys = [self.train[ni, ti + j] for j in range(1, k + 1)]
            yield x, ys

    def val_slice(self, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed strided (x, y) one-step batch from val (cheap fitness/curve
        metric — identical for every strategy)."""
        t = self.val.shape[0]
        ts = torch.linspace(0, t - 2, m).long()
        return self.val[ts], self.val[ts + 1]


def _pool_traj(x: torch.Tensor, pool: int) -> torch.Tensor:
    return F.avg_pool2d(x, pool) if pool > 1 else x.clone()


def _load_trl2d(name: str, cfg: StreamConfig) -> Segment:
    if not CACHE_PT.exists():
        raise FileNotFoundError(f"missing {CACHE_PT}")
    try:
        d = torch.load(CACHE_PT, weights_only=True, mmap=True,
                       map_location="cpu")
    except (RuntimeError, ValueError):
        d = torch.load(CACHE_PT, weights_only=True, map_location="cpu")
    spec = _TRL2D[name]
    train = torch.stack([_pool_traj(d["train"][i], cfg.pool)
                         for i in spec["train_trajs"]])
    val = _pool_traj(d["valid"][spec["val_traj"]], cfg.pool)
    return Segment(name, spec["label"], train, val, cfg.n_samples)


def _torax_profiles(disc_id: int) -> np.ndarray:
    """(T=121, 4, 27) float32: [T_e, T_i, n_e, q(edge-padded 26->27)]."""
    path = TORAX_DIR / f"discharge_{disc_id:05d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    with np.load(path) as d:
        q = d["q"]
        q = np.concatenate([q, q[:, -1:]], axis=1)  # edge-pad faces 26 -> 27
        return np.stack([d["T_e"], d["T_i"], d["n_e"], q], axis=1
                        ).astype(np.float32)


def _torax_frames(ids: list[int], mean: torch.Tensor, std: torch.Tensor,
                  cfg: StreamConfig) -> torch.Tensor:
    """(F, T, 4, H, W) expanded view (H replication costs no memory)."""
    prof = torch.from_numpy(np.stack([_torax_profiles(i) for i in ids]))
    prof = (prof - mean) / std
    f, t, c, r = prof.shape
    wide = F.interpolate(prof.reshape(f * t, c, r), size=cfg.width,
                         mode="linear", align_corners=True)
    wide = wide.reshape(f, t, c, 1, cfg.width)
    return wide.expand(f, t, c, cfg.height, cfg.width)


def _load_torax(cfg: StreamConfig) -> Segment:
    with open(SPLITS_JSON) as fh:
        splits = json.load(fh)
    train_ids = splits["train"][:cfg.torax_train_files]
    val_ids = splits["valid"][:cfg.torax_val_files]
    raw = torch.from_numpy(np.stack([_torax_profiles(i) for i in train_ids]))
    mean = raw.mean(dim=(0, 1, 3), keepdim=True)
    std = raw.std(dim=(0, 1, 3), keepdim=True).clamp(min=1e-6)
    train = _torax_frames(train_ids, mean[0], std[0], cfg)
    val = _torax_frames(val_ids, mean[0], std[0], cfg)[0]
    return Segment("C", "torax_bench_v1", train, val, cfg.n_samples)


def build_segment(name: str, cfg: StreamConfig) -> Segment:
    if name in _TRL2D:
        seg = _load_trl2d(name, cfg)
    elif name == "C":
        seg = _load_torax(cfg)
    else:
        raise ValueError(f"unknown segment {name!r}; valid: {SEGMENTS}")
    h, w = seg.train.shape[-2:]
    if (h, w) != (cfg.height, cfg.width):
        raise ValueError(f"segment {name}: frames {(h, w)} != config "
                         f"{(cfg.height, cfg.width)}")
    return seg
