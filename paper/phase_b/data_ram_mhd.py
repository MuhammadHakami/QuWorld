"""Memmap-backed, protocol-faithful data for Phase-B training on The Well MHD_64.

Port of data_ram.py (turbulent_radiative_layer_2D) to MHD_64 (3D, 64^3).

RAM decision (documented per Phase-B protocol): the full train split is
77 trajectories x 100 steps x 7 ch x 64^3 fp32 = 52.6 GiB, over the ~40 GB
in-RAM budget (121 GiB unified memory shared with a live GPU campaign).
So instead of data_ram.py's single in-RAM .pt, each split is built ONCE,
trajectory-by-trajectory (peak build RAM ~3 GiB), into a raw fp32 file of
shape (n_traj, T, 7, 64, 64, 64), then opened as a copy-on-write np.memmap
(mode="c" -> writable view, no torch warning, we never write) wrapped in a
torch tensor. Window reads fault in only the pages they touch; page cache
is reclaimable, so RSS stays bounded — same freeze-safety rationale as
data_ram.py, without HDF5 worker processes in the training loop.

Normalization uses the OFFICIAL stats.yaml that WellDataset resolves
(data/well/MHD_64/stats.yaml -> data/mhd64/stats.yaml; ZScore per field,
vectors per component, std clipped at the_well's min_denom=1e-4) so
serving is numerically equivalent to WellDataset(use_normalization=True)
— verified by build-time assertion on the first valid sample.

Channel order matches the_well's channels-first formatter: t0_fields then
t1_fields, attrs["field_names"] order, vector components last-dim x,y,z:
[density, Bx, By, Bz, vx, vy, vz]; input flattened t-major -> 28 channels.
"""

import os
import glob
import pathlib
import sys

import h5py
import numpy as np
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "phaseb_mhd_cache"
STATS = ROOT / "data" / "mhd64" / "stats.yaml"  # == data/well/MHD_64/stats.yaml
N_HIST = 4
C = 7  # density, Bx, By, Bz, vx, vy, vz
MIN_STD = 1e-4  # the_well ZScoreNormalization min_denom


def _stats():
    s = yaml.safe_load(open(STATS))
    mean = [s["mean"]["density"], *s["mean"]["magnetic_field"], *s["mean"]["velocity"]]
    std = [s["std"]["density"], *s["std"]["magnetic_field"], *s["std"]["velocity"]]
    std = [max(v, MIN_STD) for v in std]
    return (np.asarray(mean, dtype=np.float32).reshape(C, 1, 1, 1),
            np.asarray(std, dtype=np.float32).reshape(C, 1, 1, 1))


def _split_files(split: str):
    """Sorted like the_well (files_paths.sort()); returns (files, shape)."""
    files = sorted(glob.glob(str(ROOT / f"data/mhd64/data/{split}/*.hdf5")))
    assert files, f"no HDF5 files found for split '{split}'"
    n, t, res = 0, None, None
    for f in files:
        with h5py.File(f) as h:
            s = h["t0_fields/density"].shape  # (n_traj, T, 64, 64, 64)
            n += s[0]
            assert t in (None, s[1]), f"inconsistent T in {f}"
            t, res = s[1], s[2:]
    return files, (n, t, C, *res)


def _build_split(split: str) -> tuple:
    """Normalize trajectory-by-trajectory into a raw fp32 memmap file."""
    mean, std = _stats()
    files, shape = _split_files(split)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_DIR / f"{split}.f32.tmp"
    mm = np.memmap(tmp, dtype=np.float32, mode="w+", shape=shape)
    i = 0
    for f in files:
        with h5py.File(f) as h:
            for k in range(h["t0_fields/density"].shape[0]):
                den = h["t0_fields/density"][k]                # (T, 64, 64, 64)
                mag = h["t1_fields/magnetic_field"][k]         # (T, 64, 64, 64, 3)
                vel = h["t1_fields/velocity"][k]
                x = np.concatenate([den[:, None],
                                    np.moveaxis(mag, -1, 1),
                                    np.moveaxis(vel, -1, 1)], axis=1, dtype=np.float32)
                mm[i] = np.nan_to_num((x - mean) / std)
                mm.flush()  # bound dirty pages during the 53 GiB build
                i += 1
        print(f"  {split}: {i}/{shape[0]} trajectories", flush=True)
    del mm
    tmp.replace(CACHE_DIR / f"{split}.f32")
    return shape


class PreadSplit:
    """Bounded-residency accessor: reads exactly the requested frames and
    evicts them from page cache (posix_fadvise DONTNEED). Replaces the
    persistent memmap whose resident pages (up to 53 GiB) stacked on top of
    CUDA unified-memory allocations and wedged the board (OOM 2026-08-18)."""

    def __init__(self, path, shape):
        self.path, self.shape = str(path), tuple(int(s) for s in shape)
        n, t, c, d, h, w = self.shape
        self.frame_elems = c * d * h * w
        self.frame_bytes = self.frame_elems * 4
        self.fd = os.open(self.path, os.O_RDONLY)

    def frames(self, traj: int, times) -> torch.Tensor:
        """(len(times), C, D, H, W) fp32 — transient buffers only."""
        n, t = self.shape[0], self.shape[1]
        out = np.empty((len(times), self.frame_elems), dtype=np.float32)
        for j, ti in enumerate(times):
            off = (traj * t + int(ti)) * self.frame_bytes
            data = os.pread(self.fd, self.frame_bytes, off)
            out[j] = np.frombuffer(data, dtype=np.float32)
            os.posix_fadvise(self.fd, off, self.frame_bytes,
                             os.POSIX_FADV_DONTNEED)
        return torch.from_numpy(out.reshape(len(times), *self.shape[2:]))


def _open(split: str, shape) -> torch.Tensor:
    """Valid split (6.8 GiB) loads fully into RAM; train uses PreadSplit."""
    if split == "train":
        return PreadSplit(CACHE_DIR / f"{split}.f32", shape)
    arr = np.fromfile(CACHE_DIR / f"{split}.f32", dtype=np.float32).reshape(
        tuple(shape))
    return torch.from_numpy(arr)


def load(rebuild: bool = False):
    """Returns (train, valid) as (n_traj, T, 7, 64, 64, 64) normalized fp32,
    memmap-backed (copy-on-write). meta.pt is written only after the
    equivalence assertion passes, so a half-built cache is never trusted."""
    meta = CACHE_DIR / "meta.pt"
    if meta.exists() and not rebuild:
        shapes = torch.load(meta, weights_only=True)
        return _open("train", shapes["train"]), _open("valid", shapes["valid"])
    vshape = _build_split("valid")           # cheap split first: verify before
    valid = _open("valid", vshape)           # committing to the 53 GiB build
    _verify_against_official(valid)
    tshape = _build_split("train")
    torch.save({"train": tuple(tshape), "valid": tuple(vshape)}, meta)
    return _open("train", tshape), valid


def _verify_against_official(valid: torch.Tensor) -> None:
    """Assert numerical equivalence with the_well's normalized WellDataset."""
    from einops import rearrange
    from the_well.data import WellDataset
    from the_well.data.normalization import ZScoreNormalization

    ds = WellDataset(
        well_base_path=str(ROOT / "data/well"),
        well_dataset_name="MHD_64",
        well_split_name="valid", n_steps_input=N_HIST, n_steps_output=1,
        use_normalization=True, normalization_type=ZScoreNormalization)
    sample = ds[0]
    ref_x = torch.nan_to_num(sample["input_fields"])       # (4, 64, 64, 64, 7)
    ref_y = torch.nan_to_num(sample["output_fields"][0])   # (64, 64, 64, 7)
    ours_x = rearrange(valid[0, 0:N_HIST], "t c d h w -> t d h w c")
    ours_y = rearrange(valid[0, N_HIST], "c d h w -> d h w c")
    dx = (ref_x - ours_x).abs().max().item()
    dy = (ref_y - ours_y).abs().max().item()
    assert dx < 1e-4 and dy < 1e-4, \
        f"RAM loader mismatch vs the_well: input {dx}, output {dy}"
    print(f"data_ram_mhd verified vs the_well (max diff in={dx:.2e} out={dy:.2e})")


def train_batch(data, bsz: int, gen: torch.Generator, dev: str):
    n, t = data.shape[0], data.shape[1]
    ni = torch.randint(0, n, (bsz,), generator=gen)
    ti = torch.randint(N_HIST - 1, t - 1, (bsz,), generator=gen)
    if isinstance(data, PreadSplit):
        xs, ys = [], []
        for b in range(bsz):
            fr = data.frames(int(ni[b]),
                             range(int(ti[b]) - N_HIST + 1, int(ti[b]) + 2))
            xs.append(fr[:N_HIST])                     # (4, 7, 64, 64, 64)
            ys.append(fr[N_HIST])
        x = torch.stack(xs)
        y = torch.stack(ys)
    else:
        idx = ti.view(-1, 1) + torch.arange(-N_HIST + 1, 1)
        x = data[ni.view(-1, 1), idx]                  # (B, 4, 7, 64, 64, 64)
        y = data[ni, ti + 1]                           # (B, 7, 64, 64, 64)
    return x.reshape(bsz, -1, *x.shape[-3:]).to(dev), y.to(dev)


def valid_windows(data: torch.Tensor, stride_t: int = 2):
    """Yield (x, y) covering the valid split deterministically.
    x: (n, 28, 64, 64, 64) ~294 MB fp32; y: (n, 7, 64, 64, 64)."""
    n, t = data.shape[0], data.shape[1]
    for t0 in range(N_HIST - 1, t - 1, stride_t):
        idx = torch.arange(t0 - N_HIST + 1, t0 + 1)
        x = data[:, idx]                               # (n, 4, 7, 64, 64, 64)
        yield x.reshape(n, -1, *x.shape[-3:]), data[:, t0 + 1]


if __name__ == "__main__":
    train, valid = load(rebuild="--rebuild" in sys.argv)
    print(f"train {tuple(train.shape)}  valid {tuple(valid.shape)}  "
          f"(memmap-backed, copy-on-write)")
