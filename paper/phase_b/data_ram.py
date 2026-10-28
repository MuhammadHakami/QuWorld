"""In-RAM, protocol-faithful data for Phase-B training.

Replaces the_well's streaming DataLoader in the TRAINING loop (HDF5 page-cache
churn + worker processes were implicated in device-freezing memory fills).
Loads the full split into normalized fp32 tensors once, caches to a single
.pt file, and serves 4-frame-history windows by pure tensor indexing in the
main process. Normalization uses the OFFICIAL stats.yaml (ZScore per field,
velocity per component) so batches are numerically equivalent to
WellDataset(use_normalization=True) — verified by build-time assertion.

Channel order matches the_well's channels-first formatter:
fields (density, pressure, vx, vy), input flattened t-major -> 16 channels.
"""

import glob
import pathlib

import h5py
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "data" / "phaseb_cache.pt"
STATS = ROOT / "data" / "trl2d" / "stats.yaml"
N_HIST = 4


def _stats():
    s = yaml.safe_load(open(STATS))
    mean = [s["mean"]["density"], s["mean"]["pressure"], *s["mean"]["velocity"]]
    std = [s["std"]["density"], s["std"]["pressure"], *s["std"]["velocity"]]
    return (torch.tensor(mean).view(1, 1, 4, 1, 1),
            torch.tensor(std).view(1, 1, 4, 1, 1))


def _load_split(split: str) -> torch.Tensor:
    mean, std = _stats()
    trajs = []
    for f in sorted(glob.glob(str(ROOT / f"data/trl2d/data/{split}/*.hdf5"))):
        with h5py.File(f) as h:
            den = torch.from_numpy(h["t0_fields/density"][:])
            pre = torch.from_numpy(h["t0_fields/pressure"][:])
            vel = torch.from_numpy(h["t1_fields/velocity"][:])
        x = torch.stack([den, pre, vel[..., 0], vel[..., 1]], dim=2)  # (n,T,4,H,W)
        trajs.append(x)
    x = torch.cat(trajs).float()
    return torch.nan_to_num((x - mean) / std)


def load(rebuild: bool = False):
    """Returns (train, valid) as (n_traj, T, 4, 128, 384) normalized fp32."""
    if CACHE.exists() and not rebuild:
        d = torch.load(CACHE, weights_only=True, mmap=True)
        return d["train"], d["valid"]
    train, valid = _load_split("train"), _load_split("valid")
    _verify_against_official(valid)
    tmp = CACHE.with_suffix(".tmp")
    torch.save({"train": train, "valid": valid}, tmp)
    tmp.replace(CACHE)
    return train, valid


def _verify_against_official(valid: torch.Tensor) -> None:
    """Assert numerical equivalence with the_well's normalized WellDataset."""
    from einops import rearrange
    from the_well.data import WellDataset
    from the_well.data.normalization import ZScoreNormalization

    ds = WellDataset(
        well_base_path=str(ROOT / "data/well"),
        well_dataset_name="turbulent_radiative_layer_2D",
        well_split_name="valid", n_steps_input=N_HIST, n_steps_output=1,
        use_normalization=True, normalization_type=ZScoreNormalization)
    ref = torch.nan_to_num(ds[0]["input_fields"])          # (4, H, W, 4)
    ours = rearrange(valid[0, 0:N_HIST], "t c h w -> t h w c")
    diff = (ref - ours).abs().max().item()
    assert diff < 1e-4, f"RAM loader mismatch vs the_well: {diff}"
    print(f"data_ram verified vs the_well (max diff {diff:.2e})")


def train_batch(data: torch.Tensor, bsz: int, gen: torch.Generator, dev: str):
    n, t = data.shape[0], data.shape[1]
    ni = torch.randint(0, n, (bsz,), generator=gen)
    ti = torch.randint(N_HIST - 1, t - 1, (bsz,), generator=gen)
    idx = ti.view(-1, 1) + torch.arange(-N_HIST + 1, 1)
    x = data[ni.view(-1, 1), idx]                          # (B, 4, 4, H, W)
    y = data[ni, ti + 1]                                   # (B, 4, H, W)
    return x.reshape(bsz, -1, *x.shape[-2:]).to(dev), y.to(dev)


def valid_windows(data: torch.Tensor, stride_t: int = 2):
    """Yield (x, y) covering the valid split deterministically."""
    n, t = data.shape[0], data.shape[1]
    for t0 in range(N_HIST - 1, t - 1, stride_t):
        idx = torch.arange(t0 - N_HIST + 1, t0 + 1)
        x = data[:, idx]                                   # (n, 4, 4, H, W)
        yield x.reshape(n, -1, *x.shape[-2:]), data[:, t0 + 1]
