"""Direct-from-HDF5, bounded-residency data for the 2-D Well datasets
(planetswe, rayleigh_benard) — serving semantics identical to data_ram /
data_ram_mhd (official stats.yaml ZScore per field, vectors per component,
std clipped at the_well's min_denom=1e-4; N_HIST=4 history frames flattened
t-major into channels), verified against WellDataset at first use.

No cache copy: The Well HDF5 fields are stored contiguous and uncompressed,
so every frame is read with os.pread at (dataset offset + frame index *
frame bytes) and evicted from page cache right after (POSIX_FADV_DONTNEED) —
the same freeze-safety rationale as data_ram_mhd.PreadSplit without a
~450 GB fp32 copy of the train splits.

Channel order = the_well channels-first formatter: t0_fields then t1_fields,
each in the file's attrs["field_names"] order, vector components last.
"""

import glob
import os
import pathlib
import sys

import h5py
import numpy as np
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
N_HIST = 4
MIN_STD = 1e-4  # the_well ZScoreNormalization min_denom
PUBLISHED = {  # The Well paper, Table 2 (test VRMSE, one-step)
    "planetswe": {"TFNO": 0.0853, "FNO": 0.1727, "CNextU-net": 0.3724, "U-net": 0.3620},
    "rayleigh_benard": {"TFNO": 0.6566, "FNO": 0.8340, "CNextU-net": 0.6699, "U-net": 1.4860},
}


def dataset_dir(name: str) -> pathlib.Path:
    return ROOT / "data" / "well" / name


def _stats(name: str, fields):
    """Per-channel (mean, std) in the_well channel order; fields = [(fname, ncomp)]."""
    s = yaml.safe_load(open(dataset_dir(name) / "stats.yaml"))
    mean, std = [], []
    for fname, ncomp in fields:
        m, d = s["mean"][fname], s["std"][fname]
        if ncomp > 1:                              # rank-2 tensor stats are nested [[..], [..]]: flatten in C order
            m, d = np.ravel(m).tolist(), np.ravel(d).tolist()
        if ncomp == 1:
            mean.append(float(m)); std.append(float(d))
        else:
            assert len(m) == ncomp, f"{fname}: stats have {len(m)} components, file has {ncomp}"
            mean.extend(float(v) for v in m); std.extend(float(v) for v in d)
    std = [max(v, MIN_STD) for v in std]
    c = len(mean)
    return (np.asarray(mean, dtype=np.float32).reshape(c, 1, 1),
            np.asarray(std, dtype=np.float32).reshape(c, 1, 1))


class _Field:
    __slots__ = ("offset", "ncomp", "frame_bytes")

    def __init__(self, ds):
        assert ds.chunks is None and ds.compression is None, "field is chunked/compressed: pread layout invalid"
        self.offset = ds.id.get_offset()
        assert self.offset is not None, "field has no contiguous storage offset"
        shape = ds.shape                       # (n_traj, T, H, W[, d1[, d2]]); tensors flattened in C order (the_well)
        self.ncomp = int(np.prod(shape[4:])) if len(shape) > 4 else 1
        self.frame_bytes = int(np.prod(shape[2:])) * 4


class Split:
    """One Well split (all HDF5 files), addressed by global trajectory index."""

    def __init__(self, name: str, split: str, max_traj_per_file: int | None = None):
        self.name, self.split = name, split
        files = sorted(glob.glob(str(dataset_dir(name) / "data" / split / "*.hdf5")))
        assert files, f"no HDF5 files for {name}/{split}"
        self.files, self.fds, self.file_fields, self.index = [], [], [], []
        T = H = W = None
        field_list = None
        for f in files:
            with h5py.File(f) as h:
                names = [(g, str(n)) for g in ("t0_fields", "t1_fields", "t2_fields") if g in h
                         for n in h[g].attrs["field_names"]]         # t2: rank-2 tensors (e.g. active_matter D, E)
                if field_list is None:
                    field_list = names
                assert names == field_list, f"field order differs in {f}"
                fields = [_Field(h[g][n]) for g, n in names]
                shape = h[names[0][0]][names[0][1]].shape
                n_traj, t, hh, ww = shape[0], shape[1], shape[2], shape[3]
                assert T in (None, t) and H in (None, hh) and W in (None, ww), f"inconsistent shapes in {f}"
                T, H, W = t, hh, ww
            use = n_traj if max_traj_per_file is None else min(n_traj, max_traj_per_file)
            fi = len(self.files)
            self.files.append(f); self.fds.append(os.open(f, os.O_RDONLY)); self.file_fields.append(fields)
            self.index.extend((fi, k) for k in range(use))
        self.T, self.H, self.W = T, H, W
        self.fields = [(n, ff.ncomp) for (g, n), ff in zip(field_list, self.file_fields[0])]
        self.C = sum(nc for _, nc in self.fields)
        self.mean, self.std = _stats(name, self.fields)
        self.n_traj_per_file = {f: sum(1 for fi, _ in self.index if fi == i) for i, f in enumerate(self.files)}

    @property
    def n(self) -> int:
        return len(self.index)

    @property
    def shape(self):
        return (self.n, self.T, self.C, self.H, self.W)

    def frames(self, traj: int, times) -> torch.Tensor:
        """(len(times), C, H, W) normalized fp32 — transient buffers only."""
        fi, k = self.index[traj]
        fd, fields = self.fds[fi], self.file_fields[fi]
        out = np.empty((len(times), self.C, self.H, self.W), dtype=np.float32)
        for j, ti in enumerate(times):
            c0 = 0
            for ff in fields:
                off = ff.offset + (k * self.T + int(ti)) * ff.frame_bytes
                buf = os.pread(fd, ff.frame_bytes, off)
                assert len(buf) == ff.frame_bytes, f"short read in {self.files[fi]}"
                os.posix_fadvise(fd, off, ff.frame_bytes, os.POSIX_FADV_DONTNEED)
                arr = np.frombuffer(buf, dtype=np.float32)
                if ff.ncomp == 1:
                    out[j, c0] = arr.reshape(self.H, self.W)
                else:
                    out[j, c0:c0 + ff.ncomp] = np.moveaxis(arr.reshape(self.H, self.W, ff.ncomp), -1, 0)
                c0 += ff.ncomp
        out = np.nan_to_num((out - self.mean) / self.std)
        return torch.from_numpy(out)

    def window(self, traj: int, t0: int) -> tuple[torch.Tensor, torch.Tensor]:
        """x (N_HIST*C, H, W) t-major, y (C, H, W) = frame t0+1; t0 >= N_HIST-1."""
        fr = self.frames(traj, range(t0 - N_HIST + 1, t0 + 2))
        return fr[:N_HIST].reshape(-1, self.H, self.W), fr[N_HIST]

    def close(self):
        for fd in self.fds:
            os.close(fd)


def vrmse_rows(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B,) mean-over-fields VRMSE (the_well: eps 1e-7)."""
    mse = (pred - true).square().mean(dim=(2, 3))
    var = (true - true.mean(dim=(2, 3), keepdim=True)).square().mean(dim=(2, 3))
    return (mse / (var + 1e-7)).sqrt().mean(dim=1)


def train_batch(data: Split, bsz: int, gen: torch.Generator, dev: str):
    ni = torch.randint(0, data.n, (bsz,), generator=gen)
    ti = torch.randint(N_HIST - 1, data.T - 1, (bsz,), generator=gen)
    xs, ys = zip(*(data.window(int(a), int(b)) for a, b in zip(ni, ti)))
    return torch.stack(xs).to(dev), torch.stack(ys).to(dev)


def fixed_windows(data: Split, per_traj: int):
    """Deterministic (traj, t0) list: per_traj evenly spaced starts per trajectory."""
    starts = np.linspace(N_HIST - 1, data.T - 2, per_traj).round().astype(int)
    return [(k, int(t0)) for k in range(data.n) for t0 in starts]


def batched(data: Split, pairs, bsz: int):
    for i in range(0, len(pairs), bsz):
        xs, ys = zip(*(data.window(k, t0) for k, t0 in pairs[i:i + bsz]))
        yield torch.stack(xs), torch.stack(ys)


def all_test_windows(data: Split):
    """Every (traj, t0) window of the split, in the_well order (traj-major, t ascending)."""
    return [(k, t0) for k in range(data.n) for t0 in range(N_HIST - 1, data.T - 1)]


def trajectory_windows(data: Split, bsz: int):
    """Stream every window of the split trajectory by trajectory (each trajectory
    read once, sequentially, then evicted) — for full-split evaluation.
    Yields (pairs, x, y) with pairs = [(traj, t0), ...] for the batch."""
    for k in range(data.n):
        fr = data.frames(k, range(data.T))                       # (T, C, H, W)
        starts = list(range(N_HIST - 1, data.T - 1))
        for i in range(0, len(starts), bsz):
            chunk = starts[i:i + bsz]
            x = torch.stack([fr[t0 - N_HIST + 1:t0 + 1].reshape(-1, data.H, data.W) for t0 in chunk])
            y = torch.stack([fr[t0 + 1] for t0 in chunk])
            yield [(k, t0) for t0 in chunk], x, y
        del fr


def verify_against_official(name: str, split: str = "valid") -> None:
    """Assert numerical equivalence with the_well's normalized WellDataset (sample 0)."""
    from einops import rearrange
    from the_well.data import WellDataset
    from the_well.data.normalization import ZScoreNormalization

    ds = WellDataset(well_base_path=str(ROOT / "data/well"), well_dataset_name=name, well_split_name=split,
                     n_steps_input=N_HIST, n_steps_output=1, use_normalization=True,
                     normalization_type=ZScoreNormalization)
    sample = ds[0]
    ref_x = torch.nan_to_num(sample["input_fields"])      # (4, H, W, C)
    ref_y = torch.nan_to_num(sample["output_fields"][0])  # (H, W, C)
    ours = Split(name, split)
    x, y = ours.window(0, N_HIST - 1)
    ours_x = rearrange(x.reshape(N_HIST, ours.C, ours.H, ours.W), "t c h w -> t h w c")
    ours_y = rearrange(y, "c h w -> h w c")
    dx = (ref_x - ours_x).abs().max().item(); dy = (ref_y - ours_y).abs().max().item()
    ours.close()
    assert dx < 1e-4 and dy < 1e-4, f"well2d mismatch vs the_well ({name}/{split}): input {dx}, output {dy}"
    print(f"well2d {name}/{split} verified vs the_well (max diff in={dx:.2e} out={dy:.2e}); "
          f"fields {ours.fields} shape {ours.shape}")


if __name__ == "__main__":
    verify_against_official(sys.argv[1] if len(sys.argv) > 1 else "planetswe",
                            sys.argv[2] if len(sys.argv) > 2 else "valid")
