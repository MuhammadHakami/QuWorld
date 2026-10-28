"""RealPDE foil (RealPDEBench, ICLR 2026): fast window loading, normaliser and checkpoint trunks.

The windows are exactly RealPDEBench's FoilHFDataset samples (its index files, horizon 20 -> 20, 2x spatial
subsampling, p = 0 on real data). The difference is speed: each trajectory is decoded once instead of once per window.
The normaliser is RealPDEBench's GaussianNormalizer statistics (`mean_std.pt`), applied as in its eval.py.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baseline" / "RealPDEBench")); sys.path.insert(0, str(ROOT / "paper"))
import gpucap  # noqa: E402

gpucap.cap_from_env()
DATA = ROOT / "data" / "realpdebench"
CKPT = ROOT / "data" / "realpde" / "baseline_checkpoints"          # competition release (different data processing)
BENCH_CKPT = ROOT / "data" / "realpdebench_models" / "foil"        # RealPDEBench's own released checkpoints
CACHE = ROOT / "data" / "realpde_cache"
LEAKED = frozenset({"3750_10.0.h5", "3750_0.0.h5"})       # RealPDEBench test trajectories inside the competition release
DEV = torch.device("cuda")


IN_STEP, OUT_STEP, SUB_S = 20, 20, 2                        # FoilHFDataset defaults (sub_s_real = sub_s_numerical = 2)


def sim_ids(split: str, dataset_type: str = "real") -> set:
    """Trajectories of a RealPDEBench test partition: 'in_dist' | 'out_dist' | 'remain' (its params JSON keys)."""
    name = f"remain_params_{dataset_type}.json" if split == "remain" else f"{split}_test_params_{dataset_type}.json"
    return set(json.loads((DATA / "foil" / name).read_text()))


def load_windows(mode: str, dataset_type: str = "real", test_mode: str = "all", limit: int = 0):
    """-> X (N, 20, 64, 128, c), Y (N, 20, 64, 128, c) float32 CPU tensors and the sim_id per window, in RealPDEBench's
    index order: the FoilHFDataset(mode, dataset_type, test_mode) windows with N_autoregressive = 1 (verified equal to
    its __getitem__). c = 2 on real data (u, v; RealPDEBench's p channel is identically 0 there and is re-added by
    `forward`), 3 on numerical data. Arrow rows are streamed one trajectory at a time (constructing FoilHFDataset reads
    every full ~1 GB row just to map sim_ids); the full split is cached to data/realpde_cache/win_<mode>_<type>.pt."""
    dst = CACHE / f"win_{mode}_{dataset_type}.pt"
    npy = CACHE / f"win_{mode}_{dataset_type}"                    # disk-backed build (TRAIN is ~25 GB)
    if dst.exists():
        X, Y, sims = torch.load(dst, mmap=True)      # file-backed pages: reclaimable, never anonymous memory
    elif (npy.parent / (npy.name + ".done")).exists():
        sims = json.loads((npy.parent / (npy.name + "_sims.json")).read_text())
        X = torch.from_numpy(np.load(str(npy) + "_X.npy", mmap_mode="r")); Y = torch.from_numpy(np.load(str(npy) + "_Y.npy", mmap_mode="r"))
    else:
        from datasets import load_from_disk
        hf = DATA / "foil" / "hf_dataset"
        idx = json.loads((hf / f"{mode}_index_{dataset_type}.json").read_text())
        traj = load_from_disk(str(hf / dataset_type))
        row_of = {s: i for i, s in enumerate(traj["sim_id"])}
        traj = traj.select_columns(["u", "v", "shape_t", "shape_h", "shape_w"] + ([] if dataset_type == "real" else ["p"]))
        by_sim = defaultdict(list)
        for n, e in enumerate(idx):
            by_sim[e["sim_id"]].append((n, e["time_id"]))
        X = Y = None
        for sim, items in by_sim.items():
            row = traj[row_of[sim]]
            shape = (row["shape_t"], row["shape_h"], row["shape_w"])
            dec = lambda key: np.frombuffer(row[key], dtype=np.float32).reshape(shape)[:, ::SUB_S, ::SUB_S]
            chans = [dec("u"), dec("v")] + ([] if dataset_type == "real" else [dec("p")])
            full = torch.from_numpy(np.stack(chans, axis=-1))                  # copy: the Arrow row is released
            if X is None:                                                      # memmap buffers, filled trajectory by trajectory
                CACHE.mkdir(parents=True, exist_ok=True)
                X = torch.from_numpy(np.lib.format.open_memmap(str(npy) + "_X.npy", mode="w+", dtype=np.float32, shape=(len(idx), IN_STEP, *full.shape[1:])))
                Y = torch.from_numpy(np.lib.format.open_memmap(str(npy) + "_Y.npy", mode="w+", dtype=np.float32, shape=(len(idx), OUT_STEP, *full.shape[1:])))
            for n, t in items:
                X[n], Y[n] = full[t:t + IN_STEP], full[t + IN_STEP:t + IN_STEP + OUT_STEP]
            del row, chans, full
        sims = [e["sim_id"] for e in idx]
        (npy.parent / (npy.name + "_sims.json")).write_text(json.dumps(sims)); (npy.parent / (npy.name + ".done")).touch()
    keep = None
    if mode in ("val", "test") and test_mode != "all":
        allowed = sim_ids(test_mode, dataset_type); keep = [i for i, s in enumerate(sims) if s in allowed]
    if limit:
        keep = (keep if keep is not None else list(range(len(sims))))[:limit]
    if keep is not None:                              # small subsets only (smoke tests); full splits stay mmap-backed
        X, Y, sims = X[keep], Y[keep], [sims[i] for i in keep]
    return X, Y, sims


def open_pred(path: pathlib.Path, shape) -> torch.Tensor:
    """Disk-backed float32 prediction buffer (np.memmap): predictions never sit in anonymous memory."""
    arr = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float32, shape=tuple(shape))
    return torch.from_numpy(arr)


def normaliser():
    mi, mt, si, st = torch.load(DATA / "foil" / "mean_std.pt")
    si = torch.where(si == 0, torch.ones_like(si), si); st = torch.where(st == 0, torch.ones_like(st), st)
    return mi.to(DEV), mt.to(DEV), si.to(DEV), st.to(DEV)


def forward(net, x, norm, gaussian: bool):
    """Raw-units input window -> raw-units prediction (B, 20, H, W, c_in), following RealPDEBench eval.py
    (N_autoregressive = 1). Real windows carry (u, v); the models take (u, v, p) with p = 0 there."""
    c = x.shape[-1]
    if c == 2:
        x = torch.cat([x, torch.zeros_like(x[..., :1])], dim=-1)
    if not gaussian:
        return net(x)[..., :c]
    mi, mt, si, st = norm
    return (net((x - mi) / si) * st + mt)[..., :c]


def _state(model: str, stage: str = "finetune"):
    """RealPDEBench checkpoints: stage in {finetune, real, numerical}; competition ones: comp_sim_real_ft / comp_sim."""
    if stage.startswith("comp_"):
        name = f"sim_real_{model}.pth" if stage == "comp_sim_real_ft" else f"sim_{model}.pth"
        path = CKPT / ("sim_real_ft" if stage == "comp_sim_real_ft" else "sim_pretrain") / name
    else:
        path = BENCH_CKPT / model / f"{stage}.pth"
    return torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]


def build(model: str, shape_in, shape_out, variant: dict | None = None, stage: str = "finetune"):
    """Checkpoint -> RealPDEBench module; hyper-parameters read from the state_dict shapes (yaml where they agree)."""
    sd, v = _state(model, stage), variant or {}
    shape_in, shape_out = (*shape_in[:-1], 3), (*shape_out[:-1], 3)   # models take (u, v, p) even on real data
    if model == "fno":
        from realpdebench.model.fno import FNO3d
        m1, m2, m3 = sd["spectral_convs.0.weights1"].shape[2:]
        n_layers = len({k.split(".")[1] for k in sd if k.startswith("spectral_convs.")})
        net = FNO3d(modes1=m1, modes2=m2, modes3=m3, n_layers=n_layers, width=sd["fc0.weight"].shape[0],
                    shape_in=shape_in, shape_out=shape_out)
    elif model == "unet":
        from realpdebench.model.unet import Unet3d
        net = Unet3d(dim=shape_in[1], out_channels=shape_out[-1], dim_mults=tuple(v.get("dim_mults", (1, 2, 4))),
                     channels=shape_in[-1], in_time=shape_in[0], out_time=shape_out[0])
    elif model == "cno":
        from realpdebench.model.cno import CNO3d
        n_layers = len({k.split(".")[1] for k in sd if k.startswith("encoder.")})
        net = CNO3d(in_dim=shape_in[-1], out_dim=shape_out[-1], out_dim_mult=1, in_size=shape_in[2], N_layers=n_layers)
    elif model == "transolver":
        from realpdebench.model.TRANSOLVER_libs.Transolver_Structured_Mesh_3D import Model
        n_layers = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
        hidden = sd["placeholder"].shape[0]
        H, W, D = v.get("hwd", tuple(shape_in[:3]))
        net = Model(space_dim=3, n_layers=n_layers, n_hidden=hidden, n_head=sd["blocks.0.Attn.temperature"].shape[1],
                    H=H, W=W, D=D, Time_Input=False, unified_pos=False, fun_dim=0, out_dim=shape_out[-1], ref=4,
                    dropout=0.0, act="gelu", mlp_ratio=sd["blocks.0.mlp.linear_pre.0.weight"].shape[0] // hidden,
                    slice_num=sd["blocks.0.Attn.in_project_slice.weight"].shape[0])
    else:
        raise ValueError(f"unknown trunk {model}")
    net.load_state_dict(sd, strict=True)
    return net.to(DEV).eval()


@torch.no_grad()
def predict(net, X, norm, gaussian: bool, batch: int = 64, out: torch.Tensor | None = None, start: int = 0):
    """Predictions for X[start:], written into `out` (e.g. an open_pred buffer) when given, else returned."""
    if out is None:
        return torch.cat([forward(net, X[i:i + batch].clone().to(DEV), norm, gaussian).float().cpu() for i in range(0, len(X), batch)])
    for i in range(start, len(X), batch):
        out[i:i + batch] = forward(net, X[i:i + batch].clone().to(DEV), norm, gaussian).float().cpu()   # clone: see to_gpu note
    return out


def rel_l2(P, Y, c: int = 2):
    """RealPDEBench's rel-L2 (mean over windows of ||p - y|| / ||y|| on the measured channels)."""
    d = (P[..., :c] - Y[..., :c]).flatten(1).norm(dim=1)
    return (d / Y[..., :c].flatten(1).norm(dim=1)).mean().item()


METRIC_NAMES = ["rmse", "mae", "rel_l2", "r2", "ke", "f_err", "low_f", "mid_f", "high_f", "rel_low_f", "rel_mid_f",
                "rel_high_f", "freq_err"]


def official_metrics(P, Y, c: int = 2):
    """RealPDEBench eval.py path for N_autoregressive <= 4: eval_metrics on the whole set as one batch."""
    from realpdebench.utils.metrics import eval_metrics
    vals = eval_metrics(P.to(DEV), Y.to(DEV), c, P.shape[0])
    return {n: float(x) for n, x in zip(METRIC_NAMES, vals)} | {"windows": int(P.shape[0])}


def _radial_sum(F2: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
    """eval_metrics' shell sum: (b, t, h, w, c) |F|^2 -> (b, nbins, c) over i<t/2, j<h/2, k<w/2, bin floor(|ijk|)."""
    nb = min(t // 2, h // 2, w // 2)
    i, j, k = torch.meshgrid(torch.arange(t // 2), torch.arange(h // 2), torch.arange(w // 2), indexing="ij")
    it = torch.sqrt((i ** 2 + j ** 2 + k ** 2).double()).floor().long()        # == math.floor(math.sqrt(int))
    keep = it <= nb - 1
    sub = F2[:, : t // 2, : h // 2, : w // 2][:, keep]                            # (b, n_keep, c)
    return torch.zeros(F2.shape[0], nb, F2.shape[-1], device=F2.device, dtype=F2.dtype).index_add_(1, it[keep].to(F2.device), sub)


@torch.no_grad()
def official_metrics_chunked(P, Y, c: int = 2, chunk: int = 256, idx: torch.Tensor | None = None):
    """Exactly eval_metrics(P[idx], Y[idx], c, batch_size=N) (the full set as ONE batch, as eval.py does for
    N_autoregressive <= 4), accumulated over window chunks so memory stays bounded (the one-batch FFT needs ~45 GB at
    N = 5140). `idx` selects a subset without materialising it (P, Y may be disk-backed)."""
    idx = torch.arange(Y.shape[0]) if idx is None else idx
    N, t, h, w = len(idx), Y.shape[1], Y.shape[2], Y.shape[3]
    sl = lambda A, i: A[idx[i:i + chunk]][..., :c].to(DEV).double()
    ymean = sum(sl(Y, i).sum(0) for i in range(0, N, chunk)) / N                   # per-position mean over windows (R2)
    se = ae = rel = ss_res = ss_tot = ke = fq = 0.0
    errF = normF = 0.0
    for i in range(0, N, chunk):
        p, y = sl(P, i), sl(Y, i)
        d = p - y
        se += d.square().sum().item(); ae += d.abs().sum().item()
        rel += (d.flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).sum().item()
        ss_res += d.square().sum().item(); ss_tot += (y - ymean).square().sum().item()
        if c >= 2:
            kin = lambda x: 0.5 * (((x[..., 0] - x[..., 0].mean(1, keepdim=True)) ** 2).mean(1)
                                   + ((x[..., 1] - x[..., 1].mean(1, keepdim=True)) ** 2).mean(1))
            ke += (kin(p) - kin(y)).abs().sum().item()
        pF, yF = torch.fft.fftn(p, dim=[1, 2, 3]), torch.fft.fftn(y, dim=[1, 2, 3])
        errF = errF + _radial_sum((pF - yF).abs().square(), t, h, w).sum(0)
        normF = normF + _radial_sum(yF.abs().square(), t, h, w).sum(0)
        fq += (torch.fft.fftn(p.sum(dim=[2, 3, 4]), dim=1) - torch.fft.fftn(y.sum(dim=[2, 3, 4]), dim=1)).abs().sum().item()
    n_el = N * t * h * w * c
    eF = torch.sqrt(errF / N) / (t * h * w); nF = torch.sqrt(normF / N) / (t * h * w)
    nb = eF.shape[0]; lo, hi = int(np.round(nb / 3)), int(np.round(nb * 2 / 3))
    r = eF / nF
    out = dict(rmse=math.sqrt(se / n_el), mae=ae / n_el, rel_l2=rel / N, r2=1 - ss_res / ss_tot,
               ke=ke / (N * h * w) if c >= 2 else 0.0, f_err=eF.mean().item(), low_f=eF[:lo].mean().item(),
               mid_f=eF[lo:hi].mean().item(), high_f=eF[hi:].mean().item(), rel_low_f=r[:lo].mean().item(),
               rel_mid_f=r[lo:hi].mean().item(), rel_high_f=r[hi:].mean().item(), freq_err=fq / (N * t))
    return out | {"windows": int(N)}
