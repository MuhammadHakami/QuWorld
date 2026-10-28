"""Frozen-trunk caches for the allocation-strategy study (33-allocation-strategy-study.md).

    runs/capped.sh alloc-cache 24 .venv/bin/python paper/alloc/cache.py well turbulent_radiative_layer_2D
    runs/capped.sh alloc-cache 24 .venv/bin/python paper/alloc/cache.py realpde unet

For every window the frozen trunk predicts p once. The five fields d = p - x_t (predicted change), p, x_t,
dx = x_t - x_{t-1} and r = y - p (the residual) are Haar-transformed into tiles. For each tile set, in descending
order of its score, the cache stores M tile indices and their coefficients:
  proxy   per input, by the causal proxy energy sum_c d^2 / Dx_c
  static  one fixed set, by mean TRAIN residual energy (computed on a TRAIN subset first)
  random  per input, uniformly random (seeded by window)
  oracle  per input, by true weighted residual energy (diagnostic ceiling only)
It also stores, per window:
  - per-channel residual energy E_c;
  - metric denominators D_c (so every L2 metric is exact from coefficients);
  - residual means (pooled insertion), global features;
  - capture ceilings of each rule at K in KS.
TRAIN / VALID / TEST windows are fixed by seed. The trunk is fp32: CNext / U-Net / FNO are convolutional or spectral,
so NVFP4 (Linear layers only) does not apply. Every arm shares the same cached trunk.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "paper"))
import gpucap  # noqa: E402
from haar import TileBank  # noqa: E402

gpucap.cap_from_env()
DEV = torch.device("cuda")
OUT = pathlib.Path(__import__("os").environ.get("ALLOC_CACHE_OUT", ROOT / "runs" / "alloc" / "cache"))
M = 320
KS = (10, 30, 100, 300)
EPS = 1e-7
EVAL_ONLY = bool(int(__import__("os").environ.get("ALLOC_EVAL_ONLY", "0")))   # rebuild valid/test only


class FrameBank:
    """Per-frame 2-D Haar over (H, W) for fields (B, T, C, H, W); tile position = (h, w, t)."""

    def __init__(self, T: int, H: int, W: int, levels: int):
        self.b, self.T = TileBank((H, W), levels), T
        n = self.b.n
        self.n, self.n_orient, self.levels = T * n, self.b.n_orient, levels
        self.level, self.orient = self.b.level.repeat(T), self.b.orient.repeat(T)
        t = torch.arange(T).repeat_interleave(n).float()
        self.pos = torch.cat([self.b.pos[:, :2].repeat(T, 1), t[:, None]], 1)

    def flatten(self, x: torch.Tensor) -> torch.Tensor:          # (B, T, C, H, W) -> (B, T*n, C)
        B, T, C = x.shape[:3]
        return self.b.flatten(x.reshape(B * T, C, *x.shape[3:])).reshape(B, T * self.b.n, C)


class Extractor:
    """Accumulates per-window records from GPU batches of (xt, dx, p, y) laid out for `bank` (channels at dim 1 for
    TileBank; (B, T, C, H, W) for FrameBank). metric: 'vrmse' (per-channel variance of y) | 'rel_l2' (||y||^2)."""

    def __init__(self, bank, metric: str, static_idx: torch.Tensor | None, seed: int):
        self.bank, self.metric, self.static_idx, self.seed = bank, metric, static_idx, seed
        self.rec = {k: [] for k in ("E", "D", "rmean", "glob", "npix")}
        self.sets = {s: {"idx": [], "coef": []} for s in ("proxy", "static", "random", "oracle")}
        self.ceil = {f"{s}@{k}": [] for s in ("proxy", "static", "random", "oracle") for k in KS}
        self.ceil["dc"] = []
        self.static_sum, self.static_n = None, 0

    def _chan_stats(self, x):                                    # per-sample, per-channel over every other axis
        if isinstance(self.bank, FrameBank):
            x = x.transpose(1, 2)                                   # (B, C, T, H, W)
        dims = tuple(range(2, x.dim()))
        return x.mean(dims), x.var(dims, unbiased=False), x.square().sum(dims)

    @torch.no_grad()
    def add(self, xt, dx, p, y, window_ids, static_pass: bool = False):
        bank = self.bank
        d, r = p - xt, y - p
        _, vy, ey = self._chan_stats(y)
        _, vx, _ = self._chan_stats(xt if not isinstance(bank, FrameBank) else xt[:, :1])
        npix = r[0, 0].numel() if not isinstance(bank, FrameBank) else r[0, :, 0].numel()
        if self.metric == "vrmse":
            D = npix * (vy + EPS); Dx = npix * (vx + EPS)
        else:
            D = ey.sum(1, keepdim=True).expand_as(vy).clone(); Dx = D.clone()
        cr = bank.flatten(r)                                          # (B, N, C)
        er = (cr.square() / D[:, None]).sum(-1)                       # weighted residual energy per tile
        if static_pass:
            self.static_sum = er.sum(0) if self.static_sum is None else self.static_sum + er.sum(0)
            self.static_n += er.shape[0]
            return
        cd = bank.flatten(d)
        ed = (cd.square() / Dx[:, None]).sum(-1)
        coefs = torch.stack([cd, bank.flatten(p), bank.flatten(xt), bank.flatten(dx), cr], 2)   # (B, N, 5, C)
        B, N = er.shape
        g = torch.Generator(device="cpu")
        rnd = []
        for w in window_ids:
            g.manual_seed(self.seed * 1_000_003 + int(w))
            rnd.append(torch.randperm(N, generator=g)[:M])
        order = {"proxy": ed.topk(M, 1).indices, "oracle": er.topk(M, 1).indices,
                 "static": self.static_idx.to(DEV)[None].expand(B, -1), "random": torch.stack(rnd).to(DEV)}
        tot = er.sum(1)
        for s, idx in order.items():
            self.sets[s]["idx"].append(idx.int().cpu())
            self.sets[s]["coef"].append(coefs.gather(1, idx[:, :, None, None].expand(-1, -1, 5, coefs.shape[-1])).cpu())
            cap = er.gather(1, idx)
            for k in KS:
                self.ceil[f"{s}@{k}"].append((cap[:, :k].sum(1) / tot).cpu())
        E = cr.square().sum(1)                                        # (B, C) = ||r_c||^2 (orthonormal)
        rmean, _, _ = self._chan_stats(r)
        self.ceil["dc"].append(((npix * rmean.square() / D).sum(1) / tot).cpu())
        feats = []
        for f in (d, p, xt, dx):
            m, v, _ = self._chan_stats(f)
            feats += [m, v.clamp_min(1e-12).sqrt()]
        self.rec["E"].append(E.double().cpu()); self.rec["D"].append(D.double().cpu())
        self.rec["rmean"].append(rmean.cpu()); self.rec["glob"].append(torch.cat(feats, 1).cpu())
        self.rec["npix"].append(torch.full((B,), npix))

    def static_order(self) -> torch.Tensor:
        return (self.static_sum / self.static_n).topk(M).indices.cpu()

    def save(self, path: pathlib.Path, meta: dict):
        out = {"meta": meta, "sets": {s: {k: torch.cat(v) for k, v in d.items()} for s, d in self.sets.items()},
               "ceil": {k: torch.cat(v) for k, v in self.ceil.items()},
               **{k: torch.cat(v) for k, v in self.rec.items()},
               "level": self.bank.level.short(), "orient": self.bank.orient.short(), "pos": self.bank.pos}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp"); torch.save(out, tmp); tmp.replace(path)
        c = out["ceil"]
        print(f"[cache] {path.name}: N={len(out['E'])} DC {c['dc'].mean():.4f} | " + " ".join(
            f"{s}@30 {c[f'{s}@30'].mean():.4f}" for s in ("oracle", "proxy", "static", "random")), flush=True)


# ---------------------------------------------------------------- The Well (2-D, CNextU-net official checkpoints)

def well(dataset: str, n_train: int, n_valid: int, n_test: int, bsz: int, levels: int):
    sys.path.insert(0, str(ROOT / "paper" / "phase_b"))
    import well2d
    from well2d_backbone import build
    model = build("vanilla", dataset).to(DEV).eval()
    SEALED = __import__("os").environ.get("ALLOC_NO_TEST") == "1"          # doc 35 §31: TEST sealed, never opened
    TEST_ONLY = __import__("os").environ.get("ALLOC_TEST_ONLY") == "1"     # doc 35 §31e: build test.pt only, from the
    assert not (SEALED and TEST_ONLY)                                       # locked train.pt tile order; never rewrites
    if TEST_ONLY:                                                           # train.pt / valid.pt
        assert (OUT / dataset / "train.pt").exists() and (OUT / dataset / "valid.pt").exists(), "locked caches missing"
    sp = {s: well2d.Split(dataset, s) for s in (("train", "valid") if SEALED else ("valid", "test") if TEST_ONLY
                                                 else ("train", "valid", "test"))}
    C, H, W = sp["valid"].C, sp["valid"].H, sp["valid"].W
    bank = TileBank((H, W), levels)
    rng = np.random.default_rng(0)

    def windows(split, n):
        s = sp[split]
        if split == "train":
            return [(int(rng.integers(s.n)), int(rng.integers(well2d.N_HIST - 1, s.T - 1))) for _ in range(n)]
        # uniform sample of all (trajectory, start) windows, so the mean matches the full-split metric (amendment 3:
        # evenly spaced starts over-weighted the initial transient, t0 = 3, by up to 28x on rayleigh_benard)
        per = int(__import__("os").environ.get("ALLOC_VALID_PER_TRAJ", "0"))
        if split == "valid" and per:              # doc 35 §31d: n distinct starts per VALID trajectory (seed 2), so every
            g = np.random.default_rng(2)                          # trajectory is represented in V1 / V2
            return [(k, int(t)) for k in range(s.n)
                    for t in np.sort(g.choice(np.arange(well2d.N_HIST - 1, s.T - 1), min(per, s.T - well2d.N_HIST), replace=False))]
        allp = [(k, t) for k in range(s.n) for t in range(well2d.N_HIST - 1, s.T - 1)]
        if len(allp) <= n:
            return allp
        pick = np.sort(np.random.default_rng({"valid": 2, "test": 3}[split]).choice(len(allp), n, replace=False))
        return [allp[i] for i in pick]

    if TEST_ONLY:
        wins = {"test": windows("test", n_test)}
    else:
        wins = {"train": windows("train", n_train), "valid": windows("valid", n_valid)}
        if not SEALED:
            wins["test"] = windows("test", n_test)
        static_wins = wins["train"][:400]

    def run(split, ex, pairs, static_pass=False):
        t0 = time.time()
        for i in range(0, len(pairs), bsz):
            xs, ys = zip(*(sp[split].window(k, t) for k, t in pairs[i:i + bsz]))
            x, y = torch.stack(xs).to(DEV), torch.stack(ys).to(DEV)
            p = model(x)
            xt, xp = x[:, (well2d.N_HIST - 1) * C:], x[:, (well2d.N_HIST - 2) * C:(well2d.N_HIST - 1) * C]
            ex.add(xt, xt - xp, p, y, range(i, i + len(xs)), static_pass)
        print(f"[cache] {dataset} {split}{' static' if static_pass else ''}: {len(pairs)} windows {time.time()-t0:.0f}s", flush=True)

    splits = ("train", "valid") if SEALED else ("train", "valid", "test")
    with torch.no_grad():
        if TEST_ONLY:
            static_idx = torch.load(OUT / dataset / "train.pt", weights_only=False)["sets"]["static"]["idx"][0].long()
            splits = ("test",)
        elif EVAL_ONLY and (OUT / dataset / "train.pt").exists():
            static_idx = torch.load(OUT / dataset / "train.pt", weights_only=False)["sets"]["static"]["idx"][0].long()
            splits = ("valid",) if SEALED else ("valid", "test")
        else:
            st = Extractor(bank, "vrmse", None, 0); run("train", st, static_wins, static_pass=True)
            static_idx = st.static_order()
        for split in splits:
            ex = Extractor(bank, "vrmse", static_idx, {"train": 1, "valid": 2, "test": 3}[split])
            run(split, ex, wins[split])
            ex.save(OUT / dataset / f"{split}.pt", {"bench": dataset, "trunk": "cnext", "split": split, "metric": "vrmse",
                                                     "C": C, "shape": [H, W], "levels": levels, "M": M, "windows": wins[split]})


# ---------------------------------------------------------------- RealPDE foil (frozen released trunks, cached P)

def realpde(trunk: str, n_train: int, n_valid: int, n_test: int, bsz: int, levels: int):
    sys.path.insert(0, str(ROOT / "paper" / "realpde"))
    import common as RC
    rng = np.random.default_rng(0)
    bank = None
    static_idx = None
    for split, n, sname in (("train", n_train, "train"), ("val", n_valid, "valid"), ("test", n_test, "test")):
        X, Y, _ = RC.load_windows(split)
        P = RC.open_pred(RC.CACHE / f"pred_{trunk}_finetune_{split}.npy", Y.shape)
        if bank is None:
            T, H, W, C = Y.shape[1:]
            bank = FrameBank(T, H, W, levels)
        ids = np.sort(rng.choice(len(Y), size=min(n, len(Y)), replace=False)) if n < len(Y) else np.arange(len(Y))

        def take(A, b):                                  # contiguous copy first: mmap pages sent to the GPU turn anonymous
            if isinstance(A, torch.Tensor):
                return A[torch.from_numpy(b)].clone().float().to(DEV)
            return torch.from_numpy(np.ascontiguousarray(A[b])).float().to(DEV)

        def batches(sel):
            for i in range(0, len(sel), bsz):
                b = sel[i:i + bsz]
                x = take(X, b)[:, -2:]                                                  # (B, 2, H, W, C) last two inputs
                y, p = take(Y, b), take(P, b)                                           # (B, T, H, W, C)
                to = lambda a: a.permute(0, 1, 4, 2, 3).contiguous()                    # -> (B, T, C, H, W)
                xt = to(x[:, 1:]).expand(-1, y.shape[1], -1, -1, -1)
                dx = to(x[:, 1:] - x[:, :1]).expand_as(xt)
                yield i, xt, dx, to(p), to(y)

        with torch.no_grad():
            if split == "train":
                st = Extractor(bank, "rel_l2", None, 0)
                for _, xt, dx, p, y in batches(np.sort(rng.permutation(ids)[:400])):
                    st.add(xt, dx, p, y, [], static_pass=True)
                static_idx = st.static_order()
            ex = Extractor(bank, "rel_l2", static_idx, {"train": 1, "val": 2, "test": 3}[split])
            t0 = time.time()
            for i, xt, dx, p, y in batches(ids):
                ex.add(xt, dx, p, y, ids[i:i + bsz].tolist())
            print(f"[cache] realpde-{trunk} {split}: {len(ids)} windows {time.time()-t0:.0f}s", flush=True)
            ex.save(OUT / f"realpde_{trunk}" / f"{sname}.pt", {"bench": f"realpde_{trunk}", "trunk": trunk, "split": sname,
                                                               "metric": "rel_l2", "C": int(C), "shape": [T, H, W],
                                                               "levels": levels, "M": M, "windows": ids.tolist()})


# ---------------------------------------------------------------- Pelzer 1HP (static plume map, frozen 1HP_NN U-Net)

def pelzer(ckpt: str, bsz: int, levels: int):
    """Non-autoregressive: there is no previous state, so the 'state' x_t is the TRAIN-mean temperature field (d = p - x_t
    is the predicted plume anomaly) and the 'change' dx is the normalized permeability input. The study metric is
    VRMSE of the normalized temperature (L2, exact from coefficients); the benchmark's MAE in degrees C is not an L2
    metric and is not reported here."""
    sys.path.insert(0, str(ROOT / "paper" / "geothermal"))
    import train_pelzer_unet as TP
    x, y, info, tr, va, te = TP.load_1000()
    model = TP.build("vanilla")
    model.load_state_dict(torch.load(ROOT / ckpt, weights_only=True, map_location="cpu")["model_state_dict"])
    model = model.to(DEV).eval()
    bank = TileBank((256, 16), levels)
    base = y[tr].mean(0, keepdim=True).to(DEV)
    static_idx = None
    for split, idx in (("train", tr), ("valid", va), ("test", te)):
        with torch.no_grad():
            def batches(sel):
                for i in range(0, len(sel), bsz):
                    b = sel[i:i + bsz]
                    xb, yb = x[b].to(DEV), y[b].to(DEV)
                    k = xb[:, 1:2]                                        # permeability channel (gksi order)
                    yield i, base.expand(len(b), -1, -1, -1), k, model(xb), yb
            if split == "train":
                st = Extractor(bank, "vrmse", None, 0)
                for _, xt, dx, p, yb in batches(idx):
                    st.add(xt, dx, p, yb, [], static_pass=True)
                static_idx = st.static_order()
            ex = Extractor(bank, "vrmse", static_idx, {"train": 1, "valid": 2, "test": 3}[split])
            for i, xt, dx, p, yb in batches(idx):
                ex.add(xt, dx, p, yb, idx[i:i + bsz].tolist())
            ex.save(OUT / "pelzer" / f"{split}.pt", {"bench": "pelzer", "trunk": "1hp_nn_unet", "split": split, "metric": "vrmse",
                                                    "C": 1, "shape": [256, 16], "levels": levels, "M": M, "windows": idx.tolist()})


class WellND:
    """Minimal N-D Well split reader (t0 scalar fields, t1 vector fields with components last), official ZScore stats,
    channel order = the_well's channels-first formatter. Frames are read with pread and evicted (freeze safety)."""

    def __init__(self, name: str, split: str):
        import glob
        import os
        import h5py
        import yaml
        self.os = os
        files = sorted(glob.glob(str(ROOT / "data" / "well" / name / "data" / split / "*.hdf5")))
        assert files, (name, split)
        self.fds, self.fields, self.index = [], [], []
        for fi, f in enumerate(files):
            with h5py.File(f) as h:
                flds = []
                for g in ("t0_fields", "t1_fields"):
                    for n in h[g].attrs["field_names"]:
                        ds = h[g][str(n)]
                        assert ds.chunks is None and ds.compression is None
                        nd = ds.ndim - 2 - (1 if g == "t1_fields" else 0)
                        ncomp = ds.shape[-1] if g == "t1_fields" else 1
                        flds.append((str(n), ncomp, ds.id.get_offset(), int(np.prod(ds.shape[2:])) * 4))
                        S, T, ntraj = ds.shape[2:2 + nd], ds.shape[1], ds.shape[0]
                self.fields.append(flds)
            self.fds.append(os.open(f, os.O_RDONLY))
            self.index.extend((fi, k) for k in range(ntraj))
        self.S, self.T, self.n = tuple(S), T, len(self.index)
        st = yaml.safe_load(open(ROOT / "data" / "well" / name / "stats.yaml"))
        mean, std = [], []
        for n, nc, _, _ in self.fields[0]:
            m, d = st["mean"][n], st["std"][n]
            mean += [float(m)] if nc == 1 else [float(v) for v in m]
            std += [float(d)] if nc == 1 else [float(v) for v in d]
        self.C = len(mean)
        shp = (self.C,) + (1,) * len(self.S)
        self.mean = np.asarray(mean, np.float32).reshape(shp); self.std = np.maximum(np.asarray(std, np.float32), 1e-4).reshape(shp)

    def frames(self, traj: int, times) -> torch.Tensor:
        fi, k = self.index[traj]
        out = np.empty((len(times), self.C, *self.S), np.float32)
        for j, ti in enumerate(times):
            c0 = 0
            for n, nc, off0, fb in self.fields[fi]:
                off = off0 + (k * self.T + int(ti)) * fb
                buf = self.os.pread(self.fds[fi], fb, off); self.os.posix_fadvise(self.fds[fi], off, fb, self.os.POSIX_FADV_DONTNEED)
                a = np.frombuffer(buf, np.float32)
                if nc == 1:
                    out[j, c0] = a.reshape(self.S)
                else:
                    out[j, c0:c0 + nc] = np.moveaxis(a.reshape(*self.S, nc), -1, 0)
                c0 += nc
        return torch.from_numpy(np.nan_to_num((out - self.mean) / self.std))

    def window(self, traj: int, t0: int, n_hist: int = 4):
        fr = self.frames(traj, range(t0 - n_hist + 1, t0 + 2))
        return fr[:n_hist].reshape(-1, *self.S), fr[n_hist]


def well3d(dataset: str, ckpt_dir: str, n_train: int, n_valid: int, n_test: int, bsz: int, levels: int):
    sys.path.insert(0, str(ROOT / "paper" / "phase_b"))
    from cnext_hybrid import build_hybrid
    model = build_hybrid("vanilla", model_dir=str(ROOT / ckpt_dir)).to(DEV).eval()
    sp = {s: WellND(dataset, s) for s in ("train", "valid", "test")}
    C, S = sp["test"].C, sp["test"].S
    bank = TileBank(S, levels)
    rng = np.random.default_rng(0)

    def windows(split, n):
        s = sp[split]
        if split == "train":
            return [(int(rng.integers(s.n)), int(rng.integers(3, s.T - 1))) for _ in range(n)]
        per = max(1, math.ceil(n / s.n))
        starts = np.linspace(3, s.T - 2, per).round().astype(int)
        pairs = [(k, int(t)) for k in range(s.n) for t in starts]
        keep = np.linspace(0, len(pairs) - 1, min(n, len(pairs))).round().astype(int)
        return [pairs[i] for i in keep]

    wins = {"train": windows("train", n_train), "valid": windows("valid", n_valid), "test": windows("test", n_test)}

    def run(split, ex, pairs, static_pass=False):
        t0 = time.time()
        for i in range(0, len(pairs), bsz):
            xs, ys = zip(*(sp[split].window(k, t) for k, t in pairs[i:i + bsz]))
            x, y = torch.stack(xs).to(DEV), torch.stack(ys).to(DEV)
            p = model(x)
            xt, xp = x[:, 3 * C:], x[:, 2 * C:3 * C]
            ex.add(xt, xt - xp, p, y, range(i, i + len(xs)), static_pass)
        print(f"[cache] {dataset} {split}{' static' if static_pass else ''}: {len(pairs)} windows {time.time()-t0:.0f}s", flush=True)

    with torch.no_grad():
        st = Extractor(bank, "vrmse", None, 0); run("train", st, wins["train"][:400], static_pass=True)
        static_idx = st.static_order()
        for split in ("train", "valid", "test"):
            ex = Extractor(bank, "vrmse", static_idx, {"train": 1, "valid": 2, "test": 3}[split])
            run(split, ex, wins[split])
            ex.save(OUT / dataset / f"{split}.pt", {"bench": dataset, "trunk": "cnext", "split": split, "metric": "vrmse",
                                                     "C": C, "shape": list(S), "levels": levels, "M": M, "windows": wins[split]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["well", "realpde", "pelzer", "well3d"]); ap.add_argument("name")
    ap.add_argument("--train", type=int, default=3000); ap.add_argument("--valid", type=int, default=600)
    ap.add_argument("--test", type=int, default=1200); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--levels", type=int, default=5)
    a = ap.parse_args()
    if a.kind == "pelzer":
        pelzer(a.name, a.batch, a.levels)
    elif a.kind == "well3d":
        name, ck = a.name.split(":")
        well3d(name, ck, a.train, a.valid, a.test, a.batch, a.levels)
    else:
        (well if a.kind == "well" else realpde)(a.name, a.train, a.valid, a.test, a.batch, a.levels)


if __name__ == "__main__":
    main()
