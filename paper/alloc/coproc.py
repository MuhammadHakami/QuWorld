"""Routed co-processor on a frozen trunk: train on TRAIN, early-stop on VALID, score TEST once (33-allocation-...md).

    .venv/bin/python paper/alloc/coproc.py turbulent_radiative_layer_2D --strategy proxy --core ryd --K 30 --seed 1

strategy: pooled | static | proxy | learned | gain | random | oracle(diagnostic)
core:     lin (no core) | mlp | mlpX4 | ryd | ryd2d | gutz
Tile strategies write corr_i = s_i * (W_key(i) f_i + A o_i) on the K allocated tiles:
  - s_i is the tile's own scale;
  - W_key is a linear map per (level, orientation) shared by every arm;
  - o_i is the core's per-tile readout;
  - A is zero-init, so every core starts as the linear corrector.
Pooled writes one constant per channel (the audit's DC insertion): corr_c = s_c * (W g + A core(g)).
Because the Haar basis is orthonormal, the corrected metric is exact from coefficient energies
(Lean: residual_after_band_removal):
  E'_c = E_c - sum_{i in S} (r_ic^2 - (r_ic - corr_ic)^2).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "paper"))
import gpucap  # noqa: E402
from cores import make_tile_core, n_params  # noqa: E402

gpucap.cap_from_env()
DEV = torch.device("cuda")
CACHE = ROOT / "runs" / "alloc" / "cache"
OUT = ROOT / "runs" / "alloc" / "results"
N_PER = 10


# ------------------------------------------------------------------------------------------------------------ data

class Split:
    def __init__(self, bench: str, split: str):
        d = torch.load(CACHE / bench / f"{split}.pt", weights_only=False)
        self.meta = d["meta"]; self.metric = self.meta["metric"]; self.C = self.meta["C"]
        self.sets = {s: {"idx": v["idx"].long().to(DEV), "coef": v["coef"].float().to(DEV)} for s, v in d["sets"].items()}
        self.E, self.D = d["E"].to(DEV), d["D"].to(DEV)
        self.rmean, self.glob, self.npix = d["rmean"].float().to(DEV), d["glob"].float().to(DEV), d["npix"].double().to(DEV)
        self.ceil = {k: v.to(DEV) for k, v in d["ceil"].items()}
        self.level, self.orient, self.pos = d["level"].long().to(DEV), d["orient"].long().to(DEV), d["pos"].float().to(DEV)
        self.N = len(self.E)
        self.etot = (self.E / self.D).sum(1)                                  # weighted residual energy per window

    def subset(self, rows: torch.Tensor) -> "Split":
        """The same split restricted to windows `rows` (per-window tensors sliced, per-tile tensors shared)."""
        o = object.__new__(Split); o.__dict__.update(self.__dict__)
        o.sets = {s: {k: v[rows] for k, v in d.items()} for s, d in self.sets.items()}
        o.E, o.D, o.rmean, o.glob, o.npix = self.E[rows], self.D[rows], self.rmean[rows], self.glob[rows], self.npix[rows]
        o.ceil = {k: v[rows] for k, v in self.ceil.items()}
        o.N, o.etot = len(rows), self.etot[rows]
        o.meta = {**self.meta, "windows": [self.meta["windows"][int(i)] for i in rows]}
        return o

    def metric_of(self, E: torch.Tensor, rows=None) -> torch.Tensor:
        E, D = E.clamp_min(0), (self.D if rows is None else self.D[rows])
        if self.metric == "vrmse":
            return (E / D).sqrt().mean(1)
        return (E.sum(1) / D[:, 0]).sqrt()


# ------------------------------------------------------------------------------------------------------- features

class Featurizer:
    """Scale-free tile features; shared by co-processor and scorers (never reads the residual r)."""

    def __init__(self, sp: Split):
        self.levels = int(sp.level.max()) + 1; self.n_orient = int(sp.orient.max()) + 1
        self.pscale = sp.pos.max(0).values.clamp_min(1.0)
        self.gmu, self.gsd = sp.glob.mean(0), sp.glob.std(0) + 1e-6       # TRAIN statistics of the global context
        self.F = 4 * sp.C + 1 + self.levels + self.n_orient + sp.C + 3 + sp.glob.shape[1]
        self.n_keys = self.levels * self.n_orient

    def tiles(self, sp: Split, rows, idx, coef):
        """rows (B,), idx (B,K) global tile ids, coef (B,K,5,C) -> f (B,K,F), s (B,K,1), r (B,K,C), coords (B,K,4), key."""
        C = sp.C
        inp, r = coef[:, :, :4], coef[:, :, 4]
        dstd = sp.glob[rows][:, C:2 * C]                                       # per-channel spatial std of d
        floor = 1e-2 * dstd.mean(1).clamp_min(1e-8)[:, None, None]
        s = (inp[:, :, [0, 3]].square().mean((2, 3), keepdim=False)[..., None] + floor.square()).sqrt()   # (B,K,1)
        fc = (inp / s[..., None]).clamp(-20, 20).flatten(2)
        lv, ori, pos = sp.level[idx], sp.orient[idx], sp.pos[idx]
        f = torch.cat([fc, (s / floor).log() / 5,
                       nn.functional.one_hot(lv, self.levels).float(), nn.functional.one_hot(ori, self.n_orient).float(),
                       (dstd.clamp_min(1e-8).log() / 5)[:, None].expand(-1, idx.shape[1], -1),
                       pos / self.pscale,
                       ((sp.glob[rows] - self.gmu) / self.gsd).clamp(-10, 10)[:, None].expand(-1, idx.shape[1], -1)], -1)
        coords = torch.cat([pos, lv[..., None].float()], -1)
        return f, s, r, coords, lv * self.n_orient + ori


def morton(pos: torch.Tensor) -> torch.Tensor:
    p = pos.round().long().clamp(0, 1023)
    key = torch.zeros(p.shape[:-1], dtype=torch.long, device=p.device)
    for b in range(10):
        for a in range(3):
            key |= ((p[..., a] >> b) & 1) << (3 * b + a)
    return key


# ------------------------------------------------------------------------------------------------------------ models

class TileCoproc(nn.Module):
    def __init__(self, fz: Featurizer, C: int, core: str, layers: int = 4):
        super().__init__()
        self.fz, self.C, self.core_kind = fz, C, core
        self.W = nn.Parameter(torch.zeros(fz.n_keys, fz.F, C)); self.b = nn.Parameter(torch.zeros(fz.n_keys, C))
        self.core = None if core == "lin" else make_tile_core(core, fz.F, N_PER, layers)
        self.n_out = 6 if core in ("hyb", "mlp2") else 3
        self.A = nn.Linear(self.n_out, C)
        nn.init.zeros_(self.A.weight); nn.init.zeros_(self.A.bias)

    def forward(self, f, s, coords, key):
        B, K, _ = f.shape
        out = torch.einsum("bkf,bkfc->bkc", f, self.W[key]) + self.b[key]
        if self.core is not None:
            order = morton(coords[..., :3]).argsort(1)                         # locality groups of N_PER atoms
            g = lambda t: t.gather(1, order[..., None].expand(-1, -1, t.shape[-1]))
            o = self.core(g(f).reshape(B * K // N_PER, N_PER, -1), g(coords).reshape(B * K // N_PER, N_PER, -1))
            no = self.n_out
            o = torch.empty_like(o.reshape(B, K, no)).scatter(1, order[..., None].expand(-1, -1, no), o.reshape(B, K, no))
            out = out + self.A(o)
        return s * out


class PooledCoproc(nn.Module):
    """The audit's pooled insertion with K atoms = K / 10 blockade-separated registers, all reading the pooled vector."""

    def __init__(self, Fg: int, C: int, core: str, K: int = 30, layers: int = 4):
        super().__init__()
        from model import MLPCore, _count
        from qlayer import MultiRegister
        self.W = nn.Linear(Fg, C); nn.init.zeros_(self.W.weight); nn.init.zeros_(self.W.bias)
        ans = {"ryd": "rydberg", "gutz": "rydberg_mf"}
        if core == "lin":
            self.core = None
        elif core in ans:
            self.core = MultiRegister(Fg, N_PER, K // N_PER, layers, ans[core])
        elif core.startswith("mlp"):
            scale = int(core[4:]) if core.startswith("mlpX") else 1
            self.core = MLPCore(Fg, K, scale * _count(MultiRegister(Fg, N_PER, K // N_PER, layers, "rydberg")))
        else:
            raise ValueError(f"pooled strategy has no core {core}")
        self.A = nn.Linear(K, C); nn.init.zeros_(self.A.weight); nn.init.zeros_(self.A.bias)

    def forward(self, g, s):
        out = self.W(g)
        if self.core is not None:
            out = out + self.A(self.core(g))
        return s * out


class Scorer(nn.Module):
    def __init__(self, F: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(F, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, f):
        return self.net(f).squeeze(-1)


# ------------------------------------------------------------------------------------------------------ evaluation

class Runner:
    def __init__(self, a, sp: dict):
        self.a, self.sp = a, sp
        self.fz = Featurizer(sp["train"])
        g = sp["train"].glob
        self.gmu, self.gsd = g.mean(0), g.std(0) + 1e-6
        self.scorer = None

    # tile selection ----------------------------------------------------------------------------------------------
    def select(self, sp: Split, rows, K: int, strategy: str):
        base = {"static": "static", "random": "random", "oracle": "oracle"}.get(strategy, "proxy")
        idx, coef = sp.sets[base]["idx"][rows], sp.sets[base]["coef"][rows]
        if strategy in ("learned", "gain"):
            f, *_ = self.fz.tiles(sp, rows, idx, coef)
            with torch.no_grad():
                top = self.scorer(f).topk(K, 1).indices
            return idx.gather(1, top), coef.gather(1, top[..., None, None].expand(-1, -1, *coef.shape[2:]))
        return idx[:, :K], coef[:, :K]

    def gains(self, model, sp: Split, rows, strategy: str, K: int):
        """-> per-window captured fraction (B,), utilization ceiling U (B,) of the allocation, and the per-channel
        gain (B, C) in raw energy units (E'_c = E_c - gain_c; exact by orthonormality)."""
        if strategy == "pooled":
            g = (sp.glob[rows] - self.gmu) / self.gsd
            corr = model(g, sp.glob[rows][:, sp.C:2 * sp.C]).double()
            m = sp.rmean[rows].double()
            gain_c = sp.npix[rows, None] * (m.square() - (m - corr).square())
            return (gain_c / sp.D[rows]).sum(1) / sp.etot[rows], sp.ceil["dc"][rows].double(), gain_c
        idx, coef = self.select(sp, rows, K, strategy)
        f, s, r, coords, key = self.fz.tiles(sp, rows, idx, coef)
        corr, r = model(f, s, coords, key).double(), r.double()
        gain_c = (r.square() - (r - corr).square()).sum(1)                         # (B, C)
        U = (r.square() / sp.D[rows][:, None]).sum((1, 2)) / sp.etot[rows]
        return (gain_c / sp.D[rows]).sum(1) / sp.etot[rows], U, gain_c

    @torch.no_grad()
    def evaluate(self, model, sp: Split, strategy: str, K: int, bs: int = 256):
        cap, Us, mets = [], [], []
        for i in range(0, sp.N, bs):
            rows = torch.arange(i, min(i + bs, sp.N), device=DEV)
            c, U, gain_c = self.gains(model, sp, rows, strategy, K)
            cap.append(c); Us.append(U)
            mets.append(sp.metric_of(sp.E[rows] - gain_c, rows))
        cap, Us, mets = torch.cat(cap), torch.cat(Us), torch.cat(mets)
        return {"metric": mets.mean().item(), "captured": cap.mean().item(), "U": Us.mean().item(), "per": mets}

    # training ------------------------------------------------------------------------------------------------------
    def train(self, model, strategy: str, K: int, seed: int, loss_fn=None):
        a, tr, va = self.a, self.sp["train"], self.sp["valid"]
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        best = self.evaluate(model, va, strategy, K)["metric"]; start = best
        state, bad, ep = {k: v.clone() for k, v in model.state_dict().items()}, 0, 0
        for ep in range(a.epochs):
            perm = torch.randperm(tr.N, generator=gen).to(DEV)
            model.train()
            for i in range(0, tr.N - a.batch + 1, a.batch):
                cap, _, _ = self.gains(model, tr, perm[i:i + a.batch], strategy, K)
                loss = -cap.mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval()
            v = self.evaluate(model, va, strategy, K)["metric"]
            if v < best - 1e-9:
                best, state, bad = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
        model.load_state_dict(state)
        return {"valid_start": start, "valid_best": best, "epochs": ep + 1}

    def fit_scorer(self, target_fn, seed: int):
        """ListNet over each window's proxy shortlist: target distribution from target_fn(rows) (B, M) >= 0."""
        tr, va = self.sp["train"], self.sp["valid"]
        torch.manual_seed(seed + 7)
        sc = Scorer(self.fz.F).to(DEV)
        opt = torch.optim.AdamW(sc.parameters(), lr=3e-3, weight_decay=1e-4)
        gen = torch.Generator(device="cpu").manual_seed(seed + 7)

        def val():
            tot = 0.0
            with torch.no_grad():
                for i in range(0, va.N, 256):
                    rows = torch.arange(i, min(i + 256, va.N), device=DEV)
                    t = target_fn(va, rows)
                    f, *_ = self.fz.tiles(va, rows, va.sets["proxy"]["idx"][rows], va.sets["proxy"]["coef"][rows])
                    top = sc(f).topk(self.a.K, 1).indices
                    tot += (t.gather(1, top).sum(1) / t.sum(1).clamp_min(1e-12)).sum().item()
            return tot / va.N

        best, state, bad = val(), {k: v.clone() for k, v in sc.state_dict().items()}, 0
        for ep in range(30):
            perm = torch.randperm(tr.N, generator=gen).to(DEV)
            for i in range(0, tr.N - 64 + 1, 64):
                rows = perm[i:i + 64]
                t = target_fn(tr, rows)
                p = t / t.sum(1, keepdim=True).clamp_min(1e-12)
                f, *_ = self.fz.tiles(tr, rows, tr.sets["proxy"]["idx"][rows], tr.sets["proxy"]["coef"][rows])
                loss = -(p * torch.log_softmax(sc(f), 1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            v = val()
            if v > best + 1e-6:
                best, state, bad = v, {k: x.clone() for k, x in sc.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 4:
                    break
        sc.load_state_dict(state); sc.eval()
        self.scorer = sc
        return best

    def realized_gain_fn(self, model):
        """Per-tile realized gain of `model` on every proxy-shortlist tile (groups formed over the whole shortlist)."""
        def fn(sp: Split, rows):
            idx, coef = sp.sets["proxy"]["idx"][rows], sp.sets["proxy"]["coef"][rows]
            with torch.no_grad():
                f, s, r, coords, key = self.fz.tiles(sp, rows, idx, coef)
                corr = model(f, s, coords, key)
                g = ((r.square() - (r - corr).square()) / sp.D[rows][:, None].float()).sum(-1)
            return g.clamp_min(0)
        return fn


def build(runner: Runner, strategy: str, core: str, seed: int):
    torch.manual_seed(seed)
    sp = runner.sp["train"]
    if strategy == "pooled":
        return PooledCoproc(sp.glob.shape[1], sp.C, core, runner.a.K).to(DEV)
    return TileCoproc(runner.fz, sp.C, core).to(DEV)


def score_test(a, name: str) -> None:
    """doc 35 §31e phase B: the frozen phase-A model (TRAIN statistics only) scored once on the full TEST cache."""
    import hashlib
    ck = OUT / a.bench / "vsplit" / "ckpt" / f"{name}.pt"
    out = OUT / a.bench / "test" / f"{name}.json"
    assert ck.exists(), f"no frozen phase-A model {ck}"
    assert not out.exists(), f"TEST already scored once: {out}"
    sp = {"train": Split(a.bench, "train"), "test": Split(a.bench, "test")}
    run = Runner(a, sp)                                  # featurizer and context statistics from TRAIN only, as in phase A
    model = build(run, a.strategy, a.core, a.seed)
    model.load_state_dict(torch.load(ck, weights_only=False)["state"]); model.eval()
    te = run.evaluate(model, sp["test"], a.strategy, a.K)
    base = sp["test"].metric_of(sp["test"].E)
    res = {"bench": a.bench, "strategy": a.strategy, "core": a.core, "K": a.K, "seed": a.seed, "metric": sp["test"].metric,
           "trunk_test": base.mean().item(), "test": te["metric"], "test_rel_gain": 1 - te["metric"] / base.mean().item(),
           "test_captured": te["captured"], "test_U": te["U"], "n_test": sp["test"].N, "shape": list(sp["test"].meta["shape"]),
           "ckpt_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
           "test_per_window": [round(x, 7) for x in te["per"].tolist()],
           "trunk_per_window": [round(x, 7) for x in base.tolist()],
           "test_trajectory": [int(k) for k, _ in sp["test"].meta["windows"]]}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res))
    print(f"[coproc] {a.bench} {name} TEST (once): {res['trunk_test']:.5f} -> {res['test']:.5f} "
          f"({100 * res['test_rel_gain']:+.2f}%) n={res['n_test']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench"); ap.add_argument("--strategy", default="proxy"); ap.add_argument("--core", default="ryd")
    ap.add_argument("--K", type=int, default=30); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save_ckpt", action="store_true", help="doc 35 §31e phase A: with --vsplit, save the trained model "
                    "(TRAIN fit, V1 early stop) to <bench>/vsplit/ckpt/ so it is frozen before TEST exists")
    ap.add_argument("--score_test", action="store_true", help="doc 35 §31e phase B: load the frozen phase-A model and score "
                    "it ONCE on the full TEST cache (no training); refuses if the TEST result already exists")
    ap.add_argument("--vsplit", action="store_true", help="doc 35 §31c: VALID split into V1 (early stop) / V2 (report) "
                    "by alternating parity of the sorted trajectory index; TEST is never loaded")
    a = ap.parse_args()
    name = f"{a.strategy}_{a.core}_K{a.K}_s{a.seed}{a.tag}"
    if a.score_test:
        return score_test(a, name)
    out = OUT / a.bench / ("vsplit" if a.vsplit else "") / f"{name}.json"
    if out.exists():
        print(f"[coproc] exists {out.name}"); return
    t0 = time.time()
    if a.vsplit:                                   # doc 35 §31c: V1 = even ranks, V2 = odd ranks of the globally sorted
        tr, va_all = Split(a.bench, "train"), Split(a.bench, "valid")                  # (file, trajectory) ids, i.e. of
        traj = torch.tensor([int(k) for k, _ in va_all.meta["windows"]], device=DEV)   # the Split's dense global index
        sp = {"train": tr, "valid": va_all.subset(torch.nonzero(traj % 2 == 0).squeeze(1)),
              "test": va_all.subset(torch.nonzero(traj % 2 == 1).squeeze(1))}
    else:
        sp = {s: Split(a.bench, s) for s in ("train", "valid", "test")}
    run = Runner(a, sp)
    info = {}
    if a.strategy == "learned":
        info["scorer_valid_capture"] = run.fit_scorer(
            lambda s, rows: ((s.sets["proxy"]["coef"][rows][:, :, 4].square()) / s.D[rows][:, None].float()).sum(-1), a.seed)
    if a.strategy == "gain":
        stage1 = build(run, "proxy", a.core, a.seed)
        info["stage1"] = run.train(stage1, "proxy", a.K, a.seed)
        info["scorer_valid_gain_share"] = run.fit_scorer(run.realized_gain_fn(stage1), a.seed)
    model = build(run, a.strategy, a.core, a.seed)
    info["train"] = run.train(model, a.strategy, a.K, a.seed)
    te = run.evaluate(model, sp["test"], a.strategy, a.K)
    va = run.evaluate(model, sp["valid"], a.strategy, a.K)
    base_te = sp["test"].metric_of(sp["test"].E); base_va = sp["valid"].metric_of(sp["valid"].E)
    core_params = n_params(model.core) if getattr(model, "core", None) is not None else 0
    res = {"bench": a.bench, "strategy": a.strategy, "core": a.core, "K": a.K, "seed": a.seed,
           "metric": sp["test"].metric, "trunk_valid": base_va.mean().item(), "trunk_test": base_te.mean().item(),
           "valid": va["metric"], "test": te["metric"], "test_rel_gain": 1 - te["metric"] / base_te.mean().item(),
           "test_captured": te["captured"], "test_U": te["U"], "valid_captured": va["captured"], "valid_U": va["U"],
           "params_core": core_params, "params_total": n_params(model), "seconds": time.time() - t0,
           "test_per_window": [round(x, 7) for x in te["per"].tolist()], **info}
    if a.vsplit:                                   # "test" fields hold V2 (never TEST); record the split
        res.update(vsplit=True, v1_windows=sp["valid"].N, v2_windows=sp["test"].N,
                   v1_trajectories=sorted({k for k, _ in sp["valid"].meta["windows"]}),
                   v2_trajectories=sorted({k for k, _ in sp["test"].meta["windows"]}))
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.save_ckpt:
        assert a.vsplit, "--save_ckpt is phase A of §31e: TRAIN fit, V1 early stop (needs --vsplit)"
        ck = OUT / a.bench / "vsplit" / "ckpt" / f"{name}.pt"; ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "args": vars(a)}, ck)
    out.write_text(json.dumps(res))
    print(f"[coproc] {a.bench} {a.strategy} {a.core} K{a.K} s{a.seed}: TEST {res['trunk_test']:.5f} -> {res['test']:.5f} "
          f"({100 * res['test_rel_gain']:+.2f}%) captured {te['captured']:.4f} of U {te['U']:.4f} | "
          f"core {core_params} params | {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
