"""TORAX-Bench for the Optuna driver: same model, data and metric as paper/torax_bench/train_torax.py
(delta-prediction 1-D spectral trunk, Bottleneck arms, TD k=4, rollout VRMSE@8), but with the VALID split
(100 discharges) as the tuning target and TEST (100) touched only by the driver's `final` stage."""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))
from model import Bottleneck, MLPCore, _count, build_config_core, parse_config  # noqa: E402
from qlayer import MultiRegister  # noqa: E402

DEV = torch.device("cuda")
DATA = ROOT / "data" / "torax_bench_v1"
SPLITS = json.load(open(ROOT / "paper/torax_bench/splits.json"))
H, WIDTH, MODES, K = 8, 24, 8, 4
ARMS = {"none": "none", "mlp": "q30-mlpref", "q": "q30"}


def space(trial, arm: str) -> dict:
    hp = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
              steps=trial.suggest_int("steps", 500, 2000, step=250),
              batch=trial.suggest_categorical("batch", [8, 16, 32]),
              wd=trial.suggest_categorical("wd", [0.0, 1e-5, 1e-4]))
    if arm == "q":
        hp.update(L=trial.suggest_categorical("L", [2, 3, 4, 6]),
                  geo=trial.suggest_categorical("geo", ["grid", "chain", "ring", "tri"]),
                  sp=trial.suggest_categorical("sp", [0.7, 1.0, 1.5]),
                  link=trial.suggest_categorical("link", ["independent", "shared"]))
    return hp


def _load(ids):
    X, C = [], []
    for i in ids:
        with np.load(DATA / f"discharge_{i:05d}.npz") as d:
            q = np.stack([np.interp(d["rho_norm"], d["rho_face_norm"], row) for row in d["q"]])
            x = np.stack([d["T_e"], d["T_i"], d["n_e"] / 1e19, q], 1).astype(np.float64)
            act = np.repeat((d["actuators"] / np.array([1e6, 1e6, 1e22]))[:, :, None], 27, 2).astype(np.float64)
            sc = np.repeat(np.repeat(d["scenario"][None, :, None], x.shape[0], 0), 27, 2).astype(np.float64)
            X.append(x); C.append(np.concatenate([act, sc], 1))
    return np.stack(X), np.stack(C)


_CACHE: dict = {}


def data():
    if not _CACHE:
        Xtr, Ctr = _load(SPLITS["train"])
        mx, sx = Xtr.mean((0, 1, 3), keepdims=True), Xtr.std((0, 1, 3), keepdims=True) + 1e-6
        mc, sc_ = Ctr.mean((0, 1, 3), keepdims=True), Ctr.std((0, 1, 3), keepdims=True) + 1e-6
        zs = lambda a, m, s: torch.tensor((a - m) / s, dtype=torch.float32).to(DEV)
        _CACHE["train"] = (zs(Xtr, mx, sx), zs(Ctr, mc, sc_))
        for split in ("valid", "test"):
            X, C = _load(SPLITS[split]); _CACHE[split] = (zs(X, mx, sx), zs(C, mc, sc_))
    return _CACHE


class Spec1d(nn.Module):
    def __init__(s, ch, m): super().__init__(); s.m = m; s.w = nn.Parameter((1 / (ch * ch)) * torch.randn(ch, ch, m, 2))
    def forward(s, x):
        xf = torch.fft.rfft(x, norm="ortho"); out = torch.zeros_like(xf)
        out[:, :, :s.m] = torch.einsum("bim,iom->bom", xf[:, :, :s.m], torch.view_as_complex(s.w))
        return torch.fft.irfft(out, n=x.shape[-1], norm="ortho")


class Blk(nn.Module):
    def __init__(s, ch, m): super().__init__(); s.spec, s.skip, s.act = Spec1d(ch, m), nn.Conv1d(ch, ch, 1), nn.GELU()
    def forward(s, x): return s.act(s.spec(x) + s.skip(x))


def build(variant: str) -> nn.Module:
    if variant == "none": mid: nn.Module = nn.Identity()
    elif variant == "mlp30": mid = Bottleneck(WIDTH, MLPCore(WIDTH, 30, _count(MultiRegister(WIDTH, 10, 3, 4, "hea"))), 30)
    elif variant == "mr30ryd": mid = Bottleneck(WIDTH, MultiRegister(WIDTH, 10, 3, 4, "rydberg"), 30)
    elif parse_config(variant) is not None:
        core, out = build_config_core(parse_config(variant), WIDTH, 4); mid = Bottleneck(WIDTH, core, out)
    else: raise ValueError(variant)

    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.lift = nn.Conv1d(4 + 15, WIDTH, 1); s.pre = nn.Sequential(Blk(WIDTH, MODES), Blk(WIDTH, MODES))
            s.mid = mid; s.post = nn.Sequential(Blk(WIDTH, MODES), Blk(WIDTH, MODES)); s.proj = nn.Sequential(nn.Conv1d(WIDTH, 64, 1), nn.GELU(), nn.Conv1d(64, 4, 1))
        def forward(s, x, c): return x + s.proj(s.post(s.mid(s.pre(s.lift(torch.cat([x, c], 1))))))
    return Net()


def vrmse(true, pred):
    mse = ((true - pred) ** 2).mean(-1); var = true.var(-1, unbiased=False)
    return torch.sqrt(mse / (var + 1e-12)).mean().item()


@torch.no_grad()
def evaluate(model, X, C):
    T = X.shape[1]; model.eval(); one, roll = [], []
    for t0 in range(0, T - 1, 8):
        one.append(vrmse(X[:, t0 + 1], model(X[:, t0], C[:, t0 + 1])))
    for t0 in range(0, T - 1 - H, 16):
        x = X[:, t0]
        for j in range(H): x = model(x, C[:, t0 + 1 + j])
        roll.append(vrmse(X[:, t0 + H], x))
    model.train(); return float(np.mean(one)), float(np.mean(roll))


def run(variant: str, hp: dict, seed: int, split: str) -> dict:
    d = data(); Xtr, Ctr = d["train"]; Xev, Cev = d[split]
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(variant).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"]) if hp.get("wd", 0) else torch.optim.Adam(model.parameters(), lr=hp["lr"])
    N, T = Xtr.shape[0], Xtr.shape[1]; t0 = time.time(); last = None
    for step in range(hp["steps"]):
        b = torch.randint(0, N, (hp["batch"],)); ti = torch.randint(0, T - 1 - K, (hp["batch"],))
        x = Xtr[b, ti]; loss = 0.0
        for j in range(K):
            x = model(x, Ctr[b, ti + 1 + j]); loss = loss + ((x - Xtr[b, ti + 1 + j]) ** 2).mean()
        loss = loss / K; opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
    v1, v8 = evaluate(model, Xev, Cev)
    return dict(metric=v8, vrmse_1=v1, vrmse_8=v8, train_loss=last, num_params=_count(model), train_seconds=time.time() - t0, split=split)
