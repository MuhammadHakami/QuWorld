"""Pelzer 1HP U-Net for the Optuna driver: their trunk (1HP_NN U-Net, pretrained per seed by
runs/pelzer_unet_runner.sh), our arms at the deepest encoder; VALID MAE (200 runs) as the tuning target, TEST
(100 runs) touched only by `final`. Reuses paper/geothermal/train_pelzer_unet.py (data, build, train, evaluate)."""
from __future__ import annotations

import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))
sys.path.insert(0, str(ROOT / "paper" / "geothermal"))
import train_pelzer_unet as P  # noqa: E402  (imports 1HP_NN's UNet + data module)
from model import _count  # noqa: E402

ARMS = {"none": "vanilla", "mlp": "q30-mlpref", "q": "q30"}
_DATA: dict = {}


def space(trial, arm: str) -> dict:
    hp = dict(lr=trial.suggest_float("lr", 2e-5, 5e-4, log=True),
              steps=trial.suggest_int("steps", 500, 3000, step=500),
              batch=trial.suggest_categorical("batch", [50, 100]))
    if arm == "q":
        hp.update(L=trial.suggest_categorical("L", [2, 3, 4, 6]),
                  geo=trial.suggest_categorical("geo", ["grid", "chain", "ring", "tri"]),
                  sp=trial.suggest_categorical("sp", [0.7, 1.0, 1.5]),
                  link=trial.suggest_categorical("link", ["independent", "shared"]))
    return hp


def data():
    if not _DATA:
        x, y, info, tr, va, te = P.load_1000()
        _DATA.update(x=x.to(P.DEV), y=y.to(P.DEV), info=info, tr=tr, va=va, te=te)
    return _DATA


def run(variant: str, hp: dict, seed: int, split: str, trunk_seed: int | None = None) -> dict:
    """`trunk_seed` picks the pretrained U-Net (only 1337-1339 exist); `seed` drives the adapter init and batch
    order. They coincide unless a replication separates them (31-powered-replication-prereg.md)."""
    d = data()
    model = P.build(variant)
    trunk_seed = seed if trunk_seed is None else trunk_seed
    st = torch.load(ROOT / f"runs/pelzer_unet/pretrain_lr0.001_s{trunk_seed}.pt", weights_only=True, map_location="cpu")["model_state_dict"]
    if variant != "vanilla":
        st = {(k.replace("encoders.3.", "encoders.3.enc.", 1) if k.startswith("encoders.3.") else k): v for k, v in st.items()}
    missing, unexpected = model.load_state_dict(st, strict=False)
    assert not unexpected and all(m.startswith("encoders.3.bottleneck") for m in missing), (missing[:3], unexpected[:3])
    model = model.to(P.DEV)
    P.BATCH = hp["batch"]
    t0 = time.time()
    tr_info = P.train(model, d["x"], d["y"], d["tr"], d["va"], d["info"], hp["steps"], hp["lr"], seed, f"hpo-{variant}-s{seed}")
    ev = P.evaluate(model, d["x"], d["y"], d[{"valid": "va", "test": "te"}[split]], d["info"])
    return dict(metric=ev["mae_c"], mae_c=ev["mae_c"], pat_0p1=ev["pat_0p1"], best_val_mae_c=tr_info["best_val_mae_c"],
                best_step=tr_info["best_step"], num_params=_count(model), train_seconds=time.time() - t0, split=split,
                trunk_seed=trunk_seed)
