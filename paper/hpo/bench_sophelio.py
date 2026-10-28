"""Sophelio (starter UNetLite trunk, official scorer) for the Optuna driver. Each trial is a subprocess of
paper/sophelio/train_hybrid.py with --score-split val (the 20 validation shots; test untouched); `final`
uses --score-split test. Metric = -S (driver minimises)."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARMS = {"none": "vanilla", "mlp": "q30-mlpref", "q": "q30"}
OUT = ROOT / "runs" / "hpo" / "sophelio" / "trials"


def space(trial, arm: str) -> dict:
    hp = dict(lr=trial.suggest_float("lr", 2e-4, 3e-3, log=True),
              epochs=trial.suggest_int("epochs", 30, 80, step=10),
              batch=trial.suggest_categorical("batch", [128, 256, 512]))
    if arm == "q":
        hp.update(L=trial.suggest_categorical("L", [2, 3, 4, 6]),
                  geo=trial.suggest_categorical("geo", ["grid", "chain", "ring", "tri"]),
                  sp=trial.suggest_categorical("sp", [0.7, 1.0, 1.5]),
                  link=trial.suggest_categorical("link", ["independent", "shared"]))
    return hp


def run(variant: str, hp: dict, seed: int, split: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_{split}_{int(time.time())}"
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "paper/sophelio/train_hybrid.py"), "--variant", variant, "--seed", str(seed),
           "--n-shots", "200", "--epochs", str(hp["epochs"]), "--batch", str(hp["batch"]), "--lr", str(hp["lr"]),
           "--score-split", {"valid": "val", "test": "test"}[split], "--out", str(OUT), "--tag", tag]
    t0 = time.time()
    log = OUT / f"{variant}_s{seed}{tag}.log"
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"sophelio trial failed (see {log})")
    d = json.load(open(OUT / f"{variant}_s{seed}{tag}.json"))
    o = d["official"]
    return dict(metric=-o["S"], S=o["S"], r2_psi=o["r2_psi"], r2_qb=o["r2_qb"], dlcfs=o["dlcfs"], consistency=o["consistency"],
                num_params=d.get("num_params"), train_seconds=time.time() - t0, split=split)
