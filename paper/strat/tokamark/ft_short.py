"""§37 TokaMark: short continuation fine-tunes of the TokaMind-Scratch vanilla checkpoint (GPU, one lock slot).

    flock runs/strat/gpu.lock timeout 1800 env QW_GPU_GB=12 runs/capped.sh strat_tok 16 .venv-tokamark/bin/python \
        paper/strat/tokamark/ft_short.py --driver
Driver = smoke (15 batches: peak GPU memory, step rate) -> 'eq' (equal output weights, the recipe's loss) -> 'w'
(metric-aligned output weights w_k ~ 1/NRMSE_k from the TRAIN-subsample dump of the Scratch model; the task metric is
an equal-weight mean of per-signal NRMSE = sqrt(MSE_k)/sigma_k, so d NRMSE_k = d MSE_k / (2 NRMSE_k)). Both arms share
the same shots, seed, LR schedule and batch budget; their only difference is the loss weights.

TokaMind's own run_finetune.main() is used with in-memory config patches (as paper/tokamark/tokamind_hybrid.py does):
warm start from the Scratch run with all four parts loaded (output adapters included), one stage 'ft_cont' at 1/10 of
the Scratch peak LRs (2% -> 5% warmup, cosine), streaming data (no RAM cache), TRAIN = 2,000 random train shots
(seed 0), in-run val = 100 VALID shots outside the §37 sample. TEST is never loaded (the split reader drops it).
Run dirs and config snapshots go to runs/strat/tokamark/ft/.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "strat" / "tokamark"
FT_DIR = OUT / "ft"
SRC = ROOT / "runs" / "tokamark" / "tokamind" / "ft-task_1-1-scratch-vanilla_s1337_scratch"
ZARR = ROOT / "data" / "tokamark" / "zarr"
N_TRAIN, N_VAL = 2000, 100
PARTS = ("token_encoder", "backbone", "modality_heads", "output_adapters")
LR = {"token_encoder": 5e-4, "backbone": 1e-4, "modality_heads": 5e-4, "output_adapters": 5e-4}   # Scratch peak / 10
TRAIN_SECONDS = 420.0


def metric_weights() -> dict[str, float]:
    """w_k = mean_j(NRMSE_j) / NRMSE_k from the Scratch vanilla predictions on the TRAIN subsample (TRAIN fits)."""
    sys.path.insert(0, str(ROOT / "paper" / "strat" / "tokamark"))
    import numpy as np
    from analyze import load_split, predictor_table
    import json as _j
    D = load_split("train", groups=("scratch",))
    shots = np.array(sorted(int(s) for s in _j.loads((OUT / "shot_meta.json").read_text())["train_sub"]))
    i = D["members"].index("scr_vanilla")
    tab = predictor_table(D, {"m": {f: np.eye(len(D["members"]))[i] for f in D["features"]}}, shots)["all"][0]
    nr = np.nanmean(tab, axis=0)
    w = nr.mean() / nr
    return {f: float(x) for f, x in zip(D["features"], w)}


def split_reader(n_tr: int, n_va: int):
    import numpy as np
    import pandas as pd
    sample = set(json.loads((OUT / "strata.json").read_text())["valid_sample_shots"])

    def get_train_test_val_shots(max_index=None, data_splits_file_path=None, **_):
        df = pd.read_csv(data_splits_file_path, usecols=["shot_id", "train", "val"])       # test column never read
        tr = df.loc[df["train"].astype(bool), "shot_id"].to_numpy()
        va = sorted(int(s) for s in df.loc[df["val"].astype(bool), "shot_id"] if int(s) not in sample)
        tr = [int(s) for s in np.random.default_rng(0).permutation(tr)[:n_tr]]
        return tr, [], va[:n_va]
    return get_train_test_val_shots


def run_one(mode: str, batches: int) -> None:
    from paper.gpucap import cap_from_env
    cap_from_env()
    import torch
    sys.path.insert(0, str(ROOT / "paper" / "tokamark"))
    import tokamind_hybrid  # noqa: F401  (sys.path for scripts_mast)
    import run_finetune
    import mast_utils.entry_helpers as EH
    import mast_utils.config.loader as LOADER
    from mast_utils.config.finalize import compute_paths

    weights = metric_weights() if mode == "w" else None
    EH.get_train_test_val_shots = split_reader(N_TRAIN, N_VAL)

    def finalize(merged, *, phase, configs_root_path, tasks_overrides_dir):
        merged["paths"] = compute_paths(merged=merged, configs_root=configs_root_path, task_dir=tasks_overrides_dir)
        merged["run_id"] = merged["paths"]["run_id"]
        merged["paths"]["run_dir"] = str(FT_DIR / merged["run_id"])
        (FT_DIR / merged["run_id"]).mkdir(parents=True, exist_ok=True)
    LOADER.finalize_and_save_config = finalize
    orig_load = run_finetune.load_experiment_config

    def load_patched(*a, **k):
        cfg = orig_load(*a, **k)
        raw = cfg.raw
        raw["seed"] = 1337
        d = raw.setdefault("data", {})
        d.update({"local": True, "local_path": str(ZARR), "subset_of_shots": None})
        d.setdefault("cache", {})["enable"] = False
        raw.setdefault("loader", {}).update({"num_workers": 6, "batches_per_epoch": int(batches)})
        raw["model_source"].setdefault("load_parts", {}).update({p: True for p in PARTS})
        raw["train"]["loss"]["output_weights"] = weights
        raw["train"]["stages"] = [{"name": "ft_cont", "epochs": 1,
                                   "scheduler": {"grad_accum_steps": 1, "warmup_steps_fraction": 0.05},
                                   "optimizer": {"lr": dict(LR), "wd": {p: 0.01 for p in PARTS}},
                                   "freeze": {p: False for p in PARTS}}]
        import yaml
        (pathlib.Path(raw["paths"]["run_dir"]) / "strat_config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
        return cfg
    run_finetune.load_experiment_config = load_patched
    orig_train, timing = run_finetune.train_finetune, {}

    def train_timed(*a, **k):
        t0 = time.time()
        r = orig_train(*a, **k)
        timing["train_s"] = time.time() - t0
        return r
    run_finetune.train_finetune = train_timed
    sys.argv = ["run_finetune.py", "--task", "task_1-1", "--init", "warmstart", "--model", str(SRC),
                "--tag", f"strat_cont_{mode}", "--emb_profile", "dct3d"]
    t0 = time.time()
    run_finetune.main()
    res = {"mode": mode, "batches": batches, "train_s": timing.get("train_s"), "wall_s": time.time() - t0,
           "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30, "weights": weights, "lr": LR,
           "n_train_shots": N_TRAIN, "n_val_shots": N_VAL, "source": str(SRC)}
    FT_DIR.mkdir(parents=True, exist_ok=True)
    (FT_DIR / f"result_{mode}.json").write_text(json.dumps(res, indent=1))
    print("[ft_short]", json.dumps({k: v for k, v in res.items() if k != "weights"}), flush=True)


def driver(total_budget: float) -> None:
    t_start = time.time()
    py = sys.executable
    env = dict(os.environ)

    def call(mode: str, batches: int) -> dict:
        subprocess.run([py, __file__, "--mode", mode, "--batches", str(batches)], env=env, check=True)
        return json.loads((FT_DIR / f"result_{mode}.json").read_text())

    s = call("smoke", 15)
    print(f"[driver] smoke peak {s['peak_gpu_gib']:.2f} GiB, {s['train_s']:.0f}s train, {s['wall_s']:.0f}s wall", flush=True)
    if s["peak_gpu_gib"] > 10.0:
        raise SystemExit("[driver] peak GPU memory above 10 GiB: not running the arms")
    overhead = s["wall_s"] - s["train_s"]
    rate = 15 / max(s["train_s"], 1.0)                      # includes the val pass: conservative
    left = total_budget - (time.time() - t_start)
    per_arm = min(TRAIN_SECONDS, (left - 2 * overhead - 60) / 2)
    batches = int(max(30, min(1200, rate * per_arm)))
    print(f"[driver] {left:.0f}s left, arm budget {per_arm:.0f}s -> {batches} batches per arm", flush=True)
    for mode in ("eq", "w"):
        if mode == "w" and not (OUT / "preds_scratch_train.npz").exists():
            print("[driver] TRAIN dump missing: skipping the metric-weighted arm", flush=True)
            continue
        r = call(mode, batches)
        print(f"[driver] {mode}: {r['train_s']:.0f}s train, peak {r['peak_gpu_gib']:.2f} GiB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", action="store_true")
    ap.add_argument("--mode", choices=("smoke", "eq", "w"))
    ap.add_argument("--batches", type=int, default=15)
    ap.add_argument("--budget", type=float, default=1650.0)
    a = ap.parse_args()
    driver(a.budget) if a.driver else run_one(a.mode, a.batches)


if __name__ == "__main__":
    main()
