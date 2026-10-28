"""TORAX-Bench v1 CPU baselines — persistence and climatology (numpy only).

Metric: VRMSE (variance-normalized RMSE), following The Well convention.
For one (discharge, target-frame, channel):
    VRMSE = sqrt( mean_x (u - u_hat)^2 / (var_x(u) + EPS) )
with the mean/variance over the radial grid of the TRUE target frame.
Scores are averaged over target frames, then discharges, per channel.

Baselines:
  * persistence@1: u_hat(t+1) = u(t), all valid t.
  * persistence@8: u_hat(t+8) = u(t), all valid t.
  * climatology:   u_hat(t, x) = mean over TRAIN discharges at (t, x),
                   evaluated on every frame of each TEST discharge.

Usage: python paper/torax_bench/baselines.py [data_dir] [splits_json]
"""

import json
import pathlib
import sys

import numpy as np

DATA = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/torax_bench_v1")
SPLITS = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                      "paper/torax_bench/splits.json")
CHANNELS = ["T_e", "T_i", "n_e", "q"]
EPS = 1e-12


def load_split(ids: list) -> dict:
    """Stack per-channel arrays over discharges: {ch: (N, T, X) float64}."""
    stacks = {ch: [] for ch in CHANNELS}
    for i in ids:
        with np.load(DATA / f"discharge_{i:05d}.npz") as d:
            for ch in CHANNELS:
                stacks[ch].append(d[ch].astype(np.float64))
    return {ch: np.stack(v) for ch, v in stacks.items()}


def vrmse(true: np.ndarray, pred: np.ndarray) -> float:
    """Mean VRMSE over leading axes; spatial axis is the last one."""
    mse = np.mean((true - pred) ** 2, axis=-1)
    var = np.var(true, axis=-1)
    return float(np.mean(np.sqrt(mse / (var + EPS))))


def persistence(test: dict, k: int) -> dict:
    return {ch: vrmse(x[:, k:, :], x[:, :-k, :]) for ch, x in test.items()}


def climatology(train: dict, test: dict) -> dict:
    return {ch: vrmse(x, np.broadcast_to(train[ch].mean(axis=0), x.shape))
            for ch, x in test.items()}


def main() -> None:
    splits = json.loads(SPLITS.read_text())
    train = load_split(splits["train"])
    test = load_split(splits["test"])
    print(f"train N={len(splits['train'])}  test N={len(splits['test'])}  "
          f"frames/discharge={train[CHANNELS[0]].shape[1]}")

    rows = {
        "persistence@1": persistence(test, 1),
        "persistence@8": persistence(test, 8),
        "climatology": climatology(train, test),
    }
    header = f"{'baseline':<15}" + "".join(f"{ch:>10}" for ch in CHANNELS) \
        + f"{'mean':>10}"
    print(header)
    print("-" * len(header))
    for name, r in rows.items():
        mean = float(np.mean([r[ch] for ch in CHANNELS]))
        print(f"{name:<15}" + "".join(f"{r[ch]:>10.4f}" for ch in CHANNELS)
              + f"{mean:>10.4f}")


if __name__ == "__main__":
    main()
