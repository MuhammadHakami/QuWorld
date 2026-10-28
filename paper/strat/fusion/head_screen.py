"""Doc 35 §38e: VALID-sample accuracy of shape-head variants (no scorer needed): per consistency scalar R^2 against the
scorer's extraction of the true flux (fixed normaliser) and LCFS ray-radius RMSE (mm) against y_valid.

    .venv/bin/python paper/strat/fusion/head_screen.py <stem>[+<stem>...] ...    (a+b = mean of the two heads)
"""
from __future__ import annotations

import sys

import numpy as np

import fx  # noqa: I001
import strategies as ST
from gpu_pipe import NAMES


def main() -> None:
    sota, d = fx.data(); shots, _ = ST.sample(); pe, nm = ST.load_entry()
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); pos = {int(r): i for i, r in enumerate(lab["rows_valid"])}
    rows = np.concatenate([fx.valid_rows(d, s) for s in shots]); ii = [pos[int(r)] for r in rows]
    gt = np.concatenate([pe[s]["cgt"] for s in shots]); var = np.asarray(nm["c_var"]); yr = lab["y_valid"][ii, 7:].astype(np.float64)
    for arg in sys.argv[1:]:
        P = np.mean([np.load(fx.OUT / f"{st}.npz")["pred_valid"] for st in arg.split("+")], 0).astype(np.float64)[ii]
        r2 = []
        for k in range(7):
            m = np.isfinite(gt[:, k]) & np.isfinite(P[:, k]); r2.append(1 - np.mean((gt[m, k] - P[m, k]) ** 2) / var[k])
        m = np.isfinite(yr); rm = 1000 * np.sqrt(np.mean((np.where(m, yr, 0) - np.where(m, P[:, 7:], 0))[m] ** 2))
        print(f"[s38e] {arg}: mean R2 {np.mean(r2):.5f} | " + " ".join(f"{n} {v:.4f}" for n, v in zip(NAMES, r2)) + f" | rays RMSE {rm:.2f} mm", flush=True)


if __name__ == "__main__":
    main()
