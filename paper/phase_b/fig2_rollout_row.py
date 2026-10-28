"""Fig. 2 row for the TRL2D long-horizon rollout (Table 1 bottom block): density fields of the VALID-selected rollout
checkpoints (vanilla continued, matched MLP 30, register 30; seed 1337) after a 30-step autoregressive rollout from the
official start (start 0), shown at the middle step of The Well's 13:30 window.

The trajectory is the one with the median vanilla 13:30 error (representative, not chosen for the register); each
panel's number is that arm's 13:30 VRMSE on this trajectory, and the start-0 per-trajectory values are checked against
runs/phase_b2/multistart_<arm>_s1337.npz.

    .venv/bin/python paper/phase_b/fig2_rollout_row.py   -> paper/fig_cache/rows_s23/trl2d_rollout.npz
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_official import load_model  # noqa: E402
from rollout_windows import MAX_T, WINDOW_1330, load_test_trajectories  # noqa: E402

ROOT = HERE.parents[1]
RUNS = ROOT / "runs/phase_b2"
DEV = "cuda"
N_HIST = 4
SEED = 1337
ARMS = (("classical", "vanilla"), ("mlp", "mlp30"), ("qubits", "mr30ryd"))
TOL = 1e-3


@torch.no_grad()
def rollout(model, data: torch.Tensor):
    """Start-0 rollout: per-step VRMSE (MAX_T, n) and the density fields (MAX_T, n, H, W)."""
    n = data.shape[0]
    hist = data[:, :N_HIST].reshape(n, -1, *data.shape[-2:]).to(DEV)
    errs, dens = [], []
    for t in range(N_HIST, N_HIST + MAX_T):
        pred = model(hist); y = data[:, t].to(DEV)
        mse = (pred - y).square().mean(dim=(2, 3)); var = (y - y.mean(dim=(2, 3), keepdim=True)).square().mean(dim=(2, 3))
        errs.append((mse / (var + 1e-7)).sqrt().mean(dim=1).cpu().numpy()); dens.append(pred[:, 0].cpu().numpy())
        hist = torch.cat([hist[:, 4:], pred], dim=1)
    return np.stack(errs), np.stack(dens)


def main() -> None:
    data = load_test_trajectories(ROOT)
    steps = np.arange(MAX_T)[WINDOW_1330]; k = int(steps[len(steps) // 2])           # middle step of 13:30
    res = {}
    for arm, v in ARMS:
        model = load_model(str(RUNS / f"{v}_s{SEED}.pt")).to(DEV).eval()
        err, dens = rollout(model, data); w = err[WINDOW_1330].mean(0)
        ref = np.load(RUNS / f"multistart_{v}_s{SEED}.npz")["w1330"][0]
        diff = float(np.abs(ref - w).max()); print(f"[rollout] {v}: 13:30 {w.mean():.4f} | vs multistart max|diff| {diff:.1e}", flush=True)
        if diff > TOL:
            raise SystemExit(f"{v}: start-0 rollout does not match multistart_{v}_s{SEED}.npz ({diff:.2e})")
        res[arm] = (w, dens[k])
        del model; torch.cuda.empty_cache()
    wv = res["classical"][0]; i = int(np.argsort(wv)[len(wv) // 2])
    out = {"label": "TRL2D rollout", "metric": "VRMSE 13:30", "kind": "field", "sample": i, "step": k + 1,
           "truth": data[i, N_HIST + k, 0].numpy()}
    for arm, (w, d) in res.items():
        out[f"pred_{arm}"] = d[i]; out[f"m_{arm}"] = float(w[i])
    dst = ROOT / "paper/fig_cache/rows_s23/trl2d_rollout.npz"
    np.savez(dst, **{a: np.asarray(b) for a, b in out.items()})
    print(f"wrote {dst} (trajectory {i}, step {k + 1}: " + ", ".join(f"{a} {out[f'm_{a}']:.4f}" for a, _ in ARMS) + ")", flush=True)


if __name__ == "__main__":
    main()
