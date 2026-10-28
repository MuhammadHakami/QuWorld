"""Sliding-start TEST rollout evaluation (pre-registered: 19-multistart-power-plan.md).

Usage: uv run paper/phase_b/eval_rollout_multistart.py <ckpt.pt|model_dir> <name> [stride=8]

Same data, normalisation, 4-frame history, 30-step horizon and windows as
eval_rollout_pertraj.py, but rollouts start at every frame returned by
rollout_windows.start_indices (stride 8 -> 9 starts x 9 trajectories). Start 0
IS the official protocol and is checked against runs/phase_b2/pertraj_<name>.npz
when that file exists. Saves runs/phase_b2/multistart_<name>.npz with
per_step (S, MAX_T, n), w612/w1330 (S, n) and starts (S,).
"""

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_official import load_model  # noqa: E402
from rollout_windows import (MAX_T, load_test_trajectories,  # noqa: E402
                             start_indices, window_means)

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
N_HIST = 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"
REGRESSION_TOL = 1e-3   # start-0 vs the official per-trajectory evaluation


@torch.no_grad()
def rollout_from(model, data: torch.Tensor, start: int) -> np.ndarray:
    """(MAX_T, n) per-trajectory VRMSE for one start frame."""
    n = data.shape[0]
    hist = data[:, start:start + N_HIST].reshape(n, -1, *data.shape[-2:]).to(DEV)
    steps = []
    for t in range(start + N_HIST, start + N_HIST + MAX_T):
        pred = model(hist)
        y = data[:, t].to(DEV)
        mse = (pred - y).square().mean(dim=(2, 3))
        var = (y - y.mean(dim=(2, 3), keepdim=True)).square().mean(dim=(2, 3))
        steps.append((mse / (var + 1e-7)).sqrt().mean(dim=1).cpu().numpy())
        hist = torch.cat([hist[:, 4:], pred], dim=1)
    return np.stack(steps)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    ckpt, name = sys.argv[1], sys.argv[2]
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    data = load_test_trajectories(ROOT)
    starts = start_indices(data.shape[1], N_HIST, MAX_T, stride)
    model = load_model(ckpt).to(DEV).eval()
    per_step = np.stack([rollout_from(model, data, s) for s in starts])  # (S, MAX_T, n)
    w612, w1330 = window_means(per_step)
    out = ROOT / f"runs/phase_b2/multistart_{name}.npz"
    np.savez(out, per_step=per_step, w612=w612, w1330=w1330,
             starts=np.array(starts), ckpt=str(ckpt))
    print(f"ckpt:              {ckpt}")
    print(f"starts:            {starts}   n_traj={data.shape[0]}")
    print(f"start0_13_30:      {w1330[0].mean():.4f}   (official protocol)")
    print(f"allstart_13_30:    {w1330.mean():.4f}   per-start: "
          + " ".join(f"{x:.3f}" for x in w1330.mean(axis=1)))
    print(f"allstart_6_12:     {w612.mean():.4f}")
    ref = ROOT / f"runs/phase_b2/pertraj_{name}.npz"
    if ref.exists():
        r = np.load(ref)
        diff = float(np.abs(r["w1330"] - w1330[0]).max())
        status = "OK" if diff < REGRESSION_TOL else "MISMATCH"
        print(f"start0_vs_official max|diff|: {diff:.2e}  {status}")
        if status != "OK":
            raise SystemExit(f"start-0 regression failed against {ref}")
    print(f"saved:             {out}")


if __name__ == "__main__":
    main()
