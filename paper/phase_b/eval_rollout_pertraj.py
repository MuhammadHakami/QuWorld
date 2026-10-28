"""Per-trajectory TEST rollout evaluation (The Well windows 6:12 and 13:30).

Usage: uv run paper/phase_b/eval_rollout_pertraj.py <ckpt.pt|model_dir> <name>

Identical protocol to eval_rollout_test.py (same data loading, normalization,
N_HIST=4, MAX_T=30, load_model via eval_official), but records PER-TRAJECTORY
VRMSE per rollout step (mse/var computed per trajectory, never batch-averaged)
and per-trajectory means over the published windows 6:12 (v[5:12]) and 13:30
(v[12:30]). Saves arrays to runs/phase_b2/pertraj_<name>.npz for paired
significance statistics. Aggregates printed for sanity vs eval_rollout_test.py.
"""

import pathlib
import sys

import torch
from einops import rearrange  # noqa: F401  (parity with eval_rollout_test)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_official import load_model  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CKPT = sys.argv[1]
NAME = sys.argv[2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_HIST, MAX_T = 4, 30


@torch.no_grad()
def main() -> None:
    from the_well.data import WellDataset
    from the_well.data.normalization import ZScoreNormalization

    ds = WellDataset(  # noqa: F841  (parity with eval_rollout_test.py)
        well_base_path=str(ROOT / "data/well"),
        well_dataset_name="turbulent_radiative_layer_2D",
        well_split_name="test", n_steps_input=N_HIST,
        n_steps_output=1, use_normalization=True,
        normalization_type=ZScoreNormalization, max_rollout_steps=1)
    # Reconstruct full test trajectories from the dataset's underlying files
    import glob

    import h5py
    import numpy as np
    import yaml
    s = yaml.safe_load(open(ROOT / "data/trl2d/stats.yaml"))
    mean = torch.tensor([s["mean"]["density"], s["mean"]["pressure"],
                         *s["mean"]["velocity"]]).view(1, 1, 4, 1, 1)
    std = torch.tensor([s["std"]["density"], s["std"]["pressure"],
                        *s["std"]["velocity"]]).view(1, 1, 4, 1, 1)
    trajs = []
    for f in sorted(glob.glob(str(ROOT / "data/trl2d/data/test/*.hdf5"))):
        with h5py.File(f) as h:
            den = torch.from_numpy(h["t0_fields/density"][:])
            pre = torch.from_numpy(h["t0_fields/pressure"][:])
            vel = torch.from_numpy(h["t1_fields/velocity"][:])
        trajs.append(torch.stack([den, pre, vel[..., 0], vel[..., 1]], dim=2))
    data = torch.nan_to_num((torch.cat(trajs).float() - mean) / std)  # (n,T,4,H,W)

    model = load_model(CKPT).to(DEV).eval()
    n = data.shape[0]
    hist = data[:, :N_HIST].reshape(n, -1, *data.shape[-2:]).to(DEV)
    step_vrmse = []  # each entry: (n,) per-trajectory VRMSE at that step
    for t in range(N_HIST, N_HIST + MAX_T):
        pred = model(hist)
        y = data[:, t].to(DEV)
        mse = (pred - y).square().mean(dim=(2, 3))
        var = (y - y.mean(dim=(2, 3), keepdim=True)).square().mean(dim=(2, 3))
        step_vrmse.append((mse / (var + 1e-7)).sqrt().mean(dim=1).cpu().numpy())
        hist = torch.cat([hist[:, 4:], pred], dim=1)
    v = np.stack(step_vrmse)          # (MAX_T, n); row 0 = rollout step 1
    w612 = v[5:12].mean(axis=0)       # (n,) per-trajectory mean, steps 6..12
    w1330 = v[12:30].mean(axis=0)     # (n,) per-trajectory mean, steps 13..30

    out = ROOT / f"runs/phase_b2/pertraj_{NAME}.npz"
    np.savez(out, per_step=v, w612=w612, w1330=w1330, ckpt=str(CKPT))
    print(f"ckpt:            {CKPT}")
    print(f"n_traj:          {n}")
    print(f"rollout_6_12:    {w612.mean():.4f}   (aggregate sanity check)")
    print(f"rollout_13_30:   {w1330.mean():.4f}")
    print(f"pertraj_6_12:    {' '.join(f'{x:.4f}' for x in w612)}")
    print(f"pertraj_13_30:   {' '.join(f'{x:.4f}' for x in w1330)}")
    print(f"saved:           {out}")


if __name__ == "__main__":
    main()
