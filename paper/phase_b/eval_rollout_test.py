"""Official-protocol TEST rollout evaluation (The Well windows 6:12 and 13:30).

Usage: uv run paper/phase_b/eval_rollout_test.py <ckpt.pt|model_dir> [out_prefix]

Autoregressive rollout from each test trajectory's start (4-frame history),
VRMSE per step averaged over the published windows 6:12 and 13:30 —
comparable to The Well paper's Table 3 (CNextU-net TRL2D: 0.54 / 1.01).
Single-shot discipline: run once per final model.
"""

import pathlib
import sys

import torch
from einops import rearrange

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_official import load_model  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CKPT = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "rollout"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_HIST, MAX_T = 4, 30


@torch.no_grad()
def main() -> None:
    from the_well.data import WellDataset
    from the_well.data.normalization import ZScoreNormalization

    ds = WellDataset(
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
    step_vrmse = []
    for t in range(N_HIST, N_HIST + MAX_T):
        pred = model(hist)
        y = data[:, t].to(DEV)
        mse = (pred - y).square().mean(dim=(2, 3))
        var = (y - y.mean(dim=(2, 3), keepdim=True)).square().mean(dim=(2, 3))
        step_vrmse.append((mse / (var + 1e-7)).sqrt().mean().item())
        hist = torch.cat([hist[:, 4:], pred], dim=1)
    v = np.array(step_vrmse)  # index 0 = rollout step 1
    w612 = v[5:12].mean()     # steps 6..12
    w1330 = v[12:30].mean()   # steps 13..30
    print(f"ckpt:            {CKPT}")
    print(f"rollout_6_12:    {w612:.4f}   (published CNextU-net test: 0.54)")
    print(f"rollout_13_30:   {w1330:.4f}   (published: 1.01)")
    print(f"per_step: {' '.join(f'{x:.3f}' for x in v[:12])}")


if __name__ == "__main__":
    main()
