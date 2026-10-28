"""Ground-truth-vs-prediction density figure for MHD_64 (NeurIPS workshop).

Valid-split trajectory 0 from the phase_b_mhd cache (protocol-faithful
normalized fp32, see data_ram_mhd.py). Hybrid checkpoints rebuilt against
data/cnext_mhd_ckpt exactly as run_mhd_campaign.sh does. Rollout batch=1,
torch.no_grad. Mid-z density slice, lead times {1, 8}.
Outputs paper/manuscript/fig_fields_mhd.pdf + caches arrays in an .npz.
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = pathlib.Path("/home/mohhakami/Documents/paper/quworld")
SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-mohhakami-Documents-paper-quworld/"
    "af4d80b6-5cb2-4675-b69d-81a19e2a7c31/scratchpad")
sys.path.insert(0, str(ROOT / "paper/phase_b"))

import cnext_hybrid  # noqa: E402

_orig_build = cnext_hybrid.build_hybrid
_MHD_DIR = str(cnext_hybrid.ROOT / "data/cnext_mhd_ckpt")
cnext_hybrid.build_hybrid = (
    lambda variant, model_dir=_MHD_DIR, q_layers=4:
        _orig_build(variant, model_dir=model_dir, q_layers=q_layers))

from eval_official import load_model  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_HIST = 4
SHOW_STEPS = (1, 8)
MAX_STEP = 8
MIDZ = 32
NPZ = ROOT / "paper/fig_cache/fig_fields_mhd.npz"
OUT_PDF = ROOT / "paper/manuscript/fig_fields_mhd.pdf"

MODELS = {
    "Vanilla-TD": str(ROOT / "runs/phase_b_mhd/vanilla_s1337.pt"),
    "MLP-TD": str(ROOT / "runs/phase_b_mhd/mlp30_s1337.pt"),
    "Quantum-TD": str(ROOT / "runs/phase_b_mhd/mr30ryd_s1337.pt"),
}


def load_valid_traj0() -> torch.Tensor:
    meta = torch.load(ROOT / "data/phaseb_mhd_cache/meta.pt", weights_only=True)
    shape = tuple(meta["valid"])  # (10, 100, 7, 64, 64, 64)
    mm = np.memmap(ROOT / "data/phaseb_mhd_cache/valid.f32",
                   dtype=np.float32, mode="r", shape=shape)
    frames = np.array(mm[0, :N_HIST + MAX_STEP])  # (12, 7, 64, 64, 64)
    del mm
    return torch.from_numpy(frames)


@torch.no_grad()
def rollout(ckpt: str, data: torch.Tensor):
    model = load_model(ckpt).to(DEV).eval()
    hist = data[:N_HIST].reshape(1, -1, *data.shape[-3:]).to(DEV)
    slices, vr = {}, {}
    for step in range(1, MAX_STEP + 1):
        pred = model(hist)                                   # (1, 7, 64,64,64)
        y = data[N_HIST + step - 1].unsqueeze(0).to(DEV)
        mse = (pred - y).square().mean(dim=(2, 3, 4))
        var = (y - y.mean(dim=(2, 3, 4), keepdim=True)).square().mean(dim=(2, 3, 4))
        vr[step] = (mse / (var + 1e-7)).sqrt().mean().item()
        if step in SHOW_STEPS:
            slices[step] = pred[0, 0, :, :, MIDZ].float().cpu().numpy()
        hist = torch.cat([hist[:, 7:], pred], dim=1)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return slices, vr


def compute() -> dict:
    s = yaml.safe_load(open(ROOT / "data/mhd64/stats.yaml"))
    out = {"mu_d": float(s["mean"]["density"]),
           "sd_d": max(float(s["std"]["density"]), 1e-4)}
    data = load_valid_traj0()
    print(f"valid traj 0 frames: {tuple(data.shape)}")
    for st in SHOW_STEPS:
        out[f"gt_{st}"] = data[N_HIST + st - 1, 0, :, :, MIDZ].numpy()
    for name, ckpt in MODELS.items():
        print(f"rollout: {name}", flush=True)
        slices, vr = rollout(ckpt, data)
        for st in SHOW_STEPS:
            out[f"{name}_{st}"] = slices[st]
        out[f"vrmse_{name}"] = np.array([vr[k] for k in range(1, MAX_STEP + 1)])
    np.savez_compressed(NPZ, **out)
    return out


def denorm(d: dict, key: str, step: int) -> np.ndarray:
    return d[f"{key}_{step}"] * d["sd_d"] + d["mu_d"]


def plot(d: dict) -> None:
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5,
                         "figure.dpi": 200})
    rows = ["Ground truth"] + list(MODELS)
    gts = [denorm(d, "gt", s) for s in SHOW_STEPS]
    vmin = min(g.min() for g in gts)
    vmax = max(g.max() for g in gts)
    print(f"GT slice range: [{vmin:.3f}, {vmax:.3f}]")

    fig = plt.figure(figsize=(2.7, 4.15))
    gs = fig.add_gridspec(4, 3, width_ratios=[1, 1, 0.06],
                          wspace=0.05, hspace=0.08,
                          left=0.13, right=0.83, top=0.93, bottom=0.015)
    im = None
    for r, row in enumerate(rows):
        for c, step in enumerate(SHOW_STEPS):
            ax = fig.add_subplot(gs[r, c])
            key = "gt" if row == "Ground truth" else row
            im = ax.imshow(denorm(d, key, step), cmap="viridis",
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if r == 0:
                ax.set_title(f"step {step}", pad=2.5)
            if c == 0:
                ax.text(-0.10, 0.5, row, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=7)
    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("density", fontsize=7, labelpad=1)
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PDF}")


def report(d: dict) -> None:
    print("\n=== per-step VRMSE (valid traj 0, all 7 channels) ===")
    print("model        " + "".join(f"  step{s:>2}" for s in range(1, MAX_STEP + 1)))
    for name in MODELS:
        v = d[f"vrmse_{name}"]
        print(f"{name:<13}" + "".join(f"  {x:6.3f}" for x in v))
    gt8 = d["gt_8"]
    gy, gx = np.gradient(gt8)
    g_gt = float(np.hypot(gx, gy).mean())
    print(f"\nstep-8 mid-z density grad|.| GT={g_gt:.4f}")
    for name in MODELS:
        p = d[f"{name}_8"]
        gy, gx = np.gradient(p)
        g = float(np.hypot(gx, gy).mean())
        rmse = float(np.sqrt(((p - gt8) ** 2).mean()))
        print(f"  {name:<11} grad|.|={g:.4f} ({g / g_gt * 100:5.1f}% of GT)  "
              f"slice-rmse(norm)={rmse:.4f}")


def main() -> None:
    if NPZ.exists():
        print(f"using cached {NPZ}")
        d = dict(np.load(NPZ))
        d["mu_d"], d["sd_d"] = float(d["mu_d"]), float(d["sd_d"])
    else:
        d = compute()
    report(d)
    plot(d)


if __name__ == "__main__":
    main()
