"""Ground-truth-vs-prediction density figure for TRL2D (NeurIPS workshop).

Mirrors paper/phase_b/eval_rollout_test.py for data loading + normalization;
loads models via eval_official.load_model. Rollout on the FIRST test
trajectory only (tcool_0.03), batch=1, torch.no_grad.
Outputs paper/manuscript/fig_fields.pdf + caches arrays in an .npz.
"""

import glob
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import LogNorm

ROOT = pathlib.Path("/home/mohhakami/Documents/paper/quworld")
SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-mohhakami-Documents-paper-quworld/"
    "af4d80b6-5cb2-4675-b69d-81a19e2a7c31/scratchpad")
sys.path.insert(0, str(ROOT / "paper/phase_b"))
from eval_official import load_model  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_HIST = 4
SHOW_STEPS = (1, 8, 16, 24)
MAX_STEP = 24
NPZ = ROOT / "paper/fig_cache/fig_fields_trl2d.npz"
OUT_PDF = ROOT / "paper/manuscript/fig_fields.pdf"

MODELS = {  # row order below Ground truth
    "Anchor": str(ROOT / "data/cnext_ckpt"),
    "Vanilla-TD": str(ROOT / "runs/phase_b2/vanilla_s1337.pt"),
    "MLP-TD": str(ROOT / "runs/phase_b2/mlp30_s1337.pt"),
    "Quantum-TD": str(ROOT / "runs/phase_b2/mr30ryd_s1337.pt"),
}


def load_first_trajectory():
    s = yaml.safe_load(open(ROOT / "data/trl2d/stats.yaml"))
    mean = torch.tensor([s["mean"]["density"], s["mean"]["pressure"],
                         *s["mean"]["velocity"]]).view(1, 4, 1, 1)
    std = torch.tensor([s["std"]["density"], s["std"]["pressure"],
                        *s["std"]["velocity"]]).view(1, 4, 1, 1)
    first = sorted(glob.glob(str(ROOT / "data/trl2d/data/test/*.hdf5")))[0]
    with h5py.File(first) as h:
        den = torch.from_numpy(h["t0_fields/density"][0])
        pre = torch.from_numpy(h["t0_fields/pressure"][0])
        vel = torch.from_numpy(h["t1_fields/velocity"][0])
    traj = torch.stack([den, pre, vel[..., 0], vel[..., 1]], dim=1)  # (T,4,H,W)
    data = torch.nan_to_num((traj.float() - mean) / std)
    return data, float(mean[0, 0]), float(std[0, 0]), first


@torch.no_grad()
def rollout(ckpt: str, data: torch.Tensor):
    """Return ({step: normalized density pred (H,W)}, {step: VRMSE})."""
    model = load_model(ckpt).to(DEV).eval()
    hist = data[:N_HIST].reshape(1, -1, *data.shape[-2:]).to(DEV)
    dens, vr = {}, {}
    for step in range(1, MAX_STEP + 1):
        pred = model(hist)
        y = data[N_HIST + step - 1].unsqueeze(0).to(DEV)
        mse = (pred - y).square().mean(dim=(2, 3))
        var = (y - y.mean(dim=(2, 3), keepdim=True)).square().mean(dim=(2, 3))
        vr[step] = (mse / (var + 1e-7)).sqrt().mean().item()
        if step in SHOW_STEPS:
            dens[step] = pred[0, 0].float().cpu().numpy()
        hist = torch.cat([hist[:, 4:], pred], dim=1)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return dens, vr


def high_k_fraction(arr: np.ndarray, frac: float = 0.5) -> float:
    """Fraction of 2D spectral energy at |k| above frac * Nyquist."""
    f = np.fft.fftshift(np.abs(np.fft.fft2(arr - arr.mean())) ** 2)
    h, w = arr.shape
    ky = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    kx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)  # Nyquist = 0.5
    hi = f[k > frac * 0.5].sum()
    return float(hi / f.sum())


def grad_mag_mean(arr: np.ndarray) -> float:
    gy, gx = np.gradient(arr)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def compute() -> dict:
    data, mu_d, sd_d, first = load_first_trajectory()
    print(f"trajectory file: {first}  data {tuple(data.shape)}")
    out = {"mu_d": mu_d, "sd_d": sd_d}
    for s in SHOW_STEPS:
        out[f"gt_{s}"] = data[N_HIST + s - 1, 0].numpy()
    for name, ckpt in MODELS.items():
        print(f"rollout: {name} ({ckpt})", flush=True)
        dens, vr = rollout(ckpt, data)
        for s in SHOW_STEPS:
            out[f"{name}_{s}"] = dens[s]
        out[f"vrmse_{name}"] = np.array([vr[s] for s in range(1, MAX_STEP + 1)])
    np.savez_compressed(NPZ, **out)
    return out


def denorm(d: dict, key: str, step: int) -> np.ndarray:
    return d[f"{key}_{step}"] * d["sd_d"] + d["mu_d"]


def diagnostics(d: dict) -> None:
    print("\n=== per-step VRMSE (this trajectory, all 4 channels) ===")
    print("model        " + "".join(f"  step{s:>3}" for s in SHOW_STEPS))
    for name in MODELS:
        v = d[f"vrmse_{name}"]
        print(f"{name:<13}" + "".join(f"  {v[s - 1]:7.4f}" for s in SHOW_STEPS))
    print("\n=== structure diagnostics on density (normalized units) ===")
    for step in (8, 24):
        gt = d[f"gt_{step}"]
        print(f"-- step {step}: GT grad|.|={grad_mag_mean(gt):.4f} "
              f"hi-k frac={high_k_fraction(gt):.4f}")
        for name in MODELS:
            p = d[f"{name}_{step}"]
            print(f"   {name:<11} grad|.|={grad_mag_mean(p):.4f} "
                  f"({grad_mag_mean(p) / grad_mag_mean(gt) * 100:5.1f}% of GT)  "
                  f"hi-k frac={high_k_fraction(p):.4f}")


def plot(d: dict) -> None:
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5,
                         "figure.dpi": 200})
    rows = ["Ground truth"] + list(MODELS) + ["Quantum−GT"]
    # field color limits from GT over shown steps (log scale: skewed density)
    gts = [denorm(d, "gt", s) for s in SHOW_STEPS]
    vmin = max(0.5, min(g.min() for g in gts))
    vmax = max(g.max() for g in gts)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    diffs = [denorm(d, "Quantum-TD", s) - denorm(d, "gt", s) for s in SHOW_STEPS]
    dlim = float(np.percentile(np.abs(np.stack(diffs)), 99))

    fig = plt.figure(figsize=(5.4, 4.7))
    gs = fig.add_gridspec(6, 5, width_ratios=[1, 1, 1, 1, 0.05],
                          wspace=0.04, hspace=0.10,
                          left=0.075, right=0.915, top=0.945, bottom=0.01)
    im_f = im_d = None
    for r, row in enumerate(rows):
        for c, step in enumerate(SHOW_STEPS):
            ax = fig.add_subplot(gs[r, c])
            if row == "Quantum−GT":
                im_d = ax.imshow(diffs[c], cmap="RdBu_r",
                                 vmin=-dlim, vmax=dlim, interpolation="nearest")
            else:
                key = "gt" if row == "Ground truth" else row
                arr = np.maximum(denorm(d, key, step), vmin)
                im_f = ax.imshow(arr, cmap="viridis", norm=norm,
                                 interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if r == 0:
                ax.set_title(f"step {step}", pad=2.5)
            if c == 0:
                ax.text(-0.055, 0.5, row, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=7)
    cax_f = fig.add_subplot(gs[0:5, 4])
    cb = fig.colorbar(im_f, cax=cax_f)
    cb.set_label("density", fontsize=7, labelpad=1)
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cax_d = fig.add_subplot(gs[5, 4])
    cbd = fig.colorbar(im_d, cax=cax_d)
    cbd.set_label(r"$\Delta$", fontsize=7, labelpad=1)
    cbd.ax.tick_params(labelsize=6, length=2, pad=1)
    cbd.set_ticks([-dlim, 0, dlim])
    cbd.set_ticklabels([f"{-dlim:.0f}", "0", f"{dlim:.0f}"])
    fig.savefig(OUT_PDF)
    print(f"\nwrote {OUT_PDF}  (diff limits +/-{dlim:.1f})")


def main() -> None:
    if NPZ.exists():
        print(f"using cached {NPZ}")
        d = dict(np.load(NPZ))
        d["mu_d"], d["sd_d"] = float(d["mu_d"]), float(d["sd_d"])
    else:
        d = compute()
    diagnostics(d)
    plot(d)


if __name__ == "__main__":
    main()
