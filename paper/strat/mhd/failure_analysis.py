"""Doc 35 §37j: where does the final Flowers MHD_64 model fail? (VALID only, 32 stratified windows; TEST never loaded)

Per window and channel: VRMSE; radial spectra of truth, prediction and error (bands of analyze.py) -> energy ratio
pred/true (over-smoothing below 1) and error/true per band; amplitude ratio std(pred)/std(true); error localisation:
share of the window's normalised squared error inside the top-10% voxels of |grad rho| (shocks), |curl B| (current
sheets), -div v (compression) and |y - x_t| (where the field changes), against their 10% volume; mean (k = 0) bias.
Aggregated by Alfven regime and sonic Mach. Figures: worst window slices and mean spectra.

    QW_GPU_GB=6 runs/capped.sh s37j 16 uv run --no-sync python paper/strat/mhd/failure_analysis.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ.setdefault("STRAT_FLOWER_CKPT", "runs/phase_b_mhd/flower_s1337.pt")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_strat as ES  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

BANDS = [[0], [1], [2, 3], list(range(4, 8)), list(range(8, 16)), list(range(16, 32)), list(range(32, 56))]
BN = ["k0", "k1", "k2-3", "k4-7", "k8-15", "k16-31", "k32+"]
CH = ["rho", "Bx", "By", "Bz", "vx", "vy", "vz"]


def grad(f):                                                   # periodic central differences, grid spacing 1
    return [(torch.roll(f, -1, d) - torch.roll(f, 1, d)) / 2 for d in (-3, -2, -1)]


def top_mask(a, q=0.9):
    return a >= torch.quantile(a.flatten().float()[::7], q)


@torch.no_grad()
def main() -> None:
    models, meta = ES.load_models(); F = models["F"]
    valid = ES.WellND("MHD_64", "valid")
    m = torch.tensor(valid.mean, device=ES.DEV); s = torch.tensor(valid.std, device=ES.DEV)
    d = json.loads((ES.OUT / "strata.json").read_text())["samples"]["valid"]
    by = {}
    for r in d:
        by.setdefault((r["Ma"], r["tbin"]), []).append(r)
    rows = [r for key in sorted(by) for r in by[key][::5][:4]]                 # one window per trajectory, 4 per stratum
    shells = ES.shell_index().cpu()
    band = torch.zeros(ES.NSHELL, dtype=torch.long)
    for i, b in enumerate(BANDS):
        band[b] = i
    band = band[shells]
    recs, worst = [], None
    for r in rows:
        x, y = (v.unsqueeze(0).to(ES.DEV) for v in valid.window(r["traj"], r["t0"]))
        p = F(x)[0]; y0 = y[0]; xt = x[0, -ES.C:]
        var = (y0 - y0.mean((1, 2, 3), keepdim=True)).square().mean((1, 2, 3))
        e = p - y0; vr = (e.square().mean((1, 2, 3)) / (var + 1e-7)).sqrt()
        spec = lambda f: torch.zeros(ES.C, len(BANDS)).index_add_(1, band, (torch.fft.fftn(f, dim=(1, 2, 3)).abs().square().reshape(ES.C, -1).cpu() / f[0].numel()))
        Et, Ep, Ee = spec(y0 - y0.mean((1, 2, 3), keepdim=True)), spec(p - p.mean((1, 2, 3), keepdim=True)), spec(e)
        phys = y0 * s + m
        rho, B, v = phys[0], phys[1:4], phys[4:7]
        g = grad(rho); grho = torch.sqrt(sum(t.square() for t in g))
        dB = [grad(B[i]) for i in range(3)]
        J = torch.sqrt((dB[2][1] - dB[1][2]).square() + (dB[0][2] - dB[2][0]).square() + (dB[1][0] - dB[0][1]).square())
        divv = sum(grad(v[i])[i] for i in range(3))
        chg = ((y0 - xt).square() / (var[:, None, None, None] + 1e-7)).sum(0)
        se = (e.square() / (var[:, None, None, None] + 1e-7)).sum(0)                      # normalised squared error
        loc = {name: float(se[top_mask(a)].sum() / se.sum()) for name, a in
               (("shock_gradrho", grho), ("current_J", J), ("compression", -divv), ("change", chg))}
        rec = {"traj": r["traj"], "t0": r["t0"], "Ma": r["Ma"], "Ms": r["Ms"], "vrmse": vr.tolist(), "vr_mean": float(vr.mean()),
               "amp_ratio": (p.std((1, 2, 3)) / y0.std((1, 2, 3))).tolist(), "bias": (e.mean((1, 2, 3)) / (var.sqrt() + 1e-7)).tolist(),
               "Ep_over_Et": (Ep / (Et + 1e-12)).tolist(), "Ee_over_Et": (Ee / (Et + 1e-12)).tolist(), "err_band_share": (Ee / Ee.sum(1, keepdim=True)).tolist(),
               "loc": loc}
        recs.append(rec)
        if worst is None or rec["vr_mean"] > worst[0]["vr_mean"]:
            worst = (rec, y0.cpu(), p.cpu(), J.cpu(), xt.cpu())
    agg = {}
    for key, sel in (("all", recs), ("Ma0.7", [q for q in recs if q["Ma"] == 0.7]), ("Ma2", [q for q in recs if q["Ma"] == 2.0])):
        A = lambda f: np.mean([q[f] for q in sel], 0)
        agg[key] = {"vrmse_ch": dict(zip(CH, np.round(A("vrmse"), 4).tolist())), "amp_ratio": dict(zip(CH, np.round(A("amp_ratio"), 3).tolist())),
                    "Ep_over_Et_rho": dict(zip(BN, np.round(A("Ep_over_Et")[0], 3).tolist())),
                    "Ep_over_Et_Bx": dict(zip(BN, np.round(A("Ep_over_Et")[1], 3).tolist())),
                    "Ee_over_Et_rho": dict(zip(BN, np.round(A("Ee_over_Et")[0], 3).tolist())),
                    "loc_share_top10pct": {k: round(float(np.mean([q["loc"][k] for q in sel])), 3) for k in sel[0]["loc"]}}
    for ms in sorted({q["Ms"] for q in recs}):
        sel = [q for q in recs if q["Ma"] == 2.0 and q["Ms"] == ms]
        if sel:
            agg[f"Ma2_Ms{ms}"] = {"vr": round(float(np.mean([q["vr_mean"] for q in sel])), 4), "n": len(sel),
                                  "shock_share": round(float(np.mean([q["loc"]["shock_gradrho"] for q in sel])), 3)}
    (ES.OUT / "failure.json").write_text(json.dumps({"agg": agg, "windows": recs, "flower_step": meta["flower_step"]}, indent=1))
    print(json.dumps(agg, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rec, y0, p, J, xt = worst; z = y0.shape[-1] // 2
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.5))
    for row, (c, nm) in enumerate(((0, "density"), (1, "B_x"))):
        lim = float(y0[c, :, :, z].abs().max())
        for col, (img, title) in enumerate(((y0[c, :, :, z], "truth"), (p[c, :, :, z], "Flowers"), ((p - y0)[c, :, :, z], "error"),
                                            ((y0 - xt)[c, :, :, z], "true change y - x_t"))):
            im = ax[row, col].imshow(img.numpy().T, origin="lower", cmap="RdBu_r" if col >= 2 else "viridis",
                                     vmin=-lim if col >= 2 else None, vmax=lim if col >= 2 else None)
            ax[row, col].set_title(f"{nm}: {title}"); ax[row, col].axis("off"); fig.colorbar(im, ax=ax[row, col], fraction=0.046)
    fig.suptitle(f"Worst stratified VALID window: traj {rec['traj']} t0 {rec['t0']} Ma {rec['Ma']} Ms {rec['Ms']} "
                 f"VRMSE {rec['vr_mean']:.3f} (z mid-plane, normalised units)")
    fig.tight_layout(); fig.savefig(ES.OUT / "failure_worst.png", dpi=80); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, key in enumerate(("Ma0.7", "Ma2")):
        for c, nm in ((0, "rho"), (1, "Bx"), (4, "vx")):
            r = [q for q in recs if q["Ma"] == (0.7 if key == "Ma0.7" else 2.0)]
            ax[i].plot(range(len(BN)), np.mean([q["Ep_over_Et"][c] for q in r], 0), marker="o", label=f"{nm} pred/true energy")
            ax[i].plot(range(len(BN)), np.mean([q["Ee_over_Et"][c] for q in r], 0), ls="--", marker="x", label=f"{nm} error/true energy")
        ax[i].axhline(1, color="k", lw=0.5); ax[i].set_xticks(range(len(BN))); ax[i].set_xticklabels(BN); ax[i].set_yscale("log")
        ax[i].set_title(f"{key}: spectral energy ratios (mean over windows)"); ax[i].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ES.OUT / "failure_spectra.png", dpi=80); plt.close(fig)
    print("S37J_DONE", flush=True)


if __name__ == "__main__":
    main()
