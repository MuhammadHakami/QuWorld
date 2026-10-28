"""Doc 35 §37j: per-channel x band Wiener-style gain on the final Flowers dfc prediction (VALID only; TEST never loaded).

Motivation (failure.json): at |k| >= 32 Flowers' density error is ~0.9x the true signal energy (little skill) and B is
over-smoothed at |k| >= 16, while large scales are near perfect. Least-squares gains g[c, b] = <P_b, Y_b> / <P_b, P_b>
(mean-free, each window weighted by 1 / var_c as in VRMSE) are fit on the 120 stratified TRAIN windows and applied to the
160 stratified VALID windows; k = 0 keeps the conserved means (dfc). Single pass, no extra inference.

    QW_GPU_GB=6 runs/capped.sh s37j 16 uv run --no-sync python paper/strat/mhd/gain_test.py
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
from mhd_divfree import DivFreeB  # noqa: E402

BANDS = [[0], [1], [2, 3], list(range(4, 8)), list(range(8, 16)), list(range(16, 32)), list(range(32, 56))]


@torch.no_grad()
def collect(F, split_name, rows, proj, band):
    sp = ES.WellND("MHD_64", split_name)
    m = torch.tensor(sp.mean, device=ES.DEV); s = torch.tensor(sp.std, device=ES.DEV)
    out = []
    for r in rows:
        x, y = (v.unsqueeze(0).to(ES.DEV) for v in sp.window(r["traj"], r["t0"]))
        p = F(x)[0]
        p = torch.cat([p[:1], proj.project(p[None, 1:4])[0], p[4:]])
        last = x[0, -ES.C:]
        p[:4] += last[:4].mean((1, 2, 3), keepdim=True) - p[:4].mean((1, 2, 3), keepdim=True)     # dfc
        y0 = y[0]; var = (y0 - y0.mean((1, 2, 3), keepdim=True)).square().mean((1, 2, 3)) + 1e-7
        P = torch.fft.fftn(p, dim=(1, 2, 3)).reshape(ES.C, -1); Y = torch.fft.fftn(y0, dim=(1, 2, 3)).reshape(ES.C, -1)
        nb = len(BANDS)
        num = torch.zeros(ES.C, nb, device=ES.DEV).index_add_(1, band, (P * Y.conj()).real)
        den = torch.zeros(ES.C, nb, device=ES.DEV).index_add_(1, band, P.abs().square())
        out.append({"r": r, "num": (num / var[:, None]).cpu(), "den": (den / var[:, None]).cpu(), "P": P.cpu(), "Y": Y.cpu(),
                    "var": var.cpu()})
    return out


def vrmse(rec, g, band):
    Pg = rec["P"] * g[:, band.cpu()]
    mse = (Pg - rec["Y"]).abs().square().mean(1) / rec["Y"].shape[1]
    return (mse / rec["var"]).sqrt().mean().item()


def main() -> None:
    models, meta = ES.load_models(); F = models["F"]
    proj = DivFreeB(torch.nn.Identity()).to(ES.DEV)
    band = torch.zeros(ES.NSHELL, dtype=torch.long)
    for i, b in enumerate(BANDS):
        band[b] = i
    band = band[ES.shell_index().cpu()].to(ES.DEV)
    smp = json.loads((ES.OUT / "strata.json").read_text())["samples"]
    T = collect(F, "train", smp["train"], proj, band); V = collect(F, "valid", smp["valid"], proj, band)
    num = sum(t["num"] for t in T); den = sum(t["den"] for t in T)
    g = (num / den).clamp(0, 1.5); g[:, 0] = 1.0                                        # k = 0: conserved means kept
    base = np.array([vrmse(v, torch.ones_like(g), band) for v in V]); new = np.array([vrmse(v, g, band) for v in V])
    reg = np.array([0 if v["r"]["Ma"] == 0.7 else 1 for v in V]); rng = np.random.default_rng(0); bs = []
    for _ in range(4000):
        idx = np.concatenate([np.flatnonzero(reg == k)[rng.integers(0, (reg == k).sum(), (reg == k).sum())] for k in (0, 1)])
        bs.append((new[idx] - base[idx]).mean() / base[idx].mean())
    res = {"gains": g.tolist(), "valid_dfc": float(base.mean()), "valid_gain": float(new.mean()),
           "rel_pct": float(100 * (new.mean() - base.mean()) / base.mean()), "ci95_pct": [float(100 * np.percentile(bs, q)) for q in (2.5, 97.5)],
           "by_Ma": {k: [float(base[reg == i].mean()), float(new[reg == i].mean())] for i, k in enumerate(("Ma0.7", "Ma2"))},
           "frac_windows_better": float((new < base).mean())}
    (ES.OUT / "gain_test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "gains"}, indent=1))
    print("gains rho", np.round(g[0].numpy(), 3).tolist(), "Bx", np.round(g[1].numpy(), 3).tolist(), "vx", np.round(g[4].numpy(), 3).tolist())


if __name__ == "__main__":
    main()
