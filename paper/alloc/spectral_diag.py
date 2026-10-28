"""Signal-processing diagnosis of the final Well models (registered E-ft selections): Fourier, z-domain, wavelet.

    .venv/bin/python paper/alloc/spectral_diag.py <dataset> [--testk 1]

Per field c and dyadic Fourier octave b (the gate's 5 l-inf shells), with D = y - x_t (true change),
P = y_hat - x_t (predicted change), Q = x_t - x_{t-1} (last observed change), all in the Fourier domain:
  share_b    = error energy of band b / total error energy
  coh_b      = |<P,D>|^2 / (|P|^2 |D|^2)                  magnitude-squared coherence (energy-weighted over windows)
  wiener_b   = 1 - coh_b                                   floor of E_err/E_D for ANY per-band scalar gain on P
  err_b      = E_err / E_D                                 what the model reaches (>= wiener_b; gap = gain headroom)
  amp_b      = |P| / |D|                                   per-band amplitude ratio (< 1: damped)
  pole_true  = <D,Q>/<Q,Q>, pole_model = <P,Q>/<Q,Q>      teacher-forced lag-1 regression slope of the next change on the last observed change (truth vs model; not a model pole)
  ar2_b      = floor of E_err/E_D when Q is added as a second regressor (in-sample on TEST: optimistic, diagnostic)
Haar wavelets (orthonormal, 4 levels) of the error field: energy share per level (1 = finest) and top5 = share of
error energy in the largest 5% of wavelet coefficients (localization). Same for the true change, for reference.
Diagnostic only; any adaptation it suggests is fitted on TRAIN, selected on VALID and scored once on TEST elsewhere.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import finetune_gated as FG
import ft_ensemble as FE


def haar_levels(e, L=4):
    """Orthonormal 2-D Haar: returns [detail energy per level (finest first)] + [approx energy], and all coeffs."""
    out, coeffs, a = [], [], e
    for _ in range(L):
        H, W = a.shape[-2] // 2 * 2, a.shape[-1] // 2 * 2
        a = a[..., :H, :W]
        p, q, r, s = a[..., 0::2, 0::2], a[..., 1::2, 0::2], a[..., 0::2, 1::2], a[..., 1::2, 1::2]
        d = torch.stack([(p - q + r - s), (p + q - r - s), (p - q - r + s)], 0) / 2
        a = (p + q + r + s) / 2
        out.append(d.square().sum((0, -2, -1))); coeffs.append(d.permute(1, 2, 0, 3, 4).flatten(-3))
    out.append(a.square().sum((-2, -1))); coeffs.append(a.flatten(-2))
    return torch.stack(out, -1), torch.cat(coeffs, -1)                     # (B, C, L+1), (B, C, N)


def top_share(c, frac=0.05):
    e = c.square(); k = max(1, int(frac * e.shape[-1]))
    return e.topk(k, -1).values.sum(-1) / (e.sum(-1) + 1e-12)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--testk", type=int, default=1)
    a = ap.parse_args()
    sel = json.loads((FG.OUT / f"{a.dataset}_Eft.json").read_text())
    tags = sel["members"] if sel["selected"] == "E-ft" else [sel["selected"]]
    loaded = [FE.load(a.dataset, t) for t in tags]; models, test = [m for m, _ in loaded], loaded[0][1][1]
    pairs = FG.well2d.all_test_windows(test) if a.dataset == "turbulent_radiative_layer_2D" else FG.strat_pairs(test, a.testk, 3)
    C = test.C; masks = FG.band_masks((test.H, test.W)).to(FG.DEV); nb = masks.shape[0]
    acc = {k: torch.zeros(C, nb, device=FG.DEV, dtype=torch.float64) for k in ("DD", "PP", "PD", "QQ", "DQ", "PQ", "EE")}
    wl = {k: torch.zeros(C, 5, device=FG.DEV, dtype=torch.float64) for k in ("err", "inc")}
    top = {"err": [], "inc": []}
    for x, y in FG.well2d.batched(test, pairs, 4):
        x, y = x.to(FG.DEV), y.to(FG.DEV); p = torch.stack([m(x) for m in models]).mean(0)
        xt, xp = x[:, 3 * C:], x[:, 2 * C:3 * C]
        D, P, Q, E = (torch.fft.fft2(t) for t in (y - xt, p - xt, xt - xp, p - y))
        band = lambda u, v: torch.einsum("bchw,khw->ck", (u * v.conj()).real.double(), masks.double())
        for k, (u, v) in {"DD": (D, D), "PP": (P, P), "PD": (P, D), "QQ": (Q, Q), "DQ": (D, Q), "PQ": (P, Q), "EE": (E, E)}.items():
            acc[k] += band(u, v)
        for key, f in (("err", p - y), ("inc", y - xt)):
            lv, co = haar_levels(f); wl[key] += lv.double().sum(0); top[key].append(top_share(co).cpu())
    A = {k: v.cpu().numpy() for k, v in acc.items()}
    coh = A["PD"] ** 2 / (A["PP"] * A["DD"] + 1e-30)
    # two-regressor (P, Q) least squares per band: residual floor / E_D
    det = A["PP"] * A["QQ"] - A["PQ"] ** 2
    ar2_expl = (A["QQ"] * A["PD"] ** 2 - 2 * A["PQ"] * A["PD"] * A["DQ"] + A["PP"] * A["DQ"] ** 2) / (det + 1e-30)
    r = lambda z: np.round(z, 4).tolist()
    res = {"dataset": a.dataset, "selection": sel["selected"], "n": len(pairs),
           "share": r(A["EE"] / A["EE"].sum(1, keepdims=True)), "err_over_D": r(A["EE"] / A["DD"]),
           "wiener_floor": r(1 - coh), "ar2_floor": r(1 - ar2_expl / A["DD"]), "amp": r(np.sqrt(A["PP"] / A["DD"])),
           "pole_true": r(A["DQ"] / A["QQ"]), "pole_model": r(A["PQ"] / A["QQ"]),
           "wavelet_err_share": r((wl["err"] / wl["err"].sum(1, keepdim=True)).cpu().numpy()),
           "wavelet_inc_share": r((wl["inc"] / wl["inc"].sum(1, keepdim=True)).cpu().numpy()),
           "top5_err": r(torch.cat(top["err"]).mean(0).numpy()), "top5_inc": r(torch.cat(top["inc"]).mean(0).numpy())}
    out = FG.ROOT / "runs" / "failure" / f"{a.dataset}_spectral.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
