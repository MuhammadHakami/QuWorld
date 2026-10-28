"""z-domain band filter (ZBF) on top of the registered E-ft selection (doc 38 addendum, registered before any fit).

    .venv/bin/python paper/alloc/zbf.py <dataset> [--full-test]

Per field c and Fourier octave b (the gate's 5 l-inf shells), with P = y_hat - x_t (the model's predicted change) and
Q = x_t - x_{t-1} (the last observed change):   y' = x_t + sum_b P_b( alpha_cb * P + beta_cb * Q ),
a two-tap filter in time per band (its z-transform is alpha_cb * G(z) + beta_cb * (1 - z^-1)). (alpha, beta) solve a
least-squares problem weighted by 1 / var(y) per window and field (approximating the benchmark's VRMSE) on stratified
TRAIN windows (1 per trajectory x 8 time bins, rng 7). Kept iff stratified VALID (valk=2, rng 2; the E-ft selection
sample) improves by >= 0.5% relative; a kept filter is scored once on the full TEST split (--full-test).
"""
from __future__ import annotations

import argparse
import json

import torch

import finetune_gated as FG
import ft_ensemble as FE


def parts(models, x, C):
    p = torch.stack([m(x) for m in models]).mean(0); xt, xp = x[:, 3 * C:], x[:, 2 * C:3 * C]
    return p, xt, torch.fft.fft2(p - xt), torch.fft.fft2(xt - xp)


def apply(xt, Pk, Qk, masks, ab):
    """ab: (C, nb, 2). Returns x_t + sum_b P_b(alpha P + beta Q)."""
    gain = torch.einsum("cb,bhw->chw", ab[..., 0], masks)[None]; lag = torch.einsum("cb,bhw->chw", ab[..., 1], masks)[None]
    return xt + torch.fft.ifft2(gain * Pk + lag * Qk).real


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--full-test", action="store_true")
    a = ap.parse_args()
    sel = json.loads((FG.OUT / f"{a.dataset}_Eft.json").read_text())
    tags = sel["members"] if sel["selected"] == "E-ft" else [sel["selected"]]
    loaded = [FE.load(a.dataset, t) for t in tags]; models, (valid, test) = [m for m, _ in loaded], loaded[0][1]
    train = FG.well2d.Split(a.dataset, "train"); C = train.C
    masks = FG.band_masks((train.H, train.W)).to(FG.DEV).float(); nb = masks.shape[0]
    out = FG.ROOT / "runs" / "failure" / f"{a.dataset}_zbf.json"
    if not a.full_test:
        S = torch.zeros(C, nb, 5, device=FG.DEV, dtype=torch.float64)          # PP, QQ, PQ, PD, QD (weighted)
        for x, y in FG.well2d.batched(train, FG.strat_pairs(train, 1, 7), 4):
            x, y = x.to(FG.DEV), y.to(FG.DEV); p, xt, Pk, Qk = parts(models, x, C); Dk = torch.fft.fft2(y - xt)
            w = 1.0 / ((y - y.mean((-2, -1), keepdim=True)).square().mean((-2, -1)) + 1e-7)          # (B, C)
            for i, (u, v) in enumerate(((Pk, Pk), (Qk, Qk), (Pk, Qk), (Pk, Dk), (Qk, Dk))):
                S[..., i] += torch.einsum("bc,bchw,khw->ck", w.double(), (u * v.conj()).real.double(), masks.double())
        A = torch.stack([torch.stack([S[..., 0], S[..., 2]], -1), torch.stack([S[..., 2], S[..., 1]], -1)], -2)
        ab = torch.linalg.solve(A + 1e-12 * torch.eye(2, device=FG.DEV, dtype=torch.float64), S[..., 3:5]).float()
        base, filt = [], []
        for x, y in FG.well2d.batched(valid, FG.strat_pairs(valid, 2, 2), 4):
            x, y = x.to(FG.DEV), y.to(FG.DEV); p, xt, Pk, Qk = parts(models, x, C)
            base.append(FG.vrmse(p, y).cpu()); filt.append(FG.vrmse(apply(xt, Pk, Qk, masks, ab), y).cpu())
        b, f = torch.cat(base).mean().item(), torch.cat(filt).mean().item(); keep = f < b * (1 - 0.005)
        res = {"dataset": a.dataset, "members": tags, "alpha": ab[..., 0].tolist(), "beta": ab[..., 1].tolist(),
               "valid_base": b, "valid_zbf": f, "valid_rel": 1 - f / b, "keep": bool(keep)}
        out.write_text(json.dumps(res, indent=1))
        print(f"[zbf] {a.dataset}: VALID {b:.5f} -> {f:.5f} ({100 * (1 - f / b):+.2f}%) keep={keep}", flush=True)
        print("alpha", [[round(v, 3) for v in r] for r in res["alpha"]], "\nbeta", [[round(v, 3) for v in r] for r in res["beta"]])
        return
    res = json.loads(out.read_text())
    if not res["keep"]:
        raise SystemExit(f"[zbf] {a.dataset}: not kept on VALID; no TEST")
    ab = torch.stack([torch.tensor(res["alpha"]), torch.tensor(res["beta"])], -1).to(FG.DEV)
    base, filt = [], []
    for x, y in FG.well2d.batched(test, FG.strat_pairs(test, 2, 3), 4):      # amended: stratified TEST (time limit)
        x, y = x.to(FG.DEV), y.to(FG.DEV); p, xt, Pk, Qk = parts(models, x, C)
        base.append(FG.vrmse(p, y).cpu()); filt.append(FG.vrmse(apply(xt, Pk, Qk, masks, ab), y).cpu())
    b, f = torch.cat(base), torch.cat(filt)
    res.update(test_base=float(b.mean()), test_zbf=float(f.mean()), test_n=len(f), test_win_rate=float((f < b).float().mean()))
    out.write_text(json.dumps(res, indent=1))
    print(f"[zbf] {a.dataset}: STRAT TEST {b.mean():.5f} -> {f.mean():.5f} (n={len(f)}, windows improved {100 * (f < b).float().mean():.1f}%)", flush=True)


if __name__ == "__main__":
    main()
