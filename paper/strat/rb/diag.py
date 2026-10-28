"""§37 rayleigh_benard, step 1+3a: per-window diagnostics on the fixed stratified VALID sample, and x-symmetry TTA.

For every sampled window (runs/strat/rb/strata.json) it records, per field (buoyancy, pressure, u_x, u_y):
  - VRMSE of: each gated fine-tuned member (mr30ryd = best single, vanilla, mlp30), the E-ft ensemble (equal-weight
    mean of the three), persistence (x_t), the raw trunk of the best single (backbone alone), and x-symmetry TTA
    variants of each member: flip2 (id, reflection), shift4 (periodic x-shifts 0/128/256/384), tta8 (both);
  - Krogh-Vedelsby split of the members' mean normalized MSE into ensemble MSE (shared error, "bias") and ambiguity
    (member spread, "variance"): mean_i mse_i = mse_ens + mean_i |p_i - p_bar|^2, all / var(y);
  - normalized error energy per gate band (5 octaves, the doc 37 masks; Parseval so bands sum to mse/var) for
    single, single-tta8, ensemble, and the target increment (= persistence error);
  - error profiles along y (16 bins of 8 rows: walls at both ends) and along x (16 bins of 32 columns);
  - error share inside plumes (|b - <b>_x| > 1 std of the target buoyancy anomaly).
Everything is written to runs/strat/rb/diag_windows.npz; summaries are made by analyze.py (CPU).

    flock runs/strat/gpu.lock timeout 1800 env QW_GPU_GB=12 runs/capped.sh strat_rb 16 uv run --no-sync python paper/strat/rb/diag.py [--limit N]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import common as CM

TRANSFORMS = [(f, s) for f in (False, True) for s in (0, 128, 256, 384)]      # tta8; index 0 = identity
SUBSETS = {"flip2": [(False, 0), (True, 0)], "shift4": [(False, s) for s in (0, 128, 256, 384)], "tta8": TRANSFORMS}
NYB = NXB = 16


def band_err(e, masks, var):
    """(B, C, H, W) error -> (B, C, NB) band energies / (N^2 var): sums over bands to mse/var."""
    E = torch.fft.fftn(e.float(), dim=(2, 3)).abs().square()
    N = e.shape[2] * e.shape[3]
    return FGbandsum(E, masks) / (N * N) / (var[..., None] + 1e-7)


def FGbandsum(K, masks):
    return CM.FG.bandsum(K, masks)


def profiles(e, var):
    """Squared error / var averaged within y bins (dim 3) and x bins (dim 2): (B, C, 16) each."""
    se = e.square() / (var[..., None, None] + 1e-7)
    B, Cc, H, W = se.shape
    yp = se.mean(2).reshape(B, Cc, NYB, W // NYB).mean(-1)
    xp = se.mean(3).reshape(B, Cc, NXB, H // NXB).mean(-1)
    return yp, xp


def plume_share(e, y):
    """Fraction of each field's squared error inside the plume mask of the target buoyancy anomaly."""
    b = y[:, 0]
    th = b - b.mean(1, keepdim=True)                        # anomaly w.r.t. the horizontal (x) mean at each height
    m = (th.abs() > th.flatten(1).std(1)[:, None, None]).float()[:, None]
    se = e.square()
    return (se * m).flatten(2).sum(-1) / se.flatten(2).sum(-1).clamp_min(1e-30), m.flatten(1).mean(1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bsz", type=int, default=4)
    a = ap.parse_args()
    doc, pairs, sid = CM.load_strata()
    if a.limit:
        pairs, sid = pairs[:a.limit], sid[:a.limit]
    valid = CM.valid_split()
    sym = CM.XSym(valid.mean, valid.std)
    models = {m: CM.load_gated(m, valid) for m in CM.MEMBERS}
    masks = next(iter(models.values())).masks
    rec = {k: [] for k in ("vr", "kv_mbar", "kv_ens", "kv_amb", "band", "yprof", "xprof", "plume", "plume_frac")}
    names = None
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for bi, (x, y) in enumerate(well2d_batched(valid, pairs, a.bsz)):
        x, y = x.to(CM.DEV), y.to(CM.DEV); xt = x[:, 3 * CM.C:]
        var = (y - y.mean((2, 3), keepdim=True)).square().mean((2, 3))
        preds = {}
        for m, model in models.items():
            outs = [sym.invert(model(sym.apply(x, f, s)), f, s) for f, s in TRANSFORMS]
            preds[f"single_{m}"] = outs[0]
            for k, tr in SUBSETS.items():
                preds[f"{k}_{m}"] = torch.stack([outs[TRANSFORMS.index(t)] for t in tr]).mean(0)
        preds["ens"] = torch.stack([preds[f"single_{m}"] for m in models]).mean(0)
        preds["ens_tta8"] = torch.stack([preds[f"tta8_{m}"] for m in models]).mean(0)
        preds["persistence"] = xt
        preds["trunk_raw_mr30ryd"] = models[CM.BEST_SINGLE].trunk(x)
        if names is None:
            names = list(preds)
        rec["vr"].append(torch.stack([CM.vrmse_fields(preds[n], y) for n in names], 1).cpu())    # (B, P, C)
        mem = [preds[f"single_{m}"] for m in models]
        vv = var + 1e-7
        mbar = torch.stack([(p - y).square().mean((2, 3)) for p in mem]).mean(0) / vv
        amb = torch.stack([(p - preds["ens"]).square().mean((2, 3)) for p in mem]).mean(0) / vv
        rec["kv_mbar"].append(mbar.cpu()); rec["kv_amb"].append(amb.cpu())
        rec["kv_ens"].append(((preds["ens"] - y).square().mean((2, 3)) / vv).cpu())
        diag = {"single": preds[f"single_{CM.BEST_SINGLE}"], "tta8": preds[f"tta8_{CM.BEST_SINGLE}"],
                "ens": preds["ens"], "persistence": xt}
        rec["band"].append(torch.stack([band_err(p - y, masks, var) for p in diag.values()], 1).cpu())  # (B, 4, C, NB)
        yp, xp = zip(*(profiles(p - y, var) for p in diag.values()))
        rec["yprof"].append(torch.stack(yp, 1).cpu()); rec["xprof"].append(torch.stack(xp, 1).cpu())
        ps, pf = zip(*(plume_share(p - y, y) for p in diag.values()))
        rec["plume"].append(torch.stack(ps, 1).cpu()); rec["plume_frac"].append(pf[0].cpu())
        if bi == 0 or (bi + 1) % 50 == 0:
            done = (bi + 1) * a.bsz
            print(f"[diag] {done}/{len(pairs)} windows {time.time() - t0:.0f}s peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    out = {k: torch.cat(v).numpy() for k, v in rec.items()}
    out.update(pred_names=np.array(names), diag_names=np.array(["single", "tta8", "ens", "persistence"]),
               pairs=np.array(pairs), sid=sid, fields=np.array(CM.FIELDS))
    fn = CM.OUT / ("diag_windows.npz" if not a.limit else f"diag_smoke_{a.limit}.npz")
    np.savez_compressed(fn, **out)
    vr = out["vr"].mean(2).mean(0)
    print(f"[diag] wrote {fn} ({time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB); "
          "unweighted means: " + ", ".join(f"{n} {v:.5f}" for n, v in zip(names, vr)), flush=True)


def well2d_batched(split, pairs, bsz):
    return CM.well2d.batched(split, pairs, bsz)


if __name__ == "__main__":
    main()
