"""§37 rayleigh_benard: score saved single-model candidates on the fixed stratified VALID sample, plain and with
x-symmetry TTA (tta8 = {identity, reflection} x periodic x-shifts {0, 128, 256, 384}; flip2; shift4).

A candidate is `name=kind:path`, kind in
  gated : a GatedHead state dict (e.g. runs/strat/rb/ft_<tag>_ema.pt or a soup head) on the best single's trunk type;
  beta  : a head_ext.BetaHead state dict;
  soupgated:<variant> : GatedHead state for a soup trunk of <variant>.
Per-window per-field VRMSE -> runs/strat/rb/tta_<name>.npz.

    flock runs/strat/gpu.lock timeout 1800 env QW_GPU_GB=12 runs/capped.sh strat_rb 16 uv run --no-sync python paper/strat/rb/eval_tta.py ema=gated:runs/strat/rb/ft_x_ema.pt
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

import common as CM
import diag as DG
import head_ext as HE

SUBSETS = dict(DG.SUBSETS, shift2=[(False, 0), (False, 256)], fs2=[(False, 0), (True, 256)],
               fs4=[(False, 0), (True, 128), (False, 256), (True, 384)])    # cheaper 2- and 4-pass rules


def build(kind, path, valid):
    base = CM.load_gated(CM.BEST_SINGLE, valid)
    if kind == "gated":
        model = base
    elif kind == "beta":
        model = HE.BetaHead(base)
    elif kind.startswith("soupgated"):
        variant = kind.split(":")[1]
        model = CM.FG.GatedHead(HE.soup_trunk(variant), CM.C, (valid.H, valid.W), torch.ones(CM.C, CM.FG.NB), True).to(CM.DEV)
    else:
        raise SystemExit(f"unknown kind {kind}")
    model.load_state_dict(torch.load(path, weights_only=True, map_location=CM.DEV)["model"])
    return model.eval()


@torch.no_grad()
def main():
    doc, pairs, sid = CM.load_strata()
    valid = CM.valid_split(); sym = CM.XSym(valid.mean, valid.std)
    w = CM.weights_for(doc, sid)
    for spec in sys.argv[1:]:
        name, rest = spec.split("=", 1); kind, path = rest.rsplit(":", 1) if rest.count(":") > 1 else rest.split(":", 1)
        model = build(kind, path, valid); t0 = time.time(); rec = {k: [] for k in ("single", *SUBSETS)}
        for x, y in CM.well2d.batched(valid, pairs, 8):
            x, y = x.to(CM.DEV), y.to(CM.DEV)
            outs = [sym.invert(model(sym.apply(x, f, s)), f, s) for f, s in DG.TRANSFORMS]
            rec["single"].append(CM.vrmse_fields(outs[0], y).cpu())
            for k, tr in SUBSETS.items():
                rec[k].append(CM.vrmse_fields(torch.stack([outs[DG.TRANSFORMS.index(t)] for t in tr]).mean(0), y).cpu())
        res = {k: torch.cat(v).numpy() for k, v in rec.items()}
        np.savez_compressed(CM.OUT / f"tta_{name}.npz", sid=sid, **res)
        pw = {k: float((w / w.sum() * v.mean(1)).sum()) for k, v in res.items()}
        print(f"[tta] {name} ({kind}:{path}) pop-weighted {pw} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
