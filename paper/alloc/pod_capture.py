"""ADR-C step 1 (35-analog-digital-router-configuration-losses.md): does routing *compressed analog signals* (per-field
POD modes of the trunk's TRAIN residuals) instead of raw Haar tiles raise what 30 qubits can reach?

    .venv/bin/python paper/alloc/pod_capture.py <row>      # row: pelzer | rayleigh_benard | shear_flow | MHD_64

POD per field by the method of snapshots on N_SNAP training residuals (fp16 on the GPU). Allocation unit = one
(field, mode) coefficient per qubit, metric-weighted exactly as in the study (1 / D_c per window). On TEST windows
(uniform sample, same seeds as the caches) we report the utilization ceiling U@K for
  static  = the K (field, mode) units with the largest mean TRAIN energy     (fixed routing)
  oracle  = per-window top-K units by true energy                            (diagnostic ceiling)
next to the Haar tile ceilings already in runs/alloc/cache/<row>/test.pt. Orthonormal modes => U is an exact ceiling
(gain_le_ceiling); energy outside the POD span counts as unreachable.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "phase_b"))
import gpucap  # noqa: E402

gpucap.cap_from_env()
DEV = torch.device("cuda")
N_SNAP, N_TEST, KS = 1000, 600, (10, 30, 100, 300)


def sampler(row: str):
    """-> (split -> list of (x_t, y, p) GPU batches generator factory), n_fields."""
    if row == "pelzer":
        sys.path.insert(0, str(ROOT / "paper" / "geothermal"))
        import train_pelzer_unet as TP
        x, y, info, tr, va, te = TP.load_1000()
        model = TP.build("vanilla")
        model.load_state_dict(torch.load(ROOT / "runs/pelzer_unet/pretrain_lr0.001_s1337.pt", weights_only=True, map_location="cpu")["model_state_dict"])
        model = model.to(DEV).eval()
        idx = {"train": tr, "test": te}

        def gen(split, n):
            ids = idx[split][:n]
            for i in range(0, len(ids), 50):
                b = ids[i:i + 50]; xb, yb = x[b].to(DEV), y[b].to(DEV)
                yield yb, model(xb)
        return gen, 1
    import cache as CA
    if row == "MHD_64":
        from cnext_hybrid import build_hybrid
        model = build_hybrid("vanilla", model_dir=str(ROOT / "data/cnext_mhd_ckpt")).to(DEV).eval()
        sp = {s: CA.WellND("MHD_64", s) for s in ("train", "test")}
    else:
        from well2d_backbone import build
        import well2d
        model = build("vanilla", row).to(DEV).eval()
        sp = {s: well2d.Split(row, s) for s in ("train", "test")}
    C = sp["test"].C

    def gen(split, n):
        s = sp[split]; rng = np.random.default_rng({"train": 0, "test": 3}[split])
        allp = [(k, t) for k in range(s.n) for t in range(3, s.T - 1)]
        pick = np.sort(rng.choice(len(allp), min(n, len(allp)), replace=False))
        bs = 2 if row == "MHD_64" else 8
        for i in range(0, len(pick), bs):
            xs, ys = zip(*(s.window(*allp[j]) for j in pick[i:i + bs]))
            xb, yb = torch.stack(xs).to(DEV), torch.stack(ys).to(DEV)
            yield yb, model(xb)
    return gen, C


@torch.no_grad()
def main(row: str):
    gen, C = sampler(row)
    snaps = [[] for _ in range(C)]; wsum = []
    for y, p in gen("train", N_SNAP):
        r = (y - p).flatten(2)                                        # (B, C, D)
        var = y.flatten(2).var(-1, unbiased=False) + 1e-7             # VRMSE weights 1 / (npix var)
        for c in range(C):
            snaps[c].append((r[:, c] / (var[:, c, None] * r.shape[-1]).sqrt()).half())
    modes, train_energy = [], []
    for c in range(C):
        S = torch.cat(snaps[c]).float()                               # (N, D), metric-weighted residuals
        G = S @ S.T
        ev, U = torch.linalg.eigh(G)
        keep = ev > ev.max() * 1e-8
        ev, U = ev[keep].flip(0), U[:, keep].flip(1)
        Phi = (S.T @ U) / ev.sqrt()                                   # (D, m) orthonormal POD modes
        modes.append(Phi.half()); train_energy.append(ev / len(S))    # mean TRAIN energy per mode
        snaps[c] = None; del S, G, U
    unit_field = torch.cat([torch.full((len(e),), c) for c, e in enumerate(train_energy)])
    static_order = torch.cat(train_energy).argsort(descending=True).cpu()
    capO = {k: [] for k in KS}; capS = {k: [] for k in KS}; spanned = []
    for y, p in gen("test", N_TEST):
        r = (y - p).flatten(2); var = y.flatten(2).var(-1, unbiased=False) + 1e-7
        rw = r / (var * r.shape[-1]).sqrt()[..., None]                # metric-weighted residual
        tot = rw.square().sum((1, 2))
        coef = torch.cat([(rw[:, c].half() @ modes[c]).float() for c in range(C)], 1).square()   # (B, units)
        spanned.append((coef.sum(1) / tot).cpu())
        srt = coef.sort(1, descending=True).values
        for k in KS:
            capO[k].append((srt[:, :k].sum(1) / tot).cpu())
            capS[k].append((coef[:, static_order[:k].to(DEV)].sum(1) / tot).cpu())
    haar = torch.load(ROOT / "runs/alloc/cache" / row / "test.pt", weights_only=False)["ceil"]
    out = {"row": row, "n_snap": N_SNAP, "n_test": N_TEST, "pod_span_share": float(torch.cat(spanned).mean()),
           "pod_static": {k: float(torch.cat(v).mean()) for k, v in capS.items()},
           "pod_oracle": {k: float(torch.cat(v).mean()) for k, v in capO.items()},
           "haar_static": {k: float(haar[f"static@{k}"].mean()) for k in KS},
           "haar_oracle": {k: float(haar[f"oracle@{k}"].mean()) for k in KS},
           "haar_dc": float(haar["dc"].mean()),
           "static30_fields": np.bincount(unit_field[static_order[:30]].numpy(), minlength=C).tolist()}
    dst = ROOT / "runs/alloc/pod"; dst.mkdir(parents=True, exist_ok=True)
    (dst / f"{row}.json").write_text(json.dumps(out, indent=1))
    print(f"[pod] {row}: U@30 Haar static {out['haar_static'][30]:.3f} oracle {out['haar_oracle'][30]:.3f} | "
          f"POD static {out['pod_static'][30]:.3f} oracle {out['pod_oracle'][30]:.3f} | POD span {out['pod_span_share']:.3f} | "
          f"static-30 qubits per field {out['static30_fields']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
