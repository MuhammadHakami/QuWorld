"""Can a router find the hard tiles before seeing the error? (TRL2D test, GPU)

    .venv/bin/python paper/spectral/router_capture.py   ->  runs/spectral/trl2d_router_capture.json

For an atom budget K (one atom per wavelet tile), the ORACLE puts the K atoms on the K tiles with the largest
error energy; by `residual_after_band_removal` (formal/QuotaScaling/SpectralRouting.lean) the captured energy is
the ceiling for any corrector restricted to K tiles. Non-parametric routers must choose tiles from quantities
available at inference time:
  * prev_change: wavelet energy of x_t - x_{t-1} (recent dynamics, from the input history);
  * trunk_change: wavelet energy of p - x_t (the trunk's own predicted change; available after the trunk runs);
  * input_detail: wavelet energy of x_t itself (edges / fronts);
  * random: K uniformly random tiles (the floor).
Reported: share of detail error energy captured, and the router/oracle ratio, per K.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "phase_b")); sys.path.insert(0, str(ROOT / "paper" / "spectral"))
import cnext_hybrid  # noqa: E402
import well2d  # noqa: E402
from band_analysis import DEV, J, haar2d  # noqa: E402

KS = [1, 3, 10, 30, 100, 300, 1000, 3000]


def tiles(f: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> (B, n_tiles) squared detail coefficients, fixed order (level, orientation, channel, y, x)."""
    _, det = haar2d(f, J)
    B = f.shape[0]
    return torch.cat([c.reshape(B, -1) for lvl in det for c in lvl], dim=1).double().square()


def main() -> None:
    data = well2d.Split("turbulent_radiative_layer_2D", "test")
    pairs = well2d.fixed_windows(data, 20)
    C = data.C
    model = cnext_hybrid.build_hybrid("vanilla").to(DEV).eval()
    gen = torch.Generator(device=DEV).manual_seed(0)
    routers = ("oracle", "trunk_change", "prev_change", "input_detail", "random")
    cap = {r: torch.zeros(len(KS), device=DEV, dtype=torch.float64) for r in routers}
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in well2d.batched(data, pairs, 16):
            x, y = x.to(DEV), y.to(DEV)
            p = model(x)
            xt, xp = x[:, (well2d.N_HIST - 1) * C:], x[:, (well2d.N_HIST - 2) * C:(well2d.N_HIST - 1) * C]
            err = tiles(p - y)
            score = {"oracle": err, "trunk_change": tiles(p - xt), "prev_change": tiles(xt - xp),
                     "input_detail": tiles(xt), "random": torch.rand(err.shape, generator=gen, device=DEV, dtype=torch.float64)}
            tot += err.sum().item(); n += x.shape[0]
            for r in routers:
                idx = score[r].topk(max(KS), dim=1).indices
                picked = err.gather(1, idx)                              # error energy on the router's tiles
                csum = picked.cumsum(1)
                cap[r] += torch.stack([csum[:, k - 1].sum() for k in KS])
    n_tiles = err.shape[1]
    out = dict(dataset="turbulent_radiative_layer_2D", windows=n, tiles_per_sample=n_tiles, K=KS,
               capture={r: (cap[r] / tot).tolist() for r in routers})
    out["ratio_to_oracle"] = {r: (cap[r] / cap["oracle"]).tolist() for r in routers}
    dst = ROOT / "runs" / "spectral"; (dst / "trl2d_router_capture.json").write_text(json.dumps(out, indent=1))
    print(f"tiles per sample {n_tiles}; share of detail error energy captured by K tiles:")
    print("K          " + " ".join(f"{k:>7d}" for k in KS))
    for r in routers:
        print(f"{r:12s} " + " ".join(f"{v:7.3f}" for v in out["capture"][r]))
    print("router / oracle")
    for r in routers[1:]:
        print(f"{r:12s} " + " ".join(f"{v:7.3f}" for v in out["ratio_to_oracle"][r]))
    data.close()


if __name__ == "__main__":
    main()
