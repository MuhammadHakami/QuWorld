"""Where is the RealPDE trunks' error, and is it learnable? (VALID only — design decisions never look at TEST.)

    .venv/bin/python paper/realpde/error_bands.py [--models unet fno cno]  ->  runs/realpde/error_bands.json

Per radial wavenumber shell of the (t, h, w) FFT (eval_metrics' convention, |k| = floor(sqrt(i^2 + j^2 + k^2))):
  * share of each trunk's error energy and of the target energy;
  * cross-model error coherence Re<E_a, E_b> / (|E_a| |E_b|): independent trunks that share an error at a shell point
    to structure none of them captured (learnable in principle, or measurement noise common to both — the PIV noise
    floor is estimated separately from the target's own high-k energy);
  * the equal-weight ensemble's error energy per shell (how much averaging removes).
"""
from __future__ import annotations

import argparse
import json

import torch

import common as C


def shells(t: int, h: int, w: int):
    ft = torch.fft.fftfreq(t) * t; fh = torch.fft.fftfreq(h) * h; fw = torch.fft.fftfreq(w) * w
    k = torch.sqrt(ft[:, None, None] ** 2 + fh[None, :, None] ** 2 + fw[None, None, :] ** 2).floor().long()
    return k, int(k.max()) + 1


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["unet", "fno", "cno"])
    ap.add_argument("--stage", default="finetune"); ap.add_argument("--chunk", type=int, default=128)
    a = ap.parse_args()
    _, Yv, _ = C.load_windows("val")
    P = {m: C.open_pred(C.CACHE / f"pred_{m}_{a.stage}_val.npy", Yv.shape) for m in a.models}
    N, t, h, w, c = Yv.shape
    k, nk = shells(t, h, w); k = k.to(C.DEV).flatten()
    acc = lambda: torch.zeros(nk, device=C.DEV, dtype=torch.float64)
    err = {m: acc() for m in a.models}; tgt = acc(); ens = acc()
    cross = {(x, y): acc() for i, x in enumerate(a.models) for y in a.models[i + 1:]}
    for i in range(0, N, a.chunk):
        y = Yv[i:i + a.chunk].clone().to(C.DEV).double()
        F = lambda z: torch.fft.fftn(z, dim=[1, 2, 3])                          # (b, t, h, w, c)
        Fy = F(y)
        E = {m: F(P[m][i:i + a.chunk].clone().to(C.DEV).double()) - Fy for m in a.models}
        shell = lambda v: torch.zeros(nk, device=C.DEV, dtype=torch.float64).index_add_(0, k, v.sum(dim=(0, 4)).flatten())
        tgt += shell(Fy.abs().square())
        for m in a.models:
            err[m] += shell(E[m].abs().square())
        for (x, z) in cross:
            cross[(x, z)] += shell((E[x] * E[z].conj()).real)
        ens += shell((sum(E.values()) / len(E)).abs().square())
    res = {"windows": N, "shells": nk,
           "target_share": (tgt / tgt.sum()).tolist(),
           "error_share": {m: (err[m] / err[m].sum()).tolist() for m in a.models},
           "error_total": {m: err[m].sum().item() for m in a.models} | {"ensemble_equal": ens.sum().item()},
           "ensemble_error_share_of_best": (ens / min(err.values(), key=lambda e: e.sum().item())).tolist(),
           "coherence": {f"{x}~{z}": (cross[(x, z)] / (err[x] * err[z]).sqrt()).tolist() for (x, z) in cross}}
    (C.ROOT / "runs" / "realpde" / "error_bands.json").write_text(json.dumps(res, indent=1))
    third = lambda v: [sum(v[:nk // 3]), sum(v[nk // 3: 2 * nk // 3]), sum(v[2 * nk // 3:])]
    print(f"[bands] {N} VALID windows, {nk} shells; energy share low / mid / high thirds of |k|:")
    print(f"  target   " + " ".join(f"{x:.4f}" for x in third(res["target_share"])))
    for m in a.models:
        print(f"  err {m:5s}" + " ".join(f"{x:.4f}" for x in third(res["error_share"][m])) + f"   total {res['error_total'][m]:.3e}")
    print(f"  ensemble (equal) total {res['error_total']['ensemble_equal']:.3e}")
    for key, v in res["coherence"].items():
        lo, mid, hi = (sum(v[s]) / len(v[s]) for s in (slice(0, nk // 3), slice(nk // 3, 2 * nk // 3), slice(2 * nk // 3, None)))
        print(f"  coherence {key:10s} low {lo:.3f} mid {mid:.3f} high {hi:.3f}")


if __name__ == "__main__":
    main()
