"""Where do the warps fall short of the heads? GPU, 180-shot VALID sample: after each step of the frozen chain, twin R^2
of every consistency scalar against the truth, and the gap to the head (|achieved - head|) quantiles for kappa, tri, li.

    .venv/bin/python paper/strat/fusion/weak_warps.py
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fx  # noqa: E402,I001
import strategies as ST  # noqa: E402
import axis_shift as AX  # noqa: E402
import gpu_pipe as GP  # noqa: E402
from gpu_post import GPUPost  # noqa: E402


def derive_all(gp, psi):
    out = {k: [] for k in GP.NAMES}
    for c in GP.chunks(len(psi), 4096):
        o = gp.g.derive(psi[c])
        for k in GP.NAMES:
            out[k].append(o[k].cpu().numpy())
    return np.stack([np.concatenate(out[k]) for k in GP.NAMES], 1)


def main() -> None:
    sota, d = fx.data(); shots, grp = ST.sample(); pe, nm = ST.load_entry(); f, comps, mean, H = ST.entry_preds(d, sota)
    ir_, iz_ = H.index("efit_r_axis"), H.index("efit_z_axis")
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); pos = {int(r): i for i, r in enumerate(lab["rows_valid"])}
    Ph = np.mean([np.load(fx.OUT / f"head_boundary_v8_s{s}.npz")["pred_valid"] for s in range(1, 7)], 0).astype(np.float64)
    rows = np.concatenate([fx.valid_rows(d, s) for s in shots]); ph = Ph[[pos[int(r)] for r in rows]]
    C, S = zip(*(f(s) for s in shots)); C = np.concatenate([c.mean(0) for c in C]); S = np.concatenate([x.mean(0) for x in S])
    gt = np.concatenate([pe[s]["cgt"] for s in shots]); var = np.asarray(nm["c_var"])
    R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask)
    psi = torch.tensor((C @ comps + mean).reshape(-1, 65, 65), device=gp.dev); pht = torch.tensor(ph, device=gp.dev)
    ax = torch.tensor(S[:, [ir_, iz_]], device=gp.dev, dtype=torch.float64)
    steps = [("entry", ()), ("+shift", ("shift",)), ("+bw", ("bw",)), ("+kw", ("kw",)), ("+li", ("li",))]
    cur = psi
    hd = np.c_[S[:, [ir_, iz_]], ph[:, 2:7]]
    for name, st in steps:
        if st:
            cur = GP.post_all(gp, cur, ax, pht[:, 7:], pht[:, 2], pht[:, 6], steps=st)
        v = derive_all(gp, cur)
        r2 = [1 - np.nanmean((gt[:, k] - v[:, k]) ** 2) / var[k] for k in range(7)]
        gap = np.abs(v - hd)
        print(f"[warps] {name:7s} R2 " + " ".join(f"{n} {x:.4f}" for n, x in zip(GP.NAMES, r2)) +
              " | |ach-head| p50/p90/p99 kappa " + "/".join(f"{np.nanpercentile(gap[:, 2], q):.4f}" for q in (50, 90, 99)) +
              " tri_bot " + "/".join(f"{np.nanpercentile(gap[:, 4], q):.4f}" for q in (50, 90, 99)) +
              " li " + "/".join(f"{np.nanpercentile(gap[:, 6], q):.4f}" for q in (50, 90, 99)), flush=True)
    np.savez(fx.OUT / "weak_warps_final.npz", v=v, head=hd, gt=gt)


if __name__ == "__main__":
    main()
