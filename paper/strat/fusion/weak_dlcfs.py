"""Where does D_LCFS come from? GPU, VALID sample, frozen pipeline psi: location of the Hausdorff maximum (angle about
the true axis, height relative to it), its size by region, and the share of the D_LCFS sum per region.

    .venv/bin/python paper/strat/fusion/weak_dlcfs.py
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


def main() -> None:
    sota, d = fx.data(); shots, grp = ST.sample(); pe, nm = ST.load_entry(); f, comps, mean, H = ST.entry_preds(d, sota)
    ir_, iz_ = H.index("efit_r_axis"), H.index("efit_z_axis")
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); pos = {int(r): i for i, r in enumerate(lab["rows_valid"])}
    Ph = np.mean([np.load(fx.OUT / f"head_boundary_v8_s{s}.npz")["pred_valid"] for s in range(1, 7)], 0).astype(np.float64)
    rows = np.concatenate([fx.valid_rows(d, s) for s in shots]); ph = Ph[[pos[int(r)] for r in rows]]
    C, S = zip(*(f(s) for s in shots)); C = np.concatenate([c.mean(0) for c in C]); S = np.concatenate([x.mean(0) for x in S])
    R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask); pht = torch.tensor(ph, device=gp.dev)
    psi = torch.tensor((C @ comps + mean).reshape(-1, 65, 65), device=gp.dev)
    ax = torch.tensor(S[:, [ir_, iz_]], device=gp.dev, dtype=torch.float64)
    Cref, rgeo, cgt, has = GP.refs(d, shots)
    res = {}
    for name, p in (("entry", psi), ("frozen", GP.post_all(gp, psi, ax, pht[:, 7:], pht[:, 2], pht[:, 6]))):
        dd, at = [], []
        for c in GP.chunks(len(p), 2048):
            o = gp.g.derive(p[c], torch.tensor(Cref[c]), torch.tensor(rgeo[c])); dd.append(o["d"].cpu().numpy()); at.append(o["d_at"].cpu().numpy())
        res[name] = (np.concatenate(dd), np.concatenate(at))
    ra, za = cgt[:, 0], cgt[:, 1]
    for name, (dd, at) in res.items():
        m = has & np.isfinite(dd) & np.isfinite(ra)
        ang = np.degrees(np.arctan2(at[:, 1] - za, at[:, 0] - ra)); dz = at[:, 1] - za
        reg = np.select([ang < -120, ang < -60, ang < 0, ang < 60, ang < 120], ["low-in", "bottom", "low-out", "up-out", "top"], "up-in")
        print(f"[dlcfs] {name}: mean D {dd[m].mean():.5f}")
        for r in ("bottom", "low-out", "low-in", "top", "up-out", "up-in"):
            k = m & (reg == r)
            print(f"[dlcfs]   {r:8s} frames {k.mean():.3f}  mean D there {dd[k].mean() if k.any() else 0:.5f}  share of D sum {dd[k].sum() / dd[m].sum():.3f}", flush=True)
    np.savez(fx.OUT / "weak_dlcfs.npz", d_frozen=res["frozen"][0], at_frozen=res["frozen"][1], d_entry=res["entry"][0])


if __name__ == "__main__":
    main()
