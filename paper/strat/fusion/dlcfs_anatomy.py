"""Doc 35 §38aa-diag: where does our LCFS Hausdorff distance (D_LCFS) come from? (GPU, inference only)

On the 180-shot VALID sample, X8axH's final psi (gpu_build.predict, axis aimed at the J1 heads): per frame the GPU twin's
D_LCFS against the reference contour and the location of the Hausdorff maximum, classed by poloidal angle about the axis
(top / bottom tip within 30 deg of vertical, inboard, outboard) and by whether it is the reference's or our contour that is
outside; plus the share of D_LCFS sum carried by the worst 5% of frames and by the last 10% of each shot.

    .venv/bin/python paper/strat/fusion/dlcfs_anatomy.py   -> runs/strat/fusion/dlcfs_anatomy.json
"""
from __future__ import annotations

import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fx  # noqa: E402,I001
import gpu_build as GB  # noqa: E402
import gpu_pipe as GP  # noqa: E402
import strategies as ST  # noqa: E402
import axis_shift as AX  # noqa: E402
from gpu_post import GPUPost  # noqa: E402


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import ens_select as ES
    d = sota.Data(); shots, _ = ST.sample()
    class A: pass
    a = A(); a.psi, a.qb, a.axis = "psiF", "qh50", "head"; a.heads = "+".join(f"head_boundary_v8_J1_s{s}" for s in range(1, 7))
    a.qbheads = "+".join(f"head_qb_QH_s{s}" for s in range(1, 7)); a.kw = '{"_steps":["shift","bw","kw","xp","li"],"xp":{"sigma":0.15,"cap":0.10}}'
    P = GB.predict(d, shots, a, sota, ES); R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask)
    psi = torch.tensor(np.concatenate([P[s][0] for s in shots]), device="cuda"); C, rg, cg, has = GP.refs(d, shots)
    dd, at, ax = [], [], []
    for c in GP.chunks(len(psi), 2048):
        o = gp.g.derive(psi[c], torch.tensor(C[c]), torch.tensor(rg[c]))
        dd.append(o["d"].cpu().numpy()); at.append(o["d_at"].cpu().numpy()); ax.append(torch.stack([o["R_axis"], o["Z_axis"]], 1).cpu().numpy())
    dd, at, ax = np.concatenate(dd), np.concatenate(at), np.concatenate(ax); dd = np.where(has, dd, np.nan)
    ang = np.degrees(np.arctan2(at[:, 1] - ax[:, 1], at[:, 0] - ax[:, 0]))
    cls = np.where(np.abs(ang - 90) <= 30, "top", np.where(np.abs(ang + 90) <= 30, "bottom", np.where(np.abs(ang) < 60, "outboard", "inboard")))
    frac = np.concatenate([np.arange(len(P[s][1])) / max(len(P[s][1]) - 1, 1) for s in shots]); ok = np.isfinite(dd)
    tot = np.nansum(dd); rep = dict(mean=float(np.nanmean(dd)), frames=int(ok.sum()))
    rep["by_location"] = {k: dict(frames=float((cls[ok] == k).mean()), share=float(np.nansum(dd[cls == k]) / tot), mean=float(np.nanmean(dd[cls == k])))
                          for k in ("top", "bottom", "outboard", "inboard")}
    q95 = np.nanquantile(dd, 0.95); rep["worst5_share"] = float(np.nansum(dd[dd >= q95]) / tot); rep["end10_share"] = float(np.nansum(dd[frac >= 0.9]) / tot)
    rep["quantiles"] = {str(q): float(np.nanquantile(dd, q)) for q in (0.5, 0.9, 0.99)}
    print("[dlcfs]", json.dumps(rep), flush=True)
    (fx.OUT / "dlcfs_anatomy.json").write_text(json.dumps(rep, indent=1))
    print("DLCFS_ANATOMY_DONE", flush=True)


if __name__ == "__main__":
    main()
