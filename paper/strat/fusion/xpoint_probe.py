"""Doc 35 §38ab-pre: does the X-point position explain the tip errors (D_LCFS, kappa)? (GPU, inference only)

§38aa: D_LCFS maxima sit at the top / bottom tips (93%), kappa jumps a grid row there, and our psi already places the tips
better than the 64 head rays. Here, on the 180-shot VALID sample, the upper and lower X-points (minimum of |grad psi| in
the region more than 0.6 m above / below the magnetic axis, inside the envelope, parabolic sub-cell refinement) are found
in the truth and in HJl9's final psi. Read-out: rms X-point error (cm) per side; the share of frames off by more than half a
cell; the correlation of the X-point error with the per-frame D_LCFS and the kappa error. If the X-point error is a few cm
and tracks the D_LCFS tail, an X-point head (+ a saddle warp) is the next lever.

    .venv/bin/python paper/strat/fusion/xpoint_probe.py   -> runs/strat/fusion/xpoint_probe.json
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


def xpoints(gp, psi, mask, za):
    """(B, 2, 2) X-point (R, Z) for the upper [0] and lower [1] region; NaN if none."""
    gz, gr = torch.gradient(psi, spacing=(gp.g.Z, gp.g.R), dim=(1, 2)); m2 = gr ** 2 + gz ** 2
    out = torch.full((len(psi), 2, 2), float("nan"), device=psi.device, dtype=psi.dtype)
    inner = torch.zeros_like(mask); inner[2:-2, 2:-2] = mask[2:-2, 2:-2]
    for k, sgn in enumerate((1.0, -1.0)):
        reg = inner[None] & (sgn * (gp.ZZ[None] - za[:, None, None]) > 0.6)
        v = torch.where(reg, m2, torch.full_like(m2, np.inf)).flatten(1); i = v.argmin(1)
        iz, ir = i // psi.shape[2], i % psi.shape[2]; b = torch.arange(len(psi), device=psi.device)
        def par(a, c, d):                                                  # parabolic vertex offset in [-0.5, 0.5]
            den = a - 2 * c + d; return torch.where(den.abs() > 1e-30, 0.5 * (a - d) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        izc, irc = iz.clamp(1, psi.shape[1] - 2), ir.clamp(1, psi.shape[2] - 2)
        dz = par(m2[b, izc - 1, irc], m2[b, izc, irc], m2[b, izc + 1, irc]); dr = par(m2[b, izc, irc - 1], m2[b, izc, irc], m2[b, izc, irc + 1])
        ok = torch.isfinite(v[b, i])
        out[:, k, 0] = torch.where(ok, gp.g.R[0] + (irc + dr) * gp.dR, out[:, k, 0]); out[:, k, 1] = torch.where(ok, gp.g.Z[0] + (izc + dz) * gp.dZ, out[:, k, 1])
    return out


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import ens_select as ES
    d = sota.Data(); shots, _ = ST.sample()
    class A: pass
    a = A(); a.psi, a.qb, a.axis = "psiF", "qh50", "head"; a.heads = "+".join(f"head_boundary_v8_J1_s{s}" for s in range(1, 7))
    a.qbheads = "+".join(f"head_qb_QH_s{s}" for s in range(1, 7))
    a.kw = '{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
    P = GB.predict(d, shots, a, sota, ES); R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask)
    rows = np.concatenate([np.nonzero(d.shot == s)[0] for s in shots]); C, rg, cg, has = GP.refs(d, shots)
    mk = torch.tensor(mask, device="cuda"); res = {k: [] for k in ("xt", "xp", "dl", "kp", "kt")}
    pred = np.concatenate([P[s][0] for s in shots])
    for c in GP.chunks(len(rows), 4096):
        pt = torch.tensor(d.psi_of(rows[c]).reshape(-1, 65, 65), device="cuda", dtype=torch.float64); pp = torch.tensor(pred[c], device="cuda")
        ot = gp.g.derive(pt); op = gp.g.derive(pp, torch.tensor(C[c]), torch.tensor(rg[c]))
        res["xt"].append(xpoints(gp, pt, mk, ot["Z_axis"]).cpu().numpy()); res["xp"].append(xpoints(gp, pp, mk, op["Z_axis"]).cpu().numpy())
        res["dl"].append(op["d"].cpu().numpy()); res["kp"].append(op["kappa"].cpu().numpy()); res["kt"].append(ot["kappa"].cpu().numpy())
    xt, xp, dl, kp, kt = (np.concatenate(res[k]) for k in ("xt", "xp", "dl", "kp", "kt")); dl = np.where(has, dl, np.nan)
    rep = {}
    for k, side in enumerate(("upper", "lower")):
        e = np.hypot(*(xp[:, k] - xt[:, k]).T) * 100; ok = np.isfinite(e) & np.isfinite(dl)
        ez = np.abs(xp[:, k, 1] - xt[:, k, 1]) * 100; er = np.abs(xp[:, k, 0] - xt[:, k, 0]) * 100
        ke = np.abs(kp - kt)
        rep[side] = dict(rms_cm=float(np.sqrt(np.nanmean(e[ok] ** 2))), median_cm=float(np.nanmedian(e[ok])), q90_cm=float(np.nanquantile(e[ok], 0.9)),
                         off_half_cell_Z=float(np.nanmean(ez[ok] > 2.5)), off_half_cell_R=float(np.nanmean(er[ok] > 1.33)),
                         corr_dlcfs=float(np.corrcoef(e[ok], dl[ok])[0, 1]), corr_kappa_err=float(np.corrcoef(e[ok], ke[ok])[0, 1]),
                         dlcfs_tail_share_xerr_gt_q90=float(np.nansum(dl[ok][e[ok] > np.nanquantile(e[ok], 0.9)]) / np.nansum(dl[ok])))
        print(f"[xpoint] {side}: " + " ".join(f"{a_} {b:.3f}" for a_, b in rep[side].items()), flush=True)
    (fx.OUT / "xpoint_probe.json").write_text(json.dumps(rep, indent=1))
    print("XPOINT_DONE", flush=True)


if __name__ == "__main__":
    main()
