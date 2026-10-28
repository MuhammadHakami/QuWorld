"""§38m: does the xp warp gain come through lateral extreme-point displacement? (protocol agreed on the bridge, 07:36)

On the 180-shot VALID sample, frozen chain (entry psi, v8 heads, X3 warp settings), four arms after shift/bw/kw:
  base     li only (no xp)
  xp       the X3 lateral warp (re-implemented to expose its displacements; checked bit-equal to GPUPost.xp_warp)
  rad+/-   control: the same Gaussian bumps, same frames and same |displacement|, moved along the ray from the
           magnetic axis to the extreme point (outward / inward) instead of along R
Reports (paired shot bootstrap, strata of strategies.sample):
  per-extreme-point errors |R|, |Z| (top, bottom) against the reference psi through the same extractor (derive),
  pre -> arm deltas; the head ceiling |R_head - R_true|, R_head = Rgeo - tri_head * a; twin S and its parts per arm.
Rules: lateral support iff the xp R-delta CI is below 0 and the Z-delta CI lies within +-25% of |R-delta|; the
direction matters iff xp beats both radial controls on S (paired CI above 0).

    .venv/bin/python paper/strat/fusion/xp_mechanism.py   -> runs/strat/fusion/xp_mechanism.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fx  # noqa: E402,I001
import axis_shift as AX  # noqa: E402
import gpu_build as GB  # noqa: E402
import gpu_pipe as GP  # noqa: E402
import strategies as ST  # noqa: E402
from gpu_post import GPUPost  # noqa: E402

SIGMA, CAP, N_IT, DK = 0.15, 0.10, 3, 0.003
KEYS = ("R_top", "Z_top", "R_bot", "Z_bot")
CH = 4096


def lateral(gp, psi, tri_t):
    """GPUPost.xp_warp with its final displacements and acceptance mask exposed."""
    g, o0 = gp.g, gp.g.derive(psi)
    Rt, Rb = o0["Rgeo"] - tri_t[:, 0] * o0["a"], o0["Rgeo"] - tri_t[:, 1] * o0["a"]
    bump = lambda Rc, Zc: torch.exp(-((gp.RR - Rc[:, None, None]) ** 2 + (gp.ZZ - Zc[:, None, None]) ** 2) / (2 * SIGMA ** 2))
    gt, gb = bump(o0["R_top"], o0["Z_top"]), bump(o0["R_bot"], o0["Z_bot"])
    apply = lambda dt, db: gp.sample(psi, gp.RR - dt[:, None, None] * gt - db[:, None, None] * gb, gp.ZZ.expand_as(psi))
    dt = (Rt - o0["R_top"]).clamp(-CAP, CAP); db = (Rb - o0["R_bot"]).clamp(-CAP, CAP); out = apply(dt, db)
    for _ in range(N_IT - 1):
        o = g.derive(out)
        dt = torch.where(torch.isfinite(o["R_top"]), (dt + Rt - o["R_top"]).clamp(-CAP, CAP), dt)
        db = torch.where(torch.isfinite(o["R_bot"]), (db + Rb - o["R_bot"]).clamp(-CAP, CAP), db)
        out = apply(dt, db)
    o = g.derive(out)
    e0 = (o0["tri_top"] - tri_t[:, 0]).abs() + (o0["tri_bot"] - tri_t[:, 1]).abs()
    e1 = (o["tri_top"] - tri_t[:, 0]).abs() + (o["tri_bot"] - tri_t[:, 1]).abs()
    good = o0["ok"] & o["ok"] & torch.isfinite(e1) & (e1 < e0) & ((o["kappa"] - o0["kappa"]).abs() <= DK) & torch.isfinite(tri_t).all(1)
    return torch.where(good[:, None, None], out, psi), dict(o0=o0, gt=gt, gb=gb, dt=dt, db=db, good=good)


def radial(gp, psi, st, sign):
    """Same bumps, frames and |displacement|, along the axis -> extreme-point ray."""
    o0 = st["o0"]
    def unit(R, Z):
        v = torch.stack([R - o0["R_axis"], Z - o0["Z_axis"]], 1); return v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)
    ut, ub = unit(o0["R_top"], o0["Z_top"]), unit(o0["R_bot"], o0["Z_bot"])
    mt, mb = sign * st["dt"].abs(), sign * st["db"].abs()
    dR = (mt * ut[:, 0])[:, None, None] * st["gt"] + (mb * ub[:, 0])[:, None, None] * st["gb"]
    dZ = (mt * ut[:, 1])[:, None, None] * st["gt"] + (mb * ub[:, 1])[:, None, None] * st["gb"]
    out = gp.sample(psi, gp.RR - dR, gp.ZZ - dZ)
    return torch.where(st["good"][:, None, None], out, psi)


def ext(gp, psi):
    o = [gp.g.derive(psi[c]) for c in GP.chunks(len(psi), CH)]
    return {k: torch.cat([x[k] for x in o]).cpu().numpy() for k in KEYS + ("Rgeo", "a")}


def boot(per_shot, grp, B=2000, seed=0):
    """Stratified shot bootstrap of the mean: (mean, lo, hi)."""
    rng = np.random.default_rng(seed); strata = {g: np.nonzero(grp == g)[0] for g in np.unique(grp)}; m = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(v, len(v)) for v in strata.values()]); m.append(np.nanmean(per_shot[idx]))
    return float(np.nanmean(per_shot)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    t = time.time()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import ens_select as ES
    d = sota.Data(); shots, grp_d = ST.sample(); grp = np.asarray([grp_d[s] for s in shots])
    a = SimpleNamespace(psi="entry", qb="entry", heads=GB.V8, kw=json.dumps({"_steps": ["shift", "bw", "kw"]}), axis="trunk")
    P = GB.predict(d, shots, a, sota, ES)                                   # pre-xp psi (shift, bw, kw)
    rows = [np.nonzero(d.shot == s)[0] for s in shots]; allr = np.concatenate(rows)
    assert all(len(r) == len(fx.valid_rows(d, s)) for r, s in zip(rows, shots))
    ph = torch.tensor(GB.heads_gpu(d, allr, GB.V8), device="cuda"); tri_t, li_t = ph[:, 3:5], ph[:, 6]
    R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask)
    pre = torch.tensor(np.concatenate([P[s][0] for s in shots]), device="cuda")
    q, b = np.concatenate([P[s][1] for s in shots]), np.concatenate([P[s][2] for s in shots])
    arms = {k: torch.empty_like(pre) for k in ("xp", "rad+", "rad-")}; good = torch.zeros(len(pre), dtype=torch.bool, device="cuda")
    for c in GP.chunks(len(pre), CH):
        out, st = lateral(gp, pre[c], tri_t[c]); ref = gp.xp_warp(pre[c], tri_t[c], sigma=SIGMA, cap=CAP)
        assert torch.equal(out, ref), "re-implemented xp differs from GPUPost.xp_warp"
        arms["xp"][c] = out; arms["rad+"][c] = radial(gp, pre[c], st, 1.0); arms["rad-"][c] = radial(gp, pre[c], st, -1.0); good[c] = st["good"]
    gref = torch.tensor(d.psi_of(allr).reshape(-1, 65, 65), device="cuda", dtype=torch.float64)
    E = {"true": ext(gp, gref), "pre": ext(gp, pre), **{k: ext(gp, v) for k, v in arms.items()}}
    off = np.cumsum([0] + [len(r) for r in rows]); gd = good.cpu().numpy(); rep = {"accepted_frames": int(gd.sum()), "frames": len(gd)}
    def per_shot(x):
        return np.array([np.nanmean(np.where(gd[off[i]:off[i + 1]], x[off[i]:off[i + 1]], np.nan)) if gd[off[i]:off[i + 1]].any()
                         else np.nan for i in range(len(shots))])
    tri_np = tri_t.cpu().numpy(); Ep = E["pre"]
    head = {"R_top": Ep["Rgeo"] - tri_np[:, 0] * Ep["a"], "R_bot": Ep["Rgeo"] - tri_np[:, 1] * Ep["a"]}
    for k in KEYS:
        e_pre = np.abs(E["pre"][k] - E["true"][k]); rep[f"{k}_pre_mae"] = boot(per_shot(e_pre), grp)
        if k in head:
            rep[f"{k}_head_mae"] = boot(per_shot(np.abs(head[k] - E["true"][k])), grp)
        for arm in arms:
            rep[f"{k}_{arm}_delta"] = boot(per_shot(np.abs(E[arm][k] - E["true"][k]) - e_pre), grp)
    nm = ST.load_entry()[1]; sums = {}
    for name, x in (("base", pre), *arms.items()):
        y = GP.post_all(gp, x, None, None, None, li_t, steps=("li",))
        parts = GP.parts_all(gp, y, q, b, d, shots); sums[name] = np.stack([fx.shot_sums(parts[s]) for s in shots])
        rep[f"S_{name}"] = fx.composite(sums[name], nm)
    for arm in arms:
        rep[f"dS_{arm}_vs_base"] = fx.paired_boot(sums["base"], sums[arm], nm, groups=grp)
    for c in ("rad+", "rad-"):
        rep[f"dS_xp_vs_{c}"] = fx.paired_boot(sums[c], sums["xp"], nm, groups=grp)
    out = fx.OUT / "xp_mechanism.json"; out.write_text(json.dumps(rep, indent=1, default=float))
    for k, v in rep.items():
        if k.startswith(("R_", "Z_")):
            print(f"XPM {k:22s} {v[0]:+.5f} [{v[1]:+.5f}, {v[2]:+.5f}]", flush=True)
    for arm in arms:
        bt = rep[f"dS_{arm}_vs_base"]; print(f"XPM dS {arm:5s} vs base {bt['delta']:+.6f} [{bt['lo']:+.6f}, {bt['hi']:+.6f}]", flush=True)
    for c in ("rad+", "rad-"):
        bt = rep[f"dS_xp_vs_{c}"]; print(f"XPM dS xp vs {c} {bt['delta']:+.6f} [{bt['lo']:+.6f}, {bt['hi']:+.6f}]", flush=True)
    print(f"XPM_DONE frames {rep['accepted_frames']}/{rep['frames']} accepted | {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
