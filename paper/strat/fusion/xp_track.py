"""§45v diagnostic: where in the deployed F16K chain is the X-point placement lost? One shared inference on VT shots, arms = growing
prefixes of the chain (shift, bw | + xhe | + kw | + xp | + jgn | + li, li2 | + kw2); per arm: D_LCFS, the active saddle of the arm's psi
(LCFS min-|grad| vertex -> Newton saddle) against the TRUE saddle (xp_labels) and against the X-point heads (XP1 s1-3), and the share of
frames whose Hausdorff maximum sits within 10 cm of the true X-point. TRAIN-only heads as S44H / §45t.
    .venv/bin/python paper/strat/fusion/xp_track.py --n 50
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

import numpy as np
import torch

import fx

X = {"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}}
FULL = ["shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"]
CUTS = {"p2_bw": 2, "p3_xhe": 3, "p4_kw": 4, "p5_xp": 5, "p6_jgn": 6, "p8_li": 8, "p9_kw2": 9}


def shots_150():
    txt = (fx.ROOT / "runs" / "strat" / "fusion" / "s44h_jgn_end.sh").read_text()
    return [int(s) for s in re.search(r"--vt-shots[ =\"']+([0-9,]+)", txt).group(1).split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50); a0 = ap.parse_args()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import ens_select as ES
    import gpu_build as GB
    import gpu_pipe as GP
    import xp_oracle as XO
    from d_anatomy import pred_xpoint
    t = time.time(); d = sota.Data(); shots = shots_150()[:a0.n]
    XPH = "+".join(f"head_xp_XP1_s{s}" for s in (1, 2, 3))
    a = argparse.Namespace(psi="psiF", psis="PQmix~g", guard="psiF:0.25", qb="qh50", qbheads="+".join(f"head_qb_QH_s{s}" for s in range(1, 7)),
                           qbw=0.5, heads="head_boundary_v8_J1RL_s1*3+head_boundary_v8_J1QPIpi_s1+head_boundary_v8_J1QPIpi_s2+head_boundary_v8_J1QH_s1"
                           "+head_boundary_v8_J1QH_s2+head_boundary_v8_J1QH_s3", xpheads=XPH, axis="head", qbclamp=True, shrink="", kw="")
    arms = [(nm, json.dumps(dict(X, _steps=FULL[:k]))) for nm, k in CUTS.items()]
    xl = np.load(fx.OUT / "xp_labels.npz")["y"]; R, Z, mask = GB.AX.grid(); gp = GB.GPUPost(R, Z, mask); acc = {nm: [] for nm in CUTS}

    cache = {}

    def sink(tag, gshots, psi_g, q, b):
        nm = tag.split("_", 1)[1] if tag.startswith("PQmix") else tag
        rows = np.concatenate([np.nonzero(d.shot == s)[0] for s in gshots]); key = tuple(gshots)
        if key not in cache:                                                       # refs + X-point heads once per row group
            cache.clear(); cache[key] = (GP.refs(d, gshots), GB.heads_gpu(d, rows, XPH))
        (Cref, rgeo, _, has), xh_all = cache[key]
        for c in GP.chunks(len(psi_g), 2048):
            p = psi_g[c].to(torch.float64); o = gp.g.derive(p, torch.tensor(Cref[c]), torch.tensor(rgeo[c]))
            xs, r, sd = XO.saddle(gp, p, pred_xpoint(gp, p, o)); y = torch.tensor(xl[rows[c]], device=gp.dev, dtype=torch.float64)
            xh = torch.tensor(xh_all[c], device=gp.dev, dtype=torch.float64)
            low = (xs[:, 1] < 0)[:, None]; tr = torch.where(low, y[:, 0:2], y[:, 2:4]); hd = torch.where(low, xh[:, 0:2], xh[:, 2:4])
            e_t = (xs - tr).norm(dim=1); e_h = (xs - hd).norm(dim=1); h_t = (hd - tr).norm(dim=1); near = (o["d_at"] - tr).norm(dim=1) < 0.10
            ok = sd & (r < 0.2) & torch.isfinite(tr).all(1) & torch.from_numpy(has[c]).to(gp.dev)
            acc[nm].append(torch.stack([o["d"], e_t, e_h, h_t, near.double(), ok.double()], 1).cpu().numpy())

    GB.predict(d, shots, a, sota, ES, arms=arms, sink=sink, psi_sets=["PQmix~g"]); out = {}
    for nm in CUTS:
        A = np.concatenate(acc[nm]); ok = A[:, 5] > 0; D = np.nanmean(np.where(np.isfinite(A[:, 0]), A[:, 0], np.nan))
        et, eh, ht = (A[ok, j] * 1000 for j in (1, 2, 3))
        out[nm] = dict(D=float(D), n=int(ok.sum()), e_true_med=float(np.median(et)), e_true_rms=float(np.sqrt(np.mean(np.minimum(et, 150) ** 2))),
                       e_head_med=float(np.median(eh)), head_true_med=float(np.median(ht)), head_true_rms=float(np.sqrt(np.mean(np.minimum(ht, 150) ** 2))),
                       near_x=float(A[ok, 4].mean()), D_small=float(np.nanmean(A[ok][et <= 5, 0])), D_large=float(np.nanmean(A[ok][et > 5, 0])))
        v = out[nm]
        print(f"[xpt] {nm:7s} D {v['D']:.5f} | saddle-true med {v['e_true_med']:.1f} mm rms {v['e_true_rms']:.1f} | saddle-head med {v['e_head_med']:.1f} | "
              f"head-true med {v['head_true_med']:.1f} rms {v['head_true_rms']:.1f} | Dmax near X {v['near_x']:.2f} | D(err<=5mm) {v['D_small']:.5f} "
              f"D(>5mm) {v['D_large']:.5f} | n {v['n']}", flush=True)
    (fx.OUT / "xp_track.json").write_text(json.dumps(out, indent=1)); print(f"XPT_DONE {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
