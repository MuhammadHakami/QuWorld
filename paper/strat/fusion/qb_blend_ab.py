"""§48s stage A/B (doc 35), label-level: S delta of the betaN head weight 0.975 vs 0.8 (q95 stays 0.8) on A150 and B150 = 0.15 x d(mean pooled
R2 of q95, betaN), paired shot bootstrap. TRAIN-only entry trunks + the R2Hu head analogue, qbclamp as gpu_build.
    .venv/bin/python paper/strat/fusion/qb_blend_ab.py   -> runs/strat/fusion/s48s_ab.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

import fx
import gpu_build as GB
from qf_tta_probe import calib_shots
from qb_blend_probe import ANALOGUES

BASE, ARM = (0.8, 0.8), (0.8, 0.975)


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import ens_select as ES
    d = sota.Data(); _, A, B = calib_shots(d); H = sota.HEADS; iq, ib = H.index("efit_q95"), H.index("efit_beta_n")
    sel = json.loads((GB.SOTA / "ens_select_mlp.json").read_text()); clamp = json.loads((fx.OUT / "qb_clamp.json").read_text())
    out = {}
    for nm, shots in (("A150", A), ("B150", B)):
        rr = [np.nonzero(d.shot == s)[0] for s in shots]; rows = np.concatenate(rr)
        ent = [ES.member(str(fx.TRUNKS / m), d, shots) for m in sel["members"]]
        sc = np.concatenate([np.mean([m[s][1] for m in ent], 0) for s in shots])[:, [iq, ib]].astype(np.float64)
        qh = GB.heads_gpu(d, rows, ANALOGUES["R2Hu"])[:, :2].astype(np.float64)
        y = np.asarray(d.S[rows][:, [sota.LABELS.index("efit_q95"), sota.LABELS.index("efit_beta_n")]], np.float64)
        def pred(w):
            p = np.stack([sc[:, j] + w[j] * (qh[:, j] - sc[:, j]) for j in (0, 1)], 1)
            return np.stack([np.clip(p[:, 0], *clamp["q95"]), np.clip(p[:, 1], *clamp["betaN"])], 1)
        pb, pa = pred(BASE), pred(ARM); sid = np.repeat(np.arange(len(shots)), [len(r) for r in rr]); ok = np.isfinite(y).all(1)
        def stats(idx):                                   # per-shot sufficient statistics -> pooled R2 for any shot multiset
            return idx
        n_ = np.bincount(sid[ok], minlength=len(shots)).astype(float)
        sy = np.stack([np.bincount(sid[ok], y[ok, j], len(shots)) for j in (0, 1)], 1); syy = np.stack([np.bincount(sid[ok], y[ok, j] ** 2, len(shots)) for j in (0, 1)], 1)
        eb = np.stack([np.bincount(sid[ok], (pb[ok, j] - y[ok, j]) ** 2, len(shots)) for j in (0, 1)], 1)
        ea = np.stack([np.bincount(sid[ok], (pa[ok, j] - y[ok, j]) ** 2, len(shots)) for j in (0, 1)], 1)
        def s_qb(c, e):                                    # c: shot counts (bootstrap weights)
            N = c @ n_; Sy = c @ sy; Syy = c @ syy; tot = Syy - Sy ** 2 / N
            return 0.15 * float(np.mean(1 - (c @ e) / tot))
        one = np.ones(len(shots)); delta = s_qb(one, ea) - s_qb(one, eb); rng = np.random.default_rng(2026); bs = []
        for _ in range(2000):
            c = np.bincount(rng.integers(0, len(shots), len(shots)), minlength=len(shots)).astype(float); bs.append(s_qb(c, ea) - s_qb(c, eb))
        lo, hi = np.percentile(bs, [2.5, 97.5]); out[nm] = {"delta_S": delta, "lo": float(lo), "hi": float(hi), "S_qb_base": s_qb(one, eb)}
        print(f"[s48s-ab] {nm}: S_qb base {s_qb(one, eb):.6f}; arm - base {delta:+.6f} [{lo:+.6f}, {hi:+.6f}]", flush=True)
    a, b = out["A150"], out["B150"]; ok_ = a["delta_S"] > 0 and b["delta_S"] > 0 and max(a["lo"], b["lo"]) > 0
    (fx.OUT / "s48s_ab.json").write_text(json.dumps(out, indent=1))
    print(("S48S_ADOPT" if ok_ else "S48S_REJECT") + f" betaN head weight 0.975 (A150 {a['delta_S']:+.6f}, B150 {b['delta_S']:+.6f})", flush=True)


if __name__ == "__main__":
    main()
