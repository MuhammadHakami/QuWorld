"""Doc 35 §42f diagnostic: where does the q95 / betaN error live on VALID + TEST frames (TRAIN-only QH x6)? SSE share vs frame share
by shot phase, plasma current, q95, betaN, |Bt|, current ramp rate and Ip sign; caches the predictions (runs/strat/fusion/qb_vt_pred.npz).
    .venv/bin/python paper/strat/fusion/qb_anatomy.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import fx


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    d = sota.Data(); L = sota.LABELS; fn = list(d.fn); cache = fx.OUT / "qb_vt_pred.npz"
    rows = np.nonzero(d.split != 0)[0]
    if cache.exists():
        Q = np.load(cache)["Q"]
    else:
        Q = GB.heads_gpu(d, rows, "+".join(f"head_qb_QH_s{s}" for s in range(1, 7))); np.savez(cache, rows=rows, Q=Q)
    y = d.S[rows][:, [L.index("efit_q95"), L.index("efit_beta_n")]].astype(float)
    X = np.asarray(GB.fixed(d, "X_ipfix").inputs(rows, np.ones(len(fn), bool)), float); g = lambda n: X[:, fn.index(n)]
    ip, bt = g("mag_plasma_current"), g("mag_bcoil"); dip = ip - g("mag_plasma_current_lag-20ms"); pos = d.edge_feats()[rows, 1]
    sid = d.shot[rows]; sgn = np.sign(np.array([np.mean(ip[sid == s]) for s in sid]))
    strata = {"phase": (pos, [0, 0.05, 0.1, 0.3, 0.7, 0.9, 0.95, 1.01]), "|Ip| kA": (np.abs(ip), [0, 300, 600, 900, 1200, 3000]),
              "q95": (y[:, 0], [0, 3, 4, 5, 6, 8, 100]), "betaN": (y[:, 1], [-1, 0.5, 1, 1.5, 2, 2.5, 10]),
              "|Bt| kA": (np.abs(bt), [0, 80, 100, 120, 200]), "|dIp| 20ms": (np.abs(dip), [0, 5, 20, 50, 1e9]), "Ip sign": (sgn, [-2, 0, 2])}
    for j, n in enumerate(["q95", "betaN"]):
        e = (Q[:, j] - y[:, j]) ** 2; ok = np.isfinite(e); tot = e[ok].sum()
        print(f"[qba] {n}: R2 {1 - e[ok].mean() / np.var(y[ok, j]):.5f}", flush=True)
        for sn, (v, edges) in strata.items():
            parts = []
            for a, b in zip(edges[:-1], edges[1:]):
                m = ok & (v >= a) & (v < b)
                if m.sum():
                    parts.append(f"[{a:g},{b:g}) f {m.sum() / ok.sum():.3f} sse {e[m].sum() / tot:.3f}")
            print(f"   {sn}: " + " | ".join(parts), flush=True)


if __name__ == "__main__":
    main()
