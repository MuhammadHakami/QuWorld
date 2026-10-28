"""Doc 35 §42a follow-up: q/b blend weight between the entry trunks' scalars and the q/b heads, on all VALID + TEST frames (TRAIN-only
components). Deployed: qh50 = 0.5 entry + 0.5 heads. Prints mean q/b R2 per weight for each head set.
    .venv/bin/python paper/strat/fusion/qb_blend_vt.py <heads A> [<heads B> ...]     ('+'-joined stems)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

import fx


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB, ens_select as ES
    d = sota.Data(); L = sota.LABELS; shots = [int(s) for s in np.unique(d.shot[d.split != 0])]
    rows = np.concatenate([np.nonzero(d.shot == s)[0] for s in shots])
    sel = json.loads((GB.SOTA / "ens_select_mlp.json").read_text()); ent = [ES.member(str(fx.TRUNKS / m), d, shots) for m in sel["members"]]
    H = sota.HEADS; iq, ib = H.index("efit_q95"), H.index("efit_beta_n")
    sc = np.concatenate([np.mean([m[s][1] for m in ent], 0) for s in shots]); E = sc[:, [iq, ib]]
    Y = d.S[rows][:, [L.index("efit_q95"), L.index("efit_beta_n")]].astype(float); ok = np.isfinite(Y).all(1)
    r2 = lambda P: [1 - np.mean((P[ok, j] - Y[ok, j]) ** 2) / np.var(Y[ok, j]) for j in range(2)]
    print(f"[qbw] entry alone: q95 {r2(E)[0]:.5f} betaN {r2(E)[1]:.5f}", flush=True)
    for hs in sys.argv[1:]:
        Hh = GB.heads_gpu(d, rows, hs)[:, :2]
        line = []
        for w in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            q, b = r2(w * Hh + (1 - w) * E); line.append(f"w {w:.1f}: {0.5 * (q + b):.6f} (q {q:.5f} b {b:.5f})")
        print(f"[qbw] {hs[:60]}: " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
