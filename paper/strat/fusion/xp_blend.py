"""Doc 35 §42t diagnostic: on the 60 oracle shots (xp_oracle.json; frames = scored rows of the sorted shots, in order), X-point error of the
deployed-style psi saddle, of an X-point head, and of their blends w * head + (1 - w) * psi (side chosen by the psi saddle).
    .venv/bin/python paper/strat/fusion/xp_blend.py head_xp_XP1_s1
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

import fx


def main() -> None:
    stems = sys.argv[1]
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    d = sota.Data(); allv = [int(s) for s in np.unique(d.shot[d.split != 0])]
    shots = sorted(np.random.default_rng(0).choice(allv, 60, replace=False).tolist())
    rows = np.concatenate([fx.scored_rows(d, s) for s in shots])
    J = json.loads((fx.OUT / "xp_oracle.json").read_text()); A = np.array(J["rows"]); col = {c: i for i, c in enumerate(J["cols"])}
    assert len(A) == len(rows), (len(A), len(rows))
    ok = A[:, col["ok"]] > 0; xt = A[:, [col["xt_R"], col["xt_Z"]]]; xp = A[:, [col["xp_R"], col["xp_Z"]]]
    h = GB.heads_gpu(d, rows, stems); lower = xp[:, 1] < 0; xh = np.where(lower[:, None], h[:, 0:2], h[:, 2:4])
    ok &= np.isfinite(xh).all(1)
    q = lambda e: f"median {np.median(e) * 1e3:.1f} mm, p75 {np.quantile(e, 0.75) * 1e3:.1f}, p90 {np.quantile(e, 0.9) * 1e3:.1f}, rms {np.sqrt(np.mean(e ** 2)) * 1e3:.1f}"
    print(f"[xpb] {ok.sum()} frames | psi saddle: {q(np.linalg.norm(xp[ok] - xt[ok], axis=1))}", flush=True)
    for w in (0.25, 0.5, 0.75, 1.0):
        e = np.linalg.norm(w * xh[ok] + (1 - w) * xp[ok] - xt[ok], axis=1)
        print(f"[xpb] w {w:.2f}: {q(e)} | dR rms {np.sqrt(np.mean((w * xh[ok, 0] + (1 - w) * xp[ok, 0] - xt[ok, 0]) ** 2)) * 1e3:.1f} "
              f"dZ rms {np.sqrt(np.mean((w * xh[ok, 1] + (1 - w) * xp[ok, 1] - xt[ok, 1]) ** 2)) * 1e3:.1f}", flush=True)


if __name__ == "__main__":
    main()
