"""§45p: two head ensembles on all VT shots, paired shot bootstrap (same machinery as router_eval.py). Reports the mean delta, its CI and the
per-target deltas, and the robust read: the worst per-target delta (dominance: Lean Competition.dominance_any_mixture over targets).
    .venv/bin/python paper/strat/fusion/ens_eval.py shape head_boundary_v8_J1RL_s1+head_boundary_v8_J1RL_s2 head_boundary_v8_J1RL_s1+head_boundary_v8_J1RLH_s1
    .venv/bin/python paper/strat/fusion/ens_eval.py qb head_qb_QHR2c_s1+head_qb_QHR2_s1 head_qb_QHR2c_s1+head_qb_QHR2Hu_s1
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

import fx
from vt_eval import SC, r2, shot_sums


def main() -> None:
    mode, base, cand = sys.argv[1:4]
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    d = sota.Data()
    if mode == "qb":
        rows = np.nonzero(d.split != 0)[0]; y = d.S[rows][:, [sota.LABELS.index("efit_q95"), sota.LABELS.index("efit_beta_n")]].astype(np.float64); names = ["q95", "betaN"]
    else:
        vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); rows, y = vt["rows"], vt["y"][:, :7].astype(np.float64); names = SC
    nt = len(names); sh = d.shot[rows]
    pb = GB.heads_gpu(d, rows, base)[:, :nt].astype(np.float64); pc = GB.heads_gpu(d, rows, cand)[:, :nt].astype(np.float64)
    Sb, Sc = shot_sums(sh, y, pb), shot_sums(sh, y, pc); rb, rc = r2(Sb), r2(Sc)
    rng = np.random.default_rng(0); W = rng.multinomial(Sb.shape[1], np.full(Sb.shape[1], 1 / Sb.shape[1]), 2000).astype(float)
    dl = r2(Sc, W).mean(1) - r2(Sb, W).mean(1); lo, hi = np.percentile(dl, [2.5, 97.5]); per = rc - rb
    print(f"[ens] {mode}: VT shots {len(np.unique(sh))} frames {len(rows)}; base {base}: mean {rb.mean():.6f}", flush=True)
    print(f"[ens] cand {cand}: delta {rc.mean() - rb.mean():+.6f} [{lo:+.6f}, {hi:+.6f}] worst target {per.min():+.5f} ({names[int(per.argmin())]}) "
          + " ".join(f"{n} {v:+.5f}" for n, v in zip(names, per)), flush=True)
    out = {"base": base, "cand": cand, "delta": float(rc.mean() - rb.mean()), "ci95": [float(lo), float(hi)], "per": dict(zip(names, per.tolist()))}
    (fx.OUT / f"ens_{mode}_{abs(hash(cand)) % 10**8}.json").write_text(json.dumps(out, indent=1)); print("ENS_DONE", flush=True)


if __name__ == "__main__":
    main()
