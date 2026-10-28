"""§45o: Ip-sign router read-out on all VT shots. Base heads predict every frame; the canonical heads (STRAT_CANON, Ip-sign canonical
currents) replace them on reversed-Ip shots only (ip_sign.json, an inputs-only per-shot sign). Paired shot bootstrap vs the base.
    .venv/bin/python paper/strat/fusion/router_eval.py shape head_boundary_v8_J1RL_s1 head_boundary_v8_J1RLCS_s1
    .venv/bin/python paper/strat/fusion/router_eval.py qb head_qb_QHR2c_s1 head_qb_QHR2CS_s1
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

import fx
from vt_eval import SC, r2, shot_sums


def main() -> None:
    mode, base, can = sys.argv[1:4]
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    d = sota.Data(); sg = json.loads((fx.OUT / "ip_sign.json").read_text())
    if mode == "qb":
        rows = np.nonzero(d.split != 0)[0]; y = d.S[rows][:, [sota.LABELS.index("efit_q95"), sota.LABELS.index("efit_beta_n")]].astype(np.float64); names = ["q95", "betaN"]
    else:
        vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); rows, y = vt["rows"], vt["y"][:, :7].astype(np.float64); names = SC
    nt = len(names); sh = d.shot[rows]; rev = np.array([sg[str(int(s))] < 0 for s in sh])
    pb = GB.heads_gpu(d, rows, base)[:, :nt].astype(np.float64); pc = GB.heads_gpu(d, rows, can)[:, :nt].astype(np.float64)
    print(f"[router] {mode}: VT frames {len(rows)} shots {len(np.unique(sh))}, reversed frames {rev.mean():.3f} shots {len(np.unique(sh[rev]))}", flush=True)
    Sb = shot_sums(sh, y, pb); rb = r2(Sb)
    rng = np.random.default_rng(0); W = rng.multinomial(Sb.shape[1], np.full(Sb.shape[1], 1 / Sb.shape[1]), 2000).astype(float); bb = r2(Sb, W).mean(1)
    print(f"[router] base {base}: mean {rb.mean():.6f} " + " ".join(f"{n} {v:.5f}" for n, v in zip(names, rb)), flush=True)
    out = {}
    for lab, p in (("whole", pc), ("router", np.where(rev[:, None], pc, pb))):
        S = shot_sums(sh, y, p); r = r2(S); dl = r2(S, W).mean(1) - bb; lo, hi = np.percentile(dl, [2.5, 97.5])
        out[lab] = {"delta": float(r.mean() - rb.mean()), "ci95": [float(lo), float(hi)], "per": dict(zip(names, (r - rb).tolist()))}
        print(f"[router] {lab:6s} {can}: {r.mean() - rb.mean():+.6f} [{lo:+.6f}, {hi:+.6f}] " + " ".join(f"{n} {v:+.5f}" for n, v in zip(names, r - rb)), flush=True)
    (fx.OUT / f"router_{mode}_{can.replace('head_boundary_v8_', '').replace('head_qb_', '')}.json").write_text(json.dumps(out, indent=1))
    print("ROUTER_DONE", flush=True)


if __name__ == "__main__":
    main()
