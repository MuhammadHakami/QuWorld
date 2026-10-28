"""§48o stage-0 decomposition (diagnostic, labels only, calibration shots C): how much of the per-phase psi gain is the global temporal share,
and do the shares hold when the MLP side is closer to production (6 averaged trunks)? MLP sides: 'vt' = PQ2 + LS_ctrl (the VT analogue),
'prod' = 0.5 PQ2 + 0.5 mean(PSI-F s0..s6, 6 TRAIN-only seeds). Read: 2-fold OOF SSE ratio vs share 0.5 for a global share and per-phase shares.
    .venv/bin/python paper/strat/fusion/phase_probe2.py -> runs/strat/fusion/s48o_probe2.json"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

import fx
import gpu_build as GB
import phase_probe as PP
import qf_tta_probe as TP

PSIF6 = ["s38i_PSI-F.pt", "s38i_PSI-F-s1.pt", "s38i_PSI-F-s3.pt", "s38i_PSI-F-s4.pt", "s38i_PSI-F-s5.pt", "s38i_PSI-F-s6.pt"]


def oof(a, b, y, ph, so, per_phase: bool):
    us = np.unique(so); half = np.isin(so, us[np.random.default_rng(2026).permutation(len(us))[: len(us) // 2]])
    def ws(m):
        d = b[m] - a[m]; return float(np.clip((d * (y[m] - a[m])).sum() / max((d ** 2).sum(), 1e-30), 0, 1))
    sse = lambda w, m: float((((1 - w) * a[m] + w * b[m] - y[m]) ** 2).sum())
    num = den = 0.0; fitted = {}
    for h in (half, ~half):
        for p in (PP.PH if per_phase else ("all",)):
            m = (ph == p) if per_phase else np.ones(len(y), bool)
            num += sse(ws(m & ~h), m & h); den += sse(0.5, m & h)
    for p in (PP.PH if per_phase else ("all",)):
        fitted[p] = ws((ph == p) if per_phase else np.ones(len(y), bool))
    return num / den, fitted


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import ens_select as ES
    d = sota.Data(); C, _, _ = TP.calib_shots(d)
    rows = np.concatenate([np.nonzero(d.shot == s)[0] for s in C]); so = np.asarray(d.shot)[rows]; ph = PP.phase_of(so)
    ct = np.asarray(d.C[rows][:, :32], np.float64); pv = GB.heads_gpu(d, rows, TP.PSIH)[:, :32]
    side = {"vt": TP.mlp_coefs(d, C, ES)[:, :32]}
    TP.MLP = ["PQ2.pt"]; pq2 = TP.mlp_coefs(d, C, ES)[:, :32]; TP.MLP = PSIF6; pf = TP.mlp_coefs(d, C, ES)[:, :32]
    side["prod"] = 0.5 * pq2 + 0.5 * pf
    out = {}
    for nm, a in side.items():
        g, gw = oof(a, pv, ct, ph, so, False); p, pw = oof(a, pv, ct, ph, so, True)
        out[nm] = {"mlp_sse": float(((a - ct) ** 2).sum()), "v_sse": float(((pv - ct) ** 2).sum()), "global": {"oof_ratio": g, "w": gw},
                   "phase": {"oof_ratio": p, "w": pw}}
        print(f"[s48o2] MLP side {nm}: MLP/V SSE {out[nm]['mlp_sse'] / out[nm]['v_sse']:.3f} | global w {gw['all']:.2f} OOF {g:.4f} | "
              f"per-phase w {', '.join(f'{k} {v:.2f}' for k, v in pw.items())} OOF {p:.4f}", flush=True)
    (fx.OUT / "s48o_probe2.json").write_text(json.dumps(out, indent=1)); print("S48O_PROBE2_DONE", flush=True)


if __name__ == "__main__":
    main()
