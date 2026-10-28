"""§48q stage 0 (doc 35, registered before any result): regime-split mixing of the F20K members, labels only, calibration VT shots C (as §48n/o).
Regimes are label-free (from the deployed heads / inputs): Ip sign; predicted kappa terciles; predicted |Z_axis| terciles; and their product
(sign x kappa). Per split: 2-fold OOF (by shot) SSE ratio of per-regime least-squares shares vs ONE global least-squares share (so the read is
the value of the split itself, not of re-tuning the share). psi: MLP (production-like side, 0.5 PQ2 + 0.5 PSI-F x6) vs PSIV60 s1+s2 on comps 0-31;
shape: J1 part vs V60 s1+s2 on the 7 main scalars (truth-sd standardised).
    .venv/bin/python paper/strat/fusion/regime_probe.py -> runs/strat/fusion/s48q_probe.json"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

import fx
import gpu_build as GB
import phase_probe as PP
import phase_probe2 as P2
import qf_tta_probe as TP


def split_oof(a, b, y, grp, so):
    ok = np.isfinite(y).all(1); a, b, y, grp, so = a[ok], b[ok], y[ok], grp[ok], so[ok]
    us = np.unique(so); half = np.isin(so, us[np.random.default_rng(2026).permutation(len(us))[: len(us) // 2]])
    def ws(m):
        d = b[m] - a[m]; return float(np.clip((d * (y[m] - a[m])).sum() / max((d ** 2).sum(), 1e-30), 0, 1))
    sse = lambda w, m: float((((1 - w) * a[m] + w * b[m] - y[m]) ** 2).sum())
    num = den = 0.0
    for h in (half, ~half):
        den += sse(ws(~h), h)
        for g in np.unique(grp):
            m = grp == g; num += sse(ws(m & ~h), m & h)
    return num / den, {str(g): ws(grp == g) for g in np.unique(grp)}


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import canon
    import ens_select as ES
    d = sota.Data(); C, _, _ = TP.calib_shots(d)
    rows = np.concatenate([np.nonzero(d.shot == s)[0] for s in C]); so = np.asarray(d.shot)[rows]
    ct = np.asarray(d.C[rows][:, :32], np.float64); pv = GB.heads_gpu(d, rows, TP.PSIH)[:, :32]
    TP.MLP = ["PQ2.pt"]; pq2 = TP.mlp_coefs(d, C, ES)[:, :32]; TP.MLP = P2.PSIF6; pf = TP.mlp_coefs(d, C, ES)[:, :32]; pm = 0.5 * pq2 + 0.5 * pf
    z = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); zy = np.asarray(z["y"][:, :7], np.float64); pos = {r: i for i, r in enumerate(z["rows"])}
    ys = np.full((len(rows), 7), np.nan); hit = np.array([r in pos for r in rows]); ys[hit] = zy[[pos[r] for r in rows[hit]]]
    sd = np.nanstd(ys, 0); hj = GB.heads_gpu(d, rows, PP.J1)[:, :7]; hv = GB.heads_gpu(d, rows, PP.VV)[:, :7]
    sg = np.asarray(canon.shot_sign(d, public=False))[rows]
    terc = lambda v: np.digitize(v, np.quantile(v, [1 / 3, 2 / 3]))
    kap, zab = terc(hj[:, 2]), terc(np.abs(hj[:, 1]))
    splits = {"ip_sign": (sg > 0).astype(int), "kappa3": kap, "absZ3": zab, "sign_x_kappa3": (sg > 0).astype(int) * 3 + kap}
    out = {}
    for nm, g in splits.items():
        rp, wp = split_oof(pm, pv, ct, g, so); rs, wsh = split_oof(hj / sd, hv / sd, ys / sd, g, so)
        out[nm] = {"psi": {"oof_ratio_vs_global": rp, "w": wp}, "shape": {"oof_ratio_vs_global": rs, "w": wsh}}
        print(f"[s48q] {nm}: psi split/global OOF {rp:.4f} w {wp} | shape {rs:.4f} w {wsh}", flush=True)
    (fx.OUT / "s48q_probe.json").write_text(json.dumps(out, indent=1)); print("S48Q_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
