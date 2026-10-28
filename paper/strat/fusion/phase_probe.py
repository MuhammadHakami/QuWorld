"""§48o stage 0 (doc 35, registered before any result): per-shot-phase mixing of the F20K members, labels only, on the 443 calibration VT shots
C = VT \\ (A150 u B150) (as §48n). Phases: first 5 frames, middle, frames T-10..T-6, last 5.
  psi: MLP mix (PQ2 + LS_ctrl) vs temporal PSIV60 s1+s2 on components 0-31 (deployed share 0.5).
  shape (7 main scalars, standardised by the truth sd on C): J1 part (route J1RL / J1RLCS + J1QHCS) vs V60 s1+s2 (deployed V share 13.5 / 49.5).
Per phase: SSE of each, of the deployed mix and of the least-squares share w*; 2-fold out-of-fold (by shot) SSE ratio of per-phase shares vs deployed.
    .venv/bin/python paper/strat/fusion/phase_probe.py   -> runs/strat/fusion/s48o_probe.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

import fx
import gpu_build as GB
import qf_tta_probe as TP

QHCS = "head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
J1 = f"route:head_boundary_v8_J1RL_s1*24+{QHCS}|head_boundary_v8_J1RLCS_s1*24+{QHCS}"
VV = "head_boundary_v8_V60_s1+head_boundary_v8_V60_s2"
WV = 13.5 / 49.5
PH = ("first5", "mid", "last10_6", "last5")


def phase_of(shot_of: np.ndarray) -> np.ndarray:
    out = np.empty(len(shot_of), "<U8"); st = np.r_[0, np.nonzero(np.diff(shot_of))[0] + 1, len(shot_of)]
    for a, b in zip(st[:-1], st[1:]):
        i = np.arange(b - a); T = b - a
        out[a:b] = np.where(i < 5, "first5", np.where(i >= T - 5, "last5", np.where(i >= T - 10, "last10_6", "mid")))
    return out


def mix_read(a: np.ndarray, b: np.ndarray, y: np.ndarray, w0: float, ph: np.ndarray, shot_of: np.ndarray) -> dict:
    """a, b: two predictors (rows, k), y truth; w = weight on b. Deployed w0 vs per-phase least-squares w (2-fold OOF by shot)."""
    ok = np.isfinite(y).all(1); a, b, y, ph, so = a[ok], b[ok], y[ok], ph[ok], shot_of[ok]
    def wstar(m):
        d = b[m] - a[m]; return float(np.clip((d * (y[m] - a[m])).sum() / max((d ** 2).sum(), 1e-30), 0, 1))
    sse = lambda w, m: float((((1 - w) * a[m] + w * b[m] - y[m]) ** 2).sum())
    us = np.unique(so); half = np.isin(so, us[np.random.default_rng(2026).permutation(len(us))[: len(us) // 2]])
    rep, oof_dep, oof_ph = {}, 0.0, 0.0
    for p in PH:
        m = ph == p
        rep[p] = {"rows": int(m.sum()), "sse_a": sse(0, m), "sse_b": sse(1, m), "sse_deployed": sse(w0, m), "w_star": wstar(m)}
        for h in (half, ~half):
            oof_dep += sse(w0, m & h); oof_ph += sse(wstar(m & ~h), m & h)
    tot = sum(r["sse_deployed"] for r in rep.values())
    for r in rep.values():
        r["share_of_deployed_sse"] = r["sse_deployed"] / tot
    return {"phases": rep, "oof_ratio_phase_vs_deployed": oof_ph / oof_dep}


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import ens_select as ES
    t0 = time.time(); d = sota.Data(); C, _, _ = TP.calib_shots(d)
    rows = np.concatenate([np.nonzero(d.shot == s)[0] for s in C]); so = np.asarray(d.shot)[rows]; ph = phase_of(so)
    ct = np.asarray(d.C[rows][:, :32], np.float64)
    pm = TP.mlp_coefs(d, C, ES)[:, :32]; pv = GB.heads_gpu(d, rows, TP.PSIH)[:, :32]
    out = {"stage": "48o-0", "calib_shots": len(C), "rows": int(len(rows)), "psi": mix_read(pm, pv, ct, 0.5, ph, so)}
    z = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); zy = np.asarray(z["y"][:, :7], np.float64); pos = {r: i for i, r in enumerate(z["rows"])}
    ys = np.full((len(rows), 7), np.nan); hit = np.array([r in pos for r in rows]); ys[hit] = zy[[pos[r] for r in rows[hit]]]
    sd = np.nanstd(ys, 0); hj = GB.heads_gpu(d, rows, J1)[:, :7] / sd; hv = GB.heads_gpu(d, rows, VV)[:, :7] / sd
    out["shape"] = mix_read(hj, hv, ys / sd, WV, ph, so)
    for k in ("psi", "shape"):
        r = out[k]; print(f"[s48o] {k}: OOF per-phase / deployed SSE {r['oof_ratio_phase_vs_deployed']:.5f} | " + " ".join(
            f"{p} rows {v['rows']} share {v['share_of_deployed_sse']:.3f} a/b {v['sse_a'] / v['sse_b']:.3f} w* {v['w_star']:.2f}"
            for p, v in r["phases"].items()), flush=True)
    (fx.OUT / "s48o_probe.json").write_text(json.dumps(out, indent=1)); print(f"S48O_PROBE_DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
