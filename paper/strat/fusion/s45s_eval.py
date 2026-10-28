"""§45s read-out (rules in doc 35): (1) label-free: on reversed-Ip frames, Z / volume disagreement of the register part with the canonical MLP
heads, J1QH vs J1QHCS, public and VT; (2) X-point heads on VT reversed frames: XP1 vs XP1CS errors against xp_labels (finite rows), and the
label-free XPA vs XPACS disagreement on public reversed frames. The VT router read for the shape heads runs separately (router_eval.py).
    .venv/bin/python paper/strat/fusion/s45s_eval.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

import fx

QPI = "head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"
QH = "+".join(f"head_boundary_v8_J1QH_s{s}*4" for s in (1, 2, 3)); QHCS = QH.replace("J1QH_", "J1QHCS_")


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import canon
    import gpu_build as GB
    import submit_public as SP
    d = sota.Data(); pub = SP.PublicData(d.meta); out = {}
    vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); vrows = vt["rows"]; sd = np.nanstd(vt["y"][:, :7], 0)
    vrev = vrows[np.asarray(canon.shot_sign(d))[vrows] < 0]; prev = np.nonzero(np.asarray(canon.shot_sign(pub, public=True)) < 0)[0]
    cs = {"VT": "head_boundary_v8_J1RLCS_s1", "public": "+".join(f"head_boundary_v8_J1RLHACS_s{s}" for s in range(1, 5))}
    for name, dd, rows in (("VT", d, vrev), ("public", pub, prev)):
        pc = GB.heads_gpu(dd, rows, cs[name])[:, :7].astype(np.float64)
        for lab, stem in (("J1QH", QH), ("J1QHCS", QHCS)):
            p = GB.heads_gpu(dd, rows, stem)[:, :7].astype(np.float64); m = (((p - pc) / sd) ** 2).mean(0)
            out[f"{name}_{lab}_vs_cs"] = m.tolist(); print(f"[s45s] {name} {lab:6s} vs canonical MLP: Z {m[1]:.4f} volume {m[5]:.4f} mean-7 {np.nanmean(m):.4f}", flush=True)
    if os.environ.get("S45S_REG_ONLY"):                                  # register part only (the X-point twins may still be training)
        (fx.OUT / "s45s_eval_reg.json").write_text(json.dumps(out, indent=1)); print("S45S_EVAL_REG_DONE", flush=True); return
    xl = np.load(fx.OUT / "xp_labels.npz")["y"]
    for lab, stem in (("XP1", "head_xp_XP1_s1"), ("XP1CS", "head_xp_XP1CS_s1")):
        p = GB.heads_gpu(d, vrev, stem)[:, :4].astype(np.float64); y = xl[vrev].astype(np.float64); ok = np.isfinite(y)
        e = np.sqrt(np.nanmean(np.where(ok, (p - y) ** 2, np.nan), 0)) * 100; out[f"VT_{lab}_rmse_cm"] = e.tolist()
        print(f"[s45s] VT reversed {lab:6s} X-point rmse (cm) R_l {e[0]:.2f} Z_l {e[1]:.2f} R_u {e[2]:.2f} Z_u {e[3]:.2f}", flush=True)
    xa = "+".join(f"head_xp_XPA_s{s}" for s in (1, 2, 3)); xc = xa.replace("XPA_", "XPACS_")
    for name, dd, rows in (("VT", d, vrev), ("public", pub, prev)):
        a = GB.heads_gpu(dd, rows, xa)[:, :4]; c = GB.heads_gpu(dd, rows, xc)[:, :4]
        dis = np.sqrt(np.nanmean((a - c) ** 2, 0)) * 100; out[f"{name}_XPA_vs_XPACS_cm"] = dis.tolist()
        print(f"[s45s] {name} reversed XPA vs XPACS rms disagreement (cm) " + " ".join(f"{v:.2f}" for v in dis), flush=True)
    (fx.OUT / "s45s_eval.json").write_text(json.dumps(out, indent=1)); print("S45S_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
