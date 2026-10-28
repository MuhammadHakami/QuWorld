"""§46a (label-free): which F17K components are still non-canonical on reversed-Ip shots, and how far do they sit from the canonical heads
on public reversed vs normal frames? (J1QPIpi register heads, the li tree, the entry ensemble's q95 / betaN.)
    .venv/bin/python paper/strat/fusion/s46a_scan.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

import fx

QPI = "head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"
QHCS = "+".join(f"head_boundary_v8_J1QHCS_s{s}*4" for s in (1, 2, 3))
RLHA = "+".join(f"head_boundary_v8_J1RLHA_s{s}" for s in range(1, 9)); RLHACS = "+".join(f"head_boundary_v8_J1RLHACS_s{s}" for s in range(1, 5))
QBN = "+".join(f"head_qb_QHAR2Hu_s{s}" for s in (1, 2, 3)); QBR = "head_qb_QHAR2HuCS_s1+head_qb_QHAR2HuCS_s2"


def main() -> None:
    os.environ.pop("QW_LITREE", None)
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import canon
    import gpu_build as GB
    import submit_public as SP
    d = sota.Data(); pub = SP.PublicData(d.meta); sg = np.asarray(canon.shot_sign(pub, public=True))
    sd = np.nanstd(np.load(fx.OUT / "boundary_labels_v8_vt.npz")["y"][:, :7], 0)
    L = sota.LABELS; tr = np.nonzero(d.split == 0)[0][::20]; qsd = np.nanstd(d.S[tr][:, [L.index("efit_q95"), L.index("efit_beta_n")]].astype(np.float64), 0)
    rng = np.random.default_rng(0); rev = np.nonzero(sg < 0)[0]; nor = np.sort(rng.choice(np.nonzero(sg > 0)[0], 20000, replace=False))
    tree = np.load(fx.OUT / "litree.npz")["public"]; out = {}
    sub = np.load(fx.ROOT / "runs/sophelio_sota/submission_F17K/diii_d_public_test.npz"); keys = sorted(k for k in sub.files if k.endswith("_psirz"))
    qs = np.concatenate([sub[k.replace("_psirz", "_q95")] for k in keys]).astype(np.float64); bs = np.concatenate([sub[k.replace("_psirz", "_betaN")] for k in keys]).astype(np.float64)
    B = json.loads((fx.OUT / "qb_clamp.json").read_text())
    for name, rows, ref, qref in (("reversed", rev, RLHACS, QBR), ("normal", nor, RLHA, QBN)):
        pr = GB.heads_gpu(pub, rows, ref)[:, :7].astype(np.float64); o = {}
        for lab, stem in (("J1QPIpi", QPI), ("J1QHCS", QHCS)):
            p = GB.heads_gpu(pub, rows, stem)[:, :7].astype(np.float64); m = (((p - pr) / sd) ** 2).mean(0); o[lab] = m.tolist()
            print(f"[s46a] {name:8s} {lab:8s} vs canonical MLP (msd/sd^2): " + " ".join(f"{v:.4f}" for v in m) + f" | mean-7 {m.mean():.4f}", flush=True)
        t = tree[rows]; ok = np.isfinite(t); m = (((t[ok] - pr[ok, 6]) / sd[6]) ** 2).mean(); o["litree_li"] = float(m)
        print(f"[s46a] {name:8s} li tree vs canonical MLP li (msd/sd^2): {m:.4f}", flush=True)
        qh = GB.heads_gpu(pub, rows, qref)[:, :2].astype(np.float64); w = float((fx.OUT / "qb_blend_w.txt").read_text()) if (fx.OUT / "qb_blend_w.txt").exists() else 0.8
        res = []
        for j, (x, lo_hi) in enumerate(((qs[rows], B["q95"]), (bs[rows], B["betaN"]))):
            free = (x > lo_hi[0] + 1e-6) & (x < lo_hi[1] - 1e-6); ent = (x - w * qh[:, j]) / (1 - w)
            res.append(float((((ent[free] - qh[free, j]) / qsd[j]) ** 2).mean()))
        o["entry_qb"] = res; out[name] = o
        print(f"[s46a] {name:8s} entry q95 / betaN vs q/b heads (msd/sd^2): {res[0]:.4f} / {res[1]:.4f} (blend w {w})", flush=True)
    (fx.OUT / "s46a_scan.json").write_text(json.dumps(out, indent=1)); print("S46A_DONE", flush=True)


if __name__ == "__main__":
    main()
