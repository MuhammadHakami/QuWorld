"""Doc 35 §42u: second li pass applied to a built public submission (the chain's "li2" step, §41y, as a post-pass).
The li warp is the chain's last step, so re-applying it to the saved psi with the same head li targets equals the chain with li2 (up to the
float32 rounding of the saved psi). q95 / betaN are copied unchanged. Diagnostics: |li(psi) - li_head| before / after.
    .venv/bin/python paper/strat/fusion/post_li2.py --src F6 --tag F6L
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile

import numpy as np
import torch

import fx

H_F4 = "+".join(f"head_boundary_v8_J1AH_s{s}*3" for s in range(1, 9)) + "+head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--heads", default=H_F4); ap.add_argument("--cap", type=float, default=0.9); a = ap.parse_args()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB, axis_shift as AX, submit_public as SP, fold as FD
    from gpu_post import GPUPost
    t = time.time(); d = sota.Data(); pub = SP.PublicData(d.meta); shots = pub.shots(); rows = np.arange(len(pub.X))
    li_t = torch.tensor(GB.heads_gpu(pub, rows, a.heads)[:, 6], device="cuda", dtype=torch.float64)
    src = np.load(fx.ROOT / "runs" / "sophelio_sota" / f"submission_{a.src}" / f"diii_d_{FD.TAG}_test.npz")
    psi = np.concatenate([src[f"shot_{s:04d}_psirz"] for s in shots]); assert len(psi) == len(rows), (len(psi), len(rows))
    R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask); out = np.empty_like(psi); gap0, gap1 = [], []
    for i in range(0, len(psi), 2048):
        p = torch.tensor(psi[i:i + 2048], device="cuda", dtype=torch.float64); lt = li_t[i:i + 2048]
        gap0.append((gp.g.derive(p)["li"] - lt).cpu().numpy())
        q = gp.li_warp(p, lt, cap=a.cap); out[i:i + 2048] = q.cpu().numpy().astype(np.float32)
        gap1.append((gp.g.derive(q)["li"] - lt).cpu().numpy())
    gap0, gap1 = np.concatenate(gap0), np.concatenate(gap1)
    for nm, g in (("before", gap0), ("after", gap1)):
        g = np.nan_to_num(g); print(f"[li2] {nm}: rms |li - head| {np.sqrt(np.mean(g ** 2)):.4f}; frames > 0.05: {(np.abs(g) > 0.05).sum()} "
                                    f"(rms there {np.sqrt(np.mean(g[np.abs(g) > 0.05] ** 2)) if (np.abs(g) > 0.05).any() else 0:.3f})", flush=True)
    od = fx.ROOT / "runs" / "sophelio_sota" / f"submission_{a.tag}"; od.mkdir(parents=True, exist_ok=True); preds = {}; i = 0
    for s in shots:
        T = int((pub.shot == s).sum()); preds[f"shot_{s:04d}_psirz"] = out[i:i + T]; i += T
        preds[f"shot_{s:04d}_q95"] = src[f"shot_{s:04d}_q95"]; preds[f"shot_{s:04d}_betaN"] = src[f"shot_{s:04d}_betaN"]
        if not all(np.isfinite(preds[f"shot_{s:04d}_{k}"]).all() for k in ("psirz", "q95", "betaN")):
            raise RuntimeError(f"shot {s}: non-finite prediction")
    npz = od / f"diii_d_{FD.TAG}_test.npz"; np.savez_compressed(npz, **preds)
    with zipfile.ZipFile(od / "submission.zip", "w", zipfile.ZIP_STORED) as z:
        z.write(npz, arcname=npz.name)
    info = {"args": vars(a), "post_pass": "li2 (li warp re-applied, cap %.2f)" % a.cap, "frames": int(len(rows)),
            "npz_bytes": npz.stat().st_size, "seconds": round(time.time() - t)}
    (od / "build.json").write_text(json.dumps(info, indent=1)); print("POST_LI2 built", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
