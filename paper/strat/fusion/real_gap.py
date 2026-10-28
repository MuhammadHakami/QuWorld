"""Doc 35 §42u diagnostic (label-free): per consistency scalar, the gap between the scalar derived from a built public psi and the head
ensemble's prediction. rms gap^2 / head variance = the 1-R2 the realisation alone would cost if the head were exact.
    .venv/bin/python paper/strat/fusion/real_gap.py --sub F6L
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

import fx
from post_li2 import H_F4

NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--sub", required=True); ap.add_argument("--heads", default=H_F4); a = ap.parse_args()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB, axis_shift as AX, submit_public as SP, fold as FD
    from gpu_post import GPUPost
    d = sota.Data(); pub = SP.PublicData(d.meta); shots = pub.shots(); rows = np.arange(len(pub.X))
    hd = GB.heads_gpu(pub, rows, a.heads)[:, :7]
    z = np.load(fx.ROOT / "runs" / "sophelio_sota" / f"submission_{a.sub}" / f"diii_d_{FD.TAG}_test.npz")
    psi = np.concatenate([z[f"shot_{s:04d}_psirz"] for s in shots]); R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask); der = []
    for i in range(0, len(psi), 2048):
        o = gp.g.derive(torch.tensor(psi[i:i + 2048], device="cuda", dtype=torch.float64))
        der.append(np.stack([o[n].cpu().numpy() for n in NAMES], 1))
    der = np.concatenate(der); g = np.nan_to_num(der - hd); tot = 0.0
    for j, n in enumerate(NAMES):
        v = hd[:, j].var(); fl = np.mean(g[:, j] ** 2) / v; tot += fl / 7; big = np.abs(g[:, j]) > 3 * np.std(g[:, j])
        print(f"[gap] {n:8s}: rms gap {np.sqrt(np.mean(g[:, j] ** 2)):.5f}, head sd {np.sqrt(v):.4f}, realisation 1-R2 floor {fl:.5f}; "
              f"frames > 3 sd {big.sum()} carry {np.sum(g[big, j] ** 2) / np.sum(g[:, j] ** 2):.2f} of it", flush=True)
    print(f"[gap] mean-7 realisation floor {tot:.5f} (consistency points; x0.2 = S)", flush=True)


if __name__ == "__main__":
    main()
