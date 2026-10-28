"""Doc 35 §41x: does the post-processing transfer the heads into the submitted psi on the public fold? (labels not needed)
Derived scalars of the submitted psi (GPU twin of the official extraction) vs the head ensemble's own predictions, per frame.
    .venv/bin/python paper/strat/fusion/transfer_check.py <submission npz> <'+'-joined heads>  -> runs/strat/fusion/transfer_<tag>.npz
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import fx
import axis_shift as AX
from gpu_scorer import GPUScorer

SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def main() -> None:
    sub, heads, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    import submit_public as SP
    d = sota.Data(); pub = SP.PublicData(d.meta); H = GB.heads_gpu(pub, np.arange(len(pub.X)), heads)[:, :7]
    A = np.load(sub); R, Z, mask = AX.grid(); sc = GPUScorer(R, Z, mask); D = np.zeros((len(pub.X), 7))
    for s in pub.shots():
        rows = np.nonzero(pub.shot == s)[0]; P = torch.tensor(A[f"shot_{s:04d}_psirz"].astype(np.float64), device="cuda")
        for a in range(0, len(rows), 2048):
            o = sc.derive(P[a:a + 2048]); D[rows[a:a + 2048]] = np.stack([o[n].cpu().numpy() for n in SC], 1)
    np.savez(fx.OUT / f"transfer_{tag}.npz", derived=D, heads=H, shot=pub.shot)
    for j, n in enumerate(SC):
        e = D[:, j] - H[:, j]; ok = np.isfinite(e)
        print(f"[transfer] {n:8s} rms(derived - head) {np.sqrt(np.mean(e[ok] ** 2)):.5f}  head sd {np.std(H[ok, j]):.4f}  "
              f"rel {np.sqrt(np.mean(e[ok] ** 2)) / np.std(H[ok, j]):.4f}  p99 |e| {np.percentile(np.abs(e[ok]), 99):.4f}", flush=True)
    print("TRANSFER_DONE", flush=True)


if __name__ == "__main__":
    main()
