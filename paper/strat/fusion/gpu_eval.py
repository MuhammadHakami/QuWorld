"""GPU end-to-end Fusion evaluation on the 180-shot VALID sample: psi pipeline -> per-frame parts (gpu_scorer twin)
-> fx.composite / ST.compare. All frames batched (gpu_pipe). Validation target: the frozen pipeline's official sample S 0.99431
(parts_BW-Q-bkl3_v8_6s.pkl).

    .venv/bin/python paper/strat/fusion/gpu_eval.py [entry|frozen] [official]
official: also score the GPU-built psi with the official CPU scorer (fx.frame_parts, STRAT_WORKERS processes).
"""
from __future__ import annotations

import os
import pickle
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fx  # noqa: E402,I001
import strategies as ST  # noqa: E402
import axis_shift as AX  # noqa: E402
from gpu_post import GPUPost  # noqa: E402
import gpu_pipe as GP  # noqa: E402


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "frozen"
    sota, d = fx.data(); shots, grp = ST.sample(); pe, nm = ST.load_entry()
    f, comps, mean, H = ST.entry_preds(d, sota)
    iq, ib, ir_, iz_ = (H.index(h) for h in ("efit_q95", "efit_beta_n", "efit_r_axis", "efit_z_axis"))
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); pos = {int(r): i for i, r in enumerate(lab["rows_valid"])}
    Ph = np.mean([np.load(fx.OUT / f"head_boundary_v8_s{s}.npz")["pred_valid"] for s in range(1, 7)], 0).astype(np.float64)
    R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask); t = time.time()
    rows = np.concatenate([fx.valid_rows(d, s) for s in shots])
    C, S = zip(*(f(s) for s in shots)); C = np.concatenate([c.mean(0) for c in C]); S = np.concatenate([x.mean(0) for x in S])
    ph = torch.tensor(Ph[[pos[int(r)] for r in rows]], device=gp.dev)
    psi = torch.tensor((C @ comps + mean).reshape(-1, 65, 65), device=gp.dev)
    if mode == "frozen":
        psi = GP.post_all(gp, psi, torch.tensor(S[:, [ir_, iz_]], device=gp.dev, dtype=torch.float64), ph[:, 7:], ph[:, 2], ph[:, 6])
    parts = GP.parts_all(gp, psi, S[:, iq].astype(np.float64), S[:, ib].astype(np.float64), d, shots)
    if "official" in sys.argv[2:]:
        P = psi.cpu().numpy().reshape(len(psi), -1); off = np.cumsum([0] + [len(fx.valid_rows(d, s)) for s in shots])
        sl = {s: slice(off[i], off[i + 1]) for i, s in enumerate(shots)}
        po = fx.frame_parts(d, shots, lambda s: (P[sl[s]], S[sl[s], iq], S[sl[s], ib]), np.asarray(nm["c_mean"]))
        pickle.dump(po, open(fx.OUT / f"parts_GPU-{mode}-official.pkl", "wb"))
        c = fx.composite(np.stack([fx.shot_sums(po[s]) for s in shots]), nm)
        print(f"[gpu-eval] {mode} GPU-built psi, official scorer: S {c['S']:.5f} r2_psi {c['r2_psi']:.6f} dlcfs {c['dlcfs']:.5f} "
              f"cons {c['consistency']:.5f} | {time.time() - t:.0f}s", flush=True)
    c = fx.composite(np.stack([fx.shot_sums(parts[s]) for s in shots]), nm)
    print(f"[gpu-eval] {mode}: S {c['S']:.5f} r2_psi {c['r2_psi']:.6f} dlcfs {c['dlcfs']:.5f} cons {c['consistency']:.5f} | {time.time() - t:.0f}s", flush=True)
    ref = pe if mode == "entry" else pickle.load(open(fx.OUT / "parts_BW-Q-bkl3_v8_6s.pkl", "rb"))
    co = fx.composite(np.stack([fx.shot_sums(ref[s]) for s in shots]), nm)
    print(f"[gpu-eval] official CPU {mode}: S {co['S']:.5f} r2_psi {co['r2_psi']:.6f} dlcfs {co['dlcfs']:.5f} cons {co['consistency']:.5f}", flush=True)
    pickle.dump(parts, open(fx.OUT / f"parts_GPU-{mode}.pkl", "wb"))


if __name__ == "__main__":
    main()
