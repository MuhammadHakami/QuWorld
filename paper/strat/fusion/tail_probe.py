"""Doc 35 §38aa-diag: why do kappa / volume / li depart from the heads in a few frames? (GPU, no training)

On the 180-shot VALID sample, rebuild X8's final psi (gpu_build.predict), extract the scalars with the GPU twin, and compare
with the official per-frame values saved by the X8 sample run (parts_GPUS-X8.pkl) and the J1 heads. Tail frames (official
|psi - head| > 0.1 label sd) are classed: twin-official mismatch (twin within 0.02 sd of the head, official not), warp not
applied (twin also off the head), plus the share at the end of the shot (last 10%) and with failed extraction.

    .venv/bin/python paper/strat/fusion/tail_probe.py   -> runs/strat/fusion/tail_probe.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fx  # noqa: E402,I001
import gpu_build as GB  # noqa: E402
import gpu_pipe as GP  # noqa: E402
import strategies as ST  # noqa: E402
import axis_shift as AX  # noqa: E402
from gpu_post import GPUPost  # noqa: E402


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import ens_select as ES
    d = sota.Data(); shots, _ = ST.sample()
    H = "+".join(f"head_boundary_v8_J1_s{s}" for s in range(1, 7))
    class A: pass
    a = A(); a.psi, a.qb, a.heads, a.axis = "psiF", "qh50", H, "trunk"; a.qbheads = "+".join(f"head_qb_QH_s{s}" for s in range(1, 7))
    a.kw = '{"_steps":["shift","bw","kw","xp","li"],"xp":{"sigma":0.15,"cap":0.10}}'
    P = GB.predict(d, shots, a, sota, ES); R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask)
    psi = torch.tensor(np.concatenate([P[s][0] for s in shots]), device="cuda")
    tw, okt = [], []
    for c in GP.chunks(len(psi), 4096):
        o = gp.g.derive(psi[c]); tw.append(torch.stack([o[k] for k in GP.NAMES], 1).cpu().numpy()); okt.append(o["ok"].cpu().numpy())
    tw, okt = np.concatenate(tw), np.concatenate(okt)
    rows = np.concatenate([np.nonzero(d.shot == s)[0] for s in shots]); ph = GB.heads_gpu(d, rows, H)[:, :7]
    Po = pickle.load(open(fx.OUT / "parts_GPUS-X8.pkl", "rb"))
    cp = np.concatenate([Po[s]["cpred"] for s in shots]); cg = np.concatenate([Po[s]["cgt"] for s in shots])
    frac = np.concatenate([np.arange(len(P[s][1])) / max(len(P[s][1]) - 1, 1) for s in shots])
    sd = np.nanstd(np.load(fx.OUT / "boundary_labels_v8.npz")["y_train"][:, :7], 0); rep = {}
    for j, k in enumerate(GP.NAMES):
        go, gt_, eo = np.abs(cp[:, j] - ph[:, j]) / sd[j], np.abs(tw[:, j] - ph[:, j]) / sd[j], np.abs(cp[:, j] - tw[:, j]) / sd[j]
        tail = go > 0.1; mis = tail & (gt_ < 0.02); nap = tail & (gt_ >= 0.02)
        ex = ((cp[:, j] - cg[:, j]) ** 2 - (ph[:, j] - cg[:, j]) ** 2) / sd[j] ** 2; ex = np.where(np.isfinite(ex), ex, 0)
        rep[k] = dict(tail=float(tail.mean()), mismatch=float(mis.mean()), not_applied=float(nap.mean()),
                      excess_mismatch=float(ex[mis].sum() / max(ex.sum(), 1e-12)), excess_not_applied=float(ex[nap].sum() / max(ex.sum(), 1e-12)),
                      tail_end_share=float((frac[tail] >= 0.9).mean()) if tail.any() else 0.0, tail_twin_fail=float((~okt[tail]).mean()) if tail.any() else 0.0,
                      twin_off_q50=float(np.nanmedian(eo)), twin_off_q99=float(np.nanquantile(eo, 0.99)))
        print(f"[tail] {k:8s} " + " ".join(f"{a_} {b:.3f}" for a_, b in rep[k].items()), flush=True)
    (fx.OUT / "tail_probe.json").write_text(json.dumps(rep, indent=1))
    print("TAIL_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
