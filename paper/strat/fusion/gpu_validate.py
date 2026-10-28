"""Validate gpu_scorer.GPUScorer against the official per-frame values stored for the entry (parts_entry.pkl) on the
180-shot VALID sample. GPU."""
import os, sys, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np, torch
import fx, strategies as ST, axis_shift as AX
from gpu_scorer import GPUScorer
sota, d = fx.data(); shots, _ = ST.sample(); pe, nm = ST.load_entry()
f, comps, mean, H = ST.entry_preds(d, sota)
R, Z, mask = AX.grid(); g = GPUScorer(R, Z, mask)
names = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
off, tw, dd = [], [], []; t = time.time()
for s in shots:
    cs, _ = f(s); psi = torch.tensor((cs.mean(0) @ comps + mean).reshape(-1, 65, 65))
    z = np.load(fx.REF / f"ref_{d.meta['names'][s]}.npz", allow_pickle=True)
    C = [np.asarray(c, float) if c is not None else np.full((512, 2), np.nan) for c in z["contours"]]
    ref = torch.tensor(np.stack([np.pad(c, ((0, 512 - len(c)), (0, 0)), mode="edge") if len(c) < 512 else c[:512] for c in C]))
    o = g.derive(psi, ref, torch.full((len(psi),), float(z["rgeo"])))
    tw.append(torch.stack([o[k] for k in names], 1).cpu().numpy()); off.append(pe[s]["cpred"]); dd.append(np.stack([o["d"].cpu().numpy(), pe[s]["d"]], 1))
T, O, D = np.concatenate(tw), np.concatenate(off), np.concatenate(dd)
print("frames", len(T), "secs", round(time.time() - t, 1))
for j, n in enumerate(names):
    m = np.isfinite(O[:, j]) & np.isfinite(T[:, j])
    print(f"{n:8s} corr {np.corrcoef(T[m, j], O[m, j])[0,1]:.5f}  median|diff| {np.median(np.abs(T[m, j]-O[m, j])):.5f}  p99 {np.percentile(np.abs(T[m, j]-O[m, j]),99):.5f}  sd(official) {O[m, j].std():.4f}  n {m.sum()}")
m = np.isfinite(D).all(1); print(f"d        corr {np.corrcoef(D[m,0], D[m,1])[0,1]:.4f} median|diff| {np.median(np.abs(D[m,0]-D[m,1])):.5f} mean twin {D[m,0].mean():.5f} official {D[m,1].mean():.5f}")
