"""Fusion backbone / head / transfer decomposition of the consistency term (user: analyse each part to find exactly where the score is lost).
The score's consistency term = mean over 7 scalars f_j (R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li) of the pooled R^2 of
f_j(psi_pred) vs f_j(psi_gt). Our pipeline: psi trunk (backbone) -> shape heads predict f_j directly -> post-processing edits psi toward the
head values (transfer). Per scalar on the VT shots (VALID + TEST, never trained on by the TRAIN-only heads used here):
  trunk    R^2 of f_j(raw trunk psi)            (parts_VT-<raw arm>.pkl: cpred / cgt per frame)
  head     R^2 of the head ensemble's f_j       (TRAIN-only heads: head_boundary_v8_<stem>_s<k>.npz, rows_valid / pred_valid)
  final    R^2 of f_j(edited psi)               (parts_VT-<final arm>.pkl)
  transfer R^2 of f_j(edited psi) vs the head value (how faithfully the edit imposes the head)
and for the final residual: the per-shot offset share (between-shot variance / total) and the residual by shot phase (first 10%, middle,
last 10% of each shot's frames).
    python runs/strat/fusion/stage_decomp.py [raw arm] [final arm]
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

OUT = Path("runs/strat/fusion"); SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
RAW = sys.argv[1] if len(sys.argv) > 1 else "PQmix~g_ns"; FIN = sys.argv[2] if len(sys.argv) > 2 else "PQmix~g_VBxhe10e"
HEADS = (sys.argv[3].split(",") if len(sys.argv) > 3 else ["J1RL_s1", "J1QH_s1", "J1QH_s2", "J1QH_s3", "J1QRL_s1", "J1QRL_s2", "J1QRL_s3"])
shot_of = np.load("data/sophelio_proc/cache_k256/shot.npy", mmap_mode="r")
raw, fin = pickle.load(open(OUT / f"parts_VT-{RAW}.pkl", "rb")), pickle.load(open(OUT / f"parts_VT-{FIN}.pkl", "rb"))
shots = sorted(fin)
hv = [np.load(OUT / f"head_boundary_v8_{h}.npz") for h in HEADS if (OUT / f"head_boundary_v8_{h}.npz").exists()]
rows_v = hv[0]["rows_valid"]; assert all(np.array_equal(z["rows_valid"], rows_v) for z in hv)
hpred = np.mean([z["pred_valid"][:, :7] for z in hv], 0); pos = {int(r): i for i, r in enumerate(rows_v)}

G, PR, PF, PH, SH, PH_ = [], [], [], [], [], []
for s in shots:
    rows = np.nonzero(np.asarray(shot_of) == s)[0]; n = len(fin[s]["cgt"])
    assert len(rows) == n == len(raw[s]["cgt"]), (s, len(rows), n)
    idx = np.array([pos.get(int(r), -1) for r in rows]); h = np.where(idx[:, None] >= 0, hpred[np.maximum(idx, 0)], np.nan)
    G.append(fin[s]["cgt"]); PF.append(fin[s]["cpred"]); PR.append(raw[s]["cpred"]); PH.append(h); SH.append(np.full(n, s))
    PH_.append(np.arange(n) / max(n - 1, 1))
G, PF, PR, PH, SH, PHASE = (np.concatenate(v) for v in (G, PF, PR, PH, SH, PH_))
r2 = lambda p, g: 1 - np.nansum((p - g) ** 2) / np.nansum((g - np.nanmean(g)) ** 2)  # noqa: E731
res = {"raw_arm": RAW, "final_arm": FIN, "heads": HEADS[:len(hv)], "n_frames": int(len(G)), "n_shots": len(shots), "scalars": {}}
print(f"[decomp] VT {len(shots)} shots, {len(G)} frames; raw arm {RAW}, final arm {FIN}, heads {HEADS[:len(hv)]}", flush=True)
print(f"[decomp] {'scalar':>8} | {'trunk':>7} {'head':>7} {'final':>7} | {'final-vs-head':>13} | offset share | phase RMSE first10% mid last10%", flush=True)
for j, nm in enumerate(SC):
    g, pf, pr, ph = G[:, j], PF[:, j], PR[:, j], PH[:, j]
    ok = np.isfinite(g) & np.isfinite(pf) & np.isfinite(pr) & np.isfinite(ph)
    e = (pf - g)[ok]; sh = SH[ok]; us, inv = np.unique(sh, return_inverse=True)
    off = np.bincount(inv, e) / np.bincount(inv); between = np.sum((off[inv]) ** 2) / np.sum(e ** 2)       # share of SSE from per-shot means
    ph_ = PHASE[ok]; rm = lambda m: float(np.sqrt(np.mean(e[m] ** 2)))  # noqa: E731
    row = {"trunk": float(r2(pr[ok], g[ok])), "head": float(r2(ph[ok], g[ok])), "final": float(r2(pf[ok], g[ok])),
           "final_vs_head": float(r2(pf[ok], ph[ok])), "offset_share": float(between),
           "rmse_phase": [rm(ph_ < 0.1), rm((ph_ >= 0.1) & (ph_ <= 0.9)), rm(ph_ > 0.9)], "sd_gt": float(np.std(g[ok]))}
    dh, df = (ph - pr)[ok], (pf - pr)[ok]                                     # head - trunk gap, what the edit moved
    for lab_, m in (("all", np.ones(len(dh), bool)), ("first10", ph_ < 0.1), ("mid", (ph_ >= 0.1) & (ph_ <= 0.9)), ("last10", ph_ > 0.9)):
        sl_ = float(np.sum(dh[m] * df[m]) / max(np.sum(dh[m] ** 2), 1e-30))
        rh = float(np.sqrt(np.mean(((ph - g)[ok][m]) ** 2))); rf = float(np.sqrt(np.mean(((pf - g)[ok][m]) ** 2)))
        row[f"transfer_{lab_}"] = {"slope": sl_, "rmse_head": rh, "rmse_final": rf}
    big = np.abs(dh) > np.quantile(np.abs(dh), 0.9)                         # frames where the head disagrees most with the trunk
    row["transfer_top10gap"] = float(np.sum(dh[big] * df[big]) / np.sum(dh[big] ** 2))
    res["scalars"][nm] = row
    print(f"[decomp] {nm:>8} | {row['trunk']:.5f} {row['head']:.5f} {row['final']:.5f} | {row['final_vs_head']:13.5f} | {between:12.2f} | "
          + " ".join(f"{v:.4f}" for v in row["rmse_phase"]) + f" (sd {row['sd_gt']:.4f})", flush=True)
print("[decomp] transfer slope (edit move / head-trunk gap; 1 = head imposed) and RMSE head -> final by phase:", flush=True)
for nm in SC:
    r_ = res["scalars"][nm]
    print(f"[decomp] {nm:>8}: " + " | ".join(f"{k} slope {r_[f'transfer_{k}']['slope']:.2f} rmse {r_[f'transfer_{k}']['rmse_head']:.4f}->{r_[f'transfer_{k}']['rmse_final']:.4f}"
                                          for k in ("all", "first10", "mid", "last10")) + f" | top-10% gap frames slope {r_['transfer_top10gap']:.2f}", flush=True)
for k in ("trunk", "head", "final"):
    print(f"[decomp] mean over 7 ({k}): {np.mean([res['scalars'][n][k] for n in SC]):.5f}", flush=True)
json.dump(res, open(OUT / f"stage_decomp_{FIN}_{HEADS[0].split('_')[0]}.json", "w"), indent=1); print("STAGE_DECOMP_DONE", flush=True)
