"""Doc 35 §41r: TRAIN-only heads on ALL VALID + TEST frames (743 shots, 163,910 frames) instead of the 180-shot VALID sample.

Shape heads: R2 of the 7 consistency scalars (labels boundary_labels_v8_vt.npz) and ray RMSE; q/b heads (mode qb): R2 of q95 /
betaN (cache labels). Each arm is a '+'-joined head ensemble read through gpu_build.heads_gpu (same input wrappers as the build).
Paired read-out vs the first arm: difference of the mean R2 with a shot bootstrap (2,000 resamples, same shots for both arms).

    .venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1TH_s1 head_boundary_v8_J1THP_s1
    .venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QH_s1 head_qb_QHP_s1
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

import fx

SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def shot_sums(shot, y, p):
    """Per shot and target: n, sum y, sum y^2, SSE (finite labels only)."""
    us, inv = np.unique(shot, return_inverse=True); ok = np.isfinite(y) & np.isfinite(p)
    y0, e2 = np.where(ok, y, 0.0), np.where(ok, (p - y) ** 2, 0.0); out = []
    for v in (ok.astype(float), y0, y0 ** 2, e2):
        out.append(np.stack([np.bincount(inv, v[:, j], len(us)) for j in range(y.shape[1])], 1))
    return np.stack(out)                                                     # (4, shots, targets)


def r2(S, w=None):
    n, sy, syy, sse = (S[i].sum(0) if w is None else w @ S[i] for i in range(4))
    return 1 - sse / (syy - sy ** 2 / n)


def main() -> None:
    mode, arms = sys.argv[1], sys.argv[2:]
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    d = sota.Data()
    if mode == "qb":
        rows = np.nonzero(d.split != 0)[0]; y = d.S[rows][:, [sota.LABELS.index("efit_q95"), sota.LABELS.index("efit_beta_n")]].astype(np.float64); names = ["q95", "betaN"]
    else:
        vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); rows, y = vt["rows"], vt["y"].astype(np.float64); names = SC
    nt = len(names); sh = d.shot[rows]; res, sums = {}, []
    for arm in arms:
        p = GB.heads_gpu(d, rows, arm)[:, :y.shape[1]]
        S = shot_sums(sh, y[:, :nt], p[:, :nt]); sums.append(S); r = r2(S)
        res[arm] = {"r2": dict(zip(names, np.round(r, 6).tolist())), "mean": round(float(r.mean()), 6)}
        if mode != "qb":
            e = (p[:, nt:] - y[:, nt:]) ** 2; res[arm]["rays_mm"] = round(float(np.sqrt(np.nanmean(e)) * 1e3), 3)
        print(f"[vt] {arm}: mean {r.mean():.6f} " + " ".join(f"{n} {v:.5f}" for n, v in zip(names, r)) + (f" rays {res[arm].get('rays_mm')} mm" if mode != "qb" else ""), flush=True)
    rng = np.random.default_rng(0); W = rng.multinomial(sums[0].shape[1], np.full(sums[0].shape[1], 1 / sums[0].shape[1]), 2000).astype(float)
    b0 = r2(sums[0], W).mean(1)
    for arm, S in zip(arms[1:], sums[1:]):
        dl = r2(S, W).mean(1) - b0; dp = r2(S).mean() - r2(sums[0]).mean()
        res[arm]["vs_first"] = {"delta": round(float(dp), 6), "ci95": [round(float(np.percentile(dl, 2.5)), 6), round(float(np.percentile(dl, 97.5)), 6)],
                                "per_target": dict(zip(names, np.round(r2(S) - r2(sums[0]), 6).tolist()))}
        print(f"[vt] {arm} vs {arms[0]}: {dp:+.6f} [{np.percentile(dl, 2.5):+.6f}, {np.percentile(dl, 97.5):+.6f}] "
              + " ".join(f"{n} {v:+.5f}" for n, v in zip(names, r2(S) - r2(sums[0]))), flush=True)
    tag = "_".join(a.split("+")[0].replace("head_boundary_v8_", "").replace("head_qb_", "") for a in arms)
    (fx.OUT / f"vt_eval_{mode}_{tag}.json").write_text(json.dumps(res, indent=1)); print("VT_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
