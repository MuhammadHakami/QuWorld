"""Doc 35 §38r-pre: are the shape heads' consistency-scalar errors learnable from held-out VALID shots?

Heads: the X7 6-seed J1 ensemble (corrected inputs). Labels: the 7 scorer scalars from the GPU twin's extraction on the
true psi (same extractor for both VALID parts). Train part: the 190 VALID shots outside the 180-shot sample; test part:
the sample. Ridge (closed form, per target, lambda by 5-fold shot CV on the train part) on three feature sets:
  F0 = the head outputs (7 scalars + 64 rays) of the frame
  F1 = F0 + within-shot context: causal and centred rolling means (+-25 frames) of the 7 head scalars, frame position
  F2 = F1 + the top 32 principal components of the corrected magnetics inputs
Also the oracle "subtract each sample shot's mean residual" (the slow-bias share of the error).
Read-out: per-scalar R2 on the sample, head vs head + correction; output runs/strat/fusion/stack_probe.json.

    .venv/bin/python paper/strat/fusion/stack_probe.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

import fx  # noqa: I001
import axis_shift as AX
import gpu_build as GB
import gpu_pipe as GP
import strategies as ST
from gpu_post import GPUPost

HEADS = "+".join(f"head_boundary_v8_J1_s{s}" for s in range(1, 7))
WIN = 25


def labels(d, rows, gp):
    out = []
    for c in GP.chunks(len(rows), 4096):
        psi = torch.tensor(d.psi_of(rows[c]).reshape(-1, 65, 65), device="cuda", dtype=torch.float64)
        o = gp.g.derive(psi); out.append(torch.stack([o[k] for k in GP.NAMES], 1).cpu().numpy())
    return np.concatenate(out)


def context(H, shot):
    """Causal and centred rolling means of the 7 head scalars within each shot, and the frame's relative position."""
    C = np.zeros((len(H), 15))
    for s in np.unique(shot):
        i = np.nonzero(shot == s)[0]; h = H[i, :7]; n = len(i); cs = np.cumsum(np.vstack([np.zeros(7), h]), 0)
        a = np.maximum(np.arange(n) - WIN, 0); b = np.minimum(np.arange(n) + WIN + 1, n)
        C[i, :7] = (cs[np.arange(n) + 1] - cs[a]) / (np.arange(n) + 1 - a)[:, None]
        C[i, 7:14] = (cs[b] - cs[a]) / (b - a)[:, None]; C[i, 14] = np.arange(n) / max(n - 1, 1)
    return C


def ridge_cv(X, Y, groups, lams=(1e-3, 1e-2, 1e-1, 1, 10, 100)):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    ug = np.unique(groups); folds = np.array_split(np.random.default_rng(0).permutation(ug), 5); W = []
    for j in range(Y.shape[1]):
        ok = np.isfinite(Y[:, j]); best = (np.inf, None)
        for lam in lams:
            err = 0.0
            for f in folds:
                te = np.isin(groups, f) & ok; tr = ~np.isin(groups, f) & ok
                A = Xs[tr]; w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ Y[tr, j])
                err += np.sum((Xs[te] @ w - Y[te, j]) ** 2)
            best = min(best, (err, lam))
        A = Xs[ok]; lam = best[1]; W.append(np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ Y[ok, j]))
    return lambda Z: np.hstack([(Z - mu) / sd, np.ones((len(Z), 1))]) @ np.stack(W, 1)


def r2(y, p):
    ok = np.isfinite(y) & np.isfinite(p); return float(1 - np.sum((y[ok] - p[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))


def main() -> None:
    sota, d = fx.data(); smp, _ = ST.sample(); ss = set(smp)
    non = [s for s in d.shots_of(1) if s not in ss]
    parts = {}
    R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask)
    for name, shots in (("train", non), ("sample", list(smp))):
        rows = np.concatenate([fx.valid_rows(d, s) for s in shots]); shot = np.concatenate([np.full(len(fx.valid_rows(d, s)), s) for s in shots])
        H = GB.heads_gpu(d, rows, HEADS); Y = labels(d, rows, gp)
        parts[name] = dict(rows=rows, shot=shot, H=H, Y=Y, C=context(H, shot))
        print(f"[stack] {name}: {len(shots)} shots, {len(rows)} frames", flush=True)
    ck = torch.load(fx.OUT / "head_boundary_v8_J1_s1.pt", map_location="cpu", weights_only=False)
    dd = GB.fixed(d)
    Xin = {k: np.clip((dd.inputs(p["rows"], ck["fm"], ck["edge"]) - ck["mu"]) / ck["sd"], -10, 10) for k, p in parts.items()}
    U, S_, Vt = np.linalg.svd(Xin["train"] - Xin["train"].mean(0), full_matrices=False); P = Vt[:32].T; m0 = Xin["train"].mean(0)
    feats = {"F0": lambda p, k: p["H"], "F1": lambda p, k: np.hstack([p["H"], p["C"]]),
             "F2": lambda p, k: np.hstack([p["H"], p["C"], (Xin[k] - m0) @ P])}
    tr, te = parts["train"], parts["sample"]; Rtr = tr["Y"] - tr["H"][:, :7]; rep = {"names": GP.NAMES}
    rep["head"] = [r2(te["Y"][:, j], te["H"][:, j]) for j in range(7)]
    res = te["Y"] - te["H"][:, :7]; orc = te["H"][:, :7].copy()
    for s in np.unique(te["shot"]):
        i = te["shot"] == s; orc[i] += np.nanmean(res[i], 0)
    rep["oracle_shot_mean"] = [r2(te["Y"][:, j], orc[:, j]) for j in range(7)]
    for f, fn in feats.items():
        model = ridge_cv(fn(tr, "train"), Rtr, tr["shot"]); corr = te["H"][:, :7] + model(fn(te, "sample"))
        rep[f] = [r2(te["Y"][:, j], corr[:, j]) for j in range(7)]
    for k in ("head", "oracle_shot_mean", "F0", "F1", "F2"):
        print(f"[stack] {k:16s} " + " ".join(f"{n} {v:.4f}" for n, v in zip(GP.NAMES, rep[k])) + f" | mean {np.mean(rep[k]):.5f}", flush=True)
    (fx.OUT / "stack_probe.json").write_text(json.dumps(rep, indent=1))
    print("STACK_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
