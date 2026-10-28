"""Weakness map of the entry ensemble on the 180-shot VALID sample: where the psi error lives.
Per PCA coefficient (squared error share, R^2), truncation beyond K=64, time-in-shot deciles and shot concentration.

    .venv/bin/python paper/strat/fusion/weak_coef.py
"""
from __future__ import annotations

import numpy as np

import fx  # noqa: I001
import strategies as ST


def main() -> None:
    sota, d = fx.data(); shots, grp = ST.sample(); f, comps, mean, H = ST.entry_preds(d, sota)
    E, T, P, shot_err = [], [], [], {}
    for s in shots:
        rows = fx.valid_rows(d, s); cs, _ = f(s); c = cs.mean(0)
        ct = np.asarray(d.C[rows], np.float64); e = c - ct[:, :c.shape[1]]
        trunc = (ct[:, c.shape[1]:] ** 2).sum(1)
        E.append(e); T.append(trunc); P.append(d.edge_feats()[rows, 1])
        shot_err[s] = (e ** 2).sum()
    E = np.concatenate(E); T = np.concatenate(T); P = np.concatenate(P)
    tot = (E ** 2).sum() + T.sum()
    var = np.asarray(d.C[np.concatenate([fx.valid_rows(d, s) for s in shots])][:, :E.shape[1]], np.float64).var(0)
    share = (E ** 2).sum(0) / tot
    print(f"[weak] frames {len(E)}  truncation share {T.sum() / tot:.4f}")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 8), (8, 16), (16, 32), (32, 64)):
        print(f"[weak] coef [{lo},{hi}) error share {share[lo:hi].sum():.4f}  R2 " +
              " ".join(f"{1 - (E[:, k] ** 2).mean() / var[k]:.5f}" for k in range(lo, min(hi, lo + 4))))
    fe = (E ** 2).sum(1) + T
    for q in range(10):
        m = (P >= q / 10) & (P < (q + 1) / 10 + (q == 9))
        print(f"[weak] time decile {q}: frames {m.sum():5d} mean sq err {fe[m].mean():.4g} share {fe[m].sum() / fe.sum():.3f}")
    se = np.array(sorted(shot_err.values()))[::-1]
    print(f"[weak] shot concentration: top 5% shots {se[:max(1, len(se) // 20)].sum() / se.sum():.3f}, top 20% {se[:len(se) // 5].sum() / se.sum():.3f}")
    fs = np.sort(fe)[::-1]
    print(f"[weak] frame concentration: top 1% frames {fs[:len(fs) // 100].sum() / fs.sum():.3f}, top 10% {fs[:len(fs) // 10].sum() / fs.sum():.3f}")


if __name__ == "__main__":
    main()
