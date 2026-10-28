"""TokaMark 1-1: is the beta error near shot termination a learnable bias or EFIT label noise? (registered Sep 25 07:56)

    .venv/bin/python paper/tokamark/beta_diag.py runs/tokamark/preds/vanilla_s1337_scratch_truth.npz

TEST-derived diagnostic, nothing is fitted. Per beta channel:
  R    = sum_bins n_b mu_b^2 / sum e^2 over bins (shot-length quartile x relative time in shot, deciles): the in-sample
         upper bound on the MSE share any additive correction conditioned on those bins could remove (optimistic);
  jump = Var(dy_true) / Var(dy_pred) along each shot (consecutive windows): >> 1 means the labels jump where the
         model is smooth, i.e. label noise.
Decision rule (fixed before the numbers): co-processor only if R >= 0.25 in the short-shot quartile AND jump <= 2.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

CH = ("equilibrium-beta_pol", "equilibrium-beta_normal", "equilibrium-beta_tor")


def load(path):
    rec = np.load(path, allow_pickle=True)["rec"].item()
    shots = defaultdict(list)
    for key, d in rec.items():
        s, w = key.split("|")
        shots[s].append((int(w), d))
    return {s: sorted(v, key=lambda t: t[0]) for s, v in shots.items()}


def channel(shots, ch):
    lens = {s: len(v) for s, v in shots.items()}
    q = np.quantile(list(lens.values()), [0.25, 0.5, 0.75])
    rows, jt, jp = [], {}, {}
    for s, v in shots.items():
        yt = np.array([float(np.ravel(d[ch + "#true"])[0]) for _, d in v])
        yp = np.array([float(np.ravel(d[ch])[0]) for _, d in v])
        ok = np.isfinite(yt) & np.isfinite(yp)
        quart = int(np.searchsorted(q, lens[s], side="left"))
        rel = np.minimum((np.arange(len(v)) / max(len(v), 1) * 10).astype(int), 9)
        for e, dec in zip((yp - yt)[ok], rel[ok]):
            rows.append((quart, dec, e))
        if ok.sum() > 2:
            jt.setdefault(quart, []).append(np.diff(yt[ok])); jp.setdefault(quart, []).append(np.diff(yp[ok]))
    Q, D, E = (np.array(c) for c in zip(*rows))
    out = {}
    for quart in range(4):
        m = Q == quart
        e, d = E[m], D[m]
        num = sum((d == b).sum() * e[d == b].mean() ** 2 for b in range(10) if (d == b).any())
        out[f"q{quart}"] = {"n": int(m.sum()), "rmse": float(np.sqrt((e ** 2).mean())), "B_global": float(e.mean() ** 2 / (e ** 2).mean()),
                            "R_binned": float(num / (e ** 2).sum()),
                            "jump": float(np.var(np.concatenate(jt[quart])) / np.var(np.concatenate(jp[quart])))}
    return {"shot_len_quartiles": q.tolist(), **out}


def main():
    shots = load(sys.argv[1])
    res = {ch: channel(shots, ch) for ch in CH}
    short = [res[ch]["q0"] for ch in CH]
    res["decision"] = "co-processor" if all(r["R_binned"] >= 0.25 and r["jump"] <= 2 for r in short) else "label-noise: keep Scratch"
    out = pathlib.Path("runs/tokamark/results/beta_diag.json"); out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
