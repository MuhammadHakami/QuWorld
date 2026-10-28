"""Doc 35 §47 read-out: out-of-fold single-model R2 and the Lean-certified blend statistics of each arm against the control.

Per fold and consistency scalar (formal/QuotaScaling/CompetitionBlend.lean): u = control error, v = arm error on the held-out
frames, A = |u|^2, B = |v|^2, C = <u, v>; the arm helps at some positive weight iff C < A; best weight w* = (A - C) / (A + B - 2C),
best SSE A - (A - C)^2 / (A + B - 2C). Also the fixed 50/50 blend (no fitted weight). R2 = 1 - SSE / SST per target, mean of 7.

    .venv/bin/python paper/strat/fusion/quota_blend.py M Tg Hg Tp      -> one line per arm and fold + runs/strat/fusion/quota/blend.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

import fx  # noqa: I001

Q = fx.OUT / "quota"


def stats(y, pu, pv):
    out = []
    for k in range(min(7, y.shape[1])):
        m = np.isfinite(y[:, k]); yy = y[m, k].astype(np.float64); sst = ((yy - yy.mean()) ** 2).sum()
        u = pu[m, k] - yy; v = pv[m, k] - yy; A, B, C = (u * u).sum(), (v * v).sum(), (u * v).sum()
        D = A + B - 2 * C; best = A - (A - C) ** 2 / D if D > 0 else A
        out.append({"r2_ctrl": 1 - A / sst, "r2_arm": 1 - B / sst, "r2_half": 1 - ((u + v) ** 2).sum() / 4 / sst, "r2_best": 1 - best / sst,
                    "w_best": (A - C) / D if D > 0 else 0.0, "helps": bool(C < A), "rho": C / np.sqrt(A * B)})
    return out


def main() -> None:
    ctrl, arms = sys.argv[1], sys.argv[2:]; res = {}
    for a in arms:
        res[a] = []
        for f in range(3):
            pc, pa = Q / f"{ctrl}_f{f}.npz", Q / f"{a}_f{f}.npz"
            if not (pc.exists() and pa.exists()):
                continue
            zc, za = np.load(pc), np.load(pa); assert (zc["rows"] == za["rows"]).all()
            s = stats(zc["y"], zc["pred"], za["pred"]); mean = {k: float(np.mean([t[k] for t in s])) for k in ("r2_ctrl", "r2_arm", "r2_half", "r2_best", "rho")}
            res[a].append({"fold": f, "mean": mean, "per_scalar": s})
            print(f"QF_BLEND {a} vs {ctrl} fold {f}: mean7 ctrl {mean['r2_ctrl']:.5f} arm {mean['r2_arm']:.5f} ({mean['r2_arm'] - mean['r2_ctrl']:+.5f}) "
                  f"50/50 {mean['r2_half'] - mean['r2_ctrl']:+.5f} best {mean['r2_best'] - mean['r2_ctrl']:+.5f} rho {mean['rho']:.3f} | w* "
                  + " ".join(f"{t['w_best']:.2f}" for t in s), flush=True)
    (Q / "blend.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
