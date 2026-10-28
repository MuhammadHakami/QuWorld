"""Doc 35 §45i diagnostic: how much does the shape-head seed ensemble still gain at the production size?
The six TRAIN-only J1 seeds predict every VT frame (all VALID + TEST shots); for each ensemble size k = 1..6 the mean-7 R2 (and the
per-scalar R2) is averaged over all k-subsets, then fitted as R2(k) = R2_inf - c / k (seed variance shrinks as 1 / k). The fit gives the
gain of going from the 8 all-data J1RLA seeds in F14K to 16 (assumes J1RLA's seed variance ~ J1's; both share the recipe).
A family prefixed 'qb:' reads q/b heads (head_qb_<family>_s<n>) against the q95 / betaN labels of every VT frame instead.
    .venv/bin/python paper/strat/fusion/seed_curve.py [family] [n_seeds]   -> runs/strat/fusion/seed_curve_<family>.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"                                          # before fx (which hides the GPU)
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fx  # noqa: E402

CONS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def r2(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = []
    for j in range(y.shape[1]):
        ok = np.isfinite(y[:, j]); e = p[ok, j] - y[ok, j]
        out.append(1 - np.mean(e ** 2) / y[ok, j].var())
    return np.array(out)


def main() -> None:
    fam = sys.argv[1] if len(sys.argv) > 1 else "J1"; ns = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    d = sota.Data(); qb = fam.startswith("qb:"); fam = fam.removeprefix("qb:")
    if qb:
        names = ["q95", "betaN"]; rvt = np.nonzero(d.split != 0)[0]
        y = d.S[rvt][:, [sota.LABELS.index("efit_q95"), sota.LABELS.index("efit_beta_n")]].astype(np.float64); stem = "head_qb_"
    else:
        names = CONS; vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); keep = d.split[vt["rows"]] != 0
        rvt = vt["rows"][keep]; y = vt["y"][keep, :7].astype(np.float64); stem = "head_boundary_v8_"
    P = [GB.heads_gpu(d, rvt, f"{stem}{fam}_s{s}")[:, :len(names)].astype(np.float64) for s in range(1, ns + 1)]
    res = {"family": fam, "seeds": ns, "frames": int(len(rvt)), "k": {}}
    for k in range(1, ns + 1):
        rs = np.array([r2(np.mean([P[i] for i in c], 0), y) for c in itertools.combinations(range(ns), k)])
        res["k"][k] = {"mean7": float(rs.mean(1).mean()), "per": dict(zip(names, rs.mean(0).round(6).tolist())), "n_subsets": len(rs)}
        print(f"[seeds] {fam} k={k}: mean {rs.mean(1).mean():.6f} (over {len(rs)} subsets) "
              + " ".join(f"{n} {v:.5f}" for n, v in zip(names, rs.mean(0))), flush=True)
    ks = np.arange(1, ns + 1); m = np.array([res["k"][k]["mean7"] for k in ks])
    A = np.stack([np.ones_like(ks, dtype=float), -1.0 / ks], 1); (r_inf, c), *_ = np.linalg.lstsq(A, m, rcond=None)
    res["fit"] = {"r2_inf": float(r_inf), "c": float(c), "gain_8_to_16": float(c * (1 / 8 - 1 / 16)),
                  "max_abs_resid": float(np.abs(A @ np.array([r_inf, c]) - m).max())}
    print(f"[seeds] fit R2(k) = {r_inf:.6f} - {c:.6f} / k (max resid {res['fit']['max_abs_resid']:.1e}); "
          f"8 -> 16 seeds: +{c * (1 / 8 - 1 / 16):.6f} mean-7", flush=True)
    (fx.OUT / f"seed_curve_{'qb_' if qb else ''}{fam}.json").write_text(json.dumps(res, indent=1)); print("SEED_CURVE_DONE", flush=True)


if __name__ == "__main__":
    main()
