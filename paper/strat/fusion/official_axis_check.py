"""§45n label-free check: does the OFFICIAL O-point finder (fusion_scoring/o_point.py: strongest interior local maximum of
phi = AXIS_SIGN * psi in the 2-cell-eroded envelope) put the axis of our submitted public psi where our GPU twin and our heads do?
A single frame read ~1 m off costs ~0.003 of public Z_axis R2 (SST ~ 193k frames x (0.041 m)^2).
    .venv/bin/python paper/strat/fusion/official_axis_check.py <submission npz> <transfer npz>
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import numpy as np

import fx

SS = fx.ROOT / "baseline" / "sophelio-starter" / "fusion_scoring"
sys.path.insert(0, str(SS))
from o_point import find_o_point  # noqa: E402

ENV = np.load(SS / "masks" / "d3d_envelope.npz"); GR, GZ, MASK = ENV["grid_R"], ENV["grid_Z"], ENV["mask_coarse"].astype(bool)
AXIS_SIGN = -1.0                                                               # common.AXIS_SIGN["DIII-D"]; psi_sign +1 for our files


def one(args):
    sub, key = args
    P = np.load(sub)[key].astype(np.float64); out = np.zeros((len(P), 3))
    for k in range(len(P)):
        Ra, Za, iz, ir, loc = find_o_point(AXIS_SIGN * P[k], GR, GZ, MASK)
        out[k] = Ra, Za, float(loc)
    return key, out


def main() -> None:
    sub, tr = sys.argv[1], sys.argv[2]
    T = np.load(tr); D, H, sh = T["derived"], T["heads"], T["shot"]
    keys = sorted(k for k in np.load(sub).files if k.endswith("_psirz"))
    with mp.Pool(8) as pool:
        res = dict(pool.map(one, [(sub, k) for k in keys], chunksize=8))
    O = np.zeros((len(sh), 3))
    for k in keys:
        s = int(k.split("_")[1]); rows = np.nonzero(sh == s)[0]; assert len(rows) == len(res[k]), (k, len(rows), len(res[k]))
        O[rows] = res[k]
    np.savez(fx.OUT / "official_axis_F14K.npz", official=O)
    sd = np.std(H[:, 1]); n = len(H)
    for lab, ref in (("twin", D), ("heads", H)):
        for j, nm in ((0, "R_axis"), (1, "Z_axis")):
            e = O[:, j] - ref[:, j]; ok = np.isfinite(e)
            big = ok & (np.abs(e) > 0.02)
            print(f"[oa] official - {lab:5s} {nm}: rms {np.sqrt(np.mean(e[ok] ** 2)) * 1e3:.2f} mm | frames |e| > 2 cm: {big.sum()} "
                  f"(shots {len(np.unique(sh[big]))}) | their SSE / SST {np.sum(e[big] ** 2) / (n * np.var(ref[ok, j])):.5f}", flush=True)
    print(f"[oa] fallback to global argmax (no interior local max): {(O[:, 2] == 0).sum()} frames", flush=True)
    print("OA_DONE", flush=True)


if __name__ == "__main__":
    main()
