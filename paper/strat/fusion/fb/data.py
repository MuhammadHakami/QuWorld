"""Doc 35 §46: per-frame solver inputs (circuit currents, Ip, TF, Thomson midplane profile) at efit_times, from the shipped parquet
columns. MAST public test: inputs only (no labels exist there). DIII-D: the same inputs plus, for validation only, efit_psirz and the
EFIT scalars of the requested shots (equilibria from another machine; rule 3 allows them).
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from geom import DATA

MAST_EXTRA = ["magnetics_plasma_current", "magnetics_tf_current"]
D3D_EXTRA = ["magnetics_plasma_current", "magnetics_bcoil"]


def files(split: str) -> list:
    return sorted(glob.glob(f"{DATA}/{split}/*.parquet"))


def _interp(t_src, x, t):
    t_src = np.asarray(t_src, np.float64); x = np.asarray(x, np.float64)
    ok = np.isfinite(x) & np.isfinite(t_src)
    return np.interp(t, t_src[ok], x[ok]) if ok.sum() > 1 else np.full(len(t), np.nan)


def thomson_mid(r: pd.Series, t: np.ndarray):
    """MAST core Thomson (radial chord at Z = 0): ne, Te profiles (n_frames, n_ch) at the nearest Thomson time (<= 10 ms away)."""
    tt = np.asarray(r["thomson_core_times"], np.float64)
    ne = np.stack([np.asarray(v, np.float64) for v in r["thomson_core_ne"]]) if len(tt) else np.zeros((0, 0))
    Te = np.stack([np.asarray(v, np.float64) for v in r["thomson_core_Te"]]) if len(tt) else np.zeros((0, 0))
    if not len(tt):
        return None, None, None
    j = np.abs(tt[None] - t[:, None]).argmin(1); far = np.abs(tt[j] - t) > 10.0
    a, b = ne[j], Te[j]; a[far] = np.nan; b[far] = np.nan
    return np.asarray(r["thomson_core_R"], np.float64), a, b


def load(split: str, circuits: list, machine: str, paths: list | None = None, labels: bool = False):
    out = {k: [] for k in ("shot", "t", "I", "Ip", "Bt")}
    if labels:
        out.update({k: [] for k in ("psi", "q95", "betaN", "li")})
    th = []
    for s, f in enumerate(paths or files(split)):
        r = pd.read_parquet(f).iloc[0]; t = np.asarray(r["efit_times"], np.float64)
        tm = np.asarray(r["magnetics_time"], np.float64)
        out["I"].append(np.stack([_interp(tm, r[c], t) for c in circuits], 1))
        tip = np.asarray(r["magnetics_plasma_current_times"], np.float64) if "magnetics_plasma_current_times" in r.index else tm
        out["Ip"].append(_interp(tip, r["magnetics_plasma_current"], t))
        out["Bt"].append(_interp(tm, r["magnetics_tf_current" if machine == "mast" else "magnetics_bcoil"], t))
        out["shot"].append(np.full(len(t), s)); out["t"].append(t)
        if machine == "mast":
            th.append(thomson_mid(r, t))
        if labels:
            out["psi"].append(np.stack([np.stack(p).astype(np.float32) for p in r["efit_psirz"]]))
            for k, c in (("q95", "efit_q95"), ("betaN", "efit_beta_n"), ("li", "efit_li")):
                out[k].append(np.asarray(r[c], np.float64))
    res = {k: np.concatenate(v) for k, v in out.items()}
    if th:
        res["th"] = th
    return res
