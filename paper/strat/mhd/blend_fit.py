"""Doc 35 §37h step 2-3: fit the registered two-backbone band blends on the final-Flowers TRAIN shards, score them on
the VALID shards, pick by VALID, and save the chosen blend's full weights (CPU only, exact via Parseval).

Candidates (members; weights per group, least squares on TRAIN):
  FC_ch_band     F + C            (raw)  per channel x band
  FCP_Ma_ch_band F + C + P        (raw)  per Alfven regime x channel x band
  *_dfc          the same on the divergence-free members (F_raw_df, C_raw_df) with the k = 0 band of density and B
                 taken from persistence (conserved means), i.e. the §37e projections applied to the blend
Each is fit once on raw (or df) single-pass members and, for the daggered row, refit on the shg8 members.

    STRAT_SHARD_DIR=shards_final uv run --no-sync python paper/strat/mhd/blend_fit.py  ->  runs/strat/mhd/blend_final.json
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402

CONS_CH = [0, 1, 2, 3]                                   # density, Bx, By, Bz: means conserved on the periodic box


def candidates(suffix: str, base: str = "raw"):
    df = "_df" if suffix == "df" else ""
    F, C = f"F_{base}{df}", f"C_{base}{df}"
    return {"FC_ch_band": (F, [C], "ch_band"), "FCP_Ma_ch_band": (F, [C, "P"], "Ma_ch_band")}


def group_fn(name, R):
    return {"Ma_ch_band": lambda i, c, b: (R[i], c, b), "ch_band": lambda i, c, b: (c, b)}[name]


def weights(D, fit, gname, R, cons: bool):
    W = A.apply_theta(D, fit, group_fn(gname, R))
    if cons:
        W[:, :, CONS_CH, 0] = 0.0; W[:, D["idx"]["P"], CONS_CH, 0] = 1.0
    return W


def main() -> None:
    T, V = A.load("train", "trainlite"), A.load("valid", "full")
    RT, RV = A.regime(T), A.regime(V)
    out = {"shard_dir": os.environ.get("STRAT_SHARD_DIR", "shards"), "candidates": {}}
    for base, suffix in (("raw", "raw"), ("raw", "df"), ("shg8", "raw"), ("shg8", "df")):
        for cname, (anchor, free, gname) in candidates(suffix, base).items():
            fit = A.fit_theta(T, anchor, free, group_fn(gname, RT))
            for cons in ((False,) if suffix == "raw" else (True,)):
                name = f"{cname}{'_dfc' if cons else ''}{'' if base == 'raw' else '__shg8'}"
                vt = A.vr_from_mse(T, A.mse_of(T, weights(T, fit, gname, RT, cons)))
                vv = A.vr_from_mse(V, A.mse_of(V, weights(V, fit, gname, RV, cons)))
                out["candidates"][name] = {"members": [anchor] + free, "group": gname, "cons": cons,
                                           "train": float(vt.mean()), "valid": float(vv.mean()),
                                           "valid_Ma": [float(vv[RV == r].mean()) for r in (0, 1)],
                                           "theta": {json.dumps([int(x) if not isinstance(x, str) else x for x in k]): v.tolist()
                                                     for k, v in fit["theta"].items()}}
    for ref in ("F_raw", "F_raw_df"):
        out[f"valid_{ref}"] = float(A.vr_variant(V, ref).mean())
    single = [k for k in out["candidates"] if not k.endswith("__shg8")]
    best = min(single, key=lambda k: out["candidates"][k]["valid"])          # the headline: single pass per backbone
    out["picked"] = best; out["picked_shg8"] = best + "__shg8"
    (A.OUT / os.environ.get("STRAT_BLEND_OUT", "blend_final.json")).write_text(json.dumps(out, indent=1))
    for k, c in out["candidates"].items():
        print(f"{k:22s} train {c['train']:.4f} valid {c['valid']:.4f} (Ma0.7 {c['valid_Ma'][0]:.4f}, Ma2 {c['valid_Ma'][1]:.4f})")
    print(f"F_raw valid {out['valid_F_raw']:.4f}  F_raw_df {out['valid_F_raw_df']:.4f}  picked {best}")


if __name__ == "__main__":
    main()
