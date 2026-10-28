"""Where does the fusion consistency error live? (VALID diagnostic before §29e.) The entry ensemble's PCA coefficients
on 100 fixed VALID shots are scored with the official scorer as predicted, and with one band of PCA components replaced
by the truth (the cache's exact PCA-256 coefficients); a full 256-component swap gives the PCA ceiling. Reports S and the
per-scalar consistency R^2 (R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li) for each swap.

    .venv/bin/python paper/sophelio/band_oracle.py  ->  runs/sophelio_sota/band_oracle.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ens_select as ES  # noqa: E402
import sota  # noqa: E402

BANDS = {"pred": None, "0-4": (0, 4), "4-8": (4, 8), "8-16": (8, 16), "16-32": (16, 32), "32-64": (32, 64),
         "64-256": (64, 256), "all256": (0, 256)}
CUMULATIVE = {"0-8": (0, 8), "0-16": (0, 16), "0-32": (0, 32), "4-32": (4, 32), "8-64": (8, 64), "4-64": (4, 64)}


def main() -> None:
    import train_hybrid as TH
    bands = CUMULATIVE if "--cumulative" in sys.argv else BANDS
    data = sota.Data(); valid = data.shots_of(1)
    shots = [valid[i] for i in np.random.default_rng(7).permutation(len(valid))[:100]]      # ens_select's selection set
    sel = json.loads((sota.ROOT / "runs/sophelio_sota/ens_select_mlp.json").read_text())
    members = [ES.member(str(sota.ROOT / "runs/nas/sophelio/trunks" / m), data, shots) for m in sel["members"]]
    comps, mean = data.meta["comps"][:ES.KMAX].numpy(), data.meta["mean"].numpy()
    L, H = sota.LABELS, sota.HEADS
    out = {}
    for name, band in bands.items():
        gts, preds = [], []
        for s in shots:
            rows = np.nonzero(data.shot == s)[0]
            pc = np.mean([mb[s][0] for mb in members], axis=0).copy(); sc = np.mean([mb[s][1] for mb in members], axis=0)
            if band is not None:
                pc[:, band[0]:band[1]] = np.asarray(data.C[rows, band[0]:band[1]])
            gts.append(({"psi": data.psi_of(rows).reshape(-1, 65, 65).astype(np.float32),
                         "q95": data.S[rows, L.index("efit_q95")].astype(np.float64),
                         "betaN": data.S[rows, L.index("efit_beta_n")].astype(np.float64)}, data.meta["names"][s]))
            preds.append({"psirz": (pc @ comps + mean).reshape(-1, 65, 65), "q95": sc[:, H.index("efit_q95")],
                          "betaN": sc[:, H.index("efit_beta_n")]})
        r = TH.official_score(gts, preds, sota.ROOT / "runs" / "sophelio_sota" / "ref_cache")
        out[name] = {"S": r["S"], "consistency": r["consistency"], "r2_psi": r["r2_psi"], "dlcfs": r["dlcfs"],
                     "cons_each": r.get("r2_cons_each")}
        ce = r.get("r2_cons_each") or {}
        print(f"[oracle] {name:7s} S {r['S']:.5f} cons {r['consistency']:.4f} dlcfs {r['dlcfs']:.4f} | "
              + " ".join(f"{k} {v:.3f}" for k, v in ce.items()), flush=True)
    (sota.ROOT / f"runs/sophelio_sota/band_oracle{'_cum' if '--cumulative' in sys.argv else ''}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
