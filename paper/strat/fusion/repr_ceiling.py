"""Doc 35 §38aa-pre: does our psi representation cap the consistency term? (no model involved)

Our trunks predict psi as mean + the first k PCA components (k = 64 in the entry's trunks), then warps deform it. Here the
TRUE psi of the VALID-sample shots is projected onto the first k components (k = 16 / 32 / 64 / 128 / 256) and scored with
the official composite (true q95 / betaN, so only psi, D_LCFS and consistency move). If the projected truth already loses
consistency, the representation - not the heads or warps - is the ceiling.

    .venv/bin/python paper/strat/fusion/repr_ceiling.py   -> runs/strat/fusion/repr_ceiling.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

import fx


def main() -> None:
    sota, d = fx.data()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import train_hybrid as TH
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); shots = sorted(set(d.shot[lab["rows_valid"]].tolist()))
    comps, mean = d.meta["comps"].numpy().astype(np.float64), d.meta["mean"].numpy().astype(np.float64)
    L = sota.LABELS; gts, P = [], {k: [] for k in ("truth", 16, 32, 64, 128, 256)}
    for s in shots:
        rows = np.nonzero(d.shot == s)[0]; psi = d.psi_of(rows).reshape(len(rows), -1).astype(np.float64)
        q, b = d.S[rows, L.index("efit_q95")].astype(np.float64), d.S[rows, L.index("efit_beta_n")].astype(np.float64)
        gts.append(({"psi": psi.reshape(-1, 65, 65).astype(np.float32), "q95": q, "betaN": b}, d.meta["names"][s]))
        c = (psi - mean) @ comps.T
        for k in P:
            rec = psi if k == "truth" else mean + c[:, :k] @ comps[:k]
            P[k].append({"psirz": rec.reshape(-1, 65, 65).astype(np.float32), "q95": q, "betaN": b})
    rep = {}
    for k, preds in P.items():
        r = TH.official_score(gts, preds, fx.ROOT / "runs" / "sophelio_sota" / "ref_cache")
        rep[str(k)] = {m: float(r[m]) for m in ("S", "r2_psi", "dlcfs", "consistency")}
        each = r.get("cons_each") or {}
        rep[str(k)]["cons_each"] = {a: float(v) for a, v in each.items()} if isinstance(each, dict) else each
        print(f"[ceiling] k={k}: S {r['S']:.5f} r2_psi {r['r2_psi']:.6f} dlcfs {r['dlcfs']:.5f} consistency {r['consistency']:.5f} {rep[str(k)]['cons_each']}", flush=True)
    (fx.OUT / "repr_ceiling.json").write_text(json.dumps(rep, indent=1, default=float))
    print("CEILING_DONE", flush=True)


if __name__ == "__main__":
    main()
