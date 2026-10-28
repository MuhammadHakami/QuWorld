"""Doc 35 §38d step 2: C1 (psi = mean of 3 PSI-A seeds; q95/betaN, axis heads = the 7-member entry) + the frozen
post-processing, scored ONCE on the local held-out DIII-D TEST split. The frozen pipeline (entry psi + the same
post-processing, 6-seed v8 heads) is re-scored first and must reproduce 0.993428 (bw_test_kw_v8_6s.json).

    STRAT_CUDA="" .venv/bin/python paper/strat/fusion/s38d_test.py   ->  runs/strat/fusion/s38d_test.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

import fx  # noqa: I001  (CPU only)
import axis_shift as AX
import boundary_warp as BW
import kappa_warp as KW
import li_warp as LW
from bw_test import head_predict

PSI = ["s38c_PSI-A.pt", "s38c_PSI-A-s1.pt", "s38c_PSI-A-s3.pt"]


def main() -> None:
    sota, d = fx.data()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import ens_select as ES
    import train_hybrid as TH
    from common import AXIS_SIGN
    out_f = fx.OUT / "s38d_test.json"; assert not out_f.exists(), "TEST already scored once"
    rec = json.loads((fx.ROOT / "runs" / "sophelio_sota" / "test_scores.json").read_text())["_mlp"]
    prev = json.loads((fx.OUT / "bw_test_kw_v8_6s.json").read_text())["new"]["S"]
    test = d.shots_of(2)
    ent = [ES.member(str(fx.TRUNKS / m), d, test) for m in rec["members"]]
    psim = [ES.member(str(fx.ROOT / "runs" / "sophelio_sota" / p), d, test) for p in PSI]
    comps, mean = d.meta["comps"][:ES.KMAX].numpy(), d.meta["mean"].numpy()
    L, H = sota.LABELS, sota.HEADS; ir_, iz_ = H.index("efit_r_axis"), H.index("efit_z_axis")
    R, Z, mask = AX.grid(); sign = AXIS_SIGN["DIII-D"]
    sh = AX.make_shifter(R, Z, mask, sign); bw = BW.make_boundary_warp(R, Z, mask, sign)
    kw = KW.make_kappa_warp(R, Z, mask, sign); liw = LW.make_li_warp(R, Z, mask, sign)

    def post(psi, sc, ph):
        out = np.empty_like(psi)
        for k in range(len(psi)):
            a, _ = sh(psi[k], np.array([sc[k, ir_], sc[k, iz_]]), "RZ", "head")
            a, _ = bw(a, ph[k, 7:]); a, _ = kw(a, ph[k, 2]); out[k] = liw(a, ph[k, 6], full=True)[0]
        return out
    gts, base, new = [], [], []
    for s in test:
        rows = np.nonzero(d.shot == s)[0]; ph = head_predict(d, rows, "_v8", list(range(1, 7)))
        sc = np.mean([m[s][1] for m in ent], axis=0)
        pe = (np.mean([m[s][0] for m in ent], axis=0) @ comps + mean).reshape(-1, 65, 65).astype(np.float64)
        pn = (np.mean([m[s][0] for m in psim], axis=0) @ comps + mean).reshape(-1, 65, 65).astype(np.float64)
        q, b = sc[:, H.index("efit_q95")], sc[:, H.index("efit_beta_n")]
        gts.append(({"psi": d.psi_of(rows).reshape(-1, 65, 65).astype(np.float32), "q95": d.S[rows, L.index("efit_q95")].astype(np.float64),
                     "betaN": d.S[rows, L.index("efit_beta_n")].astype(np.float64)}, d.meta["names"][s]))
        base.append({"psirz": post(pe, sc, ph), "q95": q, "betaN": b}); new.append({"psirz": post(pn, sc, ph), "q95": q, "betaN": b})
    ref = fx.ROOT / "runs" / "sophelio_sota" / "ref_cache"
    rb = TH.official_score(gts, base, ref); ok = abs(rb["S"] - prev) < 1e-5
    print(f"[s38d-test] frozen pipeline re-scored S {rb['S']:.6f} (recorded {prev:.6f}) check {'ok' if ok else 'FAILED'}", flush=True)
    rn = TH.official_score(gts, new, ref)
    out_f.write_text(json.dumps({"frozen": rb, "C1post": rn, "check_ok": ok, "delta_S": rn["S"] - rb["S"]}, indent=1, default=float))
    print(f"S38D_TEST frozen {rb['S']:.6f} C1post {rn['S']:.6f} delta {rn['S'] - rb['S']:+.6f} r2_psi {rb['r2_psi']:.6f}->{rn['r2_psi']:.6f} "
          f"cons {rb['consistency']:.5f}->{rn['consistency']:.5f} dlcfs {rb['dlcfs']:.5f}->{rn['dlcfs']:.5f}", flush=True)


if __name__ == "__main__":
    main()
