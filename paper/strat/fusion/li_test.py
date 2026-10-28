"""Doc 35 §37o step 2: RZ-head shift + li-avg warp applied ONCE to the local held-out DIII-D TEST split (CPU).

The RZ-head-only pipeline is re-scored first (must reproduce §37n's 0.991553 within 1e-5), then shift + warp.

    .venv/bin/python paper/strat/fusion/li_test.py   ->  runs/strat/fusion/li_test.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

import fx  # noqa: I001  (CPU only)
import axis_shift as AX
import li_warp as LW


def main() -> None:
    sota, d = fx.data()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import ens_select as ES
    import train_hybrid as TH
    from common import AXIS_SIGN
    out_f = fx.OUT / "li_test.json"
    assert not out_f.exists(), "TEST already scored once"
    rec = json.loads((fx.ROOT / "runs" / "sophelio_sota" / "test_scores.json").read_text())["_mlp"]
    prev = json.loads((fx.OUT / "axis_shift_test.json").read_text())["shifted"]["S"]
    test = d.shots_of(2)
    members = [ES.member(str(fx.TRUNKS / m), d, test) for m in rec["members"]]
    comps, mean = d.meta["comps"][:ES.KMAX].numpy(), d.meta["mean"].numpy()
    L, H = sota.LABELS, sota.HEADS
    ir_, iz_, il_ = H.index("efit_r_axis"), H.index("efit_z_axis"), H.index("efit_li")
    R, Z, mask = AX.grid(); sign = AXIS_SIGN["DIII-D"]
    shifter = AX.make_shifter(R, Z, mask, sign); warp = LW.make_li_warp(R, Z, mask, sign)
    gts, base, new = [], [], []
    for s in test:
        rows = np.nonzero(d.shot == s)[0]
        pc = np.mean([mb[s][0] for mb in members], axis=0); sc = np.mean([mb[s][1] for mb in members], axis=0)
        psi = (pc @ comps + mean).reshape(-1, 65, 65).astype(np.float64)
        sh = np.stack([shifter(psi[k], np.array([sc[k, ir_], sc[k, iz_]]), "RZ", "head")[0] for k in range(len(psi))])
        wp = np.stack([warp(sh[k], sc[k, il_])[0] for k in range(len(sh))])
        gts.append(({"psi": d.psi_of(rows).reshape(-1, 65, 65).astype(np.float32),
                     "q95": d.S[rows, L.index("efit_q95")].astype(np.float64),
                     "betaN": d.S[rows, L.index("efit_beta_n")].astype(np.float64)}, d.meta["names"][s]))
        q, b = sc[:, H.index("efit_q95")], sc[:, H.index("efit_beta_n")]
        base.append({"psirz": sh, "q95": q, "betaN": b}); new.append({"psirz": wp, "q95": q, "betaN": b})
    ref = fx.ROOT / "runs" / "sophelio_sota" / "ref_cache"
    rb = TH.official_score(gts, base, ref)
    ok = abs(rb["S"] - prev) < 1e-5
    print(f"[s37o-test] RZ-head re-scored S {rb['S']:.6f} (§37n {prev:.6f}) check {'ok' if ok else 'FAILED'}", flush=True)
    rn = TH.official_score(gts, new, ref)
    res = {"n_test_shots": len(test), "rz_head": rb, "rz_head_li_avg": rn, "check_ok": ok, "delta_S": rn["S"] - rb["S"],
           "delta_vs_entry": rn["S"] - rec["test"]["S"]}
    out_f.write_text(json.dumps(res, indent=1, default=float))
    print(f"S37O_TEST rz_head {rb['S']:.6f} +li_avg {rn['S']:.6f} delta {rn['S'] - rb['S']:+.6f} vs entry {rn['S'] - rec['test']['S']:+.6f} "
          f"li {rb.get('r2_cons_each', {}).get('li', float('nan')):.5f}->{rn.get('r2_cons_each', {}).get('li', float('nan')):.5f}", flush=True)


if __name__ == "__main__":
    main()
