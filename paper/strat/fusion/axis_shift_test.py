"""Doc 35 §37n step 2: the VALID-picked axis-shift arm applied ONCE to the local held-out DIII-D TEST split (CPU).

Entry = ens_select_mlp.json (7 members, sigma 0; its TEST S 0.991017 is in runs/sophelio_sota/test_scores.json). The
unshifted entry is re-scored through the same path first (must reproduce 0.991017 within 1e-5), then the picked arm.

    .venv/bin/python paper/strat/fusion/axis_shift_test.py <arm>   ->  runs/strat/fusion/axis_shift_test.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

import fx  # noqa: I001  (CPU only)
import axis_shift as AX


def main() -> None:
    arm = sys.argv[1]; axes, mode = AX.ARMS[arm]
    sota, d = fx.data()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import ens_select as ES
    import train_hybrid as TH
    from common import AXIS_SIGN
    out_f = fx.OUT / "axis_shift_test.json"
    assert not out_f.exists(), "TEST already scored once"
    rec = json.loads((fx.ROOT / "runs" / "sophelio_sota" / "test_scores.json").read_text())["_mlp"]
    test = d.shots_of(2)
    members = [ES.member(str(fx.TRUNKS / m), d, test) for m in rec["members"]]
    comps, mean = d.meta["comps"][:ES.KMAX].numpy(), d.meta["mean"].numpy()
    L, H = sota.LABELS, sota.HEADS
    ir_, iz_ = H.index("efit_r_axis"), H.index("efit_z_axis")
    R, Z, mask = AX.grid(); shifter = AX.make_shifter(R, Z, mask, AXIS_SIGN["DIII-D"])
    gts, base, shifted = [], [], []
    for s in test:
        rows = np.nonzero(d.shot == s)[0]
        pc = np.mean([mb[s][0] for mb in members], axis=0); sc = np.mean([mb[s][1] for mb in members], axis=0)
        psi = (pc @ comps + mean).reshape(-1, 65, 65).astype(np.float64)
        sh = np.stack([shifter(psi[k], np.array([sc[k, ir_], sc[k, iz_]]), axes, mode)[0] for k in range(len(psi))])
        gts.append(({"psi": d.psi_of(rows).reshape(-1, 65, 65).astype(np.float32),
                     "q95": d.S[rows, L.index("efit_q95")].astype(np.float64),
                     "betaN": d.S[rows, L.index("efit_beta_n")].astype(np.float64)}, d.meta["names"][s]))
        q, b = sc[:, H.index("efit_q95")], sc[:, H.index("efit_beta_n")]
        base.append({"psirz": psi, "q95": q, "betaN": b}); shifted.append({"psirz": sh, "q95": q, "betaN": b})
    ref = fx.ROOT / "runs" / "sophelio_sota" / "ref_cache"
    rb = TH.official_score(gts, base, ref)
    ok = abs(rb["S"] - rec["test"]["S"]) < 1e-5
    print(f"[s37n-test] entry re-scored S {rb['S']:.6f} (recorded {rec['test']['S']:.6f}) check {'ok' if ok else 'FAILED'}", flush=True)
    rs = TH.official_score(gts, shifted, ref)
    res = {"arm": arm, "n_test_shots": len(test), "entry": rb, "shifted": rs, "check_ok": ok, "delta_S": rs["S"] - rb["S"]}
    out_f.write_text(json.dumps(res, indent=1, default=float))
    print(f"S37N_TEST arm={arm} entry {rb['S']:.6f} shifted {rs['S']:.6f} delta {rs['S'] - rb['S']:+.6f} "
          f"cons {rb['consistency']:.5f}->{rs['consistency']:.5f} dlcfs {rb['dlcfs']:.5f}->{rs['dlcfs']:.5f}", flush=True)


if __name__ == "__main__":
    main()
