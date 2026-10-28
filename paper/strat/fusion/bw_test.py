"""Doc 35 §37q step 2: the VALID-picked boundary-warp arm applied ONCE to the local held-out DIII-D TEST split (CPU).

The §37o pipeline (RZ-head shift + li-avg warp) is re-scored first and must reproduce li_test.json within 1e-5; the
3-seed shape heads (head_boundary_s{1,2,3}.pt) are run on the TEST rows on CPU.

    .venv/bin/python paper/strat/fusion/bw_test.py <arm>   ->  runs/strat/fusion/bw_test.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

import fx  # noqa: I001  (CPU only)
import axis_shift as AX
import li_warp as LW
import boundary_warp as BW
import kappa_warp as KW
from head_boundary import Net


def head_predict(d, rows, tag=None, seeds=None) -> np.ndarray:
    tag = os.environ.get("STRAT_LABEL_SUFFIX", "") if tag is None else tag
    seeds = [int(x) for x in os.environ.get("STRAT_HEAD_SEEDS", "1,2,3").split(",")] if seeds is None else seeds  # §37y: 1-6
    out = []
    for s in seeds:
        ck = torch.load(fx.OUT / f"head_boundary{tag}_s{s}.pt", map_location="cpu", weights_only=False)
        c = ck["cfg"]
        net = Net(len(ck["mu"]), len(ck["ymu"]), c["width"], c["depth"], c["drop"]); net.load_state_dict(ck["net"]); net.eval()
        x = torch.tensor(np.clip((d.inputs(rows, ck["fm"], ck["edge"]) - ck["mu"]) / ck["sd"], -10, 10), dtype=torch.float32)
        with torch.no_grad():
            p = torch.cat([net(x[i:i + 32768]) for i in range(0, len(x), 32768)]).numpy()
        out.append(p * ck["ysd"] + ck["ymu"])
    return np.mean(out, 0)


def main() -> None:
    arm = sys.argv[1]; assert arm in ("Q-b", "Q-bl", "Q-bl3", "Q-bkl3")
    sota, d = fx.data()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import ens_select as ES
    import train_hybrid as TH
    from common import AXIS_SIGN
    tag = os.environ.get("STRAT_LABEL_SUFFIX", "")                            # §37v: "_v5" heads vs the §37q pipeline (v1 heads)
    kwarm = arm == "Q-bkl3"                                                   # §37w: base = Q-bl3 on the same heads
    seedarm = os.environ.get("STRAT_HEAD_SEEDS", "1,2,3") != "1,2,3"          # §37y: base = the same arm with 3 seeds
    btag = os.environ.get("STRAT_BASE_TAG")                                     # §37ab: base = Q-bkl3 with the heads of this tag
    out_f = fx.OUT / (f"bw_test_kw{tag}_6s.json" if seedarm else f"bw_test_kw{tag}.json" if kwarm else f"bw_test{tag}.json")
    assert not out_f.exists(), "TEST already scored once"
    rec = json.loads((fx.ROOT / "runs" / "sophelio_sota" / "test_scores.json").read_text())["_mlp"]
    if btag is not None:
        assert kwarm and not seedarm
        prev = json.loads((fx.OUT / f"bw_test_kw{btag}.json").read_text())["new"]["S"]
    elif seedarm:
        assert kwarm, "§37y is registered on the Q-bkl3 pipeline"
        prev = json.loads((fx.OUT / f"bw_test_kw{tag}.json").read_text())["new"]["S"]
    elif kwarm:
        prev = json.loads((fx.OUT / f"bw_test{tag}.json").read_text())["new"]["S"]
    else:
        prev = (json.loads((fx.OUT / "bw_test.json").read_text())["new"]["S"] if tag
                else json.loads((fx.OUT / "li_test.json").read_text())["rz_head_li_avg"]["S"])
    test = d.shots_of(2)
    members = [ES.member(str(fx.TRUNKS / m), d, test) for m in rec["members"]]
    comps, mean = d.meta["comps"][:ES.KMAX].numpy(), d.meta["mean"].numpy()
    L, H = sota.LABELS, sota.HEADS
    ir_, iz_, il_ = H.index("efit_r_axis"), H.index("efit_z_axis"), H.index("efit_li")
    R, Z, mask = AX.grid(); sign = AXIS_SIGN["DIII-D"]
    shifter = AX.make_shifter(R, Z, mask, sign); liw = LW.make_li_warp(R, Z, mask, sign); bw = BW.make_boundary_warp(R, Z, mask, sign)
    kw = KW.make_kappa_warp(R, Z, mask, sign)
    gts, base, new = [], [], []
    for s in test:
        rows = np.nonzero(d.shot == s)[0]; ph = head_predict(d, rows)
        ph1 = head_predict(d, rows, "") if (tag and not kwarm) else None
        ph3 = head_predict(d, rows, tag, [1, 2, 3]) if seedarm else head_predict(d, rows, btag, [1, 2, 3]) if btag is not None else None
        pc = np.mean([mb[s][0] for mb in members], axis=0); sc = np.mean([mb[s][1] for mb in members], axis=0)
        psi = (pc @ comps + mean).reshape(-1, 65, 65).astype(np.float64)
        b_, n_ = [], []
        for k in range(len(psi)):
            a, _ = shifter(psi[k], np.array([sc[k, ir_], sc[k, iz_]]), "RZ", "head")
            if seedarm or btag is not None:                                     # base = Q-bkl3 with the 3-seed (or btag) heads
                b_.append(liw(kw(bw(a, ph3[k, 7:])[0], ph3[k, 2])[0], ph3[k, 6], full=True)[0])
            elif kwarm:                                                         # base = Q-bl3 with the same heads
                b_.append(liw(bw(a, ph[k, 7:])[0], ph[k, 6], full=True)[0])
            elif tag:                                                           # base = §37q Q-bl3 with the v1 heads
                b_.append(liw(bw(a, ph1[k, 7:])[0], ph1[k, 6], full=True)[0])
            else:
                b_.append(liw(a, sc[k, il_])[0])
            w, _ = bw(a, ph[k, 7:])
            if kwarm:
                w = liw(kw(w, ph[k, 2])[0], ph[k, 6], full=True)[0]
            elif arm == "Q-bl":
                w = liw(w, sc[k, il_])[0]
            elif arm == "Q-bl3":
                w = liw(w, ph[k, 6], full=True)[0]
            n_.append(w)
        gts.append(({"psi": d.psi_of(rows).reshape(-1, 65, 65).astype(np.float32),
                     "q95": d.S[rows, L.index("efit_q95")].astype(np.float64),
                     "betaN": d.S[rows, L.index("efit_beta_n")].astype(np.float64)}, d.meta["names"][s]))
        q, b = sc[:, H.index("efit_q95")], sc[:, H.index("efit_beta_n")]
        base.append({"psirz": np.stack(b_), "q95": q, "betaN": b}); new.append({"psirz": np.stack(n_), "q95": q, "betaN": b})
    ref = fx.ROOT / "runs" / "sophelio_sota" / "ref_cache"
    rb = TH.official_score(gts, base, ref)
    ok = abs(rb["S"] - prev) < 1e-5
    print(f"[s37q-test] base pipeline re-scored S {rb['S']:.6f} (recorded {prev:.6f}) check {'ok' if ok else 'FAILED'}", flush=True)
    rn = TH.official_score(gts, new, ref)
    res = {"arm": arm, "n_test_shots": len(test), "li_avg": rb, "new": rn, "check_ok": ok, "delta_S": rn["S"] - rb["S"],
           "delta_vs_entry": rn["S"] - rec["test"]["S"]}
    out_f.write_text(json.dumps(res, indent=1, default=float))
    print(f"S37Q_TEST arm={arm} li_avg {rb['S']:.6f} new {rn['S']:.6f} delta {rn['S'] - rb['S']:+.6f} vs entry {rn['S'] - rec['test']['S']:+.6f} "
          f"cons {rb['consistency']:.5f}->{rn['consistency']:.5f} dlcfs {rb['dlcfs']:.5f}->{rn['dlcfs']:.5f} psi {rb['r2_psi']:.6f}->{rn['r2_psi']:.6f}", flush=True)


if __name__ == "__main__":
    main()
