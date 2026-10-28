"""Doc 35 §38g QF: q95/betaN from trunk checkpoints (each read with the inputs it was trained on) swapped into the frozen
pipeline's sample parts (psi untouched); official composite + paired shot bootstrap vs the frozen pipeline.

    .venv/bin/python paper/strat/fusion/qb_eval.py <ckpt>[+<ckpt>...][@entry] ...
'@entry' averages the arm's q95/betaN 50/50 with the 7-member entry's. Paths are relative to runs/sophelio_sota/.
§38q: a member 'head:<stem>' reads a q95/betaN head's VALID predictions (runs/strat/fusion/<stem>.npz, head_boundary.py
mode qb); members of one arm are averaged with equal weight.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import sys

import numpy as np
import torch

import fx  # noqa: I001
import strategies as ST
import inputs_fix as IF


def main() -> None:
    sota, d = fx.data(); shots, grp = ST.sample(); pe, nm = ST.load_entry(); f, comps, mean, H = ST.entry_preds(d, sota)
    iq, ib = H.index("efit_q95"), H.index("efit_beta_n"); rows = {s: fx.valid_rows(d, s) for s in shots}
    dfix = IF.use(copy.copy(d)); g = np.asarray([grp[s] for s in shots])
    base = pickle.load(open(fx.OUT / "parts_BW-Q-bkl3_v8_6s.pkl", "rb")); sb = np.stack([fx.shot_sums(base[s]) for s in shots])
    cb = fx.composite(sb, nm); print(f"[s38g] frozen: S {cb['S']:.5f} r2_qb {cb['r2_qb']:.5f}", flush=True)
    rep = {}
    for arg in sys.argv[1:]:
        arm, ent = (arg[:-6], True) if arg.endswith("@entry") else (arg, False); preds = []
        for p in arm.split("+"):
            if p.startswith("head:"):
                z = np.load(fx.OUT / f"{p[5:]}.npz"); pos = {int(r): i for i, r in enumerate(z["rows_valid"])}
                preds.append({s: z["pred_valid"][[pos[int(r)] for r in rows[s]]].astype(np.float64) for s in shots}); continue
            path = fx.ROOT / "runs" / "sophelio_sota" / p
            dd = dfix if torch.load(path, map_location="cpu", weights_only=False)["cfg"].get("inputs") == "magfix" else d
            mp_ = fx.member_predict(path, dd, rows); preds.append({s: mp_[s][1][:, [iq, ib]] for s in shots})
        new = {}
        for s in shots:
            qb = np.mean([pr[s] for pr in preds], 0).astype(np.float64)
            if ent:
                qb = 0.5 * qb + 0.5 * f(s)[1].mean(0)[:, [iq, ib]]
            p = dict(base[s]); p["q_res"] = (p["q_gt"] - qb[:, 0]) ** 2; p["b_res"] = (p["b_gt"] - qb[:, 1]) ** 2; new[s] = p
        sn = np.stack([fx.shot_sums(new[s]) for s in shots]); cn = fx.composite(sn, nm); bt = fx.paired_boot(sb, sn, nm, groups=g)
        rep[arg] = dict(S=cn["S"], r2_qb=cn["r2_qb"], r2_q95=cn["r2_q95"], r2_betaN=cn["r2_betaN"], boot=bt)
        print(f"[s38g] {arg}: S {cn['S']:.5f} r2_qb {cn['r2_qb']:.5f} (q95 {cn['r2_q95']:.5f} betaN {cn['r2_betaN']:.5f}) "
              f"delta {bt['delta']:+.5f} [{bt['lo']:+.5f}, {bt['hi']:+.5f}]", flush=True)
    (fx.OUT / f"s38g_qb{os.environ.get('STRAT_TAG', '')}.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
