"""Doc 35 §42f diagnostic: does an arm's accuracy hold on shots never used for selection? Per-scalar consistency R2, D_LCFS and q/b
of a VT parts file split into VALID (the 180-shot sample + the rest: used for tuning) and TEST (never used for selection, like public),
and by plasma-current sign.
    .venv/bin/python paper/strat/fusion/vt_split.py <parts tag> [...]   -> stdout
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

import fx

SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    d = sota.Data(); fn = list(d.fn)
    for tag in sys.argv[1:]:
        P = pickle.load(open(fx.OUT / f"parts_VT-{tag}.pkl", "rb")); shots = sorted(P)
        split = {s: int(d.split[d.shot == s][0]) for s in shots}
        sgn = {s: np.sign(np.mean(np.asarray(d.X[np.nonzero(d.shot == s)[0], fn.index("mag_plasma_current")], float))) for s in shots}
        for nm, sel in (("VALID", lambda s: split[s] == 1), ("TEST", lambda s: split[s] == 2), ("Ip+", lambda s: sgn[s] > 0),
                        ("Ip-", lambda s: sgn[s] < 0)):
            ss = [s for s in shots if sel(s)]
            if not ss:
                continue
            cr = np.concatenate([P[s]["cres"] for s in ss]); cg = np.concatenate([P[s]["cons_gt"] if "cons_gt" in P[s] else P[s]["cgt"] for s in ss]) \
                if ("cons_gt" in P[ss[0]] or "cgt" in P[ss[0]]) else None
            dd = np.concatenate([P[s]["d"] for s in ss]); qr = np.concatenate([P[s]["q_res"] for s in ss]); br = np.concatenate([P[s]["b_res"] for s in ss])
            qg = np.concatenate([P[s]["q_gt"] for s in ss]); bg = np.concatenate([P[s]["b_gt"] for s in ss])
            line = f"[split] {tag} {nm:5s} shots {len(ss):3d} D {np.nanmean([np.nanmean(P[s]['d']) for s in ss]):.5f} " \
                   f"q95 {1 - np.nanmean(qr) / np.nanvar(qg):.5f} betaN {1 - np.nanmean(br) / np.nanvar(bg):.5f}"
            if cg is not None:
                line += " | " + " ".join(f"{n} {1 - np.nanmean(cr[:, j]) / np.nanvar(cg[:, j]):.5f}" for j, n in enumerate(SC))
            else:
                line += " | sse " + " ".join(f"{n} {np.nanmean(cr[:, j]):.3g}" for j, n in enumerate(SC))
            print(line, flush=True)


if __name__ == "__main__":
    main()
