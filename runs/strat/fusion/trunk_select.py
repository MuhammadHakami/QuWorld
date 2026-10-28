"""Label-free per-frame trunk selection (s44 quench diagnosis: on current-quench frames the raw trunk psi is far off - li 5.31 vs head 2.64,
truth 3.24 - and no stable li warp bridges it; the heads are right there). Offline emulation on stored VT parts of several psi sets that
went through the same chain: per frame, choose the arm whose post-processed f_j(psi) agrees best with the HEAD values (never the labels):
distance = sum_j ((cpred_j - head_j) / sd_j)^2 over the 7 consistency scalars (sd from the heads), or li alone. The composite S is then
recomputed from the selected frames' parts (psi residual, q/b, D_LCFS, consistency) with the scorer's fixed normalisers.
    python runs/strat/fusion/trunk_select.py ARM1 ARM2 ... (parts_VT-<arm>.pkl, same chain + heads)"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "paper/strat/fusion")
import fx  # noqa: E402
import strategies as ST  # noqa: E402

OUT = Path("runs/strat/fusion"); ARMS = sys.argv[1:] or ["PQmix~g_VB", "PQ2_VB", "K128_VB", "K256_VB", "LS_ctrl_VB", "PQ2H~g_VB", "PQL10~g_VB"]
_, nm = ST.load_entry(); P = {a: pickle.load(open(OUT / f"parts_VT-{a}.pkl", "rb")) for a in ARMS}
shots = sorted(set.intersection(*[set(p) for p in P.values()])); shot_of = np.load("data/sophelio_proc/cache_k256/shot.npy", mmap_mode="r")
hv = [np.load(OUT / f"head_boundary_v8_J1_s{k}.npz") for k in range(1, 7)]; rows_v = hv[0]["rows_valid"]
hp = np.mean([z["pred_valid"][:, :7] for z in hv], 0); pos = {int(r): i for i, r in enumerate(rows_v)}; sdh = np.nanstd(hp, 0)
KEYS = [k for k, v in P[ARMS[0]][shots[0]].items() if isinstance(v, np.ndarray)]


def merged(rule):
    out = {}
    for s in shots:
        rows = np.nonzero(np.asarray(shot_of) == s)[0]; idx = np.array([pos.get(int(r), -1) for r in rows])
        h = np.where(idx[:, None] >= 0, hp[np.maximum(idx, 0)], np.nan)
        D = []
        for a in ARMS:
            c = P[a][s]["cpred"]
            d = (np.nansum(((c - h) / sdh) ** 2, 1) if rule == "all7" else np.abs(c[:, 6] - h[:, 6]) / sdh[6])
            D.append(np.where(np.isfinite(h).any(1), d, np.inf if a != ARMS[0] else 0.0))      # no head -> first arm
        pick = np.argmin(np.stack(D), 0)
        out[s] = {k: np.stack([P[a][s][k] for a in ARMS])[pick, np.arange(len(pick))] for k in KEYS}
        out[s].update({k: v for k, v in P[ARMS[0]][s].items() if not isinstance(v, np.ndarray)})
        out[s]["_pick"] = pick
    return out


res = {}
for a in ARMS:
    c = fx.composite(np.stack([fx.shot_sums(P[a][s]) for s in shots]), nm); res[a] = c
    print(f"[tsel] {a:>14}: S {c['S']:.5f} psi {c['r2_psi']:.6f} dlcfs {c['dlcfs']:.5f} cons {c['consistency']:.5f} li {c['cons_each']['li']:.5f}", flush=True)
for rule in ("li", "all7"):
    m = merged(rule); c = fx.composite(np.stack([fx.shot_sums(m[s]) for s in shots]), nm); res[f"select_{rule}"] = c
    share = np.bincount(np.concatenate([m[s]["_pick"] for s in shots]), minlength=len(ARMS)) / sum(len(m[s]["_pick"]) for s in shots)
    print(f"[tsel] select by {rule:>4}: S {c['S']:.5f} psi {c['r2_psi']:.6f} dlcfs {c['dlcfs']:.5f} cons {c['consistency']:.5f} li {c['cons_each']['li']:.5f} | "
          f"picks {dict(zip(ARMS, np.round(share, 3)))}", flush=True)
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "cons_each"} | {"cons_each": v["cons_each"]} for k, v in res.items()},
          open(OUT / "trunk_select.json", "w"), indent=1, default=float)
print("TSEL_DONE", flush=True)
