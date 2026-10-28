"""Doc 35 §42c diagnostic: what does the head -> psi transfer cost on labelled frames?
On all VALID + TEST frames, per consistency scalar: R2 of the head ensemble's prediction and R2 of the submitted psi's derived value
(GPU twin parts of a gpu_build --split vt arm), both against the official reference values f(psi_true), plus their disagreement.
If the derived error >> the head error, an exact transfer would gain; if similar, the heads are the limit.
    .venv/bin/python paper/strat/fusion/transfer_vt.py <parts tag> <'+'-joined heads>  -> runs/strat/fusion/transfer_vt_<tag>.json
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import torch

import fx
import gpu_pipe as GP

SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def main() -> None:
    tag, heads = sys.argv[1], sys.argv[2]
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    d = sota.Data(); shots = [int(s) for s in np.unique(d.shot[d.split != 0])]
    allr = np.concatenate([fx.scored_rows(d, s) for s in shots])
    _, _, cgt, _ = GP.refs(d, shots); H = GB.heads_gpu(d, allr, heads)[:, :7]
    P = pickle.load(open(fx.OUT / f"parts_VT-{tag}.pkl", "rb")); cres = np.concatenate([P[s]["cres"] for s in shots])
    rep = {}
    for j, n in enumerate(SC):
        ok = np.isfinite(cgt[:, j]) & np.isfinite(cres[:, j]) & np.isfinite(H[:, j]); var = np.var(cgt[ok, j])
        eh = (H[ok, j] - cgt[ok, j]) ** 2; ed = cres[ok, j]
        rep[n] = dict(r2_head=float(1 - eh.mean() / var), r2_derived=float(1 - ed.mean() / var), rmse_head=float(np.sqrt(eh.mean())),
                      rmse_derived=float(np.sqrt(ed.mean())), frames=int(ok.sum()))
        print(f"[tvt] {n:8s} head R2 {rep[n]['r2_head']:.5f} rmse {rep[n]['rmse_head']:.5f} | derived R2 {rep[n]['r2_derived']:.5f} "
              f"rmse {rep[n]['rmse_derived']:.5f}", flush=True)
    rep["_mean"] = {k: float(np.mean([rep[n][k] for n in SC])) for k in ("r2_head", "r2_derived")}
    print(f"[tvt] mean-7 head {rep['_mean']['r2_head']:.5f} derived {rep['_mean']['r2_derived']:.5f}", flush=True)
    (fx.OUT / f"transfer_vt_{tag}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
