"""Fusion §41q: where the psi squared error lives on VALID (per shot: total SS, uniform-offset SS, Ip sign, frames with a
dead Ip input), for psi trunks on their own input sets. CPU only (the Spark GPU may be busy); writes err_anatomy_<tag>.json.

    CUDA_VISIBLE_DEVICES= .venv/bin/python paper/strat/fusion/err_anatomy.py s38i_PSI-F-s1.pt:X_magfix IPF_s1.pt:X_ipfix
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper" / "sophelio")); sys.path.insert(0, str(ROOT / "paper" / "strat" / "fusion"))
import sota  # noqa: E402
sota.DEV = torch.device("cpu")
import ens_select as ES  # noqa: E402
import inputs_fix  # noqa: E402

OUT = ROOT / "runs" / "strat" / "fusion"


def main() -> None:
    d = sota.Data(); shots = d.shots_of(1); comps = d.meta["comps"].numpy().astype(np.float64)[:64]; mean = d.meta["mean"].numpy().astype(np.float64)
    iip = d.fn.index("mag_plasma_current"); res = {}
    tot_acc = [0.0, np.zeros(4225), 0]
    for spec in sys.argv[1:]:
        ck, name = spec.split(":"); dd = inputs_fix.use(copy.copy(d), name=name)
        M = ES.member(str(ROOT / "runs" / "sophelio_sota" / ck), dd, shots); per = {}
        for s in shots:
            rows = np.nonzero(d.shot == s)[0]; true = d.psi_of(rows).reshape(len(rows), -1).astype(np.float64)
            e = M[s][0][:, :64].astype(np.float64) @ comps + mean - true; off = e.mean(1)
            ip = np.asarray(dd.inputs(rows, np.arange(len(d.fn)))[:, iip])
            per[int(s)] = dict(n=len(rows), ss=float((e ** 2).sum()), ss_off=float((off ** 2).sum() * e.shape[1]), ip_med=float(np.median(ip)),
                               dead=float((np.abs(ip) < 50).mean()), ss_end=float((e[int(0.9 * len(rows)):] ** 2).sum()))
            if not res:
                tot_acc[0] += float((true ** 2).sum()); tot_acc[1] += true.sum(0); tot_acc[2] += len(rows)
        res[ck] = per
    tot = tot_acc[0] - float((tot_acc[1] ** 2).sum()) / tot_acc[2]
    for ck, per in res.items():
        ss = np.array([p["ss"] for p in per.values()]); R = ss.sum()
        rev = np.array([p["ip_med"] < 0 for p in per.values()]); dead = np.array([p["dead"] > 0.2 for p in per.values()])
        off = np.array([p["ss_off"] for p in per.values()]); end = np.array([p["ss_end"] for p in per.values()]); o = np.sort(ss)[::-1]
        print(f"{ck}: R2 {1 - R / tot:.6f} | offset share {off.sum() / R:.3f} | last-10% share {end.sum() / R:.3f} | reversed-Ip shots {rev.sum()} "
              f"share {ss[rev].sum() / R:.3f} | dead-Ip shots (>20% frames |Ip|<50 kA) {dead.sum()} share {ss[dead].sum() / R:.3f} | "
              f"top-1/5/10% shots {o[0] / R:.3f} {o[:5].sum() / R:.3f} {o[:len(o) // 10].sum() / R:.3f}", flush=True)
    (OUT / "err_anatomy.json").write_text(json.dumps(dict(tot=tot, res=res)))


if __name__ == "__main__":
    main()
