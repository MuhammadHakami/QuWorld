"""Doc 35 §37p step 3: direct shape heads (3-seed mean) vs the shape the scorer extracts from the entry's predicted psi,
on the 180-shot VALID sample, both against labels from the TRUE psi through the same extraction (boundary_labels.py).

    .venv/bin/python paper/strat/fusion/head_eval.py   ->  runs/strat/fusion/head_eval.json
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

import fx  # noqa: I001
import strategies as ST
import boundary_labels as BL


def r2(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return float(1 - np.mean((y[m] - p[m]) ** 2) / np.var(y[m])) if m.sum() > 10 else float("nan")


def main() -> None:
    sota, d = fx.data()
    tag = os.environ.get("STRAT_LABEL_SUFFIX", "")                            # "_v2": §37s heads (64 rays)
    lab = np.load(fx.OUT / f"boundary_labels{tag}.npz"); va, Y = lab["rows_valid"], lab["y_valid"].astype(np.float64)
    heads = [np.load(fx.OUT / f"head_boundary{tag}_s{s}.npz") for s in (1, 2, 3)]
    assert all((h["rows_valid"] == va).all() for h in heads)
    Ph = np.mean([h["pred_valid"] for h in heads], 0).astype(np.float64)
    shots, _ = ST.sample(); f, comps, mean, H = ST.entry_preds(d, sota)
    psi = np.concatenate([(f(s)[0].mean(0) @ comps + mean) for s in shots]).astype(np.float32)
    assert len(psi) == len(va)
    with ProcessPoolExecutor(max_workers=fx.N_WORKERS, mp_context=mp.get_context("spawn")) as ex:
        Pd = np.concatenate(list(ex.map(BL._one, [(psi[i:i + 2000],) for i in range(0, len(psi), 2000)]))).astype(np.float64)
    assert Pd.shape[1] == Y.shape[1], "run with the same STRAT_NANG as the labels"
    names = list(BL.NAMES); n7 = len(names)
    out = {"scalars": {}, "radii": {}}
    for j, nm in enumerate(names):
        out["scalars"][nm] = {"head": r2(Y[:, j], Ph[:, j]), "derived": r2(Y[:, j], Pd[:, j]),
                              "avg": r2(Y[:, j], 0.5 * (Ph[:, j] + Pd[:, j])),
                              "err_corr": float(np.corrcoef(*[np.nan_to_num(a[:, j] - Y[:, j]) for a in (Ph, Pd)])[0, 1])}
    ry, rh, rd = Y[:, n7:], Ph[:, n7:], Pd[:, n7:]
    m = np.isfinite(ry) & np.isfinite(rd)
    out["radii"] = {"rms_head": float(np.sqrt(np.mean((rh[m] - ry[m]) ** 2))), "rms_derived": float(np.sqrt(np.mean((rd[m] - ry[m]) ** 2))),
                    "rms_avg": float(np.sqrt(np.mean((0.5 * (rh[m] + rd[m]) - ry[m]) ** 2))),
                    "max_head_median": float(np.nanmedian(np.nanmax(np.abs(rh - ry), 1))),
                    "max_derived_median": float(np.nanmedian(np.nanmax(np.abs(rd - ry), 1)))}
    T = np.array([len(fx.valid_rows(d, s)) for s in shots]); pos = np.concatenate([np.arange(t) / t for t in T])
    edge = (pos < 0.1) | (pos > 0.9)
    for part, sel in (("edge", edge), ("core", ~edge)):
        out[f"kappa_{part}"] = {"head": r2(Y[sel, 2], Ph[sel, 2]), "derived": r2(Y[sel, 2], Pd[sel, 2])}
    (fx.OUT / f"head_eval{tag}.json").write_text(json.dumps(out, indent=1))
    for nm, v in out["scalars"].items():
        print(f"[s37p] {nm:8s} head {v['head']:.4f} derived {v['derived']:.4f} avg {v['avg']:.4f} err-corr {v['err_corr']:.2f}", flush=True)
    print("[s37p] radii", json.dumps(out["radii"]), "kappa edge/core", out["kappa_edge"], out["kappa_core"], flush=True)
    print("S37P_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
