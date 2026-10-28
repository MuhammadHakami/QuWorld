"""RealPDE foil: weighted ensemble of the released trunks (classical lift, labelled ensemble).

    .venv/bin/python paper/realpde/ensemble.py [--models unet fno cno]  ->  runs/realpde/ensemble_finetune.json

Weights w on the simplex grid (step 0.05) are chosen by VALID rel-L2 only; TEST is scored once with the chosen w using
RealPDEBench's metric (chunked, verified equal to eval_metrics(batch = N)). Equal weights are reported alongside.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import torch

import common as C


def simplex(n: int, step: float = 0.05):
    k = int(round(1 / step))
    for c in itertools.product(range(k + 1), repeat=n - 1):
        if sum(c) <= k:
            yield np.array(list(c) + [k - sum(c)], dtype=np.float64) / k


@torch.no_grad()
def rel_l2_mix(P: list, Y, w, chunk: int = 256) -> float:
    tot = 0.0
    for i in range(0, len(Y), chunk):
        y = Y[i:i + chunk].clone().to(C.DEV).double()
        p = sum(wi * Pm[i:i + chunk].clone().to(C.DEV).double() for wi, Pm in zip(w, P) if wi > 0)
        tot += ((p - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).sum().item()
    return tot / len(Y)


class Mix:
    """Lazy weighted sum of disk-backed predictions, indexable like a tensor (for official_metrics_chunked)."""

    def __init__(self, P, w):
        self.P, self.w, self.shape = P, w, P[0].shape

    def __getitem__(self, idx):
        return sum(float(wi) * Pm[idx] for wi, Pm in zip(self.w, self.P) if wi > 0)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["unet", "fno", "cno"])
    ap.add_argument("--stage", default="finetune")
    a = ap.parse_args()
    _, Yv, _ = C.load_windows("val"); _, Yt, st = C.load_windows("test")
    ms = [(m.split(":") + [a.stage])[:2] for m in a.models]              # "unet:real" mixes training paradigms
    missing = [f"pred_{m}_{s}_{sp}" for m, s in ms for sp in ("val", "test") if not (C.CACHE / f"pred_{m}_{s}_{sp}.done").exists()]
    if missing:                                                            # open_pred would create them zero-filled
        raise FileNotFoundError(f"cached predictions not complete: {missing}")
    Pv = [C.open_pred(C.CACHE / f"pred_{m}_{s}_val.npy", Yv.shape) for m, s in ms]
    Pt = [C.open_pred(C.CACHE / f"pred_{m}_{s}_test.npy", Yt.shape) for m, s in ms]
    sub = torch.linspace(0, len(Yv) - 1, 512).long()                        # coarse grid on a fixed VALID subset
    Ys, Ps = Yv[sub], [p[sub] for p in Pv]
    grid = sorted(((rel_l2_mix(Ps, Ys, w), tuple(w)) for w in simplex(len(a.models), 0.1)))
    top = [np.array(w) for _, w in grid[:5]]
    best = min(((rel_l2_mix(Pv, Yv, w), tuple(w)) for w in top))           # refine the top 10 on all VALID windows
    w_best, w_eq = np.array(best[1]), np.full(len(a.models), 1 / len(a.models))
    import eval_trunk as E                                                   # same TEST partitions as the trunk eval
    res = {"models": a.models, "stage": a.stage, "weights_valid_best": w_best.tolist(), "valid_rel_l2_best": best[0],
           "valid_rel_l2_equal": rel_l2_mix(Pv, Yv, w_eq), "test": {}}
    for tag, w in (("valid_weighted", w_best), ("equal", w_eq)):
        res["test"][tag] = {k: C.official_metrics_chunked(Mix(Pt, w), Yt, idx=torch.nonzero(m).flatten())
                            for k, m in E.subsets(st).items()}
    (C.ROOT / "runs" / "realpde" / f"ensemble_{'+'.join(a.models).replace(':', '-')}.json").write_text(json.dumps(res, indent=1))
    T = res["test"]["valid_weighted"]
    print(f"[realpde-ens] weights {dict(zip(a.models, np.round(w_best, 2)))} | VALID {best[0]:.4f} (equal {res['valid_rel_l2_equal']:.4f}) | "
          f"TEST all {T['all']['rel_l2']:.4f} excl-2-cond {T['not_in_comp_release']['rel_l2']:.4f} in {T['in_dist']['rel_l2']:.4f} "
          f"out {T['out_dist']['rel_l2']:.4f} | equal-weight TEST {res['test']['equal']['all']['rel_l2']:.4f}", flush=True)


if __name__ == "__main__":
    main()
