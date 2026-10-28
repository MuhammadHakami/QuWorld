"""Doc 35 §37c: matched GPU retrain of the entry's best trunk config (trunk1_t5-ew2) on corrected magnetics inputs.

    flock runs/strat/gpu.lock env QW_GPU_GB=12 timeout 1800 runs/capped.sh s37c 20 .venv/bin/python \
        paper/strat/fusion/retrain_fix.py --arm {old,ip,iplag} --seed 1   ->  runs/strat/fusion/fix_<arm>_s<seed>.{pt,json}

Arms (identical config, seed, code and budget; one variable each):
  old    cached features (the shipped Ip time base and the sub-ms "lags"), lags = none   (the matched control)
  ip     Ip re-anchored on 20 kHz shots (X_magfix.npy), lags = none
  iplag  Ip re-anchored + the 13 lags in true ms (-200 ... +20), lags = sym
§37f arms (registered after §37c; same sample, disclosed as a second pass):
  oldk      old + a 20 kHz-shot indicator input (tests the "shot-type marker" reading of §37c)
  iplagk    iplag + the indicator
  iplagk50  iplagk without the -200 / -100 ms lags (lags -50 ... +20 ms only; pre-breakdown reach removed)
TRAIN fits (sota.train_eval, score_val=0). Scoring is CPU-side on the registered stratified VALID sample
(score_fix.py). TEST rows are never loaded (asserted).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "sophelio"))
from gpucap import cap_from_env  # noqa: E402

cap_from_env()
import numpy as np  # noqa: E402
import torch  # noqa: E402

import sota  # noqa: E402

OUT = ROOT / "runs" / "strat" / "fusion"
TRUNKS = ROOT / "runs" / "nas" / "sophelio" / "trunks"


ARMS = {"old": dict(fixed=False, lags="none", ind=False, min_lag=None),
        "ip": dict(fixed=True, lags="none", ind=False, min_lag=None),
        "iplag": dict(fixed=True, lags="sym", ind=False, min_lag=None),
        "oldk": dict(fixed=False, lags="none", ind=True, min_lag=None),
        "iplagk": dict(fixed=True, lags="sym", ind=True, min_lag=None),
        "iplagk50": dict(fixed=True, lags="sym", ind=True, min_lag=-50),
        "oldop01": dict(fixed=False, lags="none", ind=False, min_lag=None, opw=0.1),    # §37m O-point loss
        "oldop1": dict(fixed=False, lags="none", ind=False, min_lag=None, opw=1.0)}


class DataFix(sota.Data):
    """sota.Data whose magnetics block is replaced by the corrected one (TRAIN/VALID rows only), optionally with a
    20 kHz-shot indicator input (appended after the feature mask) and a floor on the negative lags."""

    def __init__(self, fixed: bool, ind: bool = False, min_lag: int | None = None):
        super().__init__()
        self.fixed, self.ind, self.min_lag = fixed, ind, min_lag
        if ind:
            self.is20k = np.load(OUT / "is20k.npy", mmap_mode="r")
        if fixed:
            self.mf = np.load(OUT / "X_magfix.npy", mmap_mode="r")
            cols = json.loads((OUT / "magfix.json").read_text())["columns"]
            self.cols = np.array([self.fn.index(c) for c in cols])

    def inputs(self, rows, fm, edge=False):
        assert (self.split[rows] != 2).all(), "TEST rows are never loaded in §37c"
        x = np.array(self.X[rows])
        if self.fixed:
            x[:, self.cols] = self.mf[rows]
        x = x[:, fm]
        if self.ind:
            x = np.concatenate([x, np.asarray(self.is20k[rows], np.float32)[:, None]], 1)
        return np.concatenate([x, self.edge_feats()[rows]], 1) if edge else x

    def feature_mask(self, lags: str, thomson: bool) -> np.ndarray:
        m = super().feature_mask(lags, thomson)
        if self.min_lag is not None:
            for i, n in enumerate(self.fn):
                if "_lag" in n and int(n.split("_lag")[1].replace("ms", "")) < self.min_lag:
                    m[i] = False
        return m


def data_for(arm: str) -> DataFix:
    c = ARMS[arm]
    return DataFix(c["fixed"], c["ind"], c["min_lag"])


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=tuple(ARMS))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    base = torch.load(TRUNKS / f"trunk1_t5-ew2_seed{a.seed}.pt", map_location="cpu", weights_only=False)
    cfg = dict(base["cfg"]); cfg["seed"] = a.seed
    assert cfg["lags"] == "none" and not cfg["thomson"], cfg
    cfg["lags"] = ARMS[a.arm]["lags"]; cfg["opoint_w"] = ARMS[a.arm].get("opw", 0.0)
    data = data_for(a.arm)
    tag = f"fix_{a.arm}_s{a.seed}"
    r = sota.train_eval(cfg, data, score_val=0, save=OUT / f"{tag}.pt")
    r.update(tag=tag, arm=a.arm, cfg=cfg, peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
    (OUT / f"{tag}.json").write_text(json.dumps(r, indent=1, default=float))
    print(f"S37C_TRAINED {tag}: VALID quick {r.get('valid_quick')} ({r.get('train_seconds', 0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
