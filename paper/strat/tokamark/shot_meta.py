"""§37 TokaMark: per-shot / per-window regime metadata from INPUTS only (CPU, VALID + a TRAIN subsample). TEST never loaded.

    OMP_NUM_THREADS=4 .venv-tokamark/bin/python paper/strat/tokamark/shot_meta.py
    -> runs/strat/tokamark/shot_meta.json

Reads the split CSV rows for train/validation only (the test rows are dropped before any shot id is used). Loads each
shot through TokaMark's own MastDataset (same outlier removal as the benchmark, no z-scoring so Ip is native) and
replicates TokaMarkDataset._process_shot's window clock: global_start = min first-valid time over inputs,
global_end = max last-valid time over outputs, t_cut = global_start + input_length + k * stride (window_index k).
Regimes come from summary/ip only (an input), never from EFIT targets or model errors:
  * per window: phase = ramp_up (before |Ip| first reaches 0.8 max), flat_top, ramp_down (after it last leaves 0.8 max)
  * per shot: duration (global_end - global_start), ip_max, term_drop = max over t after the peak of
    (|Ip|(t) - |Ip|(t + 5 ms)) / ip_max (a fast-termination / disruption proxy).
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "baseline" / "tokamind" / "scripts_mast"))
from mast_utils import load_task_definition  # noqa: E402
from mast_utils.tokamark_split import resolve_split_assets  # noqa: E402
from tokamark.data import initialize_MAST_dataset  # noqa: E402

OUT = ROOT / "runs" / "strat" / "tokamark"
ZARR = ROOT / "data" / "tokamark" / "zarr"
N_TRAIN_SUB = 400          # TRAIN shots for blend-weight fits (seed 0)
FLAT = 0.8
TERM_DT = 0.005


def split_shots(csv_path: str) -> tuple[list[int], list[int]]:
    df = pd.read_csv(csv_path, usecols=["shot_id", "train", "val"])   # the test column is never read
    tr = sorted(int(s) for s in df.loc[df["train"].astype(bool), "shot_id"])
    va = sorted(int(s) for s in df.loc[df["val"].astype(bool), "shot_id"])
    return tr, va


def _first_last_valid(t: np.ndarray, v: np.ndarray) -> tuple[float, float] | None:
    ok = ~np.all(np.isnan(v), axis=tuple(range(v.ndim - 1)))
    if not ok.any():
        return None
    tv = t[ok]
    return float(tv[0]), float(tv[-1])


def shot_record(sample: dict, in_keys: list[str], out_keys: list[str], seg: dict, stride: float) -> dict | None:
    starts = [r[0] for k in in_keys if (r := _first_last_valid(sample[k]["time"], sample[k]["values"]))]
    ends = []
    for k in out_keys:
        t, v = sample[k]["time"], sample[k]["values"]
        if t.size == 0 or v.size == 0:
            return None                                             # test_mode skips such shots
        if (r := _first_last_valid(t, v)):
            ends.append(r[1])
    if not starts or not ends:
        return None
    g0, g1 = float(np.min(starts)), float(np.max(ends))
    lo, hi = g0 + seg["input_length"], g1 - seg["delta"] - seg["output_length"]
    if not (np.isfinite(g0) and np.isfinite(g1)) or lo >= hi:
        return None
    t_cuts = np.arange(lo, hi, stride)
    ip_t = sample["summary-ip"]["time"]
    ip = np.abs(np.asarray(sample["summary-ip"]["values"], dtype=float).reshape(-1))
    good = np.isfinite(ip)
    if good.sum() < 10:
        return None
    ip_t, ip = ip_t[good], ip[good]
    ip_max = float(ip.max())
    centre = t_cuts - 0.5 * seg["input_length"]
    ipw = np.interp(centre, ip_t, ip, left=0.0, right=0.0) / max(ip_max, 1e-9)
    above = np.nonzero(ip >= FLAT * ip_max)[0]
    t_up, t_down = float(ip_t[above[0]]), float(ip_t[above[-1]])
    phase = np.where(centre < t_up, 0, np.where(centre > t_down, 2, 1)).astype(np.int8)   # 0 up, 1 flat, 2 down
    k_peak = int(np.argmax(ip))
    tt, yy = ip_t[k_peak:], ip[k_peak:]
    later = np.interp(tt + TERM_DT, ip_t, ip, right=0.0)
    term_drop = float(np.max((yy - later) / max(ip_max, 1e-9))) if tt.size else 0.0
    return {"g0": g0, "g1": g1, "duration": g1 - g0, "n_cuts": int(t_cuts.size), "ip_max": ip_max,
            "t_up": t_up - g0, "t_down": t_down - g0, "term_drop": term_drop,
            "phase": phase.tolist(), "ip_frac": np.round(ipw, 3).tolist()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_task = load_task_definition(task_key="task_1-1")
    seg = cfg_task["task_window_segmenter"]
    in_keys = ["-".join(k) for k in seg["input_keys"]]
    out_keys = ["-".join(k) for k in seg["output_keys"]]
    sa = resolve_split_assets("random")
    tr, va = split_shots(sa["data_splits_file_path"])
    rng = np.random.default_rng(0)
    tr_sub = sorted(int(s) for s in rng.choice(tr, size=N_TRAIN_SUB, replace=False))
    res = {"n_train_split": len(tr), "n_val_split": len(va), "train_sub": tr_sub, "valid": {}, "train": {}}
    for name, shots in (("valid", va), ("train", tr_sub)):
        ds = initialize_MAST_dataset(config_task=cfg_task, shots_list=shots, local_flag=True, use_std_scaling=False,
                                     stats_metadata_file_path=sa["stats_metadata_file_path"], use_nan_filling=False,
                                     remove_outliers=True, outlier_metadata_file=sa["outlier_metadata_file"],
                                     store_manager_settings={"base_local_zarr_path": str(ZARR)}, verbose=False)
        t0 = time.time()
        for i in range(len(ds)):
            try:
                sample = ds[i]
                sid = int(ds.get_shot_id(idx=i)) if hasattr(ds, "get_shot_id") else int(shots[i])
                rec = shot_record(sample, in_keys, out_keys, seg, float(cfg_task["stride_window"]))
            except Exception as e:  # noqa: BLE001  (a shot the benchmark would also skip; recorded, not hidden)
                rec, sid = {"error": repr(e)[:200]}, int(shots[i])
            if rec is not None:
                res[name][str(sid)] = rec
            if i % 200 == 0:
                print(f"{name} {i}/{len(ds)} {time.time() - t0:.0f}s", flush=True)
    (OUT / "shot_meta.json").write_text(json.dumps(res))
    print("wrote", OUT / "shot_meta.json", {k: len(res[k]) for k in ("valid", "train")})


if __name__ == "__main__":
    main()
