"""Sophelio: shot-edge ablation on the best stage-1 trunk config (VALID only; TEST untouched).

    .venv/bin/python paper/sophelio/edge_ablation.py [--trial 5] [--seeds 1 2]   ->  runs/sophelio_sota/edge_ablation.json

The consistency residual concentrates at ramp-up / ramp-down (cons_diag.py). Arms, same config and seed:
  base      the trial's config (a seed replica, saved under replicas.py's name so it is not trained twice)
  edge      + time-to-end and elapsed-fraction inputs (sota.Data.edge_feats; from frame times only)
  ew2       + psi loss weight 3x on the first / last 10 % of each shot
  edge-ew2  both
  ema / edge-ew2-ema   + an EMA of the weights (decay 0.999, ~3 epochs) for evaluation and saving (seed variance)
Every trunk is saved beside the NAS trunks, so the greedy ensemble selection sees all of them.
"""
from __future__ import annotations

import argparse
import json
import sys

import optuna

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
import sota  # noqa: E402

ARMS = {"base": {}, "edge": {"edge": True}, "ew2": {"edge_w": 2.0}, "edge-ew2": {"edge": True, "edge_w": 2.0},
        "ema": {"ema": 0.999}, "edge-ew2-ema": {"edge": True, "edge_w": 2.0, "ema": 0.999}}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--study", default="trunk1"); ap.add_argument("--trial", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2]); ap.add_argument("--arms", nargs="+", default=list(ARMS)[:4])
    a = ap.parse_args()
    out = sota.ROOT / "runs" / "nas" / "sophelio" / "trunks"
    s = optuna.load_study(study_name=f"sophelio-{a.study}", storage=f"sqlite:///{out.parent}/{a.study}.db")
    cfg = json.loads(s.trials[a.trial].user_attrs["cfg"])
    res_f = sota.ROOT / "runs" / "sophelio_sota" / "edge_ablation.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    data = sota.Data()
    for seed in a.seeds:
        for arm in a.arms:
            key = f"{arm}_seed{seed}" if a.trial == 5 else f"t{a.trial}-{arm}_seed{seed}"
            if key in res:
                continue
            tag = f"t{a.trial}" if arm == "base" else f"t{a.trial}-{arm}"
            r = sota.train_eval(cfg | ARMS[arm] | {"seed": seed}, data, score_val=40, save=out / f"{a.study}_{tag}_seed{seed}.pt")
            v = r["valid_official"]
            res[key] = dict(S=v["S"], consistency=v["consistency"], dlcfs=v["dlcfs"], r2_psi=v["r2_psi"], r2_qb=v["r2_qb"],
                            cons_each=v.get("r2_cons_each"), minutes=r["train_seconds"] / 60)
            res_f.write_text(json.dumps(res, indent=1))
            print(f"[edge] {key}: VALID S {v['S']:.4f} cons {v['consistency']:.4f} dlcfs {v['dlcfs']:.4f} psi {v['r2_psi']:.5f} "
                  f"({r['train_seconds']/60:.0f} min) {v.get('r2_cons_each')}", flush=True)


if __name__ == "__main__":
    main()
