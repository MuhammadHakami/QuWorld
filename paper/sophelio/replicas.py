"""Seed replicas of the top-N stage-1 NAS trunks (ensemble diversity; VALID only), saved beside the NAS trunks.

    .venv/bin/python paper/sophelio/replicas.py --study trunk1 --top 3 --seeds 1 2
"""
from __future__ import annotations

import argparse
import json
import sys

import optuna

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
import sota  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--study", default="trunk1"); ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    out = sota.ROOT / "runs" / "nas" / "sophelio" / "trunks"
    s = optuna.load_study(study_name=f"sophelio-{a.study}", storage=f"sqlite:///{out.parent}/{a.study}.db")
    done = sorted([t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: -t.value)[:a.top]
    data = sota.Data()
    for t in done:
        cfg = json.loads(t.user_attrs["cfg"])
        for seed in a.seeds:
            dst = out / f"{a.study}_t{t.number}_seed{seed}.pt"
            if dst.exists():
                continue
            r = sota.train_eval(cfg | {"seed": seed}, data, score_val=0, save=dst)
            print(f"[replica] t{t.number} seed {seed}: quick {r['valid_quick']['proxy']:.4f} ({r['train_seconds']/60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
