"""Sophelio stage-2 confirmation (VALID only): the top coupling configs of the stage-2 search re-run with 3 seeds beside
their matched classical twins ('<qcfg>-mlp': an MLP core with the register's parameter count), on the same frozen
stage-1 trunk, scored with the official scorer on the search's fixed 40 VALID shots and paired by seed. Every coupled
model is saved, so the ensemble selection can be run twice: with the register-coupled members and with their MLP twins.

    .venv/bin/python paper/nas/sophelio_confirm.py [--study coupling1] [--top 3] [--seeds 1 2 3]
        ->  runs/nas/sophelio/confirm_<study>.json, runs/nas/sophelio/trunks/coupled_c<n>-{reg,mlp}_seed<s>.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper" / "sophelio"))
import sota  # noqa: E402

OUT = ROOT / "runs" / "nas" / "sophelio"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--study", default="coupling1"); ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    s = optuna.load_study(study_name=f"sophelio-{a.study}", storage=f"sqlite:///{OUT}/{a.study}.db")
    done = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get("atoms", 99) <= 30]
    top = sorted(done, key=lambda t: -t.values[0])[:a.top]
    res_f = OUT / f"confirm_{a.study}.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    data = sota.Data()
    for t in top:
        cfg = json.loads(t.user_attrs["cfg"])
        for arm, q in (("reg", cfg["qcfg"]), ("mlp", cfg["qcfg"] + "-mlp")):
            for seed in a.seeds:
                key = f"c{t.number}-{arm}-s{seed}"
                if key in res:
                    continue
                r = sota.train_eval(cfg | {"qcfg": q, "seed": seed}, data, score_val=40,
                                    save=OUT / "trunks" / f"coupled_c{t.number}-{arm}_seed{seed}.pt")
                v = r["valid_official"]
                res[key] = dict(trial=t.number, search_S=t.values[0], quantum=cfg["quantum"], qcfg=q, S=v["S"],
                                consistency=v["consistency"], r2_psi=v["r2_psi"], trainable=r["trainable_params"])
                res_f.write_text(json.dumps(res, indent=1))
                print(f"[confirm] {key} ({cfg['quantum']} {q}): VALID S {v['S']:.4f} cons {v['consistency']:.4f} "
                      f"trainable {r['trainable_params']}", flush=True)
    from scipy.stats import ttest_rel
    for t in top:
        reg = [res[f"c{t.number}-reg-s{s}"]["S"] for s in a.seeds]; mlp = [res[f"c{t.number}-mlp-s{s}"]["S"] for s in a.seeds]
        d = np.subtract(reg, mlp)
        p = ttest_rel(reg, mlp).pvalue if len(reg) > 1 and np.std(d) > 0 else float("nan")
        print(f"[confirm] trial {t.number}: register {np.mean(reg):.4f} vs matched MLP {np.mean(mlp):.4f} "
              f"(delta {d.mean():+.4f}, paired p {p:.3f}, n={len(reg)})", flush=True)


if __name__ == "__main__":
    main()
