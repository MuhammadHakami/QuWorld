"""Doc 35 §37d (registered before any TEST decode here): RouteFinder EAL-MD official TEST files, extra decode compute.

For each (arm, strategy) job, decodes RouteFinder's 16 MD test files (data/<variant>/test/100.npz, 1,000 instances
each; PyVRP reference costs from 100_sol_pyvrp.npz, as the §23 Lightning test used) with eval_md.py's decode, and writes
the per-instance gap per variant to runs/strat/routing/test/<arm>__<strategy>/<variant>.json (resumable per variant;
the process stops before --budget seconds).

Registered jobs, in order: q30-geo=3d:g8_bf16 (harness check: must reproduce the §23 test_gap per variant within
0.01 pp), then q30-geo=3d:s8x4t05 and q30-geo=3d-mlp:s8x4t05 (4x decode compute, sampling T=0.5; daggered rows).

    flock runs/strat/gpu.lock env QW_GPU_GB=12 timeout 1800 runs/capped.sh s37d 16 .venv-vrp/bin/python \
        paper/strat/routing/test_decode.py --jobs q30-geo=3d:g8_bf16,q30-geo=3d:s8x4t05,q30-geo=3d-mlp:s8x4t05
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_md as E  # noqa: E402  (cap_from_env, chdir to baseline/routefinder, dead-end guard)
import torch  # noqa: E402
from routefinder.envs.mtdvrp import MTVRPEnv  # noqa: E402

OUT = E.STRAT / "test"
S23 = E.ROOT / "runs" / "routefinder" / "s23"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", required=True); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--budget", type=float, default=1560)
    a = ap.parse_args()
    t_end = time.time() + a.budget
    meta = json.loads((E.STRAT / "strata.json").read_text())
    variants = [s["variant"] for s in meta["strata"]]
    env = MTVRPEnv(check_solution=False, load_solutions=True)
    dev = torch.device("cuda")
    for spec in a.jobs.split(","):
        arm, strategy = spec.split(":")
        d = OUT / f"{arm.replace('@', '_')}__{strategy}"; d.mkdir(parents=True, exist_ok=True)
        todo = [v for v in variants if not (d / f"{v}.json").exists()]
        if not todo:
            continue
        policy = E.load_policy(arm, dev); cfg = E.STRATEGIES[strategy]
        for k, v in enumerate(todo):
            if time.time() > t_end:
                print(f"[s37d] budget reached before {arm} {strategy} {v}", flush=True); return
            td_all = env.load_data(f"data/{v}/test/100.npz")
            bks = td_all["costs_bks"].float().abs()
            best = []
            t0 = time.time()
            for i in range(0, td_all.batch_size[0], a.batch):
                b, _ = E.decode(policy, env, td_all[i:i + a.batch], cfg, dev, seed=7919 * (variants.index(v) + 1) + i)
                best.append(b)
            best = torch.cat(best); gap = 100 * (best - bks) / bks
            (d / f"{v}.json").write_text(json.dumps(dict(arm=arm, strategy=strategy, variant=v, n=len(best),
                                                         gap_mean=float(gap.mean()), gap=gap.tolist(), cost=best.tolist())))
            print(f"[s37d] {arm} {strategy} {v}: test gap {gap.mean():.3f}% ({time.time() - t0:.0f}s)", flush=True)
        del policy; torch.cuda.empty_cache()
    print("S37D_ALL_DONE", flush=True)


def summarize() -> None:
    meta = json.loads((E.STRAT / "strata.json").read_text()); variants = [s["variant"] for s in meta["strata"]]
    res = {}
    for d in sorted(OUT.glob("*__*")):
        g = {v: json.loads((d / f"{v}.json").read_text())["gap_mean"] for v in variants if (d / f"{v}.json").exists()}
        res[d.name] = dict(n_variants=len(g), gap_mean=sum(g.values()) / len(g) if g else None, per_variant=g)
    ref = json.loads((S23 / "eal_md_q30-geo=3d_s1.json").read_text())["test_gap"]
    chk = res.get("q30-geo=3d__g8_bf16")
    if chk and chk["n_variants"] == len(variants):
        diffs = {v: chk["per_variant"][v] - ref[v] for v in variants}
        res["harness_check"] = dict(max_abs_diff_pp=max(abs(x) for x in diffs.values()), ok=max(abs(x) for x in diffs.values()) < 0.01)
    (E.STRAT / "test_s37d.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v.get("gap_mean") if isinstance(v, dict) else v) for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    if "--summarize" in sys.argv:
        summarize()
    else:
        main()
