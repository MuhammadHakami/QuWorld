"""RRNCO ACVRP + test-time local search (doc 35 §41b; doc 36 RRNCO fix 1). Extra test-time compute: reported with a dagger,
never as like-for-like with RADAR's greedy number.

The released RRNet decodes each test instance as test.py does (dihedral x8, 101 starts); the best tour per instance is then
improved by RRNCO's own PyVRP local search (rrnco/envs/rcvrp/local_search.py) on the instance's own (asymmetric, normalised)
distance matrix, and scored with the environment's reward (real distances). Gaps against the reference costs implied by
the RRNCO paper (cost / (1 + gap)), as in rrnco_hybrid.py.

    cd baseline/rrnco && ../../.venv-vrp/bin/python ../../paper/vrp/rrnco_ls.py [--workers 16] [--trials 10] [--limit N]
        -> runs/routefinder/rrnco_ls.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RR = ROOT / "baseline" / "rrnco"
sys.path.insert(0, str(RR)); sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from rl4co.data.utils import load_npz_to_tensordict  # noqa: E402
from rl4co.utils.ops import batchify, unbatchify  # noqa: E402

import test as rrtest  # noqa: E402
from rrnco.envs.rcvrp import RCVRPEnv  # noqa: E402
from rrnco.envs.rcvrp.local_search import local_search  # noqa: E402
from rrnco.models import RRNet  # noqa: E402
from rrnco_hybrid import FILES, PUBLISHED  # noqa: E402


@torch.no_grad()
def run(policy, name: str, workers: int, trials: int, limit: int | None, bs: int = 32) -> tuple[float, float, float]:
    td = load_npz_to_tensordict(str(RR / "data" / "rcvrp" / FILES[name]))
    if limit:
        td = td[:limit]
    td["demand"] = td["demand"] / td["capacity"].unsqueeze(-1); td["capacity"] = torch.ones_like(td["capacity"])
    env = RCVRPEnv(check_solution=False, generator_params=dict(num_loc=100)); dev = next(policy.parameters()).device
    before, after, t_ls = [], [], 0.0
    for b in rrtest.get_dataloader(td, bs):
        with torch.autocast("cuda"), torch.inference_mode():
            r = env.reset(rrtest.augment(b.clone())).to(dev)
            out = policy(r, env, return_actions=True, phase="val", calc_reward=False, num_starts=101)
            rew = env.get_reward(batchify(r, 101), out["actions"]); rew = rew[0] if isinstance(rew, tuple) else rew
        B = len(b); R = unbatchify(rew, (8, 101)).reshape(B, -1); best = R.argmax(-1)
        acts = unbatchify(out["actions"], (8, 101)).reshape(B, 8 * 101, -1)
        A = acts[torch.arange(B, device=acts.device), best.to(acts.device)]
        orig = env.reset(b.clone()).cpu(); orig["distances"] = orig["distance_matrix"]
        t = time.time(); A2 = local_search(orig, A.cpu(), max_trials=trials, num_workers=workers); t_ls += time.time() - t
        r1 = env.get_reward(orig, A.cpu()); r2 = env.get_reward(orig, A2)
        r1 = r1[0] if isinstance(r1, tuple) else r1; r2 = r2[0] if isinstance(r2, tuple) else r2
        before.append(-r1.mean().item()); after.append(-torch.maximum(r1, r2).mean().item())
    return sum(before) / len(before), sum(after) / len(after), t_ls


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args(); t0 = time.time()
    model = RRNet.load_from_checkpoint(str(RR / "checkpoints/rcvrp/epoch_199.ckpt"), map_location="cpu", strict=False,
                                       load_baseline=False, weights_only=False)
    policy = model.policy.to("cuda").eval(); ref = {k: c / (1 + g / 100) for k, (c, g) in PUBLISHED.items()}; rep = {}
    for k in FILES:
        c0, c1, tls = run(policy, k, a.workers, a.trials, a.limit or None)
        rep[k] = dict(decode=c0, decode_ls=c1, gap_decode=100 * (c0 / ref[k] - 1), gap_ls=100 * (c1 / ref[k] - 1), ls_seconds=tls)
        print(f"[rrnco-ls] {k}: decode {c0:.3f} ({rep[k]['gap_decode']:.2f}%) -> +LS {c1:.3f} ({rep[k]['gap_ls']:.2f}%), LS {tls:.0f}s", flush=True)
    out = ROOT / "runs" / "routefinder" / f"rrnco_ls{'_lim' if a.limit else ''}.json"
    out.write_text(json.dumps(dict(trials=a.trials, limit=a.limit, results=rep, seconds=round(time.time() - t0)), indent=1))
    print(f"RRNCO_LS_DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
