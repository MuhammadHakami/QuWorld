"""M1 (doc 35 §28): Rydberg-blockade MWIS move selection after the neural routing policy, against matched classical
selectors, with the Shapley split of the gap reduction between the policy and the local-search layer.

Pipeline per instance: constructor (EAL-fine-tuned RouteFinder policy, greedy decoding; or nearest neighbour) -> R rounds
of [top-n improving 2-opt / relocate moves -> conflict graph -> selector picks an independent set -> apply all at once].
Selectors: rydberg (emulated blockade array, `shots` samples), exact, greedy, anneal (`shots` chains), best1 (the single
best move per round: no combinatorial selection). Variants: the unseen multi-depot ones without backhauls or time
windows (mdcvrp, mdovrp, mdvrpl, mdovrpl), where these moves keep every route feasible.

    cd baseline/routefinder && ../../.venv-vrp/bin/python ../../paper/qmethods/run_mwis_vrp.py --split val --n 200 \
        --policy ../../runs/routefinder/s23/eal_md_vanilla_s1.pt --seed 1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper/vrp"))
from paper.qmethods import mwis as M  # noqa: E402
from paper.qmethods import vrp_moves as V  # noqa: E402

VARIANTS = ("mdcvrp", "mdovrp", "mdvrpl", "mdovrpl")
SELECTORS = ("best1", "greedy", "anneal", "rydberg", "exact")


def load_split(variant: str, split: str, n: int):
    d = np.load(f"data/{variant}/{split}/100.npz"); s = np.load(f"data/{variant}/{split}/100_sol_pyvrp.npz")
    n = min(n, len(d["locs"]))                                          # VALID holds 128 instances per variant
    insts = []
    for k in range(n):
        nd = int(d["num_depots"][k][0])
        insts.append(V.instance_from_arrays(
            d["locs"][k], d["demand_linehaul"][k], d["vehicle_capacity"][k][0], nd,
            limit=d["distance_limit"][k][0] if "distance_limit" in d.files else None,
            open_route=bool(d["open_route"][k][0]) if "open_route" in d.files else False))
    return insts, -s["costs"][:n].astype(np.float64)


@torch.no_grad()
def policy_actions(policy: str, variant: str, split: str, n: int) -> np.ndarray:
    import run_rf_eal as E                                       # installs rf_compat
    env, _ = E.env_for("md")
    model = E.build("checkpoints/100/rf-transformer.ckpt", "vanilla")
    if policy != "released":
        model.policy.load_state_dict(torch.load(policy, weights_only=False, map_location="cpu")["policy"])
    model = model.to("cuda").eval()
    td = env.reset(env.load_data(f"./data/{variant}/{split}/100.npz")[:n].to("cuda"))
    out = model.policy(td, env, phase="test", decode_type="greedy", return_actions=True)
    return out["actions"].cpu().numpy()


def selector(name: str, shots: int, gen: torch.Generator):
    """Batched selector: [(moves, edges)] -> chosen move indices per instance."""
    def pick(jobs):
        if name == "best1":
            return [[0] for _ in jobs]
        graphs = [M.Graph(len(m), tuple(e), torch.tensor([x.gain for x in m], dtype=torch.float32, device="cuda"))
                  for m, e in jobs]
        if name.startswith("exact_eps"):                                  # exact_eps1 / exact_eps2: eps = 1% / 2%
            return [[int(i) for i in torch.nonzero(M.exact_eps(g.n, g.edges, g.w, int(name[9:]) / 100, gen)).flatten()]
                    for g in graphs]
        if name == "rydberg":
            xs = M.rydberg_batch(graphs, shots, generator=gen)
        elif name == "anneal":
            xs = M.anneal_batch(graphs, shots, generator=gen)
        else:
            xs = [M.SELECTORS[name](g.n, g.edges, g.w) for g in graphs]
        return [[int(i) for i in torch.nonzero(x).flatten()] for x in xs]
    return pick


def shapley_two(v_none: float, v_a: float, v_b: float, v_ab: float) -> tuple[float, float]:
    """Exact Shapley values of players a, b for a gain function v (v(empty) is the reference)."""
    phi_a = 0.5 * ((v_a - v_none) + (v_ab - v_b))
    return phi_a, (v_ab - v_none) - phi_a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=("val", "test")); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--policy", default="released"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=10); ap.add_argument("--max_n", type=int, default=18)
    ap.add_argument("--shots", type=int, default=100); ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default=str(ROOT / "runs/qmethods/m1"))
    ap.add_argument("--selectors", default=",".join(SELECTORS), help="comma list; the §28 control adds exact_eps1,exact_eps2")
    ap.add_argument("--tag", default="", help="suffix of the output JSON")
    a = ap.parse_args()
    if not 1 <= a.max_n <= M.MAX_N:
        raise SystemExit(f"--max_n must be in [1, {M.MAX_N}]")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = "released" if a.policy == "released" else pathlib.Path(a.policy).stem
    res = {"args": vars(a), "variants": {}}
    for v in a.variants.split(","):
        insts, bks = load_split(v, a.split, a.n)
        n = len(insts)
        acts = policy_actions(a.policy, v, a.split, n)
        starts = {"policy": [V.split(acts[k], insts[k].n_depot) for k in range(n)],
                  "nn": [V.nearest_neighbour(inst) for inst in insts]}
        rv = {}
        for c, sols in starts.items():
            costs = np.array([V.cost(inst, r) for inst, r in zip(insts, sols)])
            rv[f"{c}"] = {"gap": float(np.mean(costs / bks - 1) * 100), "cost": costs.tolist()}
            for s in a.selectors.split(","):
                gen = torch.Generator(device="cuda").manual_seed(a.seed)
                t0 = time.time()
                new, calls = V.improve_batch(insts, sols, selector(s, a.shots, gen), a.rounds, a.max_n)
                assert all(V.feasible(inst, r) for inst, r in zip(insts, new))
                fin = np.array([V.cost(inst, r) for inst, r in zip(insts, new)])
                rv[f"{c}+{s}"] = {"gap": float(np.mean(fin / bks - 1) * 100), "cost": fin.tolist(),
                                  "seconds": time.time() - t0, "calls": calls}
                print(f"[m1] {a.split} {v} {c}+{s}: gap {rv[f'{c}+{s}']['gap']:.3f}% "
                      f"(start {rv[c]['gap']:.3f}%) {time.time() - t0:.0f}s", flush=True)
        # credit: gain = gap reduction vs the nearest-neighbour start; players = policy, local search (per selector)
        g = lambda key: rv["nn"]["gap"] - rv[key]["gap"]
        sel = a.selectors.split(",")
        rv["shapley"] = {s: dict(zip(("policy", "local_search"), shapley_two(0.0, g("policy"), g(f"nn+{s}"), g(f"policy+{s}"))))
                         for s in sel}
        if "exact" in sel and "rydberg" in sel:
            ls_exact = rv["shapley"]["exact"]["local_search"]
            rv["quantum_share_of_exact"] = rv["shapley"]["rydberg"]["local_search"] / ls_exact if ls_exact > 0 else None
        res["variants"][v] = rv
    dst = out / f"m1_{a.split}_{tag}_s{a.seed}{a.tag}.json"
    dst.write_text(json.dumps(res)); print("wrote", dst)


if __name__ == "__main__":
    main()
