"""§37 routing: inference-time strategies for the RouteFinder EAL-MD row, scored on the generated validation strata.

Loads a §23 EAL-MD policy (runs/routefinder/s23/eal_md_<arm>_s1.pt; built exactly as paper/vrp/run_rf_eal.py builds it,
incl. the dead-end guard), decodes every instance of runs/strat/routing/val_seed0/*.npz (strata.json must exist) with
one decoding strategy, and writes the per-instance best cost (and, for greedy, the per-augmentation best) to
runs/strat/routing/scores/<arm>__<strategy>.json. The official test files are never opened.

Strategies (name -> decode):
  g8_bf16     dihedral x8, multi-start greedy, bf16 autocast   (= the §23 row's test decode: Lightning bf16-mixed)
  g8_fp16     same, fp16 autocast                                (= RouteFinder test.py default)
  g8_fp32     same, full fp32                                     (protocol-faithful precision variants)
  g32sym      32 symmetric (rotation+reflection, first = identity) copies, multi-start greedy, bf16
  s8x4        dihedral x8, multi-start SAMPLING T=1, 4 samples per start, bf16 (extra test-time compute)
  s8x4t05     same with softmax temperature 0.5

    cd <repo> && flock runs/strat/gpu.lock timeout 1800 env QW_GPU_GB=12 runs/capped.sh strat_vrp 16 \
        .venv-vrp/bin/python paper/strat/routing/eval_md.py --jobs q30-geo=3d:g8_bf16:128,vanilla:g8_bf16:128
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper"))
import torch  # noqa: E402

from gpucap import cap_from_env  # noqa: E402

CAP = cap_from_env()                                        # before any CUDA allocation
os.chdir(ROOT / "baseline" / "routefinder")                 # run_rf_eal builds from checkpoints/100/rf-transformer.ckpt
sys.path.insert(0, str(ROOT / "paper" / "vrp"))
import run_rf_eal as R  # noqa: E402  (installs rf_compat aliases and the dead-end guard, as in the §23 runs)
from rl4co.data.transforms import StateAugmentation  # noqa: E402
from rl4co.utils.ops import unbatchify  # noqa: E402
from routefinder.envs.mtdvrp import MTVRPEnv  # noqa: E402

STRAT = ROOT / "runs" / "strat" / "routing"
STRATEGIES = {
    "g8_bf16": dict(aug=8, fn="dihedral8", decode="multistart_greedy", amp=torch.bfloat16, samples=1),
    "g8_fp16": dict(aug=8, fn="dihedral8", decode="multistart_greedy", amp=torch.float16, samples=1),
    "g8_fp32": dict(aug=8, fn="dihedral8", decode="multistart_greedy", amp=None, samples=1),
    "g32sym": dict(aug=32, fn="symmetric", decode="multistart_greedy", amp=torch.bfloat16, samples=1),
    "s8x4": dict(aug=8, fn="dihedral8", decode="multistart_sampling", amp=torch.bfloat16, samples=4, temperature=1.0),
    "s8x4t05": dict(aug=8, fn="dihedral8", decode="multistart_sampling", amp=torch.bfloat16, samples=4, temperature=0.5),
}


OFFVAL_VARIANTS = "mdcvrp,mdvrptw,mdovrpbltw"    # harness check subset (official val files, never test)


def load_policy(arm: str, dev: torch.device):
    if arm == "eal0":                                        # EAL at step 0 = the pretrained RF-TE (zero-shot); its
        model = R.build("checkpoints/100/rf-transformer.ckpt", "vanilla")     # official-val gap is recorded in every
        return model.policy.to(dev).eval()                   # §23 run JSON (zero_shot_val_gap): exact harness check
    if "@" in arm:                                           # "<variant>@<runs/routefinder subdir>@<seed>" (doc 35 §39e)
        arm, sub, seed = arm.split("@")
        pt = ROOT / "runs" / "routefinder" / sub / f"eal_md_{arm}_s{seed}.pt"
    else:
        pt = ROOT / "runs" / "routefinder" / "s23" / f"eal_md_{arm}_s1.pt"
    model = R.build("checkpoints/100/rf-transformer.ckpt", arm)
    sd = torch.load(pt, map_location="cpu", weights_only=False)
    assert sd["variant"] == arm and sd["setting"] == "md", (sd["variant"], sd["setting"])
    model.policy.load_state_dict(sd["policy"], strict=True)
    return model.policy.to(dev).eval()


def decode(policy, env, td_raw, cfg: dict, dev: torch.device, seed: int):
    """Best cost per instance [B] and best per augmentation copy [B, aug] (max over starts and samples)."""
    td = env.reset(td_raw.to(dev))
    n_start = env.get_num_starts(td)
    aug = cfg["aug"]
    torch.manual_seed(seed)                                  # the symmetric copies and the samples are seeded
    td = StateAugmentation(num_augment=aug, augment_fn=cfg["fn"])(td)
    amp = torch.autocast("cuda", dtype=cfg["amp"]) if cfg["amp"] is not None else contextlib.nullcontext()
    best = None
    kw = {"temperature": cfg["temperature"]} if "temperature" in cfg else {}
    with torch.inference_mode(), amp:
        for _ in range(cfg["samples"]):
            out = policy(td.clone(), env, phase="test", decode_type=cfg["decode"], num_starts=n_start, **kw)
            r = unbatchify(out["reward"].float(), (aug, n_start)).max(-1).values        # [B, aug]
            best = r if best is None else torch.maximum(best, r)
    return (-best.max(-1).values).cpu(), (-best).cpu()


def run_one(arm: str, strategy: str, batch: int, official_val: bool, variants: str, policy, t_end: float) -> bool:
    """Score one (arm, strategy); returns False if the time budget ran out first (nothing written)."""
    meta = json.loads((STRAT / "strata.json").read_text())
    strata = [s for s in meta["strata"] if variants == "all" or s["variant"] in variants.split(",")]
    env = MTVRPEnv(check_solution=False, load_solutions=official_val)
    cfg = STRATEGIES[strategy]
    res = dict(arm=arm, strategy=strategy, cfg={k: str(v) for k, v in cfg.items()}, gpu_cap_frac=CAP, batch=batch,
               test_decode_type=getattr(policy, "test_decode_type", None), variants={})
    t0 = time.time()
    for k, s in enumerate(strata):
        if time.time() > t_end:
            print(f"[strat-vrp] time budget reached in {arm} {strategy}; not written", flush=True)
            return False
        f = (f"data/{s['variant']}/val/100.npz" if official_val else str(ROOT / s["file"]))
        td_all = env.load_data(f)[s["indices"]]
        bks = td_all["costs_bks"].float() if official_val else None
        best, per_aug = [], []
        for i in range(0, td_all.batch_size[0], batch):
            b, pa = decode(policy, env, td_all[i:i + batch], cfg, torch.device("cuda"), seed=1000 * k + i)
            best.append(b); per_aug.append(pa)
        best = torch.cat(best); per_aug = torch.cat(per_aug)
        res["variants"][s["variant"]] = dict(cost=best.tolist(), per_aug=per_aug.tolist() if cfg["aug"] <= 32 else None,
                                             mean=float(best.mean()))
        extra = ""
        if bks is not None:
            gap = 100 * (best - bks.abs()) / bks.abs()
            res["variants"][s["variant"]].update(gap=gap.tolist(), gap_mean=float(gap.mean()))
            extra = f", official-val gap {gap.mean():.3f}%"
        print(f"[strat-vrp] {arm} {strategy} {s['variant']}: mean cost {best.mean():.4f}{extra} "
              f"({time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", flush=True)
    res["seconds"] = time.time() - t0; res["dead_end_rows_cumulative"] = R.rf_compat.dead_end_count()
    out_path(arm, strategy, official_val).write_text(json.dumps(res))
    print(f"[strat-vrp] done {arm} {strategy} in {res['seconds']:.0f}s", flush=True)
    return True


def out_path(arm: str, strategy: str, official_val: bool) -> pathlib.Path:
    out = STRAT / "scores"; out.mkdir(parents=True, exist_ok=True)
    return out / f"{arm}__{strategy}{'__offval' if official_val else ''}.json"


def main() -> None:
    """--jobs 'arm:strategy:batch[:offval],...' run in order in ONE process (one lock acquisition); done jobs are
    skipped (resume), and the process stops before --budget seconds so the 30-min lock timeout is never hit."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--variants", default="all")
    ap.add_argument("--budget", type=float, default=1560)
    a = ap.parse_args()
    t_end = time.time() + a.budget
    policies = {}
    for spec in a.jobs.split(","):
        arm, strategy, batch, *flag = spec.split(":")
        official_val = bool(flag and flag[0] == "offval")
        assert strategy in STRATEGIES, strategy
        if out_path(arm, strategy, official_val).exists():
            continue
        if arm not in policies:
            policies.clear(); torch.cuda.empty_cache()
            policies[arm] = load_policy(arm, torch.device("cuda"))
        variants = OFFVAL_VARIANTS if official_val else a.variants
        if not run_one(arm, strategy, int(batch), official_val, variants, policies[arm], t_end):
            break


if __name__ == "__main__":
    main()
