"""Powered replication (31-powered-replication-prereg.md): the ms2-selected configuration of every arm, re-run
on NEW replication units, one TEST evaluation each. Resumable: a unit whose JSON exists is skipped.

    uv run --no-sync python paper/hpo/replicate.py pelzer   --trunk-seeds 1337 1338 1339 --seeds 2001 2002 2003 2004
    uv run --no-sync python paper/hpo/replicate.py sophelio --seeds 1340 1341 1342 1343 1344 1345 1346 1347
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "hpo"))
ARMS = ("none", "mlp", "q")


def frozen_config(bench: str, arm: str) -> tuple[str, dict]:
    f = ROOT / "runs" / "hpo" / bench / f"{arm}_ms2_final.json"
    if not f.exists():
        raise SystemExit(f"missing frozen configuration {f} — the ms2 selection must finish first (prereg)")
    d = json.load(open(f))
    return d["variant"], d["hp"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench"); ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--trunk-seeds", type=int, nargs="+", default=[None])
    a = ap.parse_args()
    mod = importlib.import_module(f"bench_{a.bench}")
    out = ROOT / "runs" / "replication" / a.bench; out.mkdir(parents=True, exist_ok=True)
    configs = {arm: frozen_config(a.bench, arm) for arm in ARMS}
    for trunk in a.trunk_seeds:
        for seed in a.seeds:
            for arm in ARMS:                       # arms interleaved per unit: drift in the box hits all arms alike
                tag = f"{arm}_t{trunk}_s{seed}" if trunk is not None else f"{arm}_s{seed}"
                f = out / f"{tag}.json"
                if f.exists():
                    continue
                variant, hp = configs[arm]
                kw = {"trunk_seed": trunk} if trunk is not None else {}
                res = mod.run(variant, hp, seed, "test", **kw)
                res.update(arm=arm, variant=variant, hp=hp, seed=seed, trunk_seed=trunk)
                f.write_text(json.dumps(res, indent=1))
                print(f"[rep] {a.bench} {tag}: test {res['metric']:.6f}", flush=True)


if __name__ == "__main__":
    main()
