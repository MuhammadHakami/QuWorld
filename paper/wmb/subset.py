"""Pre-registered WorldModelBench subset: 60 of the 350 instances, stratified by domain
(round-robin over a seeded permutation within each of the 7 domains), fixed seed 20260919.
Writes paper/wmb/subset60.json (list of instances, same schema as worldmodelbench.json).
Usage: python paper/wmb/subset.py [n=60]
"""
import collections, json, pathlib, random, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "baseline" / "WorldModelBench" / "worldmodelbench.json"
OUT = pathlib.Path(__file__).resolve().parent / "subset60.json"
SEED = 20260919

def main(n: int = 60) -> None:
    data = json.load(open(SRC))
    by_dom = collections.defaultdict(list)
    for x in data:
        by_dom[x["domain"]].append(x)
    rng = random.Random(SEED)
    pools = {d: rng.sample(v, len(v)) for d, v in sorted(by_dom.items())}
    picked = []
    while len(picked) < n:
        for d in sorted(pools):
            if pools[d] and len(picked) < n:
                picked.append(pools[d].pop())
    OUT.write_text(json.dumps(picked, indent=1))
    print(f"{len(picked)} instances -> {OUT}; per domain:", dict(collections.Counter(x['domain'] for x in picked)))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
