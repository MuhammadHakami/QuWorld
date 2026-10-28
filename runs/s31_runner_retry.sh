#!/bin/bash
# doc 35 §31/§31b VALID stage: sealed-TEST caches for the two prospective rows, then pooled/static/proxy linear
# correctors (TRAIN fit, V1 early stop, V2 report; §31c parity split, checked from metadata first), seeds 1-3, then the predictability score S per row. No TEST.
# RETRY copy (doc 35 §31 operational retry): waits for the first runner (PID 602570) to exit, runs ONLY cells whose result
# file is missing (a transient CUDA OOM at cache load, before training), up to 3 attempts each; same checks and S step.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; L=runs/s31/s31.log
mem() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt ${1:-30} ]; do sleep 60; done; }
fail=0
until grep -q S31_ACQUIRED $S; do grep -q "CRASH s31 download" $S && exit 1; sleep 120; done
while kill -0 602570 2>/dev/null; do sleep 30; done
.venv/bin/python runs/s31_split_check.py >> $L 2>&1 || { echo "CRASH s31 split check (§31c: stop, register a stratified split) $(date -Is)" >> $S; exit 1; }
for row in active_matter gray_scott_reaction_diffusion; do
  per=$([ $row = active_matter ] && echo 25 || echo 10)                 # §31d: balanced VALID sample per trajectory
  [ -f runs/alloc/cache/$row/valid.pt ] || { mem 35; ALLOC_NO_TEST=1 ALLOC_VALID_PER_TRAJ=$per QW_GPU_GB=16 timeout 14400 runs/capped.sh s31c 40 \
    .venv/bin/python paper/alloc/cache.py well $row >> $L 2>&1 || { echo "CRASH s31 cache $row $(date -Is)" >> $S; fail=1; continue; }; }
  [ -f runs/alloc/cache/$row/test.pt ] && { echo "CRASH s31 TEST cache exists for $row (must be sealed) $(date -Is)" >> $S; exit 1; }
  for seed in 1 2 3; do for st in pooled static proxy; do
    [ -f runs/alloc/results/$row/vsplit/${st}_lin_K30_s${seed}.json ] && continue
    ok=0; for try in 1 2 3; do
      mem 30; QW_GPU_GB=8 timeout 3600 runs/capped.sh s31v 16 .venv/bin/python paper/alloc/coproc.py $row --strategy $st --core lin \
        --K 30 --seed $seed --vsplit >> $L 2>&1 && { ok=1; break; }
      echo "RETRY s31 vsplit $row $st s$seed attempt $try failed $(date -Is)" >> $S; sleep 120
    done
    [ $ok = 1 ] || { echo "CRASH s31 vsplit $row $st s$seed (3 attempts) $(date -Is)" >> $S; fail=1; }
  done; done
done
[ $fail = 0 ] || { echo "S31_INCOMPLETE (a cache or vsplit cell failed; no S, TEST stays locked) $(date -Is)" >> $S; exit 1; }
.venv/bin/python - >> $L 2>&1 <<'PY' || { echo "S31_INCOMPLETE (missing or non-finite cell; no S, TEST stays locked) $(date -Is)" >> $S; exit 1; }
import json, math, statistics as st, pathlib, sys
R = pathlib.Path("runs/alloc/results"); out = {}
for row in ("active_matter", "gray_scott_reaction_diffusion"):       # fail closed: both caches, all 18 cells, finite
    assert (pathlib.Path("runs/alloc/cache") / row / "valid.pt").exists(), f"{row}: no VALID cache"
    assert not (pathlib.Path("runs/alloc/cache") / row / "test.pt").exists(), f"{row}: TEST cache must not exist"
    for s in ("pooled", "static", "proxy"):
        for i in (1, 2, 3):
            f = R / row / "vsplit" / f"{s}_lin_K30_s{i}.json"
            assert f.exists(), f"missing {f}"
            d = json.loads(f.read_text())
            assert d.get("vsplit") and math.isfinite(d["valid"]) and math.isfinite(d["test"]), f"bad cell {f}"
for row in ("active_matter", "gray_scott_reaction_diffusion"):
    g = lambda s, k: [json.loads((R / row / "vsplit" / f"{s}_lin_K30_s{i}.json").read_text())[k] for i in (1, 2, 3)]
    try:
        v1 = {s: st.mean(g(s, "valid")) for s in ("static", "proxy")}; route = min(v1, key=v1.get)
        S = 1 - st.mean(g(route, "test")) / st.mean(g("pooled", "test"))          # "test" = V2 in --vsplit runs
        out[row] = {"route_lin": route, "V1_mean": v1, "V2_route": st.mean(g(route, "test")), "V2_pooled": st.mean(g("pooled", "test")),
                    "S": S, "prediction": "gain in both cores" if S >= 0.01 else ("no gain >= 1%" if S <= 0 else "abstain")}
    except FileNotFoundError as e:
        sys.exit(f"missing cell: {e}")
pathlib.Path("runs/s31/predictability.json").write_text(json.dumps(out, indent=1)); print("[s31]", json.dumps(out), flush=True)
PY
echo "S31_VALID_DONE $(date -Is)" >> $S
