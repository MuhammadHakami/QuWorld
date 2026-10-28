#!/bin/bash
# Phase-C P0 step 3 + P1 runner (coordinator schedule 2026-09-19):
#   1. teacher-forced chunk cache over the PRE-REGISTERED subset
#      (runs/phase_c/teacher_subset.txt; SUBSET= env overrides; SUBSET=all
#      caches every latent) — resumable, up to 3 attempts, GPU job under
#      `systemd-run --user --scope -p MemoryMax=80G`, log runs/phase_c/teacher.log
#   2. CPU integrity pass over every cached file -> runs/phase_c/TEACHER_CACHE_OK
#   3. P1 adapter training (paper/phase_c/p1_train.py, GPU) while we still hold
#      the GPU -> runs/phase_c/P1_DONE
#   4. runs/phase_c/TEACHER_DONE — written LAST, only after every GPU process of
#      this pipeline has exited (the Physics-IQ orchestrator gates on it).
# Monitor lines go to runs/phase_c/phase_c_status; failures write
# runs/phase_c/BLOCKED and exit without either sentinel.
#
# Launch detached:
#   setsid nohup bash paper/phase_c/run_teacher.sh [teacher_cache.py args] \
#       > runs/phase_c/teacher_runner.log 2>&1 < /dev/null &
set -u
REPO=/home/mohhakami/Documents/paper/quworld
cd "$REPO" || exit 1
STATUS=runs/phase_c/phase_c_status
LOG=runs/phase_c/teacher.log
P1_LOG=runs/phase_c/p1_train.log
OUT=data/phase_c/teacher
LATENTS=data/phase_c/latents
SUBSET=${SUBSET:-runs/phase_c/teacher_subset.txt}
P1_ARGS=${P1_ARGS:-}
GPU_SCOPE="systemd-run --user --scope -q -p MemoryMax=80G"

status() { echo "$(date -Is) teacher-runner: $*" | tee -a "$STATUS"; }
blocked() { status "BLOCKED: $*"; echo "$(date -Is) $*" > runs/phase_c/BLOCKED; exit 1; }

if [ "$SUBSET" = "all" ]; then
  SUBSET_ARGS=(); EXPECTED=$(ls "$LATENTS"/*.pt 2>/dev/null | xargs -n1 basename | sed 's/\.pt$//')
else
  [ -f "$SUBSET" ] || blocked "subset file $SUBSET missing"
  SUBSET_ARGS=(--subset "$SUBSET"); EXPECTED=$(tr -s ' \n' '\n' < "$SUBSET" | grep -v '^$')
fi
N_EXPECTED=$(echo "$EXPECTED" | grep -c .)
[ "$N_EXPECTED" -gt 0 ] || blocked "empty trajectory list"
n_out() { local n=0; for s in $EXPECTED; do [ -f "$OUT/$s.pt" ] && n=$((n + 1)); done; echo $n; }
mkdir -p "$OUT"

mem_avail_gb() { awk '/MemAvailable/{printf "%d", $2/1e6}' /proc/meminfo; }
wait_for_memory() {   # DiT load transient needs ~45 GB CPU + the GPU-side 63 GB
  local need=$1 waited=0
  while [ "$(mem_avail_gb)" -lt "$need" ]; do
    [ $((waited % 600)) -eq 0 ] && status "waiting for MemAvailable >= ${need} GB (now $(mem_avail_gb) GB; other jobs on the box)"
    [ $waited -ge 10800 ] && blocked "MemAvailable stayed below ${need} GB for 3 h"
    sleep 60; waited=$((waited + 60))
  done
}

# 1. cache ------------------------------------------------------------------
rc=1
for attempt in 1 2 3; do
  wait_for_memory 55
  status "cache attempt $attempt: $(n_out)/$N_EXPECTED present; subset=$SUBSET; MemAvailable $(mem_avail_gb) GB; args: $*"
  $GPU_SCOPE uv run --with accelerate python -u paper/phase_c/teacher_cache.py \
      "${SUBSET_ARGS[@]}" "$@" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ "$(n_out)" -ge "$N_EXPECTED" ]; then break; fi
  status "cache attempt $attempt exited rc=$rc with $(n_out)/$N_EXPECTED; retry in 60 s"
  sleep 60
done
[ $rc -eq 0 ] && [ "$(n_out)" -ge "$N_EXPECTED" ] \
  || blocked "teacher_cache.py rc=$rc with $(n_out)/$N_EXPECTED after 3 attempts (see $LOG)"

# 2. integrity (CPU) ----------------------------------------------------------
status "cache python exited rc=0; integrity pass over $N_EXPECTED files (CPU)"
SUMMARY=$(EXPECTED="$EXPECTED" uv run python - <<'EOF' 2>&1 | grep -vE "^(Uninstalled|Installed) "
import os, pathlib, torch
stems = os.environ["EXPECTED"].split()
files = [pathlib.Path("data/phase_c/teacher") / f"{s}.pt" for s in stems]
exp = None; bad = []; rmse = []; secs = 0.0; nbytes = 0
for p in files:
    d = torch.load(p, weights_only=True)
    x, g, m = d["x0_pred"], d["gt"], d["meta"]
    exp = exp or (m["n_chunks"], 16, m["chunk_size"], 60, 104)
    ok = (tuple(x.shape) == exp and tuple(g.shape) == exp and x.dtype == g.dtype == torch.float16
          and bool(torch.isfinite(x).all()) and bool(torch.isfinite(g).all()))
    if not ok: bad.append(p.name)
    rmse.append(sum(m["chunk_rmse"]) / len(m["chunk_rmse"])); secs += sum(m["chunk_secs"]); nbytes += p.stat().st_size
if bad: raise SystemExit(f"INTEGRITY FAIL: {len(bad)} bad files e.g. {bad[:3]}")
print(f"{len(files)} files x0_pred/gt {exp} fp16, all finite, {nbytes/1e9:.2f} GB, mean chunk RMSE {sum(rmse)/len(rmse):.4f} "
      f"(min {min(rmse):.4f} max {max(rmse):.4f}), GPU chunk-time total {secs/60:.1f} min, shift {m['shift']} prompt {m['prompt_name']} kv_source {m['kv_source']}")
EOF
)
irc=$?
echo "$SUMMARY" | tee -a "$LOG"
[ $irc -eq 0 ] || blocked "integrity pass failed: $SUMMARY"
echo "TEACHER_CACHE_OK $(date -Is) | $SUMMARY" > runs/phase_c/TEACHER_CACHE_OK
status "TEACHER_CACHE_OK: $SUMMARY"

# 3. P1 (GPU, still ours) -----------------------------------------------------
status "P1 start (paper/phase_c/p1_train.py $P1_ARGS)"
$GPU_SCOPE uv run python -u paper/phase_c/p1_train.py $P1_ARGS >> "$P1_LOG" 2>&1
prc=$?
[ $prc -eq 0 ] && [ -f runs/phase_c/p1_results.json ] \
  || blocked "p1_train.py rc=$prc (see $P1_LOG); TEACHER cache is complete (TEACHER_CACHE_OK)"
echo "P1_DONE $(date -Is) | $(python3 -c "import json;d=json.load(open('runs/phase_c/p1_results.json'));print('valid rollout VRMSE', {a:round(v['valid_rollout']['mean'],4) for a,v in d['arms'].items()}, 'vanilla', round(d['vanilla_backbone']['rollout'],4))")" > runs/phase_c/P1_DONE
status "P1_DONE written: $(cat runs/phase_c/P1_DONE)"

# 4. TEACHER_DONE last, after all GPU processes exited -------------------------
sleep 5
echo "TEACHER_DONE $(date -Is) | cache: $SUMMARY | P1: see runs/phase_c/P1_DONE" > runs/phase_c/TEACHER_DONE
status "TEACHER_DONE written (GPU released)"
