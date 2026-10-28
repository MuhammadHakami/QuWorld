#!/bin/bash
# Phase-C Physics-IQ orchestrator v2 (run detached): vanilla -> quantum -> lora.
#
#   A. (Sep 19, user rule GPU-only) T5 prompt-context cache runs on the GPU
#      right after the GPU gate (B), before the arms; resumable, 63 unique prompts.
#   B. GPU gate: runs/phase_c/TEACHER_DONE exists AND no python process holds
#      /dev/nvidia* (polled every 120 s).
#   C. Per arm (vanilla, quantum, lora):
#        deadline (lora: Sep 22 08:00 -> physiq_lora_SKIPPED)  ->
#        adaptercheck (CPU; failure -> BLOCKED_physiq_<arm>, next arm) ->
#        pilot 3 scenarios (batch 1) -> project -> [vanilla: batchtest] ->
#        full generation (detached, resumable, 3 attempts, batch mode from
#        physiq_batch_mode) -> project -> trim -> verify -> score
#        (physiq_<arm>_scores.json + physiq_<arm>_DONE) -> compare vs done arms.
#   D. Opportunistic scorer smoke test (static-frame baseline) after all arms.
#
# GPU processes: systemd-run --user --scope -p MemoryMax=80G; scorer: 60G.
set -u
REPO=/home/mohhakami/Documents/paper/quworld
cd "$REPO" || exit 1
STATUS=runs/phase_c/phase_c_status
RUN="uv run --no-sync python paper/phase_c/physiq_runner.py"
GPU_SCOPE="systemd-run --user --scope -q -p MemoryMax=80G"
MEM_FLOOR_GB=${MEM_FLOOR_GB:-12}   # system-level floor for the generate tripwire (earlyoom fires at ~6 GB)
CPU_SCOPE="systemd-run --user --scope -q -p MemoryMax=60G"
T5_SCOPE="systemd-run --user --scope -q -p MemoryMax=14G"
POLL=120
LORA_DEADLINE="2026-09-22 08:00"
ARMS="vanilla quantum lora"

status() { echo "$(date -Is) physiq: $*" | tee -a "$STATUS"; }
mem_avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
gpu_python_pids() {
  for p in $(fuser /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm 2>/dev/null | tr -s ' ' '\n' | sort -u); do
    case "$(readlink /proc/"$p"/exe 2>/dev/null)" in *python*) echo "$p" ;; esac
  done
}
teacher_running() { pgrep -f 'teacher_cache[.]py' >/dev/null; }
raw_dir() { echo "data/phase_c/physiq/generated_videos/lingbot-world-v2-$1-op-run_01"; }
raw_count() { find "$(raw_dir "$1")" -maxdepth 1 -name '*.mp4' ! -name '*.tmp.mp4' 2>/dev/null | wc -l; }
t5_count() { ls data/phase_c/physiq/t5_context/*.pt 2>/dev/null | wc -l; }
t5_missing() { $RUN --stage t5cache --dry-run 2>/dev/null | grep -o "[0-9]* missing" | grep -o "[0-9]*" || echo 99; }
arm_log() { echo "runs/phase_c/physiq_$1.log"; }

# A. (moved after the GPU gate: T5 runs on the GPU once the teacher/P1 chain is done; user rule Sep 19: GPU only)
# B. GPU gate -------------------------------------------------------------------
status "waiting for runs/phase_c/TEACHER_DONE + free GPU (poll ${POLL}s)"
while :; do
  if [ -f runs/phase_c/TEACHER_DONE ]; then
    pids=$(gpu_python_pids)
    [ -z "$pids" ] && break
    echo "$(date -Is) TEACHER_DONE present; python GPU pids still active: $pids"
  fi
  sleep $POLL
done
status "GPU free (TEACHER_DONE present, no python GPU process); P1 results: $( [ -f runs/phase_c/p1_results.json ] && echo present || echo MISSING )"

# B2. T5 prompt-context cache on the GPU (seconds per prompt; ~14 GB while loaded) -------
while [ "$(t5_missing)" -gt 0 ]; do
  status "T5 cache (GPU): encoding ($(t5_count) cached, $(t5_missing) missing)"
  env PHYSIQ_T5_DEVICE=cuda $GPU_SCOPE $RUN --stage t5cache >> runs/phase_c/physiq_t5cache.log 2>&1
  status "T5 cache: pass finished ($(t5_count) cached, $(t5_missing) missing)"
  [ "$(t5_missing)" -gt 0 ] && sleep $POLL
done
status "T5 cache complete ($(t5_count) contexts, 0 missing)"

# C. arms ---------------------------------------------------------------------
done_arms=""
for arm in $ARMS; do
  log=$(arm_log "$arm")
  if [ "$arm" = lora ] && [ "$(date +%s)" -ge "$(date -d "$LORA_DEADLINE" +%s)" ]; then
    echo "$(date -Is) lora not started before $LORA_DEADLINE" > runs/phase_c/physiq_lora_SKIPPED
    status "lora SKIPPED (deadline $LORA_DEADLINE passed before it could start)"
    continue
  fi
  if [ -f "runs/phase_c/physiq_${arm}_DONE" ]; then
    status "$arm already DONE, skipping"; done_arms="$done_arms $arm"; continue
  fi
  status "=== arm $arm: start ==="
  if ! $RUN --stage adaptercheck --arm "$arm" >> "$log" 2>&1; then
    echo "$(date -Is) adaptercheck failed (see $log)" > "runs/phase_c/BLOCKED_physiq_$arm"
    status "$arm BLOCKED: adapter check failed (see $log); continuing with next arm"
    continue
  fi
  # pilot (batch 1) + projection
  $GPU_SCOPE $RUN --stage generate --arm "$arm" --limit 3 --batch 1 >> "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$(date -Is) pilot failed rc=$rc (see $log)" > "runs/phase_c/BLOCKED_physiq_$arm"
    status "$arm BLOCKED: pilot failed rc=$rc (see $log); continuing with next arm"; continue
  fi
  $RUN --stage project --arm "$arm" >> "$log" 2>&1
  # batch-2 test once (after the vanilla pilot, never while the teacher holds the GPU)
  if [ "$arm" = vanilla ] && [ ! -f runs/phase_c/physiq_batch_mode ]; then
    $GPU_SCOPE $RUN --stage batchtest >> runs/phase_c/physiq_batchtest.log 2>&1 \
      || { status "batchtest failed (see physiq_batchtest.log) -> batch_mode 1"; echo 1 > runs/phase_c/physiq_batch_mode; }
  fi
  # full generation (detached, resumable)
  for attempt in 1 2 3; do
    status "$arm: launching full generation (attempt $attempt, $(raw_count "$arm")/198 present, batch_mode $(cat runs/phase_c/physiq_batch_mode 2>/dev/null || echo 1))"
    setsid nohup $GPU_SCOPE $RUN --stage generate --arm "$arm" >> "$log" 2>&1 < /dev/null &
    gen_pid=$!
    # Sep 20 23:44 reset: system memory hit 5% with only this generator running (bf16 DiT + page-cached shards);
    # earlyoom could not see through the systemd scope. Tripwire: if MemAvailable < MEM_FLOOR_GB for two consecutive
    # 15 s samples, stop the generator (SIGTERM, then SIGKILL) so the attempt loop resumes it instead of losing the box.
    low=0
    while kill -0 "$gen_pid" 2>/dev/null; do
      avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
      if [ "$avail" -lt "$MEM_FLOOR_GB" ]; then low=$((low+1)); else low=0; fi
      if [ "$low" -ge 2 ]; then
        status "$arm: MemAvailable ${avail} GB < ${MEM_FLOOR_GB} GB floor twice -> stopping generator pid $gen_pid (resumable)"
        pkill -TERM -P "$gen_pid" 2>/dev/null; kill -TERM "$gen_pid" 2>/dev/null; sleep 20
        pkill -KILL -P "$gen_pid" 2>/dev/null; kill -KILL "$gen_pid" 2>/dev/null; low=0
      fi
      sleep 15
    done
    [ "$(raw_count "$arm")" -ge 198 ] && break
    status "$arm: generation exited with $(raw_count "$arm")/198 videos; retrying after ${POLL}s"
    sleep $POLL
  done
  if [ "$(raw_count "$arm")" -lt 198 ]; then
    echo "$(date -Is) generation incomplete after 3 attempts ($(raw_count "$arm")/198)" > "runs/phase_c/BLOCKED_physiq_$arm"
    status "$arm BLOCKED: generation incomplete ($(raw_count "$arm")/198); continuing with next arm"; continue
  fi
  $RUN --stage project --arm "$arm" >> "$log" 2>&1
  status "$arm: full generation complete (198/198)"
  $RUN --stage trim --arm "$arm" >> "$log" 2>&1 || { status "$arm BLOCKED: trim failed"; echo trim > "runs/phase_c/BLOCKED_physiq_$arm"; continue; }
  $RUN --stage verify --arm "$arm" >> "$log" 2>&1 || { status "$arm BLOCKED: verify failed"; echo verify > "runs/phase_c/BLOCKED_physiq_$arm"; continue; }
  $CPU_SCOPE $RUN --stage score --arms "$arm" --n-process 4 > "runs/phase_c/physiq_${arm}_score.log" 2>&1 \
    || { status "$arm BLOCKED: score failed (physiq_${arm}_score.log)"; echo score > "runs/phase_c/BLOCKED_physiq_$arm"; continue; }
  [ -f "runs/phase_c/physiq_${arm}_DONE" ] || { status "$arm BLOCKED: no DONE marker after scoring"; continue; }
  status "$arm DONE"
  for prev in $done_arms; do
    $RUN --stage compare --arms "$prev" "$arm" >> "$log" 2>&1 || status "compare $prev vs $arm failed (see $log)"
  done
  done_arms="$done_arms $arm"
done
status "arm chain finished: done=[$done_arms]"

# D. opportunistic scorer smoke test (static-frame baseline) ----------------------
[ -x runs/phase_c/physiq_scorer_smoke.sh ] && [ "$(mem_avail_gb)" -ge 45 ] && ! teacher_running \
  && runs/phase_c/physiq_scorer_smoke.sh >> runs/phase_c/physiq_scorer_smoke.log 2>&1
status "orchestrator exit"
