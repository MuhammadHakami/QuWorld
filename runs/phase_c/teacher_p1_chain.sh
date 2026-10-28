#!/bin/bash
# Phase C chain (coordinator, 2026-09-19): wait for pilot -> pick config -> 45-traj teacher cache -> P1 -> sentinels.
export PATH="$HOME/.local/bin:$PATH"
set -u
cd /home/mohhakami/Documents/paper/quworld
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/phase_c/phase_c_status
log(){ echo "$(date -Is) chain: $*" >> "$S"; }
log "chain runner started (waits for teacher pilot to exit)"
while pgrep -f "teacher_cache.py --pilot" > /dev/null; do sleep 30; done
# pick config = lowest mean rmse among pilot lines; tie -> shift 3 physics (first listed)
best=$(grep -E "teacher pilot shift=" runs/phase_c/pilot.log | sed -E 's/.*shift=([0-9.]+) prompt=([a-z]+) rmse mean ([0-9.]+).*/\3 \1 \2/' | sort -n | head -n 1)
rmse=$(echo "$best" | awk '{print $1}'); shift_v=$(echo "$best" | awk '{print $2}'); prompt=$(echo "$best" | awk '{print $3}')
[ -z "$shift_v" ] && { log "FAILED to parse pilot config"; echo "$(date -Is) chain: no pilot config" > runs/phase_c/BLOCKED; exit 1; }
log "config chosen: shift=$shift_v prompt=$prompt (pilot rmse $rmse); subset=runs/phase_c/teacher_subset.txt"
until [ "$(awk '/MemAvailable/{printf "%d", $2/1e6}' /proc/meminfo)" -ge 55 ]; do sleep 30; done
log "teacher cache start (45 traj)"
uv run --with accelerate python paper/phase_c/teacher_cache.py --subset runs/phase_c/teacher_subset.txt --shift "$shift_v" --prompt-name "$prompt" >> runs/phase_c/teacher.log 2>&1
missing=0; for stem in $(cat runs/phase_c/teacher_subset.txt); do [ -f "data/phase_c/teacher/${stem}.pt" ] || missing=$((missing+1)); done
if [ "$missing" -ne 0 ]; then log "teacher cache INCOMPLETE: $missing stems missing — retrying once"; uv run --with accelerate python paper/phase_c/teacher_cache.py --subset runs/phase_c/teacher_subset.txt --shift "$shift_v" --prompt-name "$prompt" >> runs/phase_c/teacher.log 2>&1; fi
missing=0; for stem in $(cat runs/phase_c/teacher_subset.txt); do [ -f "data/phase_c/teacher/${stem}.pt" ] || missing=$((missing+1)); done
[ "$missing" -ne 0 ] && { log "teacher cache FAILED ($missing missing)"; echo "$(date -Is) teacher cache incomplete" > runs/phase_c/BLOCKED; exit 1; }
log "teacher cache complete; P1 start (3 arms x 3 seeds, unit budget 420 s)"
while pgrep -f "teacher_cache.py" > /dev/null; do sleep 10; done
uv run python paper/phase_c/p1_train.py --unit-budget-s 420 >> runs/phase_c/p1.log 2>&1
if [ -f runs/phase_c/p1_results.json ]; then log "P1 complete"; touch runs/phase_c/P1_DONE; else log "P1 FAILED (no p1_results.json)"; echo "$(date -Is) p1 failed" > runs/phase_c/BLOCKED_p1; fi
while pgrep -f "p1_train.py" > /dev/null; do sleep 10; done
touch runs/phase_c/TEACHER_DONE; log "TEACHER_DONE written (GPU released to Physics-IQ orchestrator)"
