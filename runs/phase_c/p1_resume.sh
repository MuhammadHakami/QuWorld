#!/bin/bash
# Phase-C P1 resume after the Sep 19 22:58 board latch-off/reboot: the teacher cache (45 stems) is complete and 24/27 P1 units
# have result files, so this skips teacher_p1_chain.sh's config-pick + cache stages and re-enters p1_train.py (resumable per unit),
# then writes the same sentinels the chain would have (P1_DONE, TEACHER_DONE -> Physics-IQ / WMB orchestrators).
export PATH="$HOME/.local/bin:$PATH"; set -u; cd /home/mohhakami/Documents/paper/quworld
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/phase_c/phase_c_status; log(){ echo "$(date -Is) chain: $*" >> "$S"; }
missing=0; for stem in $(cat runs/phase_c/teacher_subset.txt); do [ -f "data/phase_c/teacher/${stem}.pt" ] || missing=$((missing+1)); done
[ "$missing" -ne 0 ] && { log "RESUME ABORT: teacher cache has $missing stems missing"; exit 1; }
log "P1 resume after reboot (teacher cache complete; $(ls runs/phase_c/p1/*.json 2>/dev/null | wc -l)/27 units done)"
uv run --no-sync python paper/phase_c/p1_train.py --unit-budget-s 420 >> runs/phase_c/p1.log 2>&1
if [ -f runs/phase_c/p1_results.json ]; then log "P1 complete"; touch runs/phase_c/P1_DONE; else log "P1 FAILED (no p1_results.json)"; echo "$(date -Is) p1 failed" > runs/phase_c/BLOCKED_p1; fi
while pgrep -f "p1_train[.]py" > /dev/null; do sleep 10; done
touch runs/phase_c/TEACHER_DONE; log "TEACHER_DONE written (GPU released to Physics-IQ orchestrator)"
