#!/bin/bash
# Post-reboot sequencer (Sep 19 22:58 latch-off; VS Code session's lanes). Keeps the "one GPU co-runner" order the runners were
# designed for: wait for P1 (time-budgeted units, keep them undisturbed) -> Pelzer U-Net redo (seed-1339 pretrain + 3 arms; the
# 18:21 PELZER_UNET_S1339_DONE line was false) -> Well 2-D chain v2 (rayleigh_benard resumes vanilla s1337 from _latest.pt, then shear_flow).
export PATH="$HOME/.local/bin:$PATH"; cd /home/mohhakami/Documents/paper/quworld; L=runs/relaunch_sep19_reboot.log
log(){ echo "$(date -Is) $*" >> "$L"; }
log "sequencer start; waiting for runs/phase_c/P1_DONE (or BLOCKED_p1)"
until [ -f runs/phase_c/P1_DONE ] || [ -f runs/phase_c/BLOCKED_p1 ]; do sleep 60; done
log "P1 finished ($(ls runs/phase_c/P1_DONE runs/phase_c/BLOCKED_p1 2>/dev/null | tr '\n' ' ')); Pelzer redo start"
bash runs/pelzer_unet/redo_s1338_s1339.sh >> runs/pelzer_unet/redo_runner.log 2>&1
log "Pelzer redo end: $(tail -n 1 runs/pelzer_unet/status)"
log "Well chain v2 start"
bash runs/well2d_chain.sh >> runs/well2d/chain_runner.log 2>&1
log "Well chain v2 end"
