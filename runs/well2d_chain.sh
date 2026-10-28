#!/bin/bash
# Well 2-D chain v2 (Sep 19 15:00): CNextU-net backbone in the main venv. One GPU co-runner rule: wait PELZER_UNET_DONE,
# then rayleigh_benard runner (train split local), then shear_flow once its train download has finished.
cd /home/mohhakami/Documents/paper/quworld; L=runs/well2d/chain.log; log() { echo "$(date -Is) $*" | tee -a $L; }
while ! grep -q PELZER_UNET_S1339_DONE runs/pelzer_unet/status 2>/dev/null; do sleep 120; done  # peer redo of seed 1339 (runs/pelzer_unet/redo_s1339.sh) writes this after the main runner; keeps one GPU co-runner
log "rayleigh_benard runner start (cnext)"; BACKBONE=cnext bash runs/well2d_runner.sh rayleigh_benard; log "rayleigh_benard runner end: $(tail -1 runs/well2d/rayleigh_benard/cnext/status)"
while ! grep -q "\[dl\] shear_flow done" runs/well_dl/download_shear_flow.log 2>/dev/null; do sleep 300; done
log "shear_flow runner start (cnext)"; BACKBONE=cnext bash runs/well2d_runner.sh shear_flow; log "shear_flow runner end: $(tail -1 runs/well2d/shear_flow/cnext/status)"
