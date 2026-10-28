#!/bin/bash
# After F15K exists (f15_build.sh: F15K_DONE / UPLOAD_F15K_DONE), rehearse the Final-phase F15K chain on the organizers' smoke fold
# (runs/final_f15k.sh; checks frames / shapes / finiteness), so F15K is ready for Oct 19 if adopted.
cd /home/mohhakami/Documents/paper/quworld; L=runs/strat/fusion
until [ -f runs/sophelio_sota/submission_F15K/diii_d_public_test.npz ] && grep -q -E "^UPLOAD_F15K_DONE|^F15K_DONE|^UPLOADED" $L/f15.log; do
  grep -q -E "^F15_FAIL|^CRASH" $L/f15.log && { echo "F15_REH_SKIP (F15 failed) $(date -Is)" >> $L/f15.log; exit 1; }
  sleep 120
done
bash runs/final_f15k.sh && echo "F15_REH_DONE $(date -Is)" >> $L/f15.log || echo "F15_REH_FAIL $(date -Is)" >> $L/f15.log
