#!/bin/bash
# After F16K exists (f16_router.sh: F16K_DONE / UPLOAD_F16K_DONE) and the F15 rehearsal has finished (both write the smoke fold's *_private
# feature files), rehearse the Final-phase F16K chain on the organizers' smoke fold with the base f16_router.sh chose (runs/final_f16k.sh).
cd /home/mohhakami/Documents/paper/quworld; L=runs/strat/fusion
until [ -f runs/sophelio_sota/submission_F16K/diii_d_public_test.npz ] && grep -q -E "^UPLOAD_F16K_DONE|^F16K_DONE|^F16_HOLD" $L/f16.log \
      && grep -q -E "^F15_REH_(DONE|FAIL|SKIP)" $L/f15.log; do
  grep -q -E "^F16_FAIL|^F16_SKIP|^CRASH" $L/f16.log && { echo "F16_REH_SKIP (F16 not built) $(date -Is)" >> $L/f16.log; exit 1; }
  sleep 120
done
B=$(grep -o "base F1[45]" $L/f16.log | tail -1 | cut -d' ' -f2)
BASE=${B:-F15} bash runs/final_f16k.sh && echo "F16_REH_DONE base ${B:-F15} $(date -Is)" >> $L/f16.log || echo "F16_REH_FAIL $(date -Is)" >> $L/f16.log
