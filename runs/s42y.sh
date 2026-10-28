#!/bin/bash
# §42y / §42z (registered): after F7's build - post_kw2 on F7's public psi (F7K), the q/b clamp on F7 and F7K (F7C, F7KC) - then the
# step-4 VT rerun (s42t_vt3r.sh). Serial, so no data-loading job runs next to a public build or the VT run. No uploads here.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42y.log
until grep -q "^F7_DONE\|^F7_FAIL\|^F7_SKIP" $L/f7.log 2>/dev/null; do sleep 30; done
if grep -q "^F7_DONE" $L/f7.log; then
  .venv/bin/python paper/strat/fusion/post_clamp.py --src F7 --tag F7C >> $LOG 2>&1 && echo "CLAMP_F7C_DONE $(date -Is)" >> $LOG || echo "CRASH clamp F7C $(date -Is)" >> $LOG
  QW_GPU_GB=16 runs/capped.sh kw2 30 .venv/bin/python paper/strat/fusion/post_kw2.py --src F7 --tag F7K >> $LOG 2>&1 && echo "KW2_DONE $(date -Is)" >> $LOG || echo "CRASH kw2 $(date -Is)" >> $LOG
  .venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F7K/diii_d_public_test.npz runs/sophelio_sota/submission_F7/diii_d_public_test.npz >> $LOG 2>&1 || true
  .venv/bin/python paper/strat/fusion/post_clamp.py --src F7K --tag F7KC >> $LOG 2>&1 && echo "CLAMP_F7KC_DONE $(date -Is)" >> $LOG || echo "CRASH clamp F7KC $(date -Is)" >> $LOG
fi
echo "S42Y_DONE $(date -Is)" >> $LOG
exec bash runs/s42t_vt3r.sh
