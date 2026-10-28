#!/bin/bash
# Sophelio candidates on ALL VALID shots (the NAS uses a fixed 40-shot subset, too noisy for final selection).
cd /home/mohhakami/Documents/paper/quworld; export QW_GPU_GB=8 SOPHELIO_SCORE_WORKERS=4
O=runs/sophelio_sota/postproc_fullvalid.log
for combo in "/home/mohhakami/Documents/paper/quworld/runs/nas/sophelio/trunks/trunk1_t5.pt" "/home/mohhakami/Documents/paper/quworld/runs/nas/sophelio/trunks/trunk1_t0.pt" "/home/mohhakami/Documents/paper/quworld/runs/nas/sophelio/trunks/trunk1_t0.pt /home/mohhakami/Documents/paper/quworld/runs/nas/sophelio/trunks/trunk1_t5.pt" "/home/mohhakami/Documents/paper/quworld/runs/nas/sophelio/trunks/trunk1_t0.pt /home/mohhakami/Documents/paper/quworld/runs/nas/sophelio/trunks/trunk1_t1.pt /home/mohhakami/Documents/paper/quworld/runs/nas/sophelio/trunks/trunk1_t2.pt /home/mohhakami/Documents/paper/quworld/runs/nas/sophelio/trunks/trunk1_t5.pt"; do
  echo "== $combo" >> $O
  runs/capped.sh postproc 20 .venv/bin/python paper/sophelio/postproc.py --trunks $combo --sigmas 0 1 --n 1000 2>&1 | grep "\[postproc\]" >> $O
done
echo POSTPROC_DONE >> $O
