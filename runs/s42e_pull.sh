#!/bin/bash
# Fusion §42e read-outs on the Spark: pull QHR2_s1 / J1R2_s1 from merphistophline when done, then vt_eval on all VT shots.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42e.log
K="ssh -o BatchMode=yes -o ConnectTimeout=10 -i $HOME/.ssh/id_ed25519_quworld"; M=mohhakami@merphistophline:Documents/paper/quworld/runs/strat/fusion
pull() {  # $1 = log name on merph, $2 = marker, $3 = file stem
  until timeout 60 $K mohhakami@merphistophline "grep -q $2 ~/Documents/paper/quworld/runs/strat/fusion/$1" 2>/dev/null; do sleep 120; done
  timeout 900 rsync -a -e "$K" $M/$3.pt $M/$3.npz $L/ && timeout 120 rsync -a -e "$K" $M/$1 $L/merph_$1
}
pull QHR2_s1.log QHR2_MERPH_DONE head_qb_QHR2_s1
until grep -q "^S42A_DONE" $L/s42a.log 2>/dev/null; do sleep 60; done
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QH_s1 head_qb_QHR_s1 head_qb_QHR2_s1 >> $LOG 2>&1 || echo "CRASH vt qb" >> $LOG
pull J1R2_s1.log J1R2_MERPH_DONE head_boundary_v8_J1R2_s1
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1THS_s1 head_boundary_v8_J1R2_s1 >> $LOG 2>&1 || echo "CRASH vt shape" >> $LOG
echo "S42E_DONE $(date -Is)" >> $LOG
