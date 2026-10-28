#!/bin/bash
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42i.log
K="ssh -o BatchMode=yes -o ConnectTimeout=10 -i $HOME/.ssh/id_ed25519_quworld"; M=mohhakami@merphistophline:Documents/paper/quworld/runs/strat/fusion
until timeout 60 $K mohhakami@merphistophline "grep -q QHR2H_MERPH_DONE ~/Documents/paper/quworld/runs/strat/fusion/QHR2H_s1.log" 2>/dev/null; do sleep 120; done
timeout 900 rsync -a -e "$K" $M/head_qb_QHR2H_s1.pt $M/head_qb_QHR2H_s1.npz $L/
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QHR2_s1 head_qb_QHR2H_s1 >> $LOG 2>&1 || echo "CRASH vt qb" >> $LOG
echo "S42I_DONE $(date -Is)" >> $LOG
