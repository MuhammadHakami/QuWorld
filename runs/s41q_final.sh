#!/bin/bash
# Fusion §41q FINAL BUILD components on ALL 7,041 labelled shots (TRAIN + VALID + TEST; Spark = the reference machine, seed sd of
# a J1 head 0.0001; learning curve: 8/9 of TRAIN costs -0.00097 mean head R2). HJl9 recipe otherwise:
#   PSI-FA s2/s1/s3 = PSI-F config + all_data; J1A s1-6 = J1 heads + STRAT_ALL; QHA s1-6 = QH q/b heads + STRAT_ALL.
# Input set for J1A / QHA from runs/strat/fusion/ipfix_decision.txt (X_ipfix iff J1I_s1 - J1MR_s1 > +0.0002 on merphistophline,
# registered before J1MR's result). PSI-FA canon from psi_canon_decision.txt (true iff PCN beats PCM on seeds 1 AND 2, mephiles); QHA STRAT_CANON (0 / 1) from
# qb_canon_decision.txt (1 iff QHC_s1 beats QHM_s1 on mean q95 / betaN R2 by > 2 QH seed sd, mephiles). J1A + fixed-coordinate
# Thomson (STRAT_THZ=1: J1T_s1 0.98695 vs J1M_s1 0.98599, ~10 sd); QHA STRAT_THZ from qb_thz_decision.txt (QHT vs QH control). No local validation is possible (VALID / TEST are in the fit): checked on the public board.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s41q_final.log
until grep -q S41Q_SPARK_DONE $L/s41q_spark.log 2>/dev/null; do sleep 60; done
while pgrep -f "[s]ota_train.py --tag PSIFA|[h]ead_boundary.py" > /dev/null; do sleep 30; done   # relaunch-safe: let an in-flight seed finish
CFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix", "all_data": true'
until [ -s $L/psi_canon_decision.txt ]; do sleep 60; done; CN=$(cat $L/psi_canon_decision.txt); echo "psi canon $CN $(date -Is)" >> $LOG
CFG="$CFG, \"canon\": $CN"
for s in 2 1 3; do
  [ -f runs/sophelio_sota/PSIFA_s$s.pt ] && continue
  flock runs/strat/gpu.lock env QW_GPU_GB=40 runs/capped.sh s41qf 40 .venv/bin/python paper/sophelio/sota_train.py --tag PSIFA_s$s --score-val 0 \
    --cfg "{$CFG, \"seed\": $s}" >> $LOG 2>&1 || echo "CRASH PSIFA s$s $(date -Is)" >> $LOG
done
until [ -s $L/ipfix_decision.txt ]; do sleep 60; done; MF=$(cat $L/ipfix_decision.txt); echo "inputs $MF $(date -Is)" >> $LOG
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_ALL=1 QW_MAGFIX=$MF QW_GPU_GB=40"
for s in 1 2 3 4 5 6; do
  [ -f $L/head_boundary_v8_J1A_s$s.pt ] && continue
  [ $s = 6 ] && continue                                              # seed 6 trains on merphistophline (runs/merph_j1a6.sh)
  flock runs/strat/gpu.lock env $E STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_OUT_TAG=_J1A runs/capped.sh s41qf 40 .venv/bin/python paper/strat/fusion/head_boundary.py $s \
    >> $LOG 2>&1 || echo "CRASH J1A s$s $(date -Is)" >> $LOG
done
until [ -s $L/qb_canon_decision.txt ] && [ -s $L/qb_thz_decision.txt ]; do sleep 60; done
QC=$(cat $L/qb_canon_decision.txt); QT=$(cat $L/qb_thz_decision.txt); echo "qb canon $QC thz $QT $(date -Is)" >> $LOG
for s in 1 2 3; do                                                    # 3 seeds (user: faster iterations)
  [ -f $L/head_qb_QHA_s$s.pt ] && continue
  flock runs/strat/gpu.lock env $E STRAT_CANON=$QC STRAT_THZ=$QT STRAT_OUT_TAG=_QHA runs/capped.sh s41qf 40 .venv/bin/python paper/strat/fusion/head_boundary.py $s qb \
    >> $LOG 2>&1 || echo "CRASH QHA s$s $(date -Is)" >> $LOG
done
echo "S41Q_FINAL_DONE $(date -Is)" >> $LOG
