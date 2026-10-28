#!/bin/bash
# Fusion queue, strictly ONE GPU job at a time (Oct 12 02:21, after the second GPU-memory collapse at 01:49 with a VT + a training;
# the kernel showed ~100 GB held by the NVIDIA driver, not by process anon memory). runs/memlog.sh records memory every 5 s.
# Order: §42m VT -> PQ2A s1 -> F4 (build + upload if §42m passes) -> PQL10 + §42n VT -> PQ2H + §42o VT -> PQ2A s3 -> J1RL + §42p -> §42k.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
PCFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix", "qphys": 2'
J=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
X='"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}'; VB="{\"_steps\":[\"shift\",\"bw\",\"kw\",\"xp\",\"jgn\",\"li\"],$X}"
trunk() {  # tag, extra cfg, log
  [ -f runs/sophelio_sota/$1.pt ] || flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh q$1 40 .venv/bin/python paper/sophelio/sota_train.py --tag $1 --score-val 0 \
    --cfg "{$PCFG, $2}" >> $3 2>&1 || echo "CRASH $1 $(date -Is)" >> $3
}
vt() {  # tag, psis, kws, log
  flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh v$1 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag $1 \
    --psi psiF --psis "$2" --guard psiF:0.25 --qb qh50 --qbheads "$Q" --heads "$J" --kws "$3" --axis head >> $4 2>&1 || echo "CRASH vt $1 $(date -Is)" >> $4
}
cmp_() { (cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py "$@") >> $LOGC 2>&1; }
LOGC=$L/s42m_vt.log; echo "[serial] §42m VT start $(date -Is)" >> $LOGC
vt PG "PQ2~g,PQmix~g" "{\"VB\":$VB,\"ns\":{\"_steps\":[\"bw\",\"kw\",\"xp\",\"jgn\",\"li\"],$X}}" $LOGC
cmp_ LS_ctrl_VB "PQ2~g_VB" "PQ2~g_ns" "PQmix~g_VB" "PQmix~g_ns" PQ2_VB; echo "S42M_VT_DONE $(date -Is)" >> $LOGC
trunk PQ2A_s1 '"all_data": true, "canon": true, "seed": 1' $L/s42m.log; echo "PQ2A_s1 $(date -Is)" >> $L/s42m.log
bash runs/s42_f4.sh                                                                   # F4 build is itself under vt.lock
trunk PQL10 '"seed": 1, "lset_w": 10' $L/s42n.log; echo "PQL10_TRAINED $(date -Is)" >> $L/s42n.log
LOGC=$L/s42n.log; vt N "PQL10~g" "{\"raw\": {\"_steps\": []}, \"VB\": $VB}" $LOGC
cmp_ "PQ2~g_VB" "PQL10~g_VB"; cmp_ PQ2_raw "PQL10~g_raw"; echo "S42N_DONE $(date -Is)" >> $LOGC
trunk PQ2H '"seed": 1, "hfull": true' $L/s42o.log; echo "PQ2H_TRAINED $(date -Is)" >> $L/s42o.log
LOGC=$L/s42o.log; vt O "PQ2H~g" "{\"raw\": {\"_steps\": []}, \"VB\": $VB}" $LOGC
cmp_ "PQ2~g_VB" "PQ2H~g_VB"; cmp_ PQ2_raw "PQ2H~g_raw"; echo "S42O_DONE $(date -Is)" >> $LOGC
trunk PQ2A_s3 '"all_data": true, "canon": true, "seed": 3' $L/s42m.log; echo "PQ2A_s3 $(date -Is)" >> $L/s42m.log; echo "S42M_DONE $(date -Is)" >> $L/s42m.log
H="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_boundary_v8_J1RL_s1.pt ] || flock runs/strat/vt.lock env $H STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_OUT_TAG=_J1RL QW_GPU_GB=30 \
  runs/capped.sh qJ1RL 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 >> $L/s42p.log 2>&1 || echo "CRASH J1RL $(date -Is)" >> $L/s42p.log
echo "J1RL_TRAINED $(date -Is)" >> $L/s42p.log
flock runs/strat/vt.lock .venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1THS_s1 head_boundary_v8_J1R2_s1 head_boundary_v8_J1RL_s1 >> $L/s42p.log 2>&1 || echo "CRASH vt shape" >> $L/s42p.log
echo "S42P_DONE $(date -Is)" >> $L/s42p.log
bash runs/s42k.sh
