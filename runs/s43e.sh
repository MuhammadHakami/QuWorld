#!/bin/bash
# §43e diagnostic (registered before any result): where is D_LCFS lost? The deployed chain (xhe gate 0 w 0.75) on the first 200 VT shots with
# (o0) head targets (J1 x6, XP1 x3), (o1) true shape labels as targets (7 scalars + 64 rays), (o2) true shape + true X-point labels.
# If o2's D_LCFS is far below #1's 0.0073 the heads are the limit; if it stays near ours (~0.0096) the realisation (warps) is the limit.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s43e.log
Jh=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
XP="head_xp_XP1_s1+head_xp_XP1_s2+head_xp_XP1_s3"
run() {   # $1 arm  $2 heads  $3 xpheads
  KWS=$(python3 -c "import json; A=json.load(open('$L/xh_arms.json')); print(json.dumps({'$1': A['$1']}))")
  flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s43e 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag XO$1 --vt-n 200 \
    --psi psiF --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "$Q" --heads "$2" --xpheads "$3" --kws "$KWS" --axis head >> $LOG 2>&1 || echo "CRASH $1 $(date -Is)" >> $LOG
}
run VBo0 "$Jh" "$XP"
run VBo1 "oracle:$Jh" "$XP"
run VBo2 "oracle:$Jh" "xporacle:$XP"
echo "S43E_DONE $(date -Is)" >> $LOG
