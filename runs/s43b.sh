#!/bin/bash
# §43b (registered before any result): J1RLE_s2 (TRAIN only, exact labels) + XPEA s1-3 (all data, exact X-point labels), two at a time ->
# vt_eval J1RL_s2 vs J1RLE_s2 (rule with §43a's seed 1: mean delta >= +0.0002) -> F11 = F8 with XPEA (alone) -> agreement vs F8. No upload.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s43b.log
J="STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix QW_GPU_GB=16"
R="STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1"
trainhead() {   # $1 tag  $2 seed  $3 mode  $4.. extra env
  local tag=$1 s=$2 mode=$3; shift 3
  env $J $R "$@" STRAT_OUT_TAG=_$tag runs/capped.sh h$tag 40 .venv/bin/python paper/strat/fusion/head_boundary.py $s $mode >> $L/s43b_$tag.log 2>&1 \
    && echo "${tag}_s${s}_TRAINED $(date -Is)" >> $LOG || echo "CRASH $tag s$s $(date -Is)" >> $LOG
}
[ -f $L/head_boundary_v9_J1RLE_s2.pt ] || trainhead J1RLE 2 shape STRAT_LABEL_SUFFIX=_v9 STRAT_SCAL_W=9.142857 &
[ -f $L/head_xp_XPEA_s1.pt ] || trainhead XPEA 1 xp STRAT_LABEL_SUFFIX=_v8 STRAT_XP_LABELS=xp_labels_v9 STRAT_ALL=1 &
wait
[ -f $L/head_xp_XPEA_s2.pt ] || trainhead XPEA 2 xp STRAT_LABEL_SUFFIX=_v8 STRAT_XP_LABELS=xp_labels_v9 STRAT_ALL=1 &
[ -f $L/head_xp_XPEA_s3.pt ] || trainhead XPEA 3 xp STRAT_LABEL_SUFFIX=_v8 STRAT_XP_LABELS=xp_labels_v9 STRAT_ALL=1 &
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1RL_s2 head_boundary_v9_J1RLE_s2 >> $LOG 2>&1 || echo "CRASH vt shape $(date -Is)" >> $LOG
wait
grep -q "^CRASH" $LOG && exit 1
KW=$(grep -o '"_steps".*' $L/f8.log | head -1 | sed 's/ 20[0-9][0-9]-.*$//'); KW="{$KW"
python3 -c "import json, sys; json.loads(sys.argv[1])" "$KW" || { echo "F11_FAIL kw parse $(date -Is)" >> $LOG; exit 1; }
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLA_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2_s$s+"; done | sed 's/+$//'); XH="head_xp_XPEA_s1+head_xp_XPEA_s2+head_xp_XPEA_s3"
echo "[f11] xh $XH kw $KW $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f11 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F11 \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head --qbclamp >> $LOG 2>&1 || { echo "F11_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F11/diii_d_public_test.npz runs/sophelio_sota/submission_F8/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F11_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
echo "F11_BUILT $(date -Is)" >> $LOG
