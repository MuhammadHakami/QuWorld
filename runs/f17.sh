#!/bin/bash
# F17 = F16K + the components that passed their registered rules (doc 35 §45s / §45t / §45u), one batch (no-oversubmit rule):
#   REGCS=1  reversed-Ip shots read the canonical register twins J1QHCS s1-3 (J1QPIpi unchanged)            (§45s register)
#   REGALL=1 J1QHCS s1-3 replace J1QH on ALL shots (compute-neutral reproduction)                           (§45w (1))
#   XPCS=1   reversed-Ip shots read the canonical X-point twins XPACS s1-3 in the xhe step                   (§45s X-point)
#   XPALL=1  XPACS s1-3 replace XPA on ALL shots                                                             (§45w (2))
#   USH=1    normal shots add J1RLHACS s1-4 to J1RLHA s1-8 at equal per-head weight (MLP share stays 0.6)   (§45u shape)
#   UQB=1    normal shots add QHAR2HuCS s1-2 to QHAR2Hu s1-3 (q / b)                                          (§45u q / b)
#   TRS=cap  post_kw2 --trs <cap in m> (continuous Rgeo shift for the column-quantised tri)                   (§45t)
# Builds alone (vt.lock), checks agreement vs F16, writes F17K. No upload here: the expected score is stated and the upload decided first.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f17.log
REGCS=${REGCS:-0}; XPCS=${XPCS:-0}; REGALL=${REGALL:-0}; XPALL=${XPALL:-0}; USH=${USH:-0}; UQB=${UQB:-0}; TRS=${TRS:-0}; TAG=${TAG:-F17}
echo "[f17] start $(date -Is) TAG=$TAG REGCS=$REGCS XPCS=$XPCS REGALL=$REGALL XPALL=$XPALL USH=$USH UQB=$UQB TRS=$TRS VH=${VH:-0} VSTEM=${VSTEM:-}" >> $LOG
REG="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2+head_boundary_v8_J1QH_s1*4+head_boundary_v8_J1QH_s2*4+head_boundary_v8_J1QH_s3*4"
REGR=$REG; [ "$REGCS" = 1 ] && REGR=${REG//J1QH_/J1QHCS_}
[ "$REGALL" = 1 ] && { REG=${REG//J1QH_/J1QHCS_}; REGR=$REG; }
CS="$(for s in 1 2 3 4; do printf "head_boundary_v8_J1RLHACS_s$s*6+"; done)"
if [ "$USH" = 1 ]; then
  NORM="$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLHA_s$s*2+"; done)$(for s in 1 2 3 4; do printf "head_boundary_v8_J1RLHACS_s$s*2+"; done)"
else
  NORM="$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLHA_s$s*3+"; done)"
fi
H="route:$NORM$REG|$CS$REGR"
if [ "${VH:-0}" = 1 ]; then                       # §47: + the VisionHOPE hybrid heads (VSTEM s1-3) on every shot, share 0.252 of the shape weight
  VS=${VSTEM:-V60}; VN=${VSEEDS:-1 2 3}; VW=$(awk -v k="$(wc -w <<< "$VN")" 'BEGIN{printf "%g", 13.5 / k}')   # share 0.25 for any seed count
  VV=$(for s in $VN; do printf "head_boundary_v8_${VS}_s$s*$VW+"; done | sed 's/+$//'); H="${H%%|*}+$VV|${H#*|}+$VV"; fi
QN=$(for s in 1 2 3; do printf "head_qb_QHAR2Hu_s$s+"; done | sed 's/+$//'); QR="head_qb_QHAR2HuCS_s1+head_qb_QHAR2HuCS_s2"
[ "$UQB" = 1 ] && QN="$QN+$QR"; Q="route:$QN|$QR"
XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"; [ "$XPCS" = 1 ] && XH="route:$XH|${XH//XPA_/XPACS_}"
[ "$XPALL" = 1 ] && XH=${XH//XPA_/XPACS_} && XH=${XH#route:} && XH=${XH%%|*}
KW=$(grep -o '"_steps".*' $L/f8.log | head -1 | sed 's/ 20[0-9][0-9]-.*$//'); KW="{$KW"; QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
echo "[f17] heads $H | qb $Q | xp $XH | kw $KW | qbw $QBW" >> $LOG
flock runs/strat/vt.lock env QW_LITREE=$PWD/$L/litree.npz:0.2 QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f17 40 .venv/bin/python paper/strat/fusion/gpu_build.py \
  --split public --tag $TAG --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head --qbclamp \
  >> $LOG 2>&1 || { echo "F17_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_$TAG/diii_d_public_test.npz runs/sophelio_sota/submission_F16/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F17_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
echo "F17_BUILT $(date -Is)" >> $LOG
TA=""; [ "$TRS" != 0 ] && TA="--trs $TRS"
QW_LITREE=$PWD/$L/litree.npz:0.2 QW_GPU_GB=12 runs/capped.sh kw2f17 30 .venv/bin/python paper/strat/fusion/post_kw2.py --src $TAG --tag ${TAG}K --heads "$H" $TA >> $LOG 2>&1 \
  || { echo "F17_FAIL kw2 $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_${TAG}K/diii_d_public_test.npz runs/sophelio_sota/submission_F16K/diii_d_public_test.npz >> $LOG 2>&1 || true
echo "F17K_DONE $(date -Is)" >> $LOG
