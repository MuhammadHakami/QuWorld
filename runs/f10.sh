#!/bin/bash
# F10 (registered 2026-10-13 before any result): the X-point step drives D_LCFS (oracle -31%, deployed xhe gate 0 w 0.75 -10% on VT); the XPA
# ensemble has 3 seeds. XPA s4-8 (same recipe, all data, alone after the smoke test) -> F10 = F8 with XPA s1-8 (alone) -> agreement vs F8.
# No VT read-out (more seeds of the same recipe); no upload here. Rebuild budget 19.5 -> 22.2 GB10-hours. Expected F10 = F8 + 0..+0.00003.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f10.log
until grep -q "^SMOKE_EXIT" $L/smoke_f8.log 2>/dev/null; do sleep 30; done
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_ALL=1 QW_MAGFIX=X_magfix"
for s in 4 5 6 7 8; do
  [ -f $L/head_xp_XPA_s$s.pt ] || env $E STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_OUT_TAG=_XPA QW_GPU_GB=16 runs/capped.sh hXPA 40 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $s xp >> $LOG 2>&1 || { echo "CRASH XPA s$s $(date -Is)" >> $LOG; exit 1; }
  echo "XPA_s${s}_TRAINED $(date -Is)" >> $LOG
done
KW=$(grep -o '"_steps".*' $L/f8.log | head -1 | sed 's/ 20[0-9][0-9]-.*$//'); KW="{$KW"
python3 -c "import json, sys; json.loads(sys.argv[1])" "$KW" || { echo "F10_FAIL kw parse $(date -Is)" >> $LOG; exit 1; }
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLA_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2_s$s+"; done | sed 's/+$//'); XH=$(for s in 1 2 3 4 5 6 7 8; do printf "head_xp_XPA_s$s+"; done | sed 's/+$//')
echo "[f10] kw $KW xh $XH $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f10 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F10 \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head --qbclamp >> $LOG 2>&1 || { echo "F10_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F10/diii_d_public_test.npz runs/sophelio_sota/submission_F8/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F10_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
echo "F10_BUILT $(date -Is)" >> $LOG
