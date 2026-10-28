#!/bin/bash
# F9 (registered 2026-10-13 before the §42x result): iff §42x passes (J1RLM_s1 vs J1RL_s1 mean-7 delta >= +0.0003 with CI > 0, s42x2.log)
# -> all-data mixup shape heads J1RLMA s1-8 (alone, after §42x) -> F9 = F8 with J1RLMA replacing J1RLA (alone) -> agreement vs F8.
# No upload here. Expected F9 public = F8 + 0.2 x 0.3-1.0 x (VT mean-7 delta) (shape-head carry 0.3x for J1RLA; more if the gain is on OOD).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f9.log
until grep -q "^S42X2_DONE\|^CRASH" $L/s42x2.log 2>/dev/null; do sleep 60; done
PASS=$(python3 - $L/s42x2.log <<'PY'
import re, sys
t = open(sys.argv[1]).read()
m = re.search(r"\[vt\] head_boundary_v8_J1RLM_s1 vs head_boundary_v8_J1RL_s1: ([-+0-9.]+) \[([-+0-9.]+), ([-+0-9.]+)\]", t)
print("yes" if m and float(m.group(1)) >= 0.0003 and float(m.group(2)) > 0 else "no")
PY
)
echo "[f9] §42x pass: $PASS $(date -Is)" >> $LOG
[ "$PASS" = yes ] || { echo "F9_SKIP $(date -Is)" >> $LOG; exit 0; }
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_ALL=1 QW_MAGFIX=X_magfix"
for s in 1 2 3 4 5 6 7 8; do
  [ -f $L/head_boundary_v8_J1RLMA_s$s.pt ] || env $E STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_MIXUP=0.2 STRAT_OUT_TAG=_J1RLMA \
    QW_GPU_GB=16 runs/capped.sh hJ1RLMA 40 .venv/bin/python paper/strat/fusion/head_boundary.py $s >> $LOG 2>&1 || { echo "CRASH J1RLMA s$s $(date -Is)" >> $LOG; exit 1; }
  echo "J1RLMA_s${s}_TRAINED $(date -Is)" >> $LOG
done
KW=$(grep -o '"_steps".*' $L/f8.log | head -1 | sed 's/ 20[0-9][0-9]-.*$//'); KW="{$KW"
python3 -c "import json, sys; json.loads(sys.argv[1])" "$KW" || { echo "F9_FAIL kw parse: $KW $(date -Is)" >> $LOG; exit 1; }
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLMA_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2_s$s+"; done | sed 's/+$//'); XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"
echo "[f9] kw $KW $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f9 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F9 \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head --qbclamp >> $LOG 2>&1 || { echo "F9_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F9/diii_d_public_test.npz runs/sophelio_sota/submission_F8/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F9_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
echo "F9_BUILT $(date -Is)" >> $LOG
