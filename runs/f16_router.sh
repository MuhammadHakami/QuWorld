#!/bin/bash
# §45o follow-up (registered in doc 35 before the router read-out): waits for S45O_DONE and applies the §45o rule per family (router VT delta
# >= +0.00008 with CI low > 0). For each passing family it trains all-data canonical twins of the F15 recipe (shape: J1RLHACS s1-4 = J1RLHA +
# STRAT_CANON=2; q/b: QHAR2HuCS s1-2 = QHAR2Hu + STRAT_CANON=2). Each seed holds a SHARED vt.lock, so a waiting public build (exclusive lock)
# starts between seeds and runs without them. Then waits for F15K's public score: base = F15 iff F15K >= 0.995893 (F14K), else F14. F16 = base
# with "route:<base stems>|<twin stems>" for the passing families (reversed-Ip shots read the twins; register heads unchanged), F16K = post_kw2,
# agreement vs the base build, upload F16K (expected base public + ~0.00004). On a Codabench storage failure it stops with F16_HOLD.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f16.log
echo "[f16] start $(date -Is)" >> $LOG
until grep -q "^S45O_DONE" $L/s45o.log; do grep -q "^CRASH" $L/s45o.log && { echo "F16_FAIL s45o crash $(date -Is)" >> $LOG; exit 1; }; sleep 120; done
read SH QB < <(.venv/bin/python - <<'PY'
import json, pathlib
o = pathlib.Path("runs/strat/fusion")
def ok(f):
    p = o / f
    if not p.exists():
        return 0
    r = json.loads(p.read_text())["router"]
    return int(r["delta"] >= 8e-5 and r["ci95"][0] > 0)
print(ok("router_shape_J1RLCS_s1.json"), ok("router_qb_QHR2CS_s1.json"))
PY
)
echo "[f16] router rule: shape $SH qb $QB $(date -Is)" >> $LOG
[ "$SH$QB" = "00" ] && { echo "F16_SKIP no family passes $(date -Is)" >> $LOG; exit 0; }
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_ALL=1 QW_MAGFIX=X_magfix"
if [ "$SH" = 1 ]; then (
  for s in 1 2 3 4; do
    [ -f $L/head_boundary_v8_J1RLHACS_s$s.pt ] || flock -s runs/strat/vt.lock env $E STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 \
      STRAT_HUBER=0.3 STRAT_CANON=2 STRAT_OUT_TAG=_J1RLHACS QW_GPU_GB=30 runs/capped.sh hJ1RLHACS 40 .venv/bin/python paper/strat/fusion/head_boundary.py $s \
      >> $L/J1RLHACS.log 2>&1 || { echo "CRASH J1RLHACS s$s $(date -Is)" >> $LOG; exit 1; }
    echo "J1RLHACS_s${s}_TRAINED $(date -Is)" >> $LOG; sleep 90
  done ) &
fi
if [ "$QB" = 1 ]; then (
  for s in 1 2; do
    [ -f $L/head_qb_QHAR2HuCS_s$s.pt ] || flock -s runs/strat/vt.lock env $E STRAT_QPHYS=2 STRAT_HUBER=0.3 STRAT_CANON=2 STRAT_OUT_TAG=_QHAR2HuCS QW_GPU_GB=20 \
      runs/capped.sh qQHAR2HuCS 30 .venv/bin/python paper/strat/fusion/head_boundary.py $s qb >> $L/QHAR2HuCS.log 2>&1 \
      || { echo "CRASH QHAR2HuCS s$s $(date -Is)" >> $LOG; exit 1; }
    echo "QHAR2HuCS_s${s}_TRAINED $(date -Is)" >> $LOG; sleep 90
  done ) &
fi
wait
grep -q "^CRASH" $LOG && { echo "F16_FAIL twins $(date -Is)" >> $LOG; exit 1; }
echo "F16_TWINS_DONE $(date -Is)" >> $LOG
until grep -q -E "^UPLOAD_F15K_DONE|^F15_FAIL|^CRASH" $L/f15.log; do sleep 120; done
S15=$(grep -o "^\[score\] F15K [0-9]* Finished {'d3d_S': '[0-9.]*" $L/f15.log | tail -1 | grep -o "[0-9.]*$")
if [ -n "$S15" ] && .venv/bin/python -c "import sys; sys.exit(0 if float('$S15') >= 0.995893 else 1)"; then B=F15; RH=J1RLHA; RQ=QHAR2Hu; else B=F14; RH=J1RLA; RQ=QHAR2; fi
echo "[f16] F15K public ${S15:-none} -> base $B $(date -Is)" >> $LOG
REG="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2+head_boundary_v8_J1QH_s1*4+head_boundary_v8_J1QH_s2*4+head_boundary_v8_J1QH_s3*4"
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_${RH}_s$s*3+"; done)"$REG"
Q=$(for s in 1 2 3; do printf "head_qb_${RQ}_s$s+"; done | sed 's/+$//'); XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"
[ "$SH" = 1 ] && H="route:$H|$(for s in 1 2 3 4; do printf "head_boundary_v8_J1RLHACS_s$s*6+"; done)$REG"
[ "$QB" = 1 ] && Q="route:$Q|head_qb_QHAR2HuCS_s1+head_qb_QHAR2HuCS_s2"
KW=$(grep -o '"_steps".*' $L/f8.log | head -1 | sed 's/ 20[0-9][0-9]-.*$//'); KW="{$KW"; QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
echo "[f16] heads $H qb $Q $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_LITREE=$PWD/$L/litree.npz:0.2 QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f16 40 .venv/bin/python paper/strat/fusion/gpu_build.py \
  --split public --tag F16 --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head --qbclamp \
  >> $LOG 2>&1 || { echo "F16_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F16/diii_d_public_test.npz runs/sophelio_sota/submission_$B/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F16_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
echo "F16_BUILT $(date -Is)" >> $LOG
QW_LITREE=$PWD/$L/litree.npz:0.2 QW_GPU_GB=12 runs/capped.sh kw2f16 30 .venv/bin/python paper/strat/fusion/post_kw2.py --src F16 --tag F16K --heads "$H" >> $LOG 2>&1 \
  || { echo "F16_FAIL kw2 $(date -Is)" >> $LOG; exit 1; }
echo "F16K_DONE $(date -Is)" >> $LOG
runs/codabench_upload.sh runs/sophelio_sota/submission_F16K/submission.zip F16K "F16K: ${B}K with Ip-sign routed canonical heads (shape $SH, q/b $QB)" >> $LOG 2>&1
grep -q "dataset create failed" runs/sophelio_sota/upload_F16K.log && { echo "F16_HOLD storage (delete a superseded submission, then upload) $(date -Is)" >> $LOG; exit 0; }
id=$(grep -o '"id":[0-9]*' runs/sophelio_sota/upload_F16K.log | head -1 | cut -d: -f2); echo "[upload] F16K id $id $(date -Is)" >> $LOG
for i in $(seq 1 120); do
  s=$(runs/codabench_status.sh $id 2>/dev/null); case "$s" in Finished*|Failed*) echo "[score] F16K $id $s $(date -Is)" >> $LOG; break;; esac; sleep 60
done
echo "UPLOAD_F16K_DONE $(date -Is)" >> $LOG
