#!/bin/bash
# §47g (registered before any V60 result): reads the §47f outcome and runs the pre-declared branch.
#   PASS (VALID ratio <= 0.92 and VT delta >= +0.0001 with CI > 0): F18K public build (alone); Development upload only if the expected public
#        gain (= VT delta x 1) is >= +0.0002; then V60A (= V60 with STRAT_ALL=1) seeds 1-3 for the Final build.
#   FAIL: the §48c 8-fold search (runs/s48.sh; V4 / V4t enqueued in it). PASS also continues with §48c after the build (user, Oct 17).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s47g.log
echo "[s47g] start $(date -Is)" >> $LOG
until grep -q '^S47F_' $L/s47f.log; do sleep 60; done
VTL=$(grep '^\[vtpp\] PQmix~g_f17V60 vs' $L/s47f.log | tail -1)
D=$(sed -n 's/.* delta \([-+0-9.]*\) \[\([-+0-9.]*\), \([-+0-9.]*\)\].*/\1 \2 \3/p' <<< "$VTL"); read -r DEL LO HI <<< "$D"
echo "[s47g] VT line: ${VTL:-none} -> delta ${DEL:-NA} lo ${LO:-NA} $(date -Is)" >> $LOG
ENV="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QW_GPU_GB=30"
if [ -n "$DEL" ] && awk -v d="$DEL" -v l="$LO" 'BEGIN{exit !(d >= 0.0001 && l > 0)}'; then
  EXP=$(awk -v d="$DEL" 'BEGIN{printf "%.6f", 0.996223 + d}')
  echo "[s47g] PASS: expected public F18K $EXP (F17K 0.996223 + VT delta $DEL x 1) $(date -Is)" >> $LOG
  UP=""; awk -v d="$DEL" 'BEGIN{exit !(d >= 0.0002)}' && UP=upload
  echo "[s47g] F18K build${UP:+ + Development upload} (no-oversubmit gate: expected gain >= +0.0002 -> ${UP:-no upload}) $(date -Is)" >> $LOG
  bash runs/f18_public.sh $UP &
  until grep -qE '^\[f18\] built|^F18_FAIL' $L/f18.log 2>/dev/null; do sleep 60; done; echo "[s47g] F18 build step: $(grep -E '^\[f18\] built|^F18_FAIL' $L/f18.log | tail -1)" >> $LOG
  echo "[s47g] PASS -> §48c search next (V60A deferred to the search winner) $(date -Is)" >> $LOG; bash runs/s48.sh
else
  echo "[s47g] FAIL branch (no passing VT line): §48c search $(date -Is)" >> $LOG
  bash runs/s48.sh                                                        # §48c: V4 / V4t are enqueued trials of the search
fi
echo "S47G_DONE $(date -Is)" >> $LOG
