#!/bin/bash
# TDC stack v3 (Sep 23 15:10): v2's MapLight+GIN `all` run deadlocked inside CatBoost (every thread in futex wait, 0 CPU for
# 1 h 40 min on clearance_hepatocyte; torch + CatBoost OpenMP in one long-lived process). Now one process per endpoint under
# a 20-min timeout with one retry; run_maplight.py skips finished endpoints. Then MapLight train-only refits for all four
# feature sets, the VALID-selected blend (paper/tdc/stack.py), and again after the CheMeleon mlp30/mr30ryd pass.
cd /home/mohhakami/Documents/paper/quworld; export QW_GPU_GB=4 PYTHONUNBUFFERED=1
S=runs/tdc_maplight/status; L=runs/tdc_ensemble/stack.log; PY=.venv-tdc/bin/python
EPS=$($PY -c "import sys; sys.path.insert(0,'paper/tdc'); import tdcdata as D; print(' '.join(D.endpoints()))" 2>/dev/null | tail -1)
for f in maplight+gin maplight+gin+chemeleon; do
  for ep in $EPS; do
    for try in 1 2; do
      timeout 1200 runs/capped.sh ml-${f//+/-} 16 $PY paper/tdc/run_maplight.py $ep --features $f >> runs/tdc_maplight/run_${f//+/_}.log 2>&1 && break
      echo "RETRY maplight $f $ep (try $try) $(date -Is)" >> $S
    done
  done
  echo "MAPLIGHT_${f//+/_}_DONE $(date -Is)" >> $S
done
timeout 7200 runs/capped.sh stack-cm 8 $PY paper/tdc/stack.py cm-valid >> $L 2>&1 || echo "CRASH stack cm-valid $(date -Is)" >> $S
for f in maplight maplight+chemeleon maplight+gin maplight+gin+chemeleon; do for ep in $EPS; do
  timeout 1800 runs/capped.sh stack-ml 16 $PY paper/tdc/stack.py ml-valid --features $f --endpoints $ep >> $L 2>&1 || echo "RETRY? stack ml-valid $f $ep $(date -Is)" >> $S
done; done
$PY paper/tdc/stack.py blend >> $L 2>&1 && echo "TDC_ENSEMBLE_STACK_DONE $(date -Is)" >> $S || echo "CRASH stack blend $(date -Is)" >> $S
until grep -q TDC_CHEMELEON_DONE runs/tdc_chemeleon/status; do sleep 600; done
timeout 7200 runs/capped.sh stack-cm2 8 $PY paper/tdc/stack.py cm-valid >> $L 2>&1 || echo "CRASH stack cm-valid 2 $(date -Is)" >> $S
$PY paper/tdc/stack.py blend >> $L 2>&1 && echo "TDC_ENSEMBLE_STACK_FINAL_DONE $(date -Is)" >> $S || echo "CRASH stack blend final $(date -Is)" >> $S
