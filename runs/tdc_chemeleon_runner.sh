#!/bin/bash
# TDC ADMET, CheMeleon trunk (Sep 23): pass 1 = vanilla on all 22 endpoints (SoTA standing), pass 2 = mlp30 + mr30ryd.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
S=runs/tdc_chemeleon/status; mkdir -p runs/tdc_chemeleon; P=.venv-tdc/bin/python
EPS=$($P -c "import sys; sys.path.insert(0,'paper/tdc'); import tdcdata as D; print(' '.join(D.endpoints()))" 2>/dev/null | tail -1)
for arms in "vanilla" "mlp30 mr30ryd"; do for ep in $EPS; do for arm in $arms; do
  [ -f runs/tdc_chemeleon/${ep}_${arm}.json ] && continue
  while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 20 ]; do sleep 60; done
  $P paper/tdc/run_chemeleon.py "$ep" "$arm" > runs/tdc_chemeleon/${ep}_${arm}.log 2>&1 \
    && echo "result $ep $arm: $(grep '\[chemeleon\]' runs/tdc_chemeleon/${ep}_${arm}.log | tail -1) $(date -Is)" >> "$S" \
    || echo "CRASH $ep $arm $(date -Is)" >> "$S"
done; done; echo "PASS_DONE $arms $(date -Is)" >> "$S"; done
echo "TDC_CHEMELEON_DONE $(date -Is)" >> "$S"
