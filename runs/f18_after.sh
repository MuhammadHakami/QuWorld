#!/bin/bash
# After the F18K Development upload has started (runs/f18_public.sh): (1) the Final-phase F18K chain on the organizers' smoke fold
# (runs/final_f16k.sh TAG=F18 REGALL=1 XPALL=1 VH=1) and the package entry point (repro/fusion/infer.sh CAND=F18K) on the same fold - every
# array must agree bit for bit; (2) resume the §48c search (runs/s48r.sh).
cd /home/mohhakami/Documents/paper/quworld; L=runs/strat/fusion; LOG=$L/f18.log
until grep -q -E "^\[upload\] F18K id|^F18_FAIL" $LOG 2>/dev/null; do sleep 60; done
QW_FOLD=diii_d_fold_smoke SUF=smoke TAG=F18 REGALL=1 XPALL=1 VH=1 bash runs/final_f16k.sh
if grep -q "^REH_DONE F18 smoke" $L/final_f16k_smoke.log; then
  QW_FOLD=diii_d_fold_smoke CAND=F18K QW_GPU_GB=30 bash repro/fusion/infer.sh F18R >> $L/final_f16k_smoke.log 2>&1
  .venv/bin/python - >> $LOG 2>&1 <<'PY'
import numpy as np
a = np.load("runs/sophelio_sota/submission_F18Ksmoke/diii_d_private_test.npz"); b = np.load("runs/sophelio_sota/submission_F18RK/diii_d_private_test.npz")
same = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
print(f"F18_REH {'BIT_EXACT' if same else 'DIFFERS'}: {len(a.files)} arrays (final_f16k.sh F18 smoke vs infer.sh CAND=F18K)")
PY
else echo "F18_REH_FAIL (see $L/final_f16k_smoke.log) $(date -Is)" >> $LOG; fi
bash runs/s48r.sh
