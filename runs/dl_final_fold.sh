#!/bin/bash
# Download the Final-phase DIII-D private fold (ganatma/fusion-equilibrium-challenge-final, published 2026-10-18 23:47 UTC) next to the
# Development folds. Inputs only (no equilibrium labels); MAST files are not fetched.
cd "$(dirname "$0")/.."; LOG=runs/strat/fusion/dl_final_fold.log
echo "[dl] start $(date -Is)" >> $LOG
for try in $(seq 1 30); do                                                   # resumes; HF rate-limits anonymous bursts (429)
.venv/bin/python - >> $LOG 2>&1 <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download('ganatma/fusion-equilibrium-challenge-final', repo_type='dataset', allow_patterns=['data/diii_d_private_test/*', 'README.md'],
                      local_dir='data/sophelio/fusion-equilibrium-challenge-final', max_workers=2)
print('done', p)
PY
rc=$?; [ $rc = 0 ] && break; echo "[dl] try $try rc $rc $(date -Is) - retry in 60 s" >> $LOG; sleep 60; done
echo "[dl] exit $rc $(date -Is) files $(ls data/sophelio/fusion-equilibrium-challenge-final/data/diii_d_private_test 2>/dev/null | wc -l)" >> $LOG
