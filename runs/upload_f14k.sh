#!/bin/bash
# F14K upload (§45a; expected public ~0.99588, range 0.99584-0.99592, stated in doc 35 before upload) + score poll.
cd /home/mohhakami/Documents/paper/quworld; LOG=runs/strat/fusion/upload_f14k.log
runs/codabench_upload.sh runs/sophelio_sota/submission_F14K/submission.zip F14K "F14K: F13K + GBDT li tree blended into the shape heads (w 0.2)" >> $LOG 2>&1
id=$(grep -o '"id":[0-9]*' runs/sophelio_sota/upload_F14K.log | head -1 | cut -d: -f2); echo "[upload] F14K id $id $(date -Is)" >> $LOG
for i in $(seq 1 120); do
  s=$(runs/codabench_status.sh $id 2>/dev/null); case "$s" in Finished*|Failed*) echo "[score] F14K $id $s $(date -Is)" >> $LOG; break;; esac; sleep 60
done
echo "UPLOAD_F14K_DONE $(date -Is)" >> $LOG
