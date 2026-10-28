#!/bin/bash
# F16K upload retry after deleting the superseded F13K (988269); then score.
cd /home/mohhakami/Documents/paper/quworld; L=runs/strat/fusion; LOG=$L/f16.log
runs/codabench_upload.sh runs/sophelio_sota/submission_F16K/submission.zip F16K "F16K: F15K with Ip-sign routed canonical heads (shape 1, q/b 1)" >> $LOG 2>&1
grep -q "dataset create failed" runs/sophelio_sota/upload_F16K.log && { echo "F16_HOLD2 storage still full $(date -Is)" >> $LOG; exit 0; }
id=$(grep -o '"id":[0-9]*' runs/sophelio_sota/upload_F16K.log | head -1 | cut -d: -f2); echo "[upload] F16K id $id $(date -Is)" >> $LOG
for i in $(seq 1 120); do
  s=$(runs/codabench_status.sh $id 2>/dev/null); case "$s" in Finished*|Failed*) echo "[score] F16K $id $s $(date -Is)" >> $LOG; break;; esac; sleep 60
done
echo "UPLOAD_F16K_DONE $(date -Is)" >> $LOG
