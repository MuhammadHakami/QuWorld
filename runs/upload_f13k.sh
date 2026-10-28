#!/bin/bash
# F13K upload (§43l; expected public ~0.99587, range 0.99584-0.99590, stated in doc 35 before upload) + score poll.
cd /home/mohhakami/Documents/paper/quworld; LOG=runs/strat/fusion/upload_f13k.log
runs/codabench_upload.sh runs/sophelio_sota/submission_F13K/submission.zip F13K "F13K: F8K + TRAIN-only J1QH register heads (independent link) as extra shape members" >> $LOG 2>&1
id=$(grep -o '"id":[0-9]*' runs/sophelio_sota/upload_F13K.log | head -1 | cut -d: -f2); echo "[upload] F13K id $id $(date -Is)" >> $LOG
for i in $(seq 1 120); do
  s=$(runs/codabench_status.sh $id 2>/dev/null); case "$s" in Finished*|Failed*) echo "[score] F13K $id $s $(date -Is)" >> $LOG; break;; esac; sleep 60
done
echo "UPLOAD_F13K_DONE $(date -Is)" >> $LOG
