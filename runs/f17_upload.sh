#!/bin/bash
# F17K Development upload (user request Oct 17) and score wait.
cd "$(dirname "$0")/.."; L=runs/strat/fusion; LOG=$L/f17.log
runs/codabench_upload.sh runs/sophelio_sota/submission_F17K/submission.zip F17K "F17K: F16K with Ip-sign canonical register / X-point heads on every shot" >> $LOG 2>&1 \
  || { echo "UPLOAD_F17K_FAIL $(date -Is)" >> $LOG; exit 1; }
id=$(grep -o '"id":[0-9]*' runs/sophelio_sota/upload_F17K.log | head -1 | cut -d: -f2); echo "[upload] F17K id $id $(date -Is)" >> $LOG
[ -n "$id" ] || { echo "UPLOAD_F17K_FAIL no id ($(tail -c 300 runs/sophelio_sota/upload_F17K.log)) $(date -Is)" >> $LOG; exit 1; }
for i in $(seq 1 150); do
  s=$(runs/codabench_status.sh $id 2>/dev/null); case "$s" in Finished*|Failed*) echo "[score] F17K $id $s $(date -Is)" >> $LOG; break;; esac; sleep 60
done
echo "UPLOAD_F17K_DONE $(date -Is)" >> $LOG
