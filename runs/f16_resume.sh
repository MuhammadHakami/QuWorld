#!/bin/bash
# Resume F16 after the disk filled during post_kw2's zip (Oct 17 ~01:40; 21 superseded submission dirs deleted, 117 GB free). F16 itself was built
# and agreement-checked at 01:27 (base F15, shape + q/b routed). Same arguments as f16_router.sh: F16K = post_kw2, upload, score.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f16.log
echo "[f16] resume after disk-full post_kw2 crash $(date -Is)" >> $LOG
REG="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2+head_boundary_v8_J1QH_s1*4+head_boundary_v8_J1QH_s2*4+head_boundary_v8_J1QH_s3*4"
H="route:$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLHA_s$s*3+"; done)$REG|$(for s in 1 2 3 4; do printf "head_boundary_v8_J1RLHACS_s$s*6+"; done)$REG"
rm -rf runs/sophelio_sota/submission_F16K
QW_LITREE=$PWD/$L/litree.npz:0.2 QW_GPU_GB=12 runs/capped.sh kw2f16 30 .venv/bin/python paper/strat/fusion/post_kw2.py --src F16 --tag F16K --heads "$H" >> $LOG 2>&1 \
  || { echo "F16_FAIL kw2 (resume) $(date -Is)" >> $LOG; exit 1; }
echo "F16K_DONE $(date -Is)" >> $LOG
runs/codabench_upload.sh runs/sophelio_sota/submission_F16K/submission.zip F16K "F16K: F15K with Ip-sign routed canonical heads (shape 1, q/b 1)" >> $LOG 2>&1
grep -q "dataset create failed" runs/sophelio_sota/upload_F16K.log && { echo "F16_HOLD storage (delete a superseded submission, then upload) $(date -Is)" >> $LOG; exit 0; }
id=$(grep -o '"id":[0-9]*' runs/sophelio_sota/upload_F16K.log | head -1 | cut -d: -f2); echo "[upload] F16K id $id $(date -Is)" >> $LOG
for i in $(seq 1 120); do
  s=$(runs/codabench_status.sh $id 2>/dev/null); case "$s" in Finished*|Failed*) echo "[score] F16K $id $s $(date -Is)" >> $LOG; break;; esac; sleep 60
done
echo "UPLOAD_F16K_DONE $(date -Is)" >> $LOG
