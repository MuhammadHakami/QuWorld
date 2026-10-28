#!/bin/bash
# Routed co-processor stacked on fine-tuned rayleigh_benard trunks, FULL test split (SoTA-comparable).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
while [ -n "$WAITPID" ] && kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done   # after the RB s3 fill-in (by PID, never pgrep -f)
for v in mr30ryd vanilla; do
  tag=rayleigh_benard:ft_${v}_s1337; dir=rayleigh_benard__ft_${v}_s1337
  [ -f runs/alloc/adrc_cache/$dir/test.pt ] || ADRC_CKPT=runs/well2d/rayleigh_benard/cnext/${v}_s1337.pt ADRC_VARIANT=$v ADRC_TEST_ALL=1 \
    QW_GPU_GB=24 timeout 14400 runs/capped.sh adrc-ft 40 .venv/bin/python paper/alloc/adrc_cache.py $tag >> runs/alloc/adrc_cache.log 2>&1 || echo "CRASH rbft cache $v $(date -Is)" >> $S
  for seed in 1 2 3; do for arm in cls_only static_lin adrc_ryd adrc_gnn adrc_mlp adrc_gutz; do
    [ -f runs/alloc/adrc_results/${dir}_${arm}_s${seed}.json ] && continue
    QW_GPU_GB=12 timeout 7200 runs/capped.sh adrc 24 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed >> runs/alloc/adrc.log 2>&1 || echo "CRASH rbft $v $arm s$seed $(date -Is)" >> $S
  done; done
  echo "RBFT_DONE $v $(date -Is)" >> $S
done
