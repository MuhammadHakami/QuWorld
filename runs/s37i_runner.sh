#!/bin/bash
# doc 35 §37i: metric-aligned Flowers fine-tune in 40-min chunks through the GPU lock, then TEST once if VALID improved.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/phase_b_mhd/flower_vft.log; S=runs/alloc/status
for i in $(seq 1 12); do
  grep -q '^target_reached' $L 2>/dev/null && break
  flock runs/strat/gpu.lock bash -c 'until [ $(awk "/MemAvailable/ {printf \"%d\", \$2/1048576}" /proc/meminfo) -ge 55 ]; do sleep 60; done
    QW_GPU_GB=40 timeout 3000 runs/capped.sh s37i 60 uv run --no-sync python paper/phase_b/finetune_flowers_vrmse.py 5 2400' >> $L 2>&1
  sleep 5
done
grep -q '^target_reached' $L || { echo "S37I INCOMPLETE $(date -Is)" >> $S; exit 1; }
start=$(grep -m1 'step=0 vrmse' $L | grep -oE 'vrmse=[0-9.]+' | cut -d= -f2); best=$(grep '^target_reached' $L | tail -1 | grep -oE 'best=[0-9.]+' | cut -d= -f2)
echo "S37I_TRAIN_DONE start=$start best=$best $(date -Is)" >> $S
if python3 -c "import sys; sys.exit(0 if float('$best') < float('$start') else 1)"; then
  flock runs/strat/gpu.lock env QW_GPU_GB=16 timeout 10800 runs/capped.sh s37itest 32 \
    uv run --no-sync python paper/phase_b/score_flowers_test.py --ckpt flower_vft_s1337.pt >> runs/phase_b_mhd/flower_vft_test.log 2>&1
  grep -q '^S34_TEST ' runs/phase_b_mhd/flower_vft_test.log && echo "S37I_TEST_DONE $(date -Is)" >> $S || echo "CRASH s37i test $(date -Is)" >> $S
else
  echo "S37I no VALID gain; TEST not opened $(date -Is)" >> $S
fi
