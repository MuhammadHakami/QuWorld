#!/bin/bash
# Ordered queue after §34 (doc 35 "User directive and revised plan", 2026-09-27 22:39). Replaces the scratchpad
# resume_after_s34.sh, which un-paused every lane at once. Order: §34 TEST once -> §30 runner (seeds 2-3) -> §35 (RB
# onset tail, 20 short runs) -> the §23 MHD row runner and the other paused lanes. The MHD-on-Flowers items (TTA rule,
# §36 gate, Flowers neck) join this queue as they are registered.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1; S=runs/alloc/status
S34_RUNNER=556889; S30=438458; LANES="4036838 357694 141818 502187"
until tail -n +190 $S | grep -qE "^(S34_TRAIN_DONE|S34 INCOMPLETE|CRASH s34)" || ! kill -0 $S34_RUNNER 2>/dev/null; do sleep 120; done
if tail -n +190 $S | grep -qE "^S34_TRAIN_DONE"; then                       # registered: TEST once, full split
  flock runs/strat/gpu.lock env QW_GPU_GB=16 timeout 10800 runs/capped.sh s34test 32 \
    uv run --no-sync python paper/phase_b/score_flowers_test.py >> runs/phase_b_mhd/flower_test.log 2>&1
  grep -q "^S34_TEST " runs/phase_b_mhd/flower_test.log && echo "S34_TEST_DONE $(date -Is)" >> $S \
    || echo "CRASH s34 test $(date -Is)" >> $S
else
  echo "QUEUE s34 ended without its target; TEST not scored, awaiting Claude $(date -Is)" >> $S
fi
kill -CONT $S30 2>/dev/null; echo "QUEUE resumed the §30 runner $(date -Is)" >> $S
while kill -0 $S30 2>/dev/null; do sleep 120; done
for seed in 4 5 6 7 8; do for core in lin mlp ryd gutz; do                  # §35: proxy, K 30, new seeds 4-8
  f=runs/alloc/results/rayleigh_benard/proxy_${core}_K30_s${seed}.json; [ -f $f ] && continue
  QW_GPU_GB=8 timeout 3600 runs/capped.sh s35 10 .venv/bin/python paper/alloc/coproc.py rayleigh_benard \
    --strategy proxy --core $core --K 30 --seed $seed >> runs/alloc/s35.log 2>&1 \
    || echo "CRASH s35 $core s$seed $(date -Is)" >> $S
done; done
echo "S35_DONE $(date -Is)" >> $S
kill -CONT $LANES 2>/dev/null; echo "QUEUE resumed the §23 MHD row runner and lanes $(date -Is)" >> $S
