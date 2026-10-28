#!/bin/bash
# doc 35 §37g (register / matched MLP seeds 2-3, bf16) then §37l (vanilla fp32 seeds 2-3); one job at a time via the GPU lock.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
run() { # arm seed outdir extra
  [ -f $3/eal_md_$1_s$2.json ] && return
  flock runs/strat/gpu.lock bash -c "cd baseline/routefinder && QW_GPU_GB=24 timeout 36000 ../../runs/capped.sh s37rf 60 ../../.venv-vrp/bin/python \
    ../../paper/vrp/run_rf_eal.py $1 --setting md --seed $2 --out ../../$3 $4" >> runs/s37gl_rf.log 2>&1 || echo "CRASH s37rf $1 s$2 $(date -Is)" >> $S
}
for seed in 2 3; do run q30-geo=3d $seed runs/routefinder/s23 ""; run q30-geo=3d-mlp $seed runs/routefinder/s23 ""; done
echo "S37G_RUNS_DONE $(date -Is)" >> $S
for seed in 2 3; do run vanilla $seed runs/routefinder/full "--epochs 10 --train-size 10000 --precision 32"; done
echo "S37L_RUNS_DONE $(date -Is)" >> $S
