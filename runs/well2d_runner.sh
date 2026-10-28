#!/bin/bash
# The Well 2-D benchmark lane (GPU): for <dataset> in {rayleigh_benard, shear_flow} (planetswe dropped: current HF data matches no released checkpoint)
#   0. verify well2d vs WellDataset; 1. official TFNO anchor: full test split single-shot (published 0.0853 / 0.6566)
#   2. arms {vanilla, mlp30, mr30ryd} x seeds {1337,1338,1339}, seed-outer, fixed protocol (pre-registered Sep 19):
#      TARGET_STEPS 6000, BATCH 4, lr 1e-4, AdamW wd 0.01, clip 1.0, valid subset 8 windows/traj, best-on-valid checkpoint
#   3. single-shot test eval of each best checkpoint -> runs/well2d/<ds>/<variant>_s<seed>_test.{csv,log}
# Memory gate before each GPU job: MemAvailable >= 25 GB. Status: runs/well2d/<ds>/status; sentinel WELL2D_<DS>_DONE.
DS=${1:-rayleigh_benard}; cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; export PATH="$HOME/.local/bin:$PATH"
export BACKBONE=${BACKBONE:-cnext}   # cnext (the_well CNextU-net, main venv, proven) | tfno (.venv-well; neuralop checkpoints not reproducible as of Sep 19)
if [ "$BACKBONE" = tfno ]; then PY=.venv-well/bin/python; else PY="uv run --no-sync python"; fi
O=runs/well2d/$DS/$BACKBONE; S=$O/status; mkdir -p $O
export BATCH=${BATCH:-4} VAL_PER_TRAJ=8 VAL_EVERY_S=300 MEM_FRACTION=0.3
STEPS=${STEPS:-6000}; BUDGET=${BUDGET:-14400}; LR=1e-4
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 25 ]; do sleep 60; done; }
echo "runner start $DS backbone $BACKBONE $(date -Is) steps $STEPS batch $BATCH" >> "$S"
[ -f runs/well2d/$DS/verified ] && touch $O/verified
[ -f $O/verified ] || { $PY paper/phase_b/well2d.py $DS valid > $O/verify.log 2>&1 && touch $O/verified || { echo "VERIFY FAILED $(date -Is)" >> "$S"; exit 1; }; }
[ -f runs/well2d/$DS/${BACKBONE}_official_test.log ] && grep -q mean_vrmse runs/well2d/$DS/${BACKBONE}_official_test.log && cp runs/well2d/$DS/${BACKBONE}_official_test.log $O/official_test.log && cp -n runs/well2d/$DS/${BACKBONE}_official_test.csv $O/official_test.csv 2>/dev/null
if [ ! -f $O/official_test.log ] || ! grep -q mean_vrmse $O/official_test.log; then
  mem_gate; echo "anchor official $BACKBONE test eval $(date -Is)" >> "$S"
  BATCH=16 $PY paper/phase_b/eval_well2d.py $DS official vanilla $O/official_test.csv > $O/official_test.log 2>&1 || echo "CRASH anchor $(date -Is)" >> "$S"
  grep -E "mean_vrmse|n_windows" $O/official_test.log | tr '\n' ' ' >> "$S"; echo >> "$S"
fi
for seed in 1337 1338 1339; do for v in vanilla mlp30 mr30ryd; do
  if ! grep -q "target_reached" $O/${v}_s${seed}.log 2>/dev/null; then
    mem_gate; echo "arm $v s$seed $(date -Is)" >> "$S"
    $PY paper/phase_b/finetune_well2d.py $DS $v $seed $BUDGET $LR $STEPS >> $O/${v}_s${seed}.log 2>&1 || echo "CRASH arm $v s$seed $(date -Is)" >> "$S"
  fi
  if grep -q "target_reached" $O/${v}_s${seed}.log 2>/dev/null && ! grep -q mean_vrmse $O/${v}_s${seed}_test.log 2>/dev/null; then
    mem_gate; echo "test $v s$seed $(date -Is)" >> "$S"
    BATCH=16 $PY paper/phase_b/eval_well2d.py $DS $O/${v}_s${seed}.pt $v $O/${v}_s${seed}_test.csv > $O/${v}_s${seed}_test.log 2>&1 || echo "CRASH test $v s$seed $(date -Is)" >> "$S"
    echo "result $v s$seed $(grep -E 'mean_vrmse' $O/${v}_s${seed}_test.log)" >> "$S"
  fi
done; done
echo "WELL2D_${DS^^}_${BACKBONE^^}_DONE $(date -Is)" >> "$S"
