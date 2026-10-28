#!/bin/bash
# Compute declaration for F20K: PSIV60A training time per step ALONE on the GB10 (1 epoch of the production recipe, all data), after §48i'.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=$PWD/runs/strat/fusion; LOG=$L/s48t.log
until grep -qE "^S48I_(DONE|FAIL)" $L/s48i2.log 2>/dev/null; do sleep 120; done
while ps -eo cmd | grep -E "quota_fusion.py|hpo_vq.py|head_boundary.py|gpu_build.py" | grep -vq grep; do sleep 120; done
J='{"target": "psi", "backbone": "vhope", "neck": "gate", "win": 8, "stride": 1, "bb_width": 128, "bb_depth": 4, "width": 2048, "depth": 4, "drop": 0.0, "q_layers": 4, "q_in": 32, "link": "independent", "lr": 0.000977105771213329, "alpha_init": 0.11830330369409621, "seq": true, "crop": 16, "shots": 256, "tokens": 12, "q_noise": 0.0, "in_noise": 0.0, "huber": 1.0, "mixer": 64, "reg": "tether3d"}'
t0=$(date +%s); env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_ALL=1 STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 \
  .venv/bin/python paper/strat/fusion/quota_fusion.py final PSIVTIME "$J" 9 1 > $L/s48t_run.log 2>&1; t1=$(date +%s)
ln=$(grep -o "step [0-9]*/[0-9]* .* [0-9]*s" $L/s48t_run.log | tail -1); st=$(sed 's/step \([0-9]*\)\/.*/\1/' <<< "$ln"); tr_=$(grep -o "[0-9]*s$" <<< "$ln" | tr -d s)
echo "S48T_DONE PSIV60A alone: ${st:-?} steps (1 epoch) training ${tr_:-?} s -> $(awk -v t="$tr_" -v s="$st" 'BEGIN{if (s > 0) printf "%.3f s/step, 22800 steps = %.2f h per seed", t / s, t / s * 22800 / 3600}') (wall incl. load $((t1 - t0)) s) $(date -Is)" >> $LOG
rm -f $L/head_boundary_v8_PSIVTIME_s9.pt $L/head_boundary_v8_PSIVTIME_s9.npz
