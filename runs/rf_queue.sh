#!/bin/bash
# RouteFinder/RRNCO queue (doc 35 §14-§16), at most 2 concurrent jobs (the memory guard killed 5-lane runs at 13 GB free).
cd /home/mohhakami/Documents/paper/quworld; S=runs/routefinder/status; RF=baseline/routefinder
gate() { while [ "$(pgrep -fc '^/home/mohhakami/Documents/paper/quworld/baseline/(routefinder|rrnco)/../../.venv-vrp/bin/python')" -ge 2 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 35 ]; do sleep 30; done; }
rf() {  # $1 variant $2 seed $3 epochs $4 train-size $5 outdir
  local out=$5 j=$5/eal_md_$1_s$2.json; [ -f $j ] && return; pgrep -f "run_rf_eal.py $1 --setting md --seed $2 --epochs $3 " > /dev/null && return
  gate; (cd $RF && timeout 28800 ../../runs/capped.sh rfq 30 ../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py $1 --setting md --seed $2 --epochs $3 --train-size $4 --precision 32 --out ../../$out > ../../$out/$1_s$2.log 2>&1 \
    && echo "RF_Q_DONE $1 s$2 e$3 $(date -Is)" >> ../../$S || echo "CRASH rf-q $1 s$2 e$3 $(date -Is)" >> ../../$S) &
  sleep 20
}
mkdir -p runs/routefinder/scaling runs/routefinder/routed runs/routefinder/full
for v in vanilla q30 q30-mlp; do rf $v 1 1 2000 runs/routefinder/scaling; done
for v in route30 route30-mlp route10 route10-mlp route5 route5-mlp route20 route20-mlp; do rf $v 1 1 2000 runs/routefinder/routed; done
for v in q5-part=1x5 q5-part=1x5-mlp q10-part=1x10 q10-part=1x10-mlp q20-part=2x10 q20-part=2x10-mlp; do rf $v 1 1 2000 runs/routefinder/scaling; done
gate; (cd baseline/rrnco && for d in in_distribution in_distribution_cluster; do timeout 2400 ../../runs/capped.sh rrnco-test 30 ../../.venv-vrp/bin/python test.py --problem rcvrp --datasets data/rcvrp/rcvrp_n100_seed3333_$d.npz --checkpoint checkpoints/rcvrp/epoch_199.ckpt --batch_size 32 > ../../runs/routefinder/rrnco_test_$d.log 2>&1; echo "RRNCO_TEST_EXIT $d $? $(date -Is)" >> ../../$S; done) &
sleep 20
for spec in "vanilla 1" "route30 1" "route30-mlp 1" "vanilla 2" "route30 2" "route30-mlp 2" "vanilla 3" "route30 3" "route30-mlp 3" "q30 1" "q30-mlp 1" "route10 1" "route10-mlp 1"; do set -- $spec; rf $1 $2 10 10000 runs/routefinder/full; done
wait; echo "RF_QUEUE_DONE $(date -Is)" >> $S
