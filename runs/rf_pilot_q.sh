#!/bin/bash
# RouteFinder pilot + RRNCO queue (Sep 25 22:15): one extra lane, only with >= 30 GB free (memguard v4 kills pilots first).
cd /home/mohhakami/Documents/paper/quworld; S=runs/routefinder/status; RF=baseline/routefinder; P=runs/routefinder/crn
PY='^/home/mohhakami/Documents/paper/quworld/baseline/(routefinder|rrnco)/../../.venv-vrp/bin/python'
gate() { while [ "$(pgrep -fc "$PY")" -ge 4 ] || [ "$(pgrep -fc "$PY.* --epochs 1 ")" -ge 1 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 30; done; }
rf() {  # $1 variant; retried up to 3 times if killed
  local j=$P/eal_md_$1_s1.json n=0
  while [ ! -f $j ] && [ $n -lt 3 ]; do n=$((n+1)); gate
    (cd $RF && timeout 7200 ../../runs/capped.sh rfp 30 ../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py $1 --setting md --seed 1 --epochs 1 --train-size 2000 --precision 32 --out ../../$P > ../../$P/$1_s1.log 2>&1 \
      && echo "RF_PILOT_DONE $1 $(date -Is)" >> ../../$S || echo "CRASH rf-pilot $1 try$n $(date -Is)" >> ../../$S)
  done
}
for v in route10-mlp route20 route20-mlp route5 route5-mlp; do rf $v; done
gate; (cd baseline/rrnco && for d in in_distribution in_distribution_cluster out_of_distribution; do timeout 3600 ../../runs/capped.sh rrnco-test 30 ../../.venv-vrp/bin/python test.py --problem rcvrp --datasets data/rcvrp/rcvrp_n100_seed3333_$d.npz --checkpoint checkpoints/rcvrp/epoch_199.ckpt --batch_size 32 > ../../runs/routefinder/rrnco_test_$d.log 2>&1; echo "RRNCO_TEST_EXIT $d $? $(date -Is)" >> ../../$S; done)
for v in route30-fgn route30-mlp-fgn; do rf $v; done
echo "RF_PILOT_Q_DONE $(date -Is)" >> $S
