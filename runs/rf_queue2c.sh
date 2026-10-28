#!/bin/bash
# RouteFinder/RRNCO queue v2c (doc 35 §19, Sep 25 21:45): the vanilla-based qubit ladder first (CRN pilots, then the
# full-protocol ladder seed 1), full seeds 2-3, RRNCO anchor, then the §18 FGN arms, pooled pilots last. 4-lane gate.
# 21:30: vanilla full seed 1 re-queued first (memguard v3 killed it at 76 min; v4 kills cheap jobs first).
cd /home/mohhakami/Documents/paper/quworld; S=runs/routefinder/status; RF=baseline/routefinder
gate() { while [ "$(pgrep -fc '^/home/mohhakami/Documents/paper/quworld/baseline/(routefinder|rrnco)/../../.venv-vrp/bin/python')" -ge 4 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 25 ]; do sleep 30; done; }
rf() {  # $1 variant $2 seed $3 epochs $4 train-size $5 outdir
  local out=$5 j=$5/eal_md_$1_s$2.json; [ -f $j ] && return; pgrep -f "run_rf_eal.py $1 --setting md --seed $2 --epochs $3 " > /dev/null && return
  gate; (cd $RF && timeout 36000 ../../runs/capped.sh rfq 30 ../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py $1 --setting md --seed $2 --epochs $3 --train-size $4 --precision 32 --out ../../$out > ../../$out/$1_s$2.log 2>&1 \
    && echo "RF_Q2_DONE $1 s$2 e$3 $(date -Is)" >> ../../$S || echo "CRASH rf-q2 $1 s$2 e$3 $(date -Is)" >> ../../$S) &
  sleep 45
}
P=runs/routefinder/crn; F=runs/routefinder/full; mkdir -p $P $F
rf vanilla 1 10 10000 $F
for v in route10-mlp route20 route20-mlp route5 route5-mlp; do rf $v 1 1 2000 $P; done
for v in route20 route20-mlp route10 route10-mlp route5 route5-mlp; do rf $v 1 10 10000 $F; done
for s in 2 3; do for v in vanilla route30 route30-mlp; do rf $v $s 10 10000 $F; done; done
gate; (cd baseline/rrnco && for d in in_distribution in_distribution_cluster out_of_distribution; do timeout 3600 ../../runs/capped.sh rrnco-test 30 ../../.venv-vrp/bin/python test.py --problem rcvrp --datasets data/rcvrp/rcvrp_n100_seed3333_$d.npz --checkpoint checkpoints/rcvrp/epoch_199.ckpt --batch_size 32 > ../../runs/routefinder/rrnco_test_$d.log 2>&1; echo "RRNCO_TEST_EXIT $d $? $(date -Is)" >> ../../$S; done) &
sleep 45
for v in route30-fgn route30-mlp-fgn; do rf $v 1 1 2000 $P; done
for s in 1 2 3; do for v in route30-fgn route30-mlp-fgn; do rf $v $s 10 10000 $F; done; done
rf vanilla 1 1 2000 $P
for v in q30 q30-mlp q10-part=1x10 q10-part=1x10-mlp q20-part=2x10 q20-part=2x10-mlp q5-part=1x5 q5-part=1x5-mlp; do rf $v 1 1 2000 $P; done
wait; echo "RF_QUEUE2C_DONE $(date -Is)" >> $S
