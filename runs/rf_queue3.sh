#!/bin/bash
# RouteFinder queue v3 (doc 35 §18): noise-conditioned routed configuration; shares queue v2's 4-lane gate.
cd /home/mohhakami/Documents/paper/quworld; S=runs/routefinder/status; RF=baseline/routefinder
gate() { while [ "$(pgrep -fc '^/home/mohhakami/Documents/paper/quworld/baseline/(routefinder|rrnco)/../../.venv-vrp/bin/python')" -ge 4 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 25 ]; do sleep 37; done; }
rf() {  # $1 variant $2 seed $3 epochs $4 train-size $5 outdir
  local out=$5 j=$5/eal_md_$1_s$2.json; [ -f $j ] && return; pgrep -f "run_rf_eal.py $1 --setting md --seed $2 --epochs $3 " > /dev/null && return
  gate; (cd $RF && timeout 36000 ../../runs/capped.sh rfq 30 ../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py $1 --setting md --seed $2 --epochs $3 --train-size $4 --precision 32 --out ../../$out > ../../$out/$1_s$2.log 2>&1 \
    && echo "RF_Q3_DONE $1 s$2 e$3 $(date -Is)" >> ../../$S || echo "CRASH rf-q3 $1 s$2 e$3 $(date -Is)" >> ../../$S) &
  sleep 45
}
P=runs/routefinder/crn; F=runs/routefinder/full; mkdir -p $P $F
for v in route30-fgn route30-mlp-fgn; do rf $v 1 1 2000 $P; done
for s in 1 2 3; do for v in route30-fgn route30-mlp-fgn; do rf $v $s 10 10000 $F; done; done
wait; echo "RF_QUEUE3_DONE $(date -Is)" >> $S
