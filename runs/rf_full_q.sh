#!/bin/bash
# RouteFinder full-protocol queue (doc 35 §19 amendment Sep 25 22:15): at most 3 full runs (--epochs 10) at once, 4 RF
# processes in total, >= 25 GB free. Order: the vanilla-based qubit ladder (K = 20, 10; register and MLP) at seed 1, then
# seeds 2-3 of vanilla / route30 / route30-mlp, then K = 5 at seed 1, then the §18 FGN arms if time remains.
cd /home/mohhakami/Documents/paper/quworld; S=runs/routefinder/status; RF=baseline/routefinder; F=runs/routefinder/full
PY='^/home/mohhakami/Documents/paper/quworld/baseline/(routefinder|rrnco)/../../.venv-vrp/bin/python'
gate() { while [ "$(pgrep -fc "$PY")" -ge 4 ] || [ "$(pgrep -fc "$PY.* --epochs 10 ")" -ge 3 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 25 ]; do sleep 30; done; }
rf() {  # $1 variant $2 seed
  local j=$F/eal_md_$1_s$2.json; [ -f $j ] && return; pgrep -f "run_rf_eal.py $1 --setting md --seed $2 --epochs 10 " > /dev/null && return
  gate; (cd $RF && timeout 36000 ../../runs/capped.sh rfq 30 ../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py $1 --setting md --seed $2 --epochs 10 --train-size 10000 --precision 32 --out ../../$F > ../../$F/$1_s$2.log 2>&1 \
    && echo "RF_FULL_DONE $1 s$2 $(date -Is)" >> ../../$S || echo "CRASH rf-full $1 s$2 $(date -Is)" >> ../../$S) &
  sleep 60
}
mkdir -p $F
rf vanilla 1
for v in route20 route20-mlp route10; do rf $v 1; done
for v in vanilla route30 route30-mlp; do rf $v 2; done           # 01:05 (Gemini's counter-case, accepted): replicate the
rf route10-mlp 1                                                  # headline null (seeds 2-3) before the §18 FGN arms
for v in vanilla route30 route30-mlp; do rf $v 3; done
for s in 1 2 3; do for v in route30-fgn route30-mlp-fgn; do rf $v $s; done; done
for v in route5 route5-mlp; do rf $v 1; done
wait; echo "RF_FULL_Q_DONE $(date -Is)" >> $S
