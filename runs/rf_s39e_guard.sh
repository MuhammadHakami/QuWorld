#!/bin/bash
# mephiles: stop §39e after seed 3's classical decode (seeds 4-5 run on the Spark).
D=~/Documents/paper/quworld/runs/strat/routing/test/vanilla_s23_mephiles_3__s8x4t05
until [ $(ls $D/*.json 2>/dev/null | wc -l) -ge 16 ]; do sleep 20; done
pkill -f "bash runs/[r]f_s39e.sh"; pkill -f "[t]est_decode.py --budget 70000"
echo "S39E_GUARD stopped after seed 3 $(date -Is)" >> ~/Documents/paper/quworld/runs/strat/routing/s39e.log
