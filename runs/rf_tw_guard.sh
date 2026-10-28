#!/bin/bash
# mephiles: stop the §39d runner after seed 3 (seeds 4-5 run on the Spark).
O=~/Documents/paper/quworld/runs/routefinder/bf16_mephiles
until [ -f $O/eal_md_route30-tw_s3.json ]; do sleep 10; done
pkill -f "bash runs/[r]f_tw.sh"; sleep 3; pkill -f "[r]un_rf_eal.py route30-tw --setting md --seed 4"
echo "RF_TW_GUARD stopped after s3 $(date -Is)" >> $O/runner_tw.log
