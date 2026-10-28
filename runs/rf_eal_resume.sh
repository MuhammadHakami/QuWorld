#!/bin/bash
# RouteFinder EAL: resume after the RealPDE trunk evaluation finishes (memory), under OOM priority (capped.sh).
cd /home/mohhakami/Documents/paper/quworld
while ps -eo args | grep -q "[e]val_trunk.py --models"; do sleep 120; done
runs/capped.sh rf-eal 32 bash runs/rf_eal_runner.sh > runs/routefinder/runner.log 2>&1
