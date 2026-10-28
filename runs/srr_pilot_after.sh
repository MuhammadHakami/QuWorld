#!/bin/bash
# Start the SRR pilot (runner v2) only after the RealPDE trunk eval, the TokaMark ensembles and Sophelio NAS stage 1 are done (memory).
cd /home/mohhakami/Documents/paper/quworld
until grep -q TOKAMIND_ENS_DONE runs/tokamark/status 2>/dev/null && grep -q NAS2_TRUNK_DONE runs/sophelio_sota/status 2>/dev/null && ! ps -eo args | grep -q "[e]val_trunk.py --models"; do sleep 120; done
exec bash runs/srr_pilot_runner2.sh
