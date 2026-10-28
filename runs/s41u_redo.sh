#!/bin/bash
# Fusion §41u: J1QPIpiAH seed 1 died in the 15:45 OOM (the §41y-b VT job); after seed 2 finishes, rerun the §41u runner (it skips
# finished seeds, so it retrains seed 1 only).
cd /home/mohhakami/Documents/paper/quworld; L=runs/strat/fusion
until grep -q S41U_DONE $L/J1QPIpiAH_s1.log 2>/dev/null; do sleep 120; done
runs/s41u_qpiah.sh
