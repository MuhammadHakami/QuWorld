#!/bin/bash
# §45t relaunch: the 150-shot run was killed by memguard (avail 19 GB with 4 trainings resident); start it once the three J1QHCS
# register trainings have written their checkpoints (only the X-point training remains then).
cd /home/mohhakami/Documents/paper/quworld; L=runs/strat/fusion
until [ -f $L/head_boundary_v8_J1QHCS_s1.pt ] && [ -f $L/head_boundary_v8_J1QHCS_s2.pt ] && [ -f $L/head_boundary_v8_J1QHCS_s3.pt ]; do sleep 60; done
sleep 120; exec runs/s45t_trs.sh 150
