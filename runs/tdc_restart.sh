#!/bin/bash
# TDC CheMeleon runner restart after the Sep 23 02:26 OOM/reboot, memory-capped (resumable: finished endpoints skipped).
cd /home/mohhakami/Documents/paper/quworld; export QW_GPU_GB=8
echo "RESTART after OOM/reboot (capped 12G) $(date -Is)" >> runs/tdc_chemeleon/status
runs/capped.sh tdc 12 bash runs/tdc_chemeleon_runner.sh > runs/tdc_chemeleon/runner.log 2>&1
