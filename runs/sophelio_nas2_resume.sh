#!/bin/bash
# Resume the two-stage Sophelio NAS (guard-killed 08:10) once the RealPDE trunk eval is done and >= 40 GB is free.
cd /home/mohhakami/Documents/paper/quworld
until ! ps -eo args | grep -q "[e]val_trunk.py --models" && [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 40 ]; do sleep 120; done
echo "nas2 resume $(date -Is)" >> runs/sophelio_sota/status
exec bash runs/sophelio_nas2_runner.sh
