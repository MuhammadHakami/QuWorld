#!/bin/bash
# Pause the step-4 XP1 head training while F6's XPA heads train (F6 reaches the board sooner); resume after XPA_s3 (or F6 failure).
cd /home/mohhakami/Documents/paper/quworld; L=runs/strat/fusion; LOG=$L/s42t_vt3.log
P=""
for p in $(pgrep -f '[h]ead_boundary.py'); do tr '\0' ' ' < /proc/$p/environ | grep -q 'STRAT_OUT_TAG=_XP1 ' && P="$P $p"; done
trap '[ -n "$P" ] && kill -CONT $P 2>/dev/null' EXIT
[ -n "$P" ] && kill -STOP $P && echo "[pause] XP1 pids$P paused for F6's XPA $(date -Is)" >> $LOG
until grep -q "^XPA_s3_TRAINED\|^CRASH\|^F6_\|resumed F5c" $L/f6_xh.log 2>/dev/null; do
  # a new XP1 process (next seed) started by the runner is paused as well
  for p in $(pgrep -f '[h]ead_boundary.py'); do tr '\0' ' ' < /proc/$p/environ | grep -q 'STRAT_OUT_TAG=_XP1 ' && ! echo " $P " | grep -q " $p " && kill -STOP $p && P="$P $p"; done
  sleep 30
done
kill -CONT $P 2>/dev/null; echo "[pause] XP1 resumed $(date -Is)" >> $LOG; P=""
