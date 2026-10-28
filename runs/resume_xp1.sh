#!/bin/bash
# Keep the step-4 XP1 training paused until F6's public build finishes (<= 2 Fusion GPU jobs: F5c + F6 build), then resume it.
cd /home/mohhakami/Documents/paper/quworld; L=runs/strat/fusion; LOG=$L/s42t_vt3.log; P=862913
trap 'kill -CONT $P 2>/dev/null' EXIT
until grep -q "^F6_DONE\|^F6_FAIL\|^CRASH\|\[f6\] expected" $L/f6_xh.log 2>/dev/null; do sleep 30; done
kill -CONT $P 2>/dev/null; echo "[pause] XP1 resumed after F6's build $(date -Is)" >> $LOG
