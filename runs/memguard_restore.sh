#!/bin/bash
# After F7's build ends: memguard back from 22 GB (lowered for the F7 build next to ~21 GB of VS Code servers) to 25 GB.
cd /home/mohhakami/Documents/paper/quworld; L=runs/strat/fusion
until grep -q "^F7_DONE\|^F7_FAIL\|^F7_SKIP" $L/f7.log 2>/dev/null; do sleep 20; done
for p in $(pgrep -f "^bash runs/memguard.sh"); do kill $p; done
setsid nohup bash runs/memguard.sh 25 > /dev/null 2>&1 < /dev/null &
echo "memguard restored to 25 at $(date -Is)" >> runs/memguard_notes.txt
