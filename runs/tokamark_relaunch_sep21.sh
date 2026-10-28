#!/bin/bash
# Sep 21 00:35: wait for the Physics-IQ quanttest to release the GPU, then run the CNN-trunk lane
# (runs/tokamark_runner.sh: 3 arms, writes TOKAMARK_DONE) which the TokaMind runner chains behind.
cd /home/mohhakami/Documents/paper/quworld
while pgrep -f "physiq_runner.py --stage quanttest" > /dev/null; do sleep 60; done
setsid nohup bash runs/tokamark_runner.sh > /dev/null 2>&1 < /dev/null &
sleep 5
pgrep -f "^bash runs/tokamind_runner.sh" > /dev/null || setsid nohup bash runs/tokamind_runner.sh > /dev/null 2>&1 < /dev/null &
echo "$(date -Is) relaunched tokamark_runner + tokamind_runner" >> runs/tokamark/status
