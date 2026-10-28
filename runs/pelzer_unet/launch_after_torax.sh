#!/bin/bash
# one GPU co-runner at a time next to the DiT teacher: wait for TORAX_DONE, then run the Pelzer U-Net track
cd /home/mohhakami/Documents/paper/quworld
while ! grep -q TORAX_DONE runs/torax/status 2>/dev/null; do sleep 60; done
bash runs/pelzer_unet_runner.sh
