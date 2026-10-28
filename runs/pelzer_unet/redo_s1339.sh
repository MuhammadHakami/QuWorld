#!/bin/bash
# Pelzer U-Net: redo the seed-1339 pretraining (killed by the memory guard at step 600) and its three arms after the main runner ends.
export PATH="$HOME/.local/bin:$PATH"; cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
O=runs/pelzer_unet; S=$O/status
while ! grep -q PELZER_UNET_DONE $S; do sleep 120; done
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 20 ]; do sleep 60; done; }
LR=$(cat $O/best_lr); LRTAG=$(python3 -c "print(f'{float(\"$LR\"):g}')")
if [ ! -f $O/pretrain_lr${LRTAG}_s1339.json ]; then mem_gate; echo "redo pretrain lr$LR s1339 $(date -Is)" >> "$S"
  uv run --no-sync python paper/geothermal/train_pelzer_unet.py pretrain $LR 1339 3000 > $O/pretrain_lr${LRTAG}_s1339.log 2>&1 || echo "CRASH redo pretrain s1339 $(date -Is)" >> "$S"; fi
for v in vanilla mlp30 mr30ryd; do
  [ -f $O/${v}_s1339.json ] && continue; mem_gate; echo "redo arm $v s1339 $(date -Is)" >> "$S"
  uv run --no-sync python paper/geothermal/train_pelzer_unet.py arm $v 1339 1500 1e-4 $O/pretrain_lr${LRTAG}_s1339.pt > $O/${v}_s1339.log 2>&1 || echo "CRASH redo arm $v s1339 $(date -Is)" >> "$S"
done
echo "PELZER_UNET_S1339_DONE $(date -Is)" >> "$S"
