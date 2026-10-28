#!/bin/bash
# Pelzer U-Net redo (takes over the dead owner session's redo_s1339.sh, Sep 19 17:40): seed-1339 pretraining (killed by the
# memory guard at step 600) + its three arms, THEN the seed-1338 mlp30/mr30ryd arms that memguard shed at 17:29/17:31.
# Writes PELZER_UNET_S1339_DONE only after the whole 3x3 block is complete (the Well chain v2 waits on it -> one GPU co-runner).
export PATH="$HOME/.local/bin:$PATH"; cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
O=runs/pelzer_unet; S=$O/status
while ! grep -q PELZER_UNET_DONE $S; do sleep 120; done
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 14 ]; do sleep 60; done; }  # 14 GB: arms are ~2-3 GB; teacher holds ~110 GB; guard kills at 9
LR=$(cat $O/best_lr); LRTAG=$(python3 -c "print(f'{float(\"$LR\"):g}')")
if [ ! -f $O/pretrain_lr${LRTAG}_s1339.json ]; then mem_gate; echo "redo pretrain lr$LR s1339 $(date -Is)" >> "$S"
  uv run --no-sync python paper/geothermal/train_pelzer_unet.py pretrain $LR 1339 3000 > $O/pretrain_lr${LRTAG}_s1339.log 2>&1 || echo "CRASH redo pretrain s1339 $(date -Is)" >> "$S"; fi
for v in vanilla mlp30 mr30ryd; do
  [ -f $O/${v}_s1339.json ] && continue; mem_gate; echo "redo arm $v s1339 $(date -Is)" >> "$S"
  uv run --no-sync python paper/geothermal/train_pelzer_unet.py arm $v 1339 1500 1e-4 $O/pretrain_lr${LRTAG}_s1339.pt > $O/${v}_s1339.log 2>&1 || echo "CRASH redo arm $v s1339 $(date -Is)" >> "$S"
done
for v in mlp30 mr30ryd; do
  [ -f $O/${v}_s1338.json ] && continue; mem_gate; echo "redo arm $v s1338 $(date -Is)" >> "$S"
  uv run --no-sync python paper/geothermal/train_pelzer_unet.py arm $v 1338 1500 1e-4 $O/pretrain_lr${LRTAG}_s1338.pt > $O/${v}_s1338.log 2>&1 || echo "CRASH redo arm $v s1338 $(date -Is)" >> "$S"
done
# Sep 19 23:10: the 18:21 PELZER_UNET_S1339_DONE line was written after memguard had killed the s1339 pretrain (17:47) -> sentinel now conditional
missing=""; for f in vanilla_s1339 mlp30_s1339 mr30ryd_s1339 mlp30_s1338 mr30ryd_s1338; do [ -f $O/$f.json ] || missing="$missing $f"; done
if [ -z "$missing" ]; then echo "PELZER_UNET_S1339_DONE $(date -Is)" >> "$S"; else echo "REDO INCOMPLETE (missing:$missing) $(date -Is)" >> "$S"; fi
