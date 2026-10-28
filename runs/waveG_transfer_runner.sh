#!/bin/bash
# Wave G transfer (29-graph-configuration-plan.md, "what comes after"): the promoted register configuration
# q30-link=shared and its matched control (q30-link=shared-mlp) enter TORAX, Pelzer U-Net and Sophelio as a
# configuration-only adaptation, 3 seeds each, on exactly the protocols of the existing rows (their vanilla /
# mlp30 / mr30ryd arms are already on file). Chained behind WAVEG_GATE_DONE; same GPU gate as wave G.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; export PATH="$HOME/.local/bin:$PATH"
S=runs/waveG/status; echo "transfer runner start $(date -Is)" >> "$S"
while ! grep -q "^WAVEG_GATE_DONE" "$S" 2>/dev/null; do sleep 300; done
gate() { while :; do a=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  pgrep -f "physiq_runner[.]py --stage generate" > /dev/null && { sleep 120; continue; }
  pgrep -f "wmb/generate[.]py" > /dev/null && { sleep 120; continue; }
  [ "$a" -lt 40 ] && { sleep 60; continue; }; break; done; }
CFGS="q30-link=shared q30-link=shared-mlp"
# TORAX (1000 steps, batch 16, k=4) -----------------------------------------------------------------
for seed in 1337 1338 1339; do for v in $CFGS; do
  f=runs/torax/${v}_k4_s${seed}.json; [ -f "$f" ] && continue
  gate; echo "torax $v s$seed $(date -Is)" >> "$S"
  uv run --no-sync python paper/torax_bench/train_torax.py "$v" "$seed" 1000 4 runs/torax > runs/torax/${v}_k4_s${seed}.log 2>&1 || { echo "CRASH torax $v s$seed $(date -Is)" >> "$S"; continue; }
  echo "result torax $v s$seed: $(python3 -c "import json; d=json.load(open('$f')); print(f\"vrmse8={d['test_vrmse_8']:.4f} vrmse1={d['test_vrmse_1']:.4f} params={d['num_params']}\")") $(date -Is)" >> "$S"
done; done
echo "TRANSFER_TORAX_DONE $(date -Is)" >> "$S"
# Pelzer U-Net (arms from the seed's pretrain checkpoint; ARM_STEPS/ARM_LR as the existing row) ---------
O=runs/pelzer_unet; ARM_STEPS=1500; ARM_LR=1e-4; LRTAG=0.001
for seed in 1337 1338 1339; do for v in $CFGS; do
  [ -f $O/${v}_s${seed}.json ] && continue
  gate; echo "pelzer $v s$seed $(date -Is)" >> "$S"
  uv run --no-sync python paper/geothermal/train_pelzer_unet.py arm $v $seed $ARM_STEPS $ARM_LR $O/pretrain_lr${LRTAG}_s${seed}.pt > $O/${v}_s${seed}.log 2>&1 || { echo "CRASH pelzer $v s$seed $(date -Is)" >> "$S"; continue; }
  echo "result pelzer $v s$seed: $(python3 -c "import json; d=json.load(open('$O/${v}_s${seed}.json')); t=d['test']; print(f\"mae_c={t['mae_c']:.5f} pat01={t['pat_0p1']:.4f}\")") $(date -Is)" >> "$S"
done; done
echo "TRANSFER_PELZER_DONE $(date -Is)" >> "$S"
# Sophelio (starter protocol: 200 shots, 50 epochs, official scorer) ---------------------------------
O=runs/sophelio
for seed in 1337 1338 1339; do for v in $CFGS; do
  [ -f $O/${v}_s${seed}.json ] && continue
  gate; echo "sophelio $v s$seed $(date -Is)" >> "$S"
  uv run --no-sync python paper/sophelio/train_hybrid.py --variant $v --seed $seed --n-shots 200 --epochs 50 > $O/${v}_s${seed}.log 2>&1 || { echo "CRASH sophelio $v s$seed $(date -Is)" >> "$S"; continue; }
  echo "result sophelio $v s$seed: $(python3 -c "import json; d=json.load(open('$O/${v}_s${seed}.json')); o=d['official']; print(f\"S={o['S']:.4f} r2psi={o['r2_psi']:.4f} cons={o['consistency']:.4f} dlcfs={o['dlcfs']:.4f}\")") $(date -Is)" >> "$S"
done; done
echo "TRANSFER_DONE $(date -Is)" >> "$S"
