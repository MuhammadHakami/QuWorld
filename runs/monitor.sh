#!/bin/bash
# 20-minute monitor (user Oct 3): status, distance to SoTA and weak spots for the three lanes.
# Fusion (Spark) | Learn2Design (merphistophline ROCm + mephiles 3080 Ti) | host health. Read-only.
cd /home/mohhakami/Documents/paper/quworld
M="timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=10 -i $HOME/.ssh/id_ed25519_quworld mohhakami@merphistophline"
P="timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=10 moka@mephiles-1"
echo "=== $(date '+%F %H:%M') ==="
echo "--- FUSION (Spark) ---"
timeout 90 python3 runs/codabench_board.py 2>/dev/null | grep -E "^ +1 |mohhakami|gap to" | cut -c1-150
for f in s38s shot_bias_lin shot_bias_mlp shot_bias_reg; do
  [ -f runs/strat/fusion/$f.log ] && echo "[$f] $(grep -av Warn runs/strat/fusion/$f.log | grep -E 'step|SB_RESULT|\[s38g\]|GPU_SAMPLE|GPU_TEST|DONE|CRASH|Traceback|\[sb\]' | tail -1 | cut -c1-170)"
done
echo "Spark: $(free -g | awk '/Mem/{print "avail "$7"G"}'), GPU $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader), memguard $(pgrep -f '[m]emguard.sh' >/dev/null && echo on || echo OFF)"
L=runs/strat/fusion/s38ac_hpo.log; [ -f $L ] && { echo "QPI-KAN search: $(grep -c "^QK_TRIAL" $L) trials done, $(grep -c "TrialPruned\|pruned" $L) pruned; best so far:"; grep "^QK_TRIAL" $L | sort -k5 -g -r | head -3 | cut -c1-150; grep "^\[qk\] t[0-9]* epoch" $L | tail -1 | cut -c1-120; }
echo "--- LEARN2DESIGN (deadline 15 Oct AoE; Round 2 #1 mean -0.211, best baseline 0.504) ---"
$P 'bash -c "cd ~/Documents/paper/quworld/runs/l2d; for f in sub_*.log wvc_*.log; do grep -h \"l2d-sub\|l2d-wvc\" \$f 2>/dev/null | tail -2 | sed -E \"s/\\{([^}]*)\\}/{...}/\" ; done | cut -c1-110; for f in sub_*.json; do python3 -c \"import json,sys; d=json.load(open(sys.argv[1])); c=d[\\\"curve\\\"]; k=sorted(c,key=int)[-1]; print(sys.argv[1], \\\"final@\\\"+k, round(c[k],4) if c[k] is not None else None)\" \$f; done; echo mephiles GPU \$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader); pgrep -fa \"[r]un_submission|[w]arm_vs_cold\" | sed -E \"s/.*(--run [0-9]+).*--tag ([a-z0-9]+).*/running \\1 \\2/\""' 2>/dev/null
$M 'bash -c "cd ~/Documents/paper/quworld/runs/l2d; for f in sub_*.json; do python3 -c \"import json,sys; d=json.load(open(sys.argv[1])); c=d[\\\"curve\\\"]; k=sorted(c,key=int)[-1]; print(sys.argv[1], \\\"final@\\\"+k, round(c[k],4) if c[k] is not None else None)\" \$f; done 2>/dev/null; pgrep -fa \"[r]un_submission|[w]arm_vs_cold\" | sed -E \"s/.*(--run [0-9]+).*--tag ([a-z0-9]+).*/running \\1 \\2/\"; df -h ~ | tail -1 | awk \"{print \\\"merphistophline disk free \\\"\\\$4}\""' 2>/dev/null
