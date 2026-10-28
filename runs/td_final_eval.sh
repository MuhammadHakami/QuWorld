#!/bin/bash
# Doc 38 secondary (registered 06:12 Sep 26): evaluate step-10,000 checkpoints of all arms, seeds 1337-1340.
# Pinned to the A725 efficiency cores so the TD replication keeps the X925 cores; s1340 mlp30/vanilla wait for REPL_DONE.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/phase_b2/replication_status; PY=.venv/bin/python
for s in 1337 1338 1339 1340; do for v in mr30ryd mlp30 vanilla; do
  n=${v}final_s${s}; [ -f runs/phase_b2/multistart_$n.npz ] && continue
  if [ $s = 1340 ] && [ $v != mr30ryd ]; then until grep -q "REPL_DONE ${v}_s1340" $S; do sleep 60; done; fi
  ck=runs/phase_b2/final/${v}_s${s}.pt
  taskset -c 0-4,10-14 $PY paper/phase_b/eval_rollout_pertraj.py $ck $n > runs/phase_b2/rollout_$n.log 2>&1 || echo "CRASH final pertraj $n $(date -Is)" >> $S
  taskset -c 0-4,10-14 $PY paper/phase_b/eval_rollout_multistart.py $ck $n > runs/phase_b2/multistart_$n.log 2>&1 || echo "CRASH final multistart $n $(date -Is)" >> $S
  echo "FINAL_EVAL_DONE $n $(date -Is)" >> $S
done; done
echo "FINAL_EVAL_ALL_DONE $(date -Is)" >> $S
