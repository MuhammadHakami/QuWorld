#!/bin/bash
# RouteFinder EAL row (logistics): their EAL-full fine-tuning to unseen multi-depot variants + {vanilla, mlp30, mr30ryd}
# bottleneck on the encoder's node embeddings, 3 seeds. Re-queued Sep 23 after the replication runner was deferred.
# Starts after the HPO chain (TDC moved to a CheMeleon trunk, Sep 23). Resumable: existing JSONs are skipped.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
S=runs/routefinder/status
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 25 ]; do sleep 60; done; }
while ! grep -q HPO_MS2_NONE_DONE runs/hpo/status 2>/dev/null; do sleep 300; done
[ -f runs/routefinder/EAL_SMOKE_OK ] || { echo "SKIP rf-eal: no EAL_SMOKE_OK $(date -Is)" >> "$S"; exit 0; }
echo "rf-eal runner start $(date -Is)" >> "$S"
for seed in 1 2 3; do for v in vanilla mlp30 mr30ryd; do
  [ -f runs/routefinder/eal_md_${v}_s${seed}.json ] && continue
  mem_gate; echo "rf-eal md $v s$seed $(date -Is)" >> "$S"
  (cd baseline/routefinder && ../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py $v --setting md --seed $seed) \
    >> runs/routefinder/eal_md.log 2>&1 && echo "result rf-eal md $v s$seed: $(grep 'mean test gap' runs/routefinder/eal_md.log | tail -1) $(date -Is)" >> "$S" \
    || echo "CRASH rf-eal md $v s$seed $(date -Is)" >> "$S"
done; done
echo "RF_EAL_DONE $(date -Is)" >> "$S"
