#!/bin/bash
# TDC ADMET Group row (drug discovery, user directive Sep 21). Official leaderboard protocol: 22 endpoints,
# TDC's own train_val/test split, its per-seed validation split, 5 seeds, its evaluator. Frozen ChemBERTa-77M
# trunk (embeddings precomputed once in data/tdc/emb), arms differ only in the bottleneck on the pooled
# embedding. Waits for the HPO chain so the two do not contend for the GPU. Resumable (skips existing JSONs).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
S=runs/tdc/status; P=.venv-tdc/bin/python
while ! grep -q HPO_MS2_NONE_DONE runs/hpo/status 2>/dev/null; do sleep 120; done
echo "tdc runner start $(date -Is)" >> "$S"
for ep in $($P -c "import sys; sys.path.insert(0,'paper/tdc'); import tdcdata as D; print(' '.join(D.endpoints()))" 2>/dev/null | tail -1); do
  for arm in vanilla mlp30 mr30ryd; do
    [ -f runs/tdc/${ep}_${arm}.json ] && continue
    while :; do a=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo); [ "$a" -lt 20 ] && { sleep 60; continue; }; break; done
    $P paper/tdc/run_admet.py "$ep" "$arm" > runs/tdc/${ep}_${arm}.log 2>&1 \
      && echo "result $ep $arm: $(grep TEST runs/tdc/${ep}_${arm}.log | tail -1) $(date -Is)" >> "$S" \
      || echo "CRASH $ep $arm $(date -Is)" >> "$S"
  done
  echo "endpoint $ep done $(date -Is)" >> "$S"
done
echo "TDC_DONE $(date -Is)" >> "$S"
