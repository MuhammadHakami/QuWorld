#!/bin/bash
# doc 35 §23: single-shot TEST of the MHD_64 3-D register / matched 3-D MLP pair (seed 1337), exactly as
# paper/phase_b/run_mhd_campaign.sh scored the earlier neck models (eval_official.py with the MHD checkpoint patched in).
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
OUT=runs/phase_b_mhd; S=runs/alloc/status
for name in q30-geo=3d_s1337 q30-geo=3d-mlp_s1337; do
  csv="$OUT/test_${name}.csv"; [ -f "$csv" ] && continue
  flock runs/strat/gpu.lock env WELL_DATASET=MHD_64 EVAL_BATCH=8 QW_GPU_GB=28 timeout 3600 runs/capped.sh s23mhdtest 60 \
    uv run --no-sync python -c "
import sys
sys.path.insert(0, 'paper/phase_b')
import cnext_hybrid
_orig = cnext_hybrid.build_hybrid
_mhd = str(cnext_hybrid.ROOT / 'data/cnext_mhd_ckpt')
cnext_hybrid.build_hybrid = (lambda variant, model_dir=_mhd, q_layers=4: _orig(variant, model_dir=model_dir, q_layers=q_layers))
import eval_official
eval_official.main()
" "$OUT/${name}.pt" "$csv" > "$OUT/test_${name}.log" 2>&1 || echo "EVAL_CRASH s23 ${name} $(date -Is)" >> $S
done
echo "S23_MHD_TEST_DONE $(date -Is)" >> $S
