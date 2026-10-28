#!/bin/bash
# §38e rebuild: H1 heads (width 4096, depth 3) x 6 seeds, inputs per the §38g G1 rule (registered before G1's result):
# G1 mean R^2 > H1 s1 (0.98415) + 0.0002 -> corrected inputs (magfix), else cached.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q S38G_DONE $L/s38g_qb.log 2>/dev/null; do sleep 60; done
G1=$(grep "head_boundary_v8_G1_s1:" $L/s38g_heads.log | sed 's/.*mean R2 \([0-9.]*\).*/\1/')
INP=$(python3 -c "print('magfix' if float('${G1:-0}') > 0.98415 + 0.0002 else '')")
TAG=$([ "$INP" = magfix ] && echo _H1F || echo _H1)
echo "G1 mean R2 ${G1:-NA} -> inputs '${INP:-cached}' tag $TAG $(date -Is)" > $L/s38e6.log
for seed in 1 2 3 4 5 6; do
  [ $seed = 1 ] && continue                     # seed 1 exists: H1_s1 (cached) or G1_s1 (corrected)
  [ -f $L/head_boundary_v8${TAG}_s$seed.npz ] && continue
  flock runs/strat/gpu.lock env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 \
    STRAT_DEPTH=3 STRAT_INPUTS=$INP STRAT_OUT_TAG=$TAG QW_GPU_GB=24 runs/capped.sh s38e6 40 .venv/bin/python paper/strat/fusion/head_boundary.py $seed \
    >> $L/s38e6.log 2>&1 || echo "CRASH seed $seed $(date -Is)" >> $L/s38e6.log
done
S=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8${TAG}_s$s+"; done | sed 's/+$//')
[ "$TAG" = _H1F ] && S=$(echo $S | sed 's/head_boundary_v8_H1F_s1/head_boundary_v8_G1_s1/')
.venv/bin/python paper/strat/fusion/head_screen.py "$S" "head_boundary_v8_s1+head_boundary_v8_s2+head_boundary_v8_s3+head_boundary_v8_s4+head_boundary_v8_s5+head_boundary_v8_s6" >> $L/s38e6.log 2>&1
echo "S38E6_DONE $S" >> $L/s38e6.log
