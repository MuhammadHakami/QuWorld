#!/bin/bash
# §44f li accept-if-better guard (quench frames: the li warp overshot, final 5.99 vs head 2.64): paired VT arms F8K vs F8K + li guard on the
# 16 worst-li VT shots (35.5% of the VT li SSE in the F7-era arm; the 150-shot sample had no overshoot frames), newest TRAIN-only heads.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; LOG=runs/strat/fusion/s44f_liguard_vt.log
until grep -q "^S44_TRANSFER_DONE" runs/strat/fusion/s44_transfer_vt.log 2>/dev/null || grep -q "^CRASH" runs/strat/fusion/s44_transfer_vt.log 2>/dev/null; do sleep 30; done
env QW_GPU_GB=30 runs/capped.sh s44fvt 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag S44FW --psi psiF --psis "PQmix~g" \
  --guard psiF:0.25 --qb qh50 --qbheads "head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6" --heads "head_boundary_v8_J1RL_s1*3+head_boundary_v8_J1QPIpi_s1+head_boundary_v8_J1QPIpi_s2+head_boundary_v8_J1QH_s1+head_boundary_v8_J1QH_s2+head_boundary_v8_J1QH_s3" --xpheads head_xp_XP1_s1+head_xp_XP1_s2+head_xp_XP1_s3 --kws '{"F8K": {"_steps": ["shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"], "xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.01}}, "F8Kg": {"_steps": ["shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"], "xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9, "guard": true}, "xhe": {"gate": 0.01}}}' --axis head \
  --qbclamp --vt-shots "1184,164,6520,6960,849,5170,6945,3659,2716,3578,4732,2669,275,3136,5405,2196" >> $LOG 2>&1 || { echo "CRASH s44f vt $(date -Is)" >> $LOG; exit 1; }
for arm in F8K F8Kg; do
  .venv/bin/python runs/strat/fusion/stage_decomp.py "PQmix~g_ns" "PQmix~g_$arm" J1RL_s1,J1RL_s1,J1RL_s1,J1QPIpi_s1,J1QPIpi_s2,J1QH_s1,J1QH_s2,J1QH_s3 >> $LOG 2>&1
done
echo "S44F_DONE $(date -Is)" >> $LOG
