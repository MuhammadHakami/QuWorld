#!/bin/bash
# §47c: V s1-3 TRAIN-only full length (two at a time), then the §45y-protocol VT read f17eq vs f17V (V share 0.252).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s47c.log
V='{"backbone":"vhope","seq":true,"crop":32,"shots":128,"bb_width":128,"bb_depth":2,"tokens":8}'
echo "[s47c] start $(date -Is)" >> $LOG
tr() { [ -f $L/head_boundary_v8_V_s$1.pt ] && return 0
  env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QW_GPU_GB=30 \
    runs/capped.sh s47cV$1 40 .venv/bin/python paper/strat/fusion/quota_fusion.py final V "$V" $1 20 > $L/s47c_V_s$1.log 2>&1
  echo "[s47c] V s$1 exit $? $(date -Is)" >> $LOG; }
tr 1 & tr 2 & wait; tr 3
for s in 1 2 3; do [ -f $L/head_boundary_v8_V_s$s.pt ] || { echo "S47C_FAIL training $(date -Is)" >> $LOG; exit 1; }; done
SHOTS=$(grep -o -- '--vt-shots "[0-9,]*"' $L/s44h_jgn_end.sh | cut -d'"' -f2)
C='"shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"'; X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
QPI="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"; QHCS="head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
N="head_boundary_v8_J1RL_s1*24"; R="head_boundary_v8_J1RLCS_s1*24"; VV="head_boundary_v8_V_s1*4.5+head_boundary_v8_V_s2*4.5+head_boundary_v8_V_s3*4.5"
for arm in f17eq f17V; do
  [ -f "$L/parts_VT-PQmix~g_$arm.pkl" ] && continue          # f17eq exists from §45y (same protocol, 12:35)
  if [ $arm = f17eq ]; then H="route:$N+$QPI+$QHCS|$R+$QPI+$QHCS"; else H="route:$N+$QPI+$QHCS+$VV|$R+$QPI+$QHCS+$VV"; fi
  flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s47c 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag S47C$arm --psi psiF \
    --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6" \
    --heads "$H" --xpheads head_xp_XP1CS_s1 --kws "{\"$arm\": {\"_steps\": [$C], $X}}" --axis head --qbclamp --vt-shots "$SHOTS" >> $LOG 2>&1 \
    || { echo "S47C_FAIL vt $arm $(date -Is)" >> $LOG; exit 1; }
done
.venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmix~g_f17eq" "PQmix~g_f17V" >> $LOG 2>&1
echo "S47C_DONE $(date -Is)" >> $LOG
