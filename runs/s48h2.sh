#!/bin/bash
# §48h stage 2 (doc 35, registered before any full-length result): PSIV60 s1-2 (60 epochs, all TRAIN) -> VT arms f20Pa (F18K heads, psi PQmixV~g)
# and f20Pb (minus J1QPIpi) vs PQmix~g_f17V60; adopt F20K iff f20Pb CI lower > 0 and D_LCFS not worse.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=$PWD/runs/strat/fusion; LOG=$L/s48h.log
E="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2"
J='{"target": "psi", "backbone": "vhope", "neck": "gate", "win": 8, "stride": 1, "bb_width": 128, "bb_depth": 4, "width": 2048, "depth": 4, "drop": 0.0, "q_layers": 4, "q_in": 32, "link": "independent", "lr": 0.000977105771213329, "alpha_init": 0.11830330369409621, "seq": true, "crop": 16, "shots": 256, "tokens": 12, "q_noise": 0.0, "in_noise": 0.0, "huber": 1.0, "mixer": 64, "reg": "tether3d"}'
echo "[s48h2] start $(date -Is)" >> $LOG
tr1() { [ -f $L/head_boundary_v8_PSIV60_s$1.pt ] && return 0
  env $E QW_GPU_GB=30 runs/capped.sh s48hP$1 40 .venv/bin/python paper/strat/fusion/quota_fusion.py final PSIV60 "$J" $1 60 > $L/s48h_PSIV60_s$1.log 2>&1
  echo "[s48h2] PSIV60 s$1 exit $? $(date -Is)" >> $LOG; }
tr1 1 &
until grep -qE "^S48H_(ADVANCE|STOP|FAIL)" $LOG; do sleep 120; done
tr1 2 & wait
[ -f $L/head_boundary_v8_PSIV60_s1.pt ] && [ -f $L/head_boundary_v8_PSIV60_s2.pt ] || { echo "S48H2_FAIL training $(date -Is)" >> $LOG; exit 1; }
SHOTS=$(grep -o -- '--vt-shots "[0-9,]*"' $L/s44h_jgn_end.sh | cut -d'"' -f2)
C='"shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"'; X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
QPI="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"; QHCS="head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
N="head_boundary_v8_J1RL_s1*24"; R_="head_boundary_v8_J1RLCS_s1*24"; VV="head_boundary_v8_V60_s1*6.75+head_boundary_v8_V60_s2*6.75"
for arm in f20Pa f20Pb; do
  if [ $arm = f20Pa ]; then H="route:$N+$QPI+$QHCS+$VV|$R_+$QPI+$QHCS+$VV"; else H="route:$N+$QHCS+$VV|$R_+$QHCS+$VV"; fi
  flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s48h$arm 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag S48H$arm --psi psiF \
    --psis "PQmixV~g" --guard psiF:0.25 --qb qh50 --qbheads "head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6" \
    --heads "$H" --xpheads head_xp_XP1CS_s1 --kws "{\"$arm\": {\"_steps\": [$C], $X}}" --axis head --qbclamp --vt-shots "$SHOTS" >> $LOG 2>&1 \
    || { echo "S48H2_FAIL vt $arm $(date -Is)" >> $LOG; exit 1; }
  .venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmix~g_f17V60" "PQmixV~g_$arm" >> $LOG 2>&1
done
line=$(grep "PQmixV~g_f20Pb vs PQmix~g_f17V60" $LOG | tail -1); lo=$(sed 's/.*\[\([-+0-9.]*\),.*/\1/' <<< "$line"); dl=$(grep -o "dlcfs [-+0-9.]*" <<< "$line" | cut -d' ' -f2)
if awk -v l="$lo" -v d="$dl" 'BEGIN{exit !(l > 0 && d <= 0)}'; then echo "S48H2_ADOPT F20K (f20Pb CI lower $lo, dlcfs $dl) $(date -Is)" >> $LOG
else echo "S48H2_REJECT (f20Pb CI lower ${lo:-?}, dlcfs ${dl:-?}; F18K stays) $(date -Is)" >> $LOG; fi
echo "S48H2_DONE $(date -Is)" >> $LOG
