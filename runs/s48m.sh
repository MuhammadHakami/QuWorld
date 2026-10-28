#!/bin/bash
# §48m (doc 35, registered before any result): third temporal psi seed - PSIV60 s3 (TRAIN only), VT f20Pd (PQmixV32x3) vs f20Pc on A150 (CI lower > 0)
# and B150 (point > 0).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=$PWD/runs/strat/fusion; LOG=$L/s48m.log
echo "[s48m] start $(date -Is)" >> $LOG
J='{"target": "psi", "backbone": "vhope", "neck": "gate", "win": 8, "stride": 1, "bb_width": 128, "bb_depth": 4, "width": 2048, "depth": 4, "drop": 0.0, "q_layers": 4, "q_in": 32, "link": "independent", "lr": 0.000977105771213329, "alpha_init": 0.11830330369409621, "seq": true, "crop": 16, "shots": 256, "tokens": 12, "q_noise": 0.0, "in_noise": 0.0, "huber": 1.0, "mixer": 64, "reg": "tether3d"}'
[ -f $L/head_boundary_v8_PSIV60_s3.pt ] || env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QW_GPU_GB=30 \
  runs/capped.sh s48mP3 40 .venv/bin/python paper/strat/fusion/quota_fusion.py final PSIV60 "$J" 3 60 > $L/s48m_PSIV60_s3.log 2>&1
[ -f $L/head_boundary_v8_PSIV60_s3.pt ] || { echo "S48M_FAIL training $(date -Is)" >> $LOG; exit 1; }
C='"shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"'; X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
QHCS="head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
N="head_boundary_v8_J1RL_s1*24"; R_="head_boundary_v8_J1RLCS_s1*24"; VV="head_boundary_v8_V60_s1*6.75+head_boundary_v8_V60_s2*6.75"
H="route:$N+$QHCS+$VV|$R_+$QHCS+$VV"; QB="head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6"
vt() { flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s48m 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag $1 --psi psiF \
  --psis "$2" --guard psiF:0.25 --qb qh50 --qbheads "$QB" --heads "$H" --xpheads head_xp_XP1CS_s1 --kws "{\"$3\": {\"_steps\": [$C], $X}}" --axis head --qbclamp --vt-shots "$4" >> $LOG 2>&1; }
A=$(grep -o -- '--vt-shots "[0-9,]*"' $L/s44h_jgn_end.sh | cut -d'"' -f2); B=$(cat $L/vt_shots_B150.txt)
vt S48MA "PQmixV32x3~g" f20Pd "$A" || { echo "S48M_FAIL vt A $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmixV32~g_f20Pc" "PQmixV32x3~g_f20Pd" >> $LOG 2>&1
lo=$(grep "PQmixV32x3~g_f20Pd vs PQmixV32~g_f20Pc" $LOG | tail -1 | sed 's/.*\[\([-+0-9.]*\),.*/\1/')
vt S48MB "PQmixV32~g,PQmixV32x3~g" f20B "$B" || { echo "S48M_FAIL vt B $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmixV32~g_f20B" "PQmixV32x3~g_f20B" >> $LOG 2>&1
dB=$(grep "PQmixV32x3~g_f20B vs PQmixV32~g_f20B" $LOG | tail -1 | grep -o "delta [-+0-9.]*" | cut -d' ' -f2)
if awk -v l="$lo" -v d="$dB" 'BEGIN{exit !(l > 0 && d > 0)}'; then echo "S48M_ADOPT F21K (A150 CI lower $lo, B150 delta $dB) $(date -Is)" >> $LOG
else echo "S48M_REJECT (A150 CI lower ${lo:-?}, B150 delta ${dB:-?}; F20K stays) $(date -Is)" >> $LOG; fi
