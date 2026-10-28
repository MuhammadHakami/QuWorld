#!/bin/bash
# §48h stage 3 (registered with stage 2; amended: keys on the f20Pc decision): only after S48H2C_ADOPT - production PSIV60A s1-2 (all data, STRAT_ALL=1, 60 epochs, the PSIV60 recipe),
# then the F20 smoke rehearsal (organizers' smoke fold, runs/final_f16k.sh TAG=F20 PV=1) as for F17 / F18. REJECT -> nothing.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=$PWD/runs/strat/fusion; LOG=$L/s48h.log
until grep -qE "^S48H2C_(ADOPT|REJECT|FAIL)" $LOG; do sleep 300; done
grep -q "^S48H2C_ADOPT" $LOG || { echo "[s48h3] stage 2 did not adopt - no production training $(date -Is)" >> $LOG; exit 0; }
E="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_ALL=1 STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2"
J='{"target": "psi", "backbone": "vhope", "neck": "gate", "win": 8, "stride": 1, "bb_width": 128, "bb_depth": 4, "width": 2048, "depth": 4, "drop": 0.0, "q_layers": 4, "q_in": 32, "link": "independent", "lr": 0.000977105771213329, "alpha_init": 0.11830330369409621, "seq": true, "crop": 16, "shots": 256, "tokens": 12, "q_noise": 0.0, "in_noise": 0.0, "huber": 1.0, "mixer": 64, "reg": "tether3d"}'
for s in 1 2; do [ -f $L/head_boundary_v8_PSIV60A_s$s.pt ] || { env $E QW_GPU_GB=30 runs/capped.sh s48hA$s 40 .venv/bin/python paper/strat/fusion/quota_fusion.py \
  final PSIV60A "$J" $s 60 > $L/s48h_PSIV60A_s$s.log 2>&1; echo "[s48h3] PSIV60A s$s exit $? $(date -Is)" >> $LOG; } & sleep 60; done; wait
[ -f $L/head_boundary_v8_PSIV60A_s1.pt ] && [ -f $L/head_boundary_v8_PSIV60A_s2.pt ] || { echo "S48H3_FAIL training $(date -Is)" >> $LOG; exit 1; }
env QW_FOLD=diii_d_fold_smoke SUF=smoke TAG=F20 REGALL=1 XPALL=1 VH=1 PV=2 bash runs/final_f16k.sh \
  && grep -q "^REH_DONE F20 smoke" $L/final_f16k_smoke.log && echo "S48H3_READY F20K production weights + smoke rehearsal $(date -Is)" >> $LOG \
  || echo "S48H3_FAIL rehearsal (see final_f16k_smoke.log) $(date -Is)" >> $LOG
