#!/bin/bash
# §43g = the deferred §41u (registered 2026-10-11; recipe and rule unchanged): register heads J1QPIpiAH s1-2 (PI-KAN + simulated 30-atom
# Rydberg layer, J1QPIpi recipe) on ALL labelled shots with thz + hfull; replace the TRAIN-only J1QPIpi s1-2 (register share 1/7) if training
# is finite. Seed 2 starts once seed 1's memory is known (SEED2_PARALLEL=1 -> both now). Then F12 = F8 with them (alone) -> agreement vs F8.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s43g.log
PK='{"block": "mlp", "width": 4096, "depth": 4, "drop": 0.1, "lr": 0.0025322139193052723, "w_pi": 0.1, "q_layers": 4, "q_in": 128, "link": "cascade", "alpha_mult": 1.0, "alpha_init": 0.1}'
reg() {
  [ -f $L/head_boundary_v8_J1QPIpiAH_s$1.pt ] && return 0
  env STRAT_CUDA=0 STRAT_ALL=1 STRAT_THZ=1 STRAT_HFULL=1 QW_MAGFIX=X_magfix QW_GPU_GB=24 runs/capped.sh qpiah$1 40 .venv/bin/python paper/strat/fusion/hpo_qpikan.py \
    --fixed "$PK" --tag J1QPIpiAH --epochs 60 --seed $1 > $L/J1QPIpiAH_s$1.log 2>&1 && echo "QPIAH_s$1_TRAINED $(date -Is)" >> $LOG || echo "CRASH QPIAH s$1 $(date -Is)" >> $LOG
}
reg 1 &
until [ -f $L/qpiah_seed2_go ] || ! kill -0 $! 2>/dev/null; do sleep 30; done
reg 2 &
wait
grep -q "^CRASH" $LOG && exit 1
KW=$(grep -o '"_steps".*' $L/f8.log | head -1 | sed 's/ 20[0-9][0-9]-.*$//'); KW="{$KW"
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLA_s$s*3+"; done)"head_boundary_v8_J1QPIpiAH_s1*2+head_boundary_v8_J1QPIpiAH_s2*2"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2_s$s+"; done | sed 's/+$//'); XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"
echo "[f12] heads $H $(date -Is)" >> $LOG
export CUDA_VISIBLE_DEVICES=0
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f12 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F12 \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head --qbclamp >> $LOG 2>&1 || { echo "F12_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F12/diii_d_public_test.npz runs/sophelio_sota/submission_F8/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F12_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
echo "F12_BUILT $(date -Is)" >> $LOG
