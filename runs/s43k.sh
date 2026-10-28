#!/bin/bash
# §43k (registered before any result): seed-CV confirmation of the §43j best trial (10) at full length (60 epochs, all TRAIN), same
# inputs (J1RL physics) and the same register path (QW_QFAST=1, no compile) for both arms, so the comparison is like for like:
#   J1QH s1-3 = trial 10 (mlp 2048 x 4, drop 0.1, register 4 layers, q_in 64, independent, lr 1.25e-3, no physics loss, alpha 0.1)
#   J1QRL s2-3 = the deployed register recipe with the same inputs (J1QRL_s1 exists: VT 0.985861)
# VT read-out: singles (seed means / spread) + ensembles A = J1RL s1*3 + s2*3 + J1QPIpi_s1 vs A' with J1QH_s1 in its place.
# Rule: adopt (all-data J1QHA heads -> next build) iff min(J1QH) > mean(J1QRL s1-3) AND the A' - A ensemble CI > 0.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s43k.log
QH='{"block": "mlp", "width": 2048, "depth": 4, "drop": 0.1, "q_layers": 4, "q_in": 64, "link": "independent", "lr": 0.0012505271791067153, "alpha_mult": 1.0, "alpha_init": 0.1, "w_pi": 0.0}'
PK='{"block": "mlp", "width": 4096, "depth": 4, "drop": 0.1, "lr": 0.0025322139193052723, "w_pi": 0.1, "q_layers": 4, "q_in": 128, "link": "cascade", "alpha_mult": 1.0, "alpha_init": 0.1}'
RL="STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2"
reg() {   # $1 tag $2 seed $3 params
  local tag=$1 s=$2 p=$3
  [ -f $L/head_boundary_v8_${tag}_s$s.pt ] && return 0
  env STRAT_CUDA=0 QW_QFAST=1 QW_MAGFIX=X_magfix QW_GPU_GB=20 $RL runs/capped.sh q$tag$s 40 .venv/bin/python paper/strat/fusion/hpo_qpikan.py \
    --fixed "$p" --tag $tag --epochs 60 --seed $s > $L/${tag}_s$s.log 2>&1 && echo "${tag}_s${s}_TRAINED $(date -Is)" >> $LOG || echo "CRASH $tag s$s $(date -Is)" >> $LOG
}
echo "[s43k] start $(date -Is)" >> $LOG
reg J1QH 1 "$QH" & reg J1QH 2 "$QH" & reg J1QH 3 "$QH" & wait
reg J1QRL 2 "$PK" & reg J1QRL 3 "$PK" & wait
grep -q "^CRASH" $LOG && exit 1
H=head_boundary_v8; E="$H"_J1RL_s1*3+"$H"_J1RL_s2*3
flock runs/strat/vt.lock .venv/bin/python paper/strat/fusion/vt_eval.py shape ${H}_J1QRL_s1 ${H}_J1QRL_s2 ${H}_J1QRL_s3 ${H}_J1QH_s1 ${H}_J1QH_s2 ${H}_J1QH_s3 >> $LOG 2>&1 \
  || { echo "CRASH vt singles $(date -Is)" >> $LOG; exit 1; }
flock runs/strat/vt.lock .venv/bin/python paper/strat/fusion/vt_eval.py shape "$E+${H}_J1QPIpi_s1*1" "$E+${H}_J1QH_s1*1" "$E+${H}_J1QH_s1*1+${H}_J1QH_s2*1+${H}_J1QH_s3*1" >> $LOG 2>&1 \
  || { echo "CRASH vt ensembles $(date -Is)" >> $LOG; exit 1; }
echo "S43K_DONE $(date -Is)" >> $LOG
