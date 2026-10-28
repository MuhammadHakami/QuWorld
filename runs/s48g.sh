#!/bin/bash
# §48g (doc 35, registered 2026-10-19 before any result): boundary extension around the §48c winner t20 (one factor per arm, rest = t20).
#   E1 crop 8 x 512 shots (starts at once, third GPU job next to the W60 trainings); E2 tokens 16 and E3 bb_depth 6 start after §48d ends.
#   Stage A (search folds 0-2): advance iff mean gain > t20 + 0.0001 AND beats t20 on >= 2/3 folds; dropped after fold 0 if < t20 f0 - 0.0002.
#   Stage B (held-out folds 3-5): beats t20 on >= 2/3 folds with a higher mean; two passing arms -> their combination is read as well.
#   Stage C (X60 training + paired VT read) is launched separately after S48G_DONE names a winner.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH
L=$PWD/runs/strat/fusion; LOG=$L/s48g.log; P=paper/strat/fusion/hpo_vq.py
E="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QW_GPU_GB=24"
echo "[s48g] start $(date -Is)" >> $LOG
T20F0=$(.venv/bin/python -c "
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
st = optuna.load_study(study_name='s48_vq', storage='sqlite:///$L/quota/hpo_vq.db'); print(st.trials[20].user_attrs['f0']['gain'])")
[ -n "$T20F0" ] || { echo "S48G_FAIL t20 reference not readable $(date -Is)" >> $LOG; exit 1; }

arm() {                                                                       # arm <name> '<json overrides>' <folds> [prune]
  local nm=$1 js=$2 folds=$3 prune=${4:-0} k g
  for k in ${folds//,/ }; do
    env $E runs/capped.sh s48g$nm$k 36 .venv/bin/python $P probe $nm "$js" $k >> $L/s48g_$nm.log 2>&1
    g=$(grep "^HPO_PROBE $nm f$k " $L/s48g_$nm.log | tail -1 | awk '{print $5}')
    echo "[s48g] $nm f$k gain ${g:-NONE} $(date -Is)" >> $LOG
    [ -n "$g" ] || { echo "[s48g] $nm f$k FAILED (see s48g_$nm.log)" >> $LOG; return 1; }
    if [ "$prune" = 1 ] && [ "$k" = 0 ] && awk -v g="$g" -v t="$T20F0" 'BEGIN{exit !(g < t - 0.0002)}'; then
      echo "[s48g] $nm dropped after fold 0 ($g < t20 $T20F0 - 0.0002)" >> $LOG; return 0; fi
  done
}

arm E1 '{"crop_shots": "8x512"}' 0,1,2 1 &
until grep -qE "^S48D_(DONE|STOP|FAIL)" $L/s48d.log; do sleep 300; done
arm E2 '{"tokens": 16}' 0,1,2 1 & sleep 60
arm E3 '{"bb_depth": 6}' 0,1,2 1 &
wait

stage() {                                                                     # stage A|B -> prints S48G_<stage> lines, returns the passing arms
  .venv/bin/python - "$1" "$2" <<'PY'
import json, sys
import numpy as np
sys.path.insert(0, "paper/strat/fusion")
import optuna
import hpo_vq as H
optuna.logging.set_verbosity(optuna.logging.WARNING)
stage, arms = sys.argv[1], [a for a in sys.argv[2].split(",") if a]
folds = (0, 1, 2) if stage == "A" else (3, 4, 5)
if stage == "A":
    t = optuna.load_study(study_name=H.STUDY, storage=H.DB).trials[20]; ref = [t.user_attrs[f"f{k}"]["gain"] for k in folds]; margin = 1e-4
else:
    ref = [r[0] for r in json.load(open(H.Q / "hpo_vq.json"))["hold"]["best"]]; margin = 0.0
ok = []
for nm in arms:
    g = [H.gain(f"H8_{nm}", k)[0] for k in folds if (H.Q / f"H8_{nm}_f{k}.npz").exists()]
    win = len(g) == 3 and np.mean(g) > np.mean(ref) + margin and sum(a > b for a, b in zip(g, ref)) >= 2
    print(f"S48G_{stage} {nm} " + " ".join(f"{x:+.5f}" for x in g) + f" mean {np.mean(g) if g else float('nan'):+.5f} vs t20 "
          + " ".join(f"{x:+.5f}" for x in ref) + f" mean {np.mean(ref):+.5f} -> {'PASS' if win else 'fail'}", file=sys.stderr)
    if win:
        ok.append((float(np.mean(g)), nm))
print(",".join(nm for _, nm in sorted(ok, reverse=True)))
PY
}
A=$(stage A E1,E2,E3 2>>$LOG); echo "[s48g] stage A passing: ${A:-none} $(date -Is)" >> $LOG
[ -n "$A" ] || { echo "S48G_DONE winner none (t20 / W60 stays) $(date -Is)" >> $LOG; exit 0; }
declare -A OV=([E1]='"crop_shots": "8x512"' [E2]='"tokens": 16' [E3]='"bb_depth": 6')
for nm in ${A//,/ }; do arm $nm "{${OV[$nm]}}" 3,4,5 & sleep 60; done; wait
B=$(stage B $A 2>>$LOG)
if [ "$(tr ',' '\n' <<< "$B" | grep -c .)" -ge 2 ]; then                      # two or more pass: their combination is read too
  set -- ${B//,/ }; CJ="{${OV[$1]}, ${OV[$2]}}"; echo "[s48g] combination EC = $1 + $2: $CJ" >> $LOG
  arm EC "$CJ" 3,4,5; C=$(stage B EC 2>>$LOG); [ -n "$C" ] && B="EC,$B"
fi
W=$(stage B ${B:-none} 2>/dev/null | cut -d, -f1)
echo "S48G_DONE winner ${W:-none} $(date -Is)" >> $LOG
