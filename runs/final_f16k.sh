#!/bin/bash
# Final-phase build of the F16K configuration (§45o: the base K build with Ip-sign routing - reversed-Ip shots read the all-data canonical
# twins J1RLHACS s1-4 (shape, register part unchanged) and QHAR2HuCS s1-2 (q / b); BASE=F15 (J1RLHA / QHAR2Hu, default) or BASE=F14 (J1RLA / QHAR2)) for a test fold (default: rehearsal on the organizers' smoke fold diii_d_fold_smoke, 5 shots;
# Oct 19: QW_FOLD=diii_d_private_test SUF=priv runs/final_f16k.sh; F17: TAG=F17 REGCS=1 [XPCS=1] or REGALL=1 XPALL=1 (§45w) - §45s canonical register / X-point twins on reversed-Ip shots): every inputs-only
# feature block (final_fold.sh), the li-tree inputs + SAVED all-data tree (gbdt_shape dumppub / litreepred -> litree_private.npz),
# the F16 build with exactly f16_router.sh's arguments, F16K = post_kw2, and an output check (frames per shot = efit_times, finite, shapes).
# Files land as *_private (fold.py convention); the real private fold overwrites them on Oct 19.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 QW_FOLD=${QW_FOLD:-diii_d_fold_smoke}; SUF=${SUF:-smoke}; L=runs/strat/fusion; LOG=$L/final_f16k_$SUF.log
echo "[reh] start $(date -Is)" >> $LOG
bash runs/final_fold.sh >> $LOG 2>&1 || { echo "REH_FAIL features $(date -Is)" >> $LOG; exit 1; }


export QW_LITREE=/home/mohhakami/Documents/paper/quworld/runs/strat/fusion/$([ "$QW_FOLD" = diii_d_public_test ] && echo litree.npz || echo litree_private.npz):0.2
KW=$(grep -o '"_steps".*' $L/f8.log | head -1 | sed 's/ 20[0-9][0-9]-.*$//'); KW="{$KW"; QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
BASE=${BASE:-F15}; [ "$BASE" = F15 ] && { RH=J1RLHA; RQ=QHAR2Hu; } || { RH=J1RLA; RQ=QHAR2; }
REG="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2+head_boundary_v8_J1QH_s1*4+head_boundary_v8_J1QH_s2*4+head_boundary_v8_J1QH_s3*4"
{ [ "${WH:-0}" = 1 ] || [ "${PV:-0}" != 0 ]; } && REG=${REG#"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2+"}   # §48d F19K / §48h F20K: the PI-KAN register heads leave (budget)
PSIS=psiMixA; [ "${PV:-0}" = 1 ] && PSIS=psiMixAV; [ "${PV:-0}" = 2 ] && PSIS=psiMixAV32; [ "${PV:-0}" = 3 ] && PSIS=psiMixAV32w95   # §48h: + temporal psi heads PSIV60A s1-2 (PV=2 = F20K: comps 0-31, share 0.5; PV=3 = F21K: share 0.95, §48p)
TAG=${TAG:-F16}; REGR=$REG; [ "${REGCS:-0}" = 1 ] && REGR=${REG//J1QH_/J1QHCS_}
[ "${REGALL:-0}" = 1 ] && { REG=${REG//J1QH_/J1QHCS_}; REGR=$REG; }   # §45w (1)
H="route:$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_${RH}_s$s*3+"; done)$REG|$(for s in 1 2 3 4; do printf "head_boundary_v8_J1RLHACS_s$s*6+"; done)$REGR"
Q="route:$(for s in 1 2 3; do printf "head_qb_${RQ}_s$s+"; done | sed 's/+$//')|head_qb_QHAR2HuCS_s1+head_qb_QHAR2HuCS_s2"; XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"; [ "${XPCS:-0}" = 1 ] && XH="route:$XH|${XH//XPA_/XPACS_}"
[ "${XPALL:-0}" = 1 ] && XH=${XH//XPA_/XPACS_}   # §45w (2)
if [ "${VH:-0}" = 1 ]; then                       # §47c: + the VisionHOPE hybrid heads V s1-3 on every shot (share 0.252 of the shape weight)
  VS=${VSTEM:-V60}; VN=${VSEEDS:-1 2}; VW=$(awk -v k="$(wc -w <<< "$VN")" 'BEGIN{printf "%g", 13.5 / k}')   # F18K: V60 s1-2 at 6.75 (share 0.25)
  VV=$(for s in $VN; do printf "head_boundary_v8_${VS}_s$s*$VW+"; done | sed 's/+$//')
  [ "${WH:-0}" = 1 ] && VV="$VV+head_boundary_v8_W60_s1*6.75+head_boundary_v8_W60_s2*6.75"   # §48d F19K: + the §48c winner t20 trained 60 epochs
  H="${H%%|*}+$VV|${H#*|}+$VV"; fi
echo "[reh] $TAG base $BASE heads $H qb $Q xp $XH" >> $LOG
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh rehf16 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag $TAG$SUF \
  --psi $PSIS --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head --qbclamp >> $LOG 2>&1 || { echo "REH_FAIL build $(date -Is)" >> $LOG; exit 1; }
QW_GPU_GB=12 runs/capped.sh rehkw2f16 30 .venv/bin/python paper/strat/fusion/post_kw2.py --src $TAG$SUF --tag ${TAG}K$SUF --heads "$H" >> $LOG 2>&1 || { echo "REH_FAIL kw2 $(date -Is)" >> $LOG; exit 1; }
TAG=$TAG SUF=$SUF QW_FOLD=$QW_FOLD .venv/bin/python - >> $LOG 2>&1 <<'PY' || { echo "REH_FAIL check $(date -Is)" >> $LOG; exit 1; }
import glob, os, numpy as np, pyarrow.parquet as pq
suf, fold, tg = os.environ["SUF"], os.environ["QW_FOLD"], os.environ["TAG"]; tag = "public" if fold == "diii_d_public_test" else "private"
z = np.load(f"runs/sophelio_sota/submission_{tg}K{suf}/diii_d_{tag}_test.npz"); raw = sorted(glob.glob(f"data/sophelio/fusion-equilibrium-challenge/data/{fold}/*.parquet"))
assert len(z.files) == 3 * len(raw), (len(z.files), len(raw)); bad = 0
for i, r in enumerate(raw):
    T = len(pq.read_table(r, columns=["efit_times"]).column(0)[0].as_py()); p, q, b = (z[f"shot_{i:04d}_{k}"] for k in ("psirz", "q95", "betaN"))
    ok = p.shape == (T, 65, 65) and q.shape == (T,) and b.shape == (T,) and np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(b).all(); bad += not ok
print(f"[reh] {len(raw)} shots checked, {bad} bad (frames = efit_times, 65x65, finite)"); assert bad == 0
PY
echo "REH_DONE $TAG $SUF $(date -Is)" >> $LOG
