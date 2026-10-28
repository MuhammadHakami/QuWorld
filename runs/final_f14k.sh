#!/bin/bash
# Final-phase build of the F14K configuration for a test fold (default: rehearsal on the organizers' smoke fold diii_d_fold_smoke, 5 shots;
# Oct 19: QW_FOLD=diii_d_private_test SUF=priv runs/final_f14k.sh): every inputs-only
# feature block (final_fold.sh), the li-tree inputs + SAVED all-data tree (gbdt_shape dumppub / litreepred -> litree_private.npz),
# the F14 build with exactly f14.sh's arguments, F14K = post_kw2, and an output check (frames per shot = efit_times, finite, shapes).
# Files land as *_private (fold.py convention); the real private fold overwrites them on Oct 19.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 QW_FOLD=${QW_FOLD:-diii_d_fold_smoke}; SUF=${SUF:-smoke}; L=runs/strat/fusion; LOG=$L/final_f14k_$SUF.log
echo "[reh] start $(date -Is)" >> $LOG
bash runs/final_fold.sh >> $LOG 2>&1 || { echo "REH_FAIL features $(date -Is)" >> $LOG; exit 1; }


export QW_LITREE=/home/mohhakami/Documents/paper/quworld/runs/strat/fusion/$([ "$QW_FOLD" = diii_d_public_test ] && echo litree.npz || echo litree_private.npz):0.2
KW=$(grep -o '"_steps".*' $L/f8.log | head -1 | sed 's/ 20[0-9][0-9]-.*$//'); KW="{$KW"; QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLA_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2+head_boundary_v8_J1QH_s1*4+head_boundary_v8_J1QH_s2*4+head_boundary_v8_J1QH_s3*4"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2_s$s+"; done | sed 's/+$//'); XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh rehf14 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F14$SUF \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head --qbclamp >> $LOG 2>&1 || { echo "REH_FAIL build $(date -Is)" >> $LOG; exit 1; }
QW_GPU_GB=12 runs/capped.sh rehkw2 30 .venv/bin/python paper/strat/fusion/post_kw2.py --src F14$SUF --tag F14K$SUF --heads "$H" >> $LOG 2>&1 || { echo "REH_FAIL kw2 $(date -Is)" >> $LOG; exit 1; }
SUF=$SUF QW_FOLD=$QW_FOLD .venv/bin/python - >> $LOG 2>&1 <<'PY' || { echo "REH_FAIL check $(date -Is)" >> $LOG; exit 1; }
import glob, os, numpy as np, pyarrow.parquet as pq
suf, fold = os.environ["SUF"], os.environ["QW_FOLD"]; tag = "public" if fold == "diii_d_public_test" else "private"
z = np.load(f"runs/sophelio_sota/submission_F14K{suf}/diii_d_{tag}_test.npz"); raw = sorted(glob.glob(f"data/sophelio/fusion-equilibrium-challenge/data/{fold}/*.parquet"))
assert len(z.files) == 3 * len(raw), (len(z.files), len(raw)); bad = 0
for i, r in enumerate(raw):
    T = len(pq.read_table(r, columns=["efit_times"]).column(0)[0].as_py()); p, q, b = (z[f"shot_{i:04d}_{k}"] for k in ("psirz", "q95", "betaN"))
    ok = p.shape == (T, 65, 65) and q.shape == (T,) and b.shape == (T,) and np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(b).all(); bad += not ok
print(f"[reh] {len(raw)} shots checked, {bad} bad (frames = efit_times, 65x65, finite)"); assert bad == 0
PY
echo "REH_DONE $SUF $(date -Is)" >> $LOG
