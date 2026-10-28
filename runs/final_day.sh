#!/bin/bash
# Fusion Final phase (Oct 19-26) runbook for the Final configuration (CAND=F18K default = F17K + VisionHOPE + register heads V60 s1-2; CAND=F17K) (doc 35): download the private fold, build F17K on it with the rehearsed runner,
# check it, and upload ONLY when CODE_URL (the public reproduction-package reference the Final phase requires) is given.
#   runs/final_day.sh                      # download + build + check (no upload)
#   CODE_URL='<forge url>@<40-char sha>' runs/final_day.sh upload     # + Final upload (PHASE 29581) and score wait
cd "$(dirname "$0")/.."; L=runs/strat/fusion; LOG=$L/final_day.log; FOLD=${FOLD:-diii_d_private_test}; SUF=${SUF:-priv}
CAND=${CAND:-F18K}; case "$CAND" in F21K) T=F21; VA="VH=1 PV=3";; F20K) T=F20; VA="VH=1 PV=2";; F19K) T=F19; VA="VH=1 WH=1";; F18K) T=F18; VA="VH=1";; F17K) T=F17; VA="VH=0";; *) echo "CAND must be F21K, F20K, F19K, F18K or F17K"; exit 2;; esac
D=data/sophelio/fusion-equilibrium-challenge; echo "[final] start $(date -Is) fold $FOLD cand $CAND" >> $LOG
if ! ls $D/data/$FOLD/*.parquet > /dev/null 2>&1; then
  .venv/bin/hf download Sophelio/fusion-equilibrium-challenge --repo-type dataset --include "data/$FOLD/*" --local-dir $D >> $LOG 2>&1 \
    || { echo "FINAL_FAIL download $(date -Is)" >> $LOG; exit 1; }
fi
n=$(ls $D/data/$FOLD/*.parquet 2>/dev/null | wc -l); echo "[final] $n shot files in $D/data/$FOLD" >> $LOG; [ "$n" -gt 0 ] || { echo "FINAL_FAIL no fold files" >> $LOG; exit 1; }
if ! grep -q "^REH_DONE $T $SUF" $L/final_f16k_$SUF.log 2>/dev/null; then
  env QW_FOLD=$FOLD SUF=$SUF TAG=$T REGALL=1 XPALL=1 $VA bash runs/final_f16k.sh
  grep -q "^REH_DONE $T $SUF" $L/final_f16k_$SUF.log || { echo "FINAL_FAIL build (see $L/final_f16k_$SUF.log) $(date -Is)" >> $LOG; exit 1; }
fi
S=runs/sophelio_sota/submission_${T}K$SUF; Z=$S/submission.zip
[ -f $Z ] || .venv/bin/python -c "import zipfile; z = zipfile.ZipFile('$Z', 'w'); z.write('$S/diii_d_private_test.npz', 'diii_d_private_test.npz'); z.close()"
.venv/bin/python -c "import zipfile; n = zipfile.ZipFile('$Z').namelist(); assert n == ['diii_d_private_test.npz'], n" \
  || { echo "FINAL_FAIL zip layout $(date -Is)" >> $LOG; exit 1; }
echo "FINAL_BUILT $Z $(stat -c %s $Z) bytes sha256 $(sha256sum $Z | cut -c1-16) $(date -Is)" >> $LOG
[ "$1" = upload ] || exit 0
[ -n "$CODE_URL" ] || { echo "FINAL_HOLD no CODE_URL (the Final phase requires the public package reference) $(date -Is)" >> $LOG; exit 0; }
M="${CAND}: hybrid MLP + 30-atom Rydberg-register heads, Ip-sign canonical twins, psi edits toward the heads"
[ "$CAND" = F21K ] && M="F21K: VisionHOPE temporal psi trunks (95/5 with the MLP psi trunks) + VisionHOPE / input-dependent Rydberg-register shape heads, Ip-sign canonical twins, psi edits toward the heads"
[ "$CAND" = F20K ] && M="F20K: VisionHOPE temporal psi trunks (50/50 with the MLP psi trunks) + VisionHOPE / input-dependent Rydberg-register shape heads, Ip-sign canonical twins, psi edits toward the heads"
[ "$CAND" = F19K ] && M="F19K: VisionHOPE + input-dependent 30-atom Rydberg-register heads (V60 + searched W60) with MLP / register heads, Ip-sign canonical twins, psi edits toward the heads"
[ "$CAND" = F18K ] && M="F18K: VisionHOPE + input-dependent 30-atom Rydberg-register heads with MLP / register heads, Ip-sign canonical twins, psi edits toward the heads"
PHASE=29581 CODE_URL="$CODE_URL" runs/codabench_upload.sh $Z ${T}Kpriv "$M" >> $LOG 2>&1 \
  || { echo "FINAL_FAIL upload $(date -Is)" >> $LOG; exit 1; }
id=$(grep -o '"id":[0-9]*' runs/sophelio_sota/upload_${T}Kpriv.log | head -1 | cut -d: -f2); echo "[final] ${T}Kpriv id $id $(date -Is)" >> $LOG
for i in $(seq 1 180); do
  s=$(runs/codabench_status.sh $id 2>/dev/null); case "$s" in Finished*|Failed*) echo "[final] status $id $s $(date -Is)" >> $LOG; break;; esac; sleep 60
done
echo "FINAL_DAY_DONE $(date -Is)" >> $LOG
