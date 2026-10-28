#!/bin/bash
# F18K = F17K's exact public build (runs/f17.sh TAG=F18 REGALL=1 XPALL=1) + the VisionHOPE hybrid heads VSTEM s1-3 (VH=1), then the
# Development upload ONLY with the argument "upload" (decided after the registered VT read and the expected-score statement).
cd /home/mohhakami/Documents/paper/quworld; L=runs/strat/fusion; LOG=$L/f18.log; export VSTEM=${VSTEM:-V60} VSEEDS=${VSEEDS:-1 2}   # §47f amended: seeds 1-2 (6.75 each)
S=runs/sophelio_sota/submission_F18K; Z=$S/submission.zip
if [ ! -f $S/diii_d_public_test.npz ]; then
  TAG=F18 REGALL=1 XPALL=1 VH=1 bash runs/f17.sh; grep -q '^F17K_DONE' <(tail -1 $L/f17.log) || { echo "F18_FAIL build $(date -Is)" >> $LOG; exit 1; }
fi
[ -f $Z ] || .venv/bin/python -c "import zipfile; z = zipfile.ZipFile('$Z', 'w'); z.write('$S/diii_d_public_test.npz', 'diii_d_public_test.npz'); z.close()"
.venv/bin/python -c "import zipfile; n = zipfile.ZipFile('$Z').namelist(); assert n == ['diii_d_public_test.npz'], n" || { echo "F18_FAIL zip $(date -Is)" >> $LOG; exit 1; }
echo "[f18] built $Z VSTEM=$VSTEM VSEEDS=$VSEEDS $(date -Is)" >> $LOG
[ "${1:-}" = upload ] || { echo "F18_BUILT (no upload) $(date -Is)" >> $LOG; exit 0; }
runs/codabench_upload.sh $Z F18K "F18K: F17K + VisionHOPE hybrid heads (VisionHOPE block on the shot map + gate + 30-atom register)" >> $LOG 2>&1 \
  || { echo "F18_FAIL upload $(date -Is)" >> $LOG; exit 1; }
id=$(grep -o '"id":[0-9]*' runs/sophelio_sota/upload_F18K.log | head -1 | cut -d: -f2); echo "[upload] F18K id $id $(date -Is)" >> $LOG
for i in $(seq 1 150); do
  s=$(runs/codabench_status.sh $id 2>/dev/null); case "$s" in Finished*|Failed*) echo "[score] F18K $id $s $(date -Is)" >> $LOG; break;; esac; sleep 60
done
echo "F18_DONE $(date -Is)" >> $LOG
