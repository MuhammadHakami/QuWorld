#!/bin/bash
# Fusion Final, unattended: from 2026-10-19 00:00 UTC wait for the DIII-D private fold (runs/dl_final_fold.sh), build the candidate (runs/final_day.sh)
# and upload it ONCE with the public package link - only when the release carries every weight listed in the package's WEIGHTS.sha256, the
# package commit is on GitHub AND the package's default candidate (repro/fusion/infer.sh) is the one built (the package must be complete, public
# and describe the submitted model).
#   candidate: F21K once §48p is adopted and rehearsed (S48P_READY in s48p.log; release f20k-weights - the same weights), else F20K once §48h is adopted and its production weights + smoke rehearsal are done (S48H3_READY in s48h.log; release f20k-weights),
#   else F19K if §48d adopted it (S48D_ADOPT; release f19k-weights), else F18K (release f18k-weights); CAND / TAGW override.
#   the build runs alone (memguard): it waits until §48d has ended and no training job (hpo_vq.py, quota_fusion.py, head_boundary.py,
#   hpo_qpikan.py) is running; backstop 2026-10-21 00:00Z.
#   setsid nohup runs/final_auto.sh &        -> runs/strat/fusion/final_auto.log
cd "$(dirname "$0")/.."; LOG=runs/strat/fusion/final_auto.log; REPO=MuhammadHakami/fusion-equilibrium; PKG=$HOME/Documents/fusion-equilibrium
L=runs/strat/fusion; CAND0=$CAND; TAGW0=$TAGW
echo "[auto] start $(date -Is) cand ${CAND0:-auto} release ${TAGW0:-auto}" >> $LOG
until [ "$(date -u +%s)" -ge "$(date -u -d 2026-10-19T00:00:00Z +%s)" ]; do sleep 1800; done
while true; do
  C=F18K; grep -q "^S48D_ADOPT" $L/s48d.log 2>/dev/null && C=F19K
  grep -q "^S48H3_READY" $L/s48h.log 2>/dev/null && C=F20K                 # §48h: adopted AND production PSIV60A + smoke rehearsal done
  grep -q "^S48P_READY" $L/s48p.log 2>/dev/null && C=F21K                  # §48p: F20K at temporal psi share 0.95 (same weights) + rehearsal + infer.sh check
  export CAND=${CAND0:-$C}; T=${CAND%K}
  TAGW=${TAGW0:-$(tr 'A-Z' 'a-z' <<< "$CAND")-weights}; [ -z "$TAGW0" ] && [ "$CAND" = F21K ] && TAGW=f20k-weights   # F21K ships the F20K weights
  grep -q "^\[final\] ${T}Kpriv id" $L/final_day.log 2>/dev/null && { echo "[auto] $CAND already uploaded $(date -Is)" >> $LOG; exit 0; }
  SHA=$(git -C $PKG rev-parse HEAD); CODE_URL=https://github.com/$REPO@$SHA; NW=$(wc -l < $PKG/repro/fusion/WEIGHTS.sha256)
  PC=$(grep -o 'CAND=${CAND:-F[0-9][0-9]K}' $PKG/repro/fusion/infer.sh | grep -o 'F[0-9][0-9]K')
  ONGH=$(curl -s -o /dev/null -w '%{http_code}' https://api.github.com/repos/$REPO/commits/$SHA)
  # the Final inputs live in their own dataset (ganatma/fusion-equilibrium-challenge-final, starter README "Final phase"), fetched by
  # runs/dl_final_fold.sh and linked as data/$FD; the fold counts once the download finished and every file is present
  FD=sophelio/fusion-equilibrium-challenge/data/diii_d_private_test; FOLD=''
  grep -q "^\[dl\] exit 0 .* files 879$" $L/dl_final_fold.log 2>/dev/null && [ "$(ls data/$FD/*.parquet 2>/dev/null | wc -l)" = 879 ] && FOLD=diii_d_private_test
  NA=$(curl -s https://api.github.com/repos/$REPO/releases/tags/$TAGW | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('assets', [])))" 2>/dev/null)
  echo "[auto] $(date -Is) cand $CAND fold '${FOLD}' release $TAGW assets ${NA:-0} / $NW, package $SHA (default $PC) on GitHub: $ONGH" >> $LOG
  if grep -q "^GITHUB_TOKEN=." .env && { [ "${NA:-0}" != "$NW" ] || [ "$ONGH" != 200 ]; }; then        # token added by the user: push + release ourselves
    TAGW=$TAGW bash runs/publish_release.sh; echo "[auto] publish exit $? $(date -Is) ($(tail -1 $L/publish.log))" >> $LOG; sleep 120; continue
  fi
  CB=$(curl -s -m 30 -o /dev/null -w '%{http_code}' https://www.codabench.org/)
  BUSY=1; grep -qE "^S48D_(DONE|STOP|FAIL)" $L/s48d.log 2>/dev/null && BUSY=0
  ps -eo cmd | grep -E "hpo_vq.py|quota_fusion.py|head_boundary.py|hpo_qpikan.py" | grep -vq grep && BUSY=1
  [ "$(date -u +%s)" -ge "$(date -u -d 2026-10-21T00:00:00Z +%s)" ] && BUSY=0
  [ "$BUSY" = 1 ] && echo "[auto] $(date -Is) GPU busy (§48d not ended or a training job running) - build deferred" >> $LOG
  if [ "$BUSY" = 0 ] && [ -n "$FOLD" ] && [ "${FOLD#AMBIGUOUS}" = "$FOLD" ]; then
    if [ "${NA:-0}" = "$NW" ] && [ "$ONGH" = 200 ] && [ "$PC" = "$CAND" ] && [ "$CB" != 200 ]; then
      echo "[auto] $(date -Is) ready to upload but Codabench is unreachable (http $CB) - retry" >> $LOG
    elif [ "${NA:-0}" = "$NW" ] && [ "$ONGH" = 200 ] && [ "$PC" = "$CAND" ]; then
      set -a; . ./.env; set +a; for sid in 991617 994544; do                  # superseded F14K / F15K: free the 15 GB storage quota
        curl -s -o /dev/null -w "[auto] delete $sid http %{http_code}\n" -X DELETE -H "Authorization: Token $CODABENCH_TOKEN" \
          "https://www.codabench.org/api/submissions/$sid/" >> $LOG; done
      FOLD=$FOLD CODE_URL=$CODE_URL bash runs/final_day.sh upload; echo "[auto] final_day exit $? $(date -Is)" >> $LOG
    else
      FOLD=$FOLD bash runs/final_day.sh; echo "[auto] $CAND built, waiting for the release / package before uploading $(date -Is)" >> $LOG
    fi
  fi
  sleep 600
done
