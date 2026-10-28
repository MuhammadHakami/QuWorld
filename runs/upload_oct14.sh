#!/bin/bash
# Oct 14 UTC-day uploads (registered 2026-10-13 21:00 local, expected scores stated in doc 35 before upload): at 03:01 local (00:01 UTC)
# F7KC (kw2 probe: F7C + kw2 + revised q95 bound; expected 0.99579-0.99583), F8 (final candidate; expected ~0.99585, range 0.99578-0.99592),
# 3 min apart (F11 dropped 22:47: §43c showed exact X-point labels neutral on VT, +0.000004); deletes 982249 (F5c) and 981699 (F6L) (reports saved);
# then polls the three scores into upload_oct14.log.
cd /home/mohhakami/Documents/paper/quworld; LOG=runs/strat/fusion/upload_oct14.log
until [ "$(date -u +%Y-%m-%d)" = "2026-10-14" ]; do sleep 30; done; sleep 60
set -a; . ./.env; set +a
for id in 982249 981699; do curl -s --max-time 120 -X DELETE -H "Authorization: Token $CODABENCH_TOKEN" "https://www.codabench.org/api/submissions/$id/" -o /dev/null -w "delete $id http %{http_code}\n" >> $LOG; done
IDS=""
for t in F7KC F8; do
  case $t in F7KC) M="F7KC: F7C + second kappa pass (kw2 + li x2) + q95 clamp at the label range";; F8) M="F8: final candidate (QHAR2 q/b, clamp, X-point step gate 0 w 0.75, li2)";; F11) M="F11: F8 + X-point heads on exact labels";; esac
  runs/codabench_upload.sh runs/sophelio_sota/submission_$t/submission.zip $t "$M" >> $LOG 2>&1
  id=$(grep -o '"id":[0-9]*' runs/sophelio_sota/upload_$t.log | head -1 | cut -d: -f2); echo "[upload] $t id $id $(date -Is)" >> $LOG; IDS="$IDS $t:$id"; sleep 180
done
for i in $(seq 1 360); do
  left=0
  for p in $IDS; do t=${p%%:*}; id=${p##*:}; grep -q "^\[score\] $t " $LOG && continue
    s=$(runs/codabench_status.sh $id 2>/dev/null); case "$s" in Finished*|Failed*) echo "[score] $t $id $s $(date -Is)" >> $LOG;; *) left=1;; esac
  done
  [ $left = 0 ] && break; sleep 60
done
echo "UPLOAD_OCT14_DONE $(date -Is)" >> $LOG
