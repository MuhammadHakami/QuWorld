#!/bin/bash
# Publish the Fusion reproduction package: push ~/Documents/fusion-equilibrium (main) and upload the weights listed in WEIGHTS.sha256 (49 for F18K) as assets of release
# f18k-weights. Token from the repo .env (GITHUB_TOKEN, fine-grained, Contents: read and write); never printed or stored in git config.
#   runs/publish_release.sh            -> runs/strat/fusion/publish.log, last line PUBLISH_DONE <CODE_URL>
cd "$(dirname "$0")/.."; Q=$PWD; LOG=$Q/runs/strat/fusion/publish.log; set -a; . ./.env; set +a
: "${GITHUB_TOKEN:?add GITHUB_TOKEN to .env}"; REPO=MuhammadHakami/fusion-equilibrium; TAG=${TAGW:-f18k-weights}; P=$HOME/Documents/fusion-equilibrium
API=https://api.github.com/repos/$REPO; AUTH="Authorization: Bearer $GITHUB_TOKEN"; echo "[pub] start $(date -Is)" >> $LOG
cd $P && git -c credential.helper= -c credential.helper='!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f' push origin main >> $LOG 2>&1 \
  || { echo "PUBLISH_FAIL push $(date -Is)" >> $LOG; exit 1; }
SHA=$(git rev-parse HEAD); echo "[pub] pushed $SHA" >> $LOG
RID=$(curl -s -H "$AUTH" $API/releases/tags/$TAG | python3 -c "import json,sys; print(json.load(sys.stdin).get('id', ''))")
[ -n "$RID" ] || RID=$(curl -s -X POST -H "$AUTH" $API/releases -d "{\"tag_name\": \"$TAG\", \"target_commitish\": \"$SHA\", \"name\": \"$TAG\", \"body\": \"Trained weights of the Fusion Final entry (package default CAND in repro/fusion/infer.sh); sha256 in repro/fusion/WEIGHTS.sha256, fetch with repro/fusion/fetch_weights.py.\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('id', ''))")
[ -n "$RID" ] || { echo "PUBLISH_FAIL release $(date -Is)" >> $LOG; exit 1; }
have=$(curl -s -H "$AUTH" "$API/releases/$RID/assets?per_page=100" | python3 -c "import json,sys; print(' '.join(a['name'] for a in json.load(sys.stdin)))")
while read h n rel; do
  b=$(basename $rel); case " $have " in *" $b "*) continue;; esac
  code=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "$AUTH" -H "Content-Type: application/octet-stream" --data-binary @$Q/$rel \
    "https://uploads.github.com/repos/$REPO/releases/$RID/assets?name=$b")
  echo "[pub] $b http $code" >> $LOG; [ "$code" = 201 ] || { echo "PUBLISH_FAIL asset $b $(date -Is)" >> $LOG; exit 1; }
done < $P/repro/fusion/WEIGHTS.sha256
n=$(curl -s -H "$AUTH" "$API/releases/$RID/assets?per_page=100" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")
echo "[pub] release $TAG assets $n / $(wc -l < $P/repro/fusion/WEIGHTS.sha256)" >> $LOG
echo "PUBLISH_DONE https://github.com/$REPO@$SHA $(date -Is)" >> $LOG
