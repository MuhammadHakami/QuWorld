#!/bin/bash
# Codabench 17739 upload of a Fusion submission zip: Development phase 29580 by default; Final: PHASE=29581 CODE_URL=<public package ref> (required there). Token from .env (never printed).
#   runs/codabench_upload.sh <submission.zip> <dataset-name> <method-name>   -> runs/sophelio_sota/upload_<dataset-name>.log
cd /home/mohhakami/Documents/paper/quworld; set -a; . ./.env; set +a
ZIP=$1; NAME=$2; METHOD=$3; SZ=$(stat -c %s "$ZIP"); LOG=runs/sophelio_sota/upload_$NAME.log; T=$(mktemp -d)
[ "${PHASE:-29580}" = 29581 ] && [ -z "${CODE_URL:-}" ] && { echo "Final phase needs CODE_URL (public reproduction package reference)"; exit 2; }
echo "start $(date -Is) size $SZ zip $ZIP" > $LOG
curl -s --max-time 120 -X POST -H "Authorization: Token $CODABENCH_TOKEN" -H "Content-Type: application/json" "https://www.codabench.org/api/datasets/" \
  -d "{\"type\": \"submission\", \"name\": \"$NAME\", \"request_sassy_file_name\": \"submission.zip\", \"file_size\": $SZ, \"competition\": 17739}" > $T/ds.json
KEY=$(python3 -c "import json; print(json.load(open('$T/ds.json'))['key'])") || { echo "dataset create failed: $(head -c 300 $T/ds.json)" >> $LOG; exit 1; }
URL=$(python3 -c "import json; print(json.load(open('$T/ds.json'))['sassy_url'])")
echo "dataset key $KEY" >> $LOG; t0=$(date +%s)
code=$(curl -s -o $T/put.txt -w "%{http_code}" --max-time 14400 -X PUT -H "Content-Type: application/zip" --upload-file "$ZIP" "$URL")
echo "PUT http $code in $(( $(date +%s) - t0 ))s" >> $LOG; [ "$code" = 200 ] || exit 1
curl -s --max-time 120 -w " [http %{http_code}]\n" -X PUT -H "Authorization: Token $CODABENCH_TOKEN" "https://www.codabench.org/api/datasets/completed/$KEY/" >> $LOG
curl -s --max-time 120 -w " [http %{http_code}]\n" -X POST -H "Authorization: Token $CODABENCH_TOKEN" -H "Content-Type: application/json" "https://www.codabench.org/api/submissions/" \
  -d "{\"phase\": ${PHASE:-29580}, \"data\": \"$KEY\", \"fact_sheet_answers\": {\"method_name\": \"$METHOD\", \"code_url\": \"${CODE_URL:-}\", \"compute_light\": false}}" | head -c 600 >> $LOG
echo; echo "UPLOAD_DONE $(date -Is)" >> $LOG; rm -rf $T
