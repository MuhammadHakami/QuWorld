#!/bin/bash
# Learn2Design-2026 upload (submit.learn2design2026.com, competition 4, Final round phase 14). Token L2D_TOKEN from .env
# (never printed). Only the LAST submission before 16 Oct 11:59 UTC is evaluated.
#   runs/l2d_upload.sh <submission.zip> <name> <method>   -> runs/l2d/final/upload_<name>.log
cd /home/mohhakami/Documents/paper/quworld; set -a; . ./.env; set +a
ZIP=$1; NAME=$2; METHOD=$3; SZ=$(stat -c %s "$ZIP"); LOG=runs/l2d/final/upload_$NAME.log; T=$(mktemp -d); API=https://submit.learn2design2026.com/api
echo "start $(date -Is) size $SZ zip $ZIP" > $LOG
curl -s --max-time 120 -X POST -H "Authorization: Token $L2D_TOKEN" -H "Content-Type: application/json" "$API/datasets/" \
  -d "{\"type\": \"submission\", \"name\": \"$NAME\", \"request_sassy_file_name\": \"submission.zip\", \"file_size\": $SZ, \"competition\": 4}" > $T/ds.json
KEY=$(python3 -c "import json; print(json.load(open('$T/ds.json'))['key'])") || { echo "dataset create failed: $(head -c 300 $T/ds.json)" >> $LOG; exit 1; }
URL=$(python3 -c "import json; print(json.load(open('$T/ds.json'))['sassy_url'])")
echo "dataset key $KEY" >> $LOG; t0=$(date +%s)
code=$(curl -s -o $T/put.txt -w "%{http_code}" --max-time 3600 -X PUT -H "Content-Type: application/zip" --upload-file "$ZIP" "$URL")
echo "PUT http $code in $(( $(date +%s) - t0 ))s" >> $LOG; [ "$code" = 200 ] || exit 1
curl -s --max-time 120 -w " [http %{http_code}]\n" -X PUT -H "Authorization: Token $L2D_TOKEN" "$API/datasets/completed/$KEY/" >> $LOG
curl -s --max-time 120 -w " [http %{http_code}]\n" -X POST -H "Authorization: Token $L2D_TOKEN" -H "Content-Type: application/json" "$API/submissions/" \
  -d "{\"phase\": 14, \"data\": \"$KEY\", \"fact_sheet_answers\": {\"method_summary\": \"$METHOD\", \"uses_pretrained_assets\": true}}" | head -c 600 >> $LOG
echo; echo "UPLOAD_DONE $(date -Is)" >> $LOG; rm -rf $T
