#!/bin/bash
# Status and scores of a Codabench submission (token from .env, never printed):  runs/codabench_status.sh <id>
cd /home/mohhakami/Documents/paper/quworld; set -a; . ./.env; set +a; T=$(mktemp)
for i in 1 2 3; do
  curl -s --max-time 60 -H "Authorization: Token $CODABENCH_TOKEN" "https://www.codabench.org/api/submissions/$1/" > $T
  python3 -c "
import json, sys
try:
    d = json.load(open('$T'))
except Exception:
    sys.exit(1)
print(d.get('status'), {s.get('column_key', s.get('index')): s.get('score') for s in d.get('scores', [])})" && break
  sleep 5
done; rm -f $T
