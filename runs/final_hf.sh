#!/bin/bash
# Final-phase uploader, website route (Codabench down since Oct 22; organizers' HF discussion #1 on ganatma/fusion-equilibrium-challenge-final,
# Oct 23): push the candidate's diii_d_private_test.npz to a PRIVATE HF dataset repo with the organizers' push_predictions.py (starter 823f6ee),
# then enter the pointer on https://fusion-equilibrium-challenge.sophelio.io/final-submission/ (POST /final-submit.php). Closes 2026-10-28T00:00Z.
# Needs in .env (values never printed):
#   HF_TOKEN       write token (upload)            HF_REPO   owner/name of the private predictions dataset repo
#   HF_READ_TOKEN  fine-grained READ token scoped to HF_REPO (goes into the manifest; the push script refuses broader tokens)
#   CODABENCH_USER + FINAL_EMAIL   optional: only if both are set does this script submit the website form itself (the user's opt-in);
#                                  otherwise it stops after the push and the user pastes submission/manifest.json into the page.
# One entry per candidate: the receipt file blocks a second form post, and runs/final_auto.sh (Codabench route) is retired so the same build
# cannot use two of the three shared slots.
#   setsid nohup bash runs/final_hf.sh > /dev/null 2>&1 &      (CAND=F21K default; log runs/strat/fusion/final_hf.log)
cd /home/mohhakami/Documents/paper/quworld
CAND=${CAND:-F21K}; DIR=runs/sophelio_sota/submission_${CAND}priv; L=runs/strat/fusion; LOG=$L/final_hf.log
REC=$L/final_receipt_${CAND}.json; MAN=$DIR/manifest.json; CLOSE=$(date -d 2026-10-28T00:00:00Z +%s)
envv() { grep -m1 "^$1=" .env 2>/dev/null | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//'; }
echo "[hf] start $(date -Is) cand $CAND dir $DIR" >> $LOG
[ -s $DIR/diii_d_private_test.npz ] || { echo "[hf] FINAL_HF_FAIL no $DIR/diii_d_private_test.npz" >> $LOG; exit 1; }
while [ "$(date +%s)" -lt "$CLOSE" ]; do
  [ -s "$REC" ] && { echo "[hf] receipt present ($REC); done $(date -Is)" >> $LOG; exit 0; }
  REPO=$(envv HF_REPO)
  if [ ! -s "$MAN" ] || ! grep -q "\"repo_id\": \"$REPO\"" "$MAN"; then
    if [ -n "$(envv HF_TOKEN)" ] && [ -n "$(envv HF_READ_TOKEN)" ] && [ -n "$REPO" ]; then
      echo "[hf] pushing $DIR to $REPO $(date -Is)" >> $LOG
      HF_TOKEN="$(envv HF_TOKEN)" HF_READ_TOKEN="$(envv HF_READ_TOKEN)" HF_HUB_DISABLE_PROGRESS_BARS=1 \
        .venv/bin/python runs/sophelio/push_predictions.py --repo "$REPO" --dir $DIR --out $DIR/submission_pointer.zip >> $LOG 2>&1
      rc=$?; echo "[hf] push exit $rc $(date -Is)" >> $LOG
      [ $rc = 0 ] && [ -s "$MAN" ] && echo "[hf] PUSH_DONE $CAND $REPO@$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$MAN")" >> $LOG
      [ $rc = 0 ] || { sleep 300; continue; }
    else
      echo "[hf] $(date -Is) waiting for HF_TOKEN / HF_READ_TOKEN / HF_REPO in .env" >> $LOG; sleep 300; continue
    fi
  fi
  if [ -n "$(envv CODABENCH_USER)" ] && [ -n "$(envv FINAL_EMAIL)" ]; then
    CBU="$(envv CODABENCH_USER)" FEM="$(envv FINAL_EMAIL)" TEAM="$(envv FINAL_TEAM)" .venv/bin/python - "$MAN" "$REC" "$CAND" >> $LOG 2>&1 <<'PY'
import json, os, sys, urllib.request
man, rec, cand = sys.argv[1:4]
m = json.load(open(man))
body = {"codabenchUser": os.environ["CBU"], "email": os.environ["FEM"], "repoId": m["repo_id"], "revision": m["revision"],
        "token": m.get("token") or "", "teamName": os.environ.get("TEAM", ""), "notes": f"Challenge 1 only; entry {cand}", "website": "",
        "ackRules": True}
req = urllib.request.Request("https://fusion-equilibrium-challenge.sophelio.io/final-submit.php", data=json.dumps(body).encode(),
                             headers={"Content-Type": "text/plain;charset=utf-8"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        code, p = r.status, json.loads(r.read().decode() or "null")
except urllib.error.HTTPError as e:
    code, p = e.code, None
    try: p = json.loads(e.read().decode())
    except Exception: pass
keep = {k: p.get(k) for k in ("ok", "receiptId", "receivedAt", "filesFound", "websiteEntries", "hfCheck", "duplicate", "error")} if isinstance(p, dict) else {}
print(f"[hf] form http {code}: {json.dumps(keep)}")
if code == 200 and keep.get("ok") is True:
    json.dump({**keep, "cand": cand, "repo_id": m["repo_id"], "revision": m["revision"]}, open(rec, "w"), indent=1)
    print(f"[hf] FINAL_SUBMITTED {cand} receipt {keep.get('receiptId')}")
PY
    [ -s "$REC" ] || echo "[hf] form not accepted; retry in 300 s $(date -Is)" >> $LOG
  else
    echo "[hf] $(date -Is) pushed; manifest at $MAN - paste it into the Final submission page (or set CODABENCH_USER + FINAL_EMAIL)" >> $LOG
  fi
  sleep 300
done
echo "[hf] Final closed $(date -Is)" >> $LOG
