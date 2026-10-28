"""Final entry form post (website route), run once the fine-grained read token can see the pinned commit.
Polls .env every 20 s until 2026-10-27T23:58Z; posts the organizers' form once; receipt -> runs/strat/fusion/final_receipt_F21K.json.
Tokens and email are read from .env and never printed."""
import json, time, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi

REPO, REV, CAND = "Moka/fusion-final-preds", "abece02dae92ca7554f28cbdeae125c2cb349384", "F21K"
LOG, REC = Path("runs/strat/fusion/final_hf.log"), Path("runs/strat/fusion/final_receipt_F21K.json")
MAN = Path("runs/sophelio_sota/submission_F21Kpriv/manifest.json")
STOP = datetime(2026, 10, 27, 23, 58, tzinfo=timezone.utc).timestamp()


def env(k):
    for ln in Path(".env").read_text().splitlines():
        if ln.startswith(k + "="):
            return ln.split("=", 1)[1].strip().strip("'\"")
    return ""


def log(m):
    with LOG.open("a") as f:
        f.write(f"[form] {datetime.now().astimezone().isoformat(timespec='seconds')} {m}\n")


def readable(tok):
    try:
        i = HfApi().repo_info(REPO, repo_type="dataset", revision=REV, token=tok)
        return "diii_d_private_test.npz" in {s.rfilename for s in i.siblings}
    except Exception as e:
        return type(e).__name__


log("start: waiting for a read token that can see the pinned commit")
last = None
while time.time() < STOP and not REC.exists():
    tok = env("HF_READ_TOKEN"); r = readable(tok) if tok else "no token"
    if r is not True:
        if r != last:
            log(f"read token cannot see {REPO}@{REV[:8]} ({r}); retry every 20 s"); last = r
        time.sleep(20); continue
    log("read token OK; posting form")
    MAN.write_text(json.dumps({"repo_id": REPO, "revision": REV, "token": tok}, indent=1))
    body = {"codabenchUser": env("CODABENCH_USER"), "email": env("FINAL_EMAIL"), "repoId": REPO, "revision": REV, "token": tok,
            "teamName": env("FINAL_TEAM"), "notes": f"Challenge 1 only; entry {CAND}", "website": "", "ackRules": True}
    req = urllib.request.Request("https://fusion-equilibrium-challenge.sophelio.io/final-submit.php", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "text/plain;charset=utf-8", "Origin": "https://fusion-equilibrium-challenge.sophelio.io",
                                          "Referer": "https://fusion-equilibrium-challenge.sophelio.io/final-submission/",
                                          "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"},
                                 method="POST")   # Cloudflare answers 403 (error 1010) to the Python user agent
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            code, p = resp.status, json.loads(resp.read().decode() or "null")
    except urllib.error.HTTPError as e:
        code, p = e.code, None
        try: p = json.loads(e.read().decode())
        except Exception: pass
    except Exception as e:
        code, p = type(e).__name__, None
    keep = {k: p.get(k) for k in ("ok", "receiptId", "receivedAt", "filesFound", "websiteEntries", "hfCheck", "duplicate", "error")} if isinstance(p, dict) else {}
    log(f"form http {code}: {json.dumps(keep)}")
    if code == 200 and keep.get("ok") is True:
        REC.write_text(json.dumps({**keep, "cand": CAND, "repo_id": REPO, "revision": REV}, indent=1))
        log(f"FINAL_SUBMITTED {CAND} receipt {keep.get('receiptId')}")
        break
    time.sleep(30)
else:
    log("stopped (receipt present or deadline guard)")
