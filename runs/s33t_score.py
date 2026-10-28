"""Doc 35 §33t: score the frozen §33 checkpoints once on TEST (score only: no optimiser, no retraining).

For each of the 16 registered checkpoints (stage A and B2loss; adrc_ryd and adrc_mlp; seeds 1-2; shear_flow and
rayleigh_benard) plus the cls_only references (stage A, seeds 1-2), the checkpoint is rebuilt with its saved model, gate,
route and QW_* environment (diag_s33.rebuild). VALID is replayed and must match the saved VALID value to 1e-6, or the
script refuses. All 20 replays and metadata checks pass before any TEST file is opened (Codex 084535).
Then an exclusive claim file (O_EXCL) is created, and each checkpoint gets one journalled TEST pass (STARTED and
FINISHED lines, fsync). Refuses if the claim, the journal or the output exists (a crash needs documented recovery,
never an implicit rerun) or if runs/s33/lock/test_lock.sha256 does not verify.

    .venv/bin/python runs/s33t_score.py   ->  runs/s33/test_scores.json
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper/alloc"))
import adrc  # noqa: E402
import diag_s33  # noqa: E402

CK, RES = ROOT / "runs/alloc/ckpt_s33", ROOT / "runs/alloc/adrc_results_s33"
OUT, LOCK = ROOT / "runs/s33/test_scores.json", ROOT / "runs/s33/lock/test_lock.sha256"
CLAIM, JOURNAL = ROOT / "runs/s33/test_claim.json", ROOT / "runs/s33/test_journal.jsonl"
ROWS = ("shear_flow__gate_vanilla_s1337", "rayleigh_benard__gatest_vanilla_s1337")
SEEDS = (1, 2)
RUNS = [("A", "adrc_ryd_gate_rs_dyn"), ("A", "adrc_mlp_gate_rs_dyn"), ("A", "cls_only_gate"),
        ("B2loss", "adrc_ryd_gate_rs_dyn"), ("B2loss", "adrc_mlp_gate_rs_dyn")]
TOL = 1e-6


def lock_ok() -> None:
    body = "".join(LOCK.read_text().splitlines(keepends=True)[1:])          # first line is the timestamp
    r = subprocess.run(["sha256sum", "-c", "--quiet", "-"], input=body, text=True, cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        sys.exit(f"refuse: §33t lock does not verify:\n{r.stdout}{r.stderr}")


def load(cell: str, row: str, arm: str, seed: int):
    name = f"{row}_{arm}_s{seed}"
    ck = torch.load(CK / cell / f"{name}.pt", weights_only=False)
    a, env = ck["args"], ck.get("env", {})
    ok = (a["row"] == row and a["seed"] == seed and a["arm"] == arm.split("_gate")[0] and ck["K"] == 30
          and (env.get("QW_LOSS") == "vrmse") == (cell == "B2loss"))
    if not ok:
        sys.exit(f"refuse: checkpoint metadata does not match {cell}/{name}: args={a} K={ck['K']} env={env}")
    return name, ck


def replay_valid(cell: str, row: str, arm: str, seed: int) -> dict:
    name, ck = load(cell, row, arm, seed)
    model, gate, va, route = diag_s33.rebuild(ck)
    v = diag_s33.valid_vrmse(model, gate, va, route)
    js = json.loads((RES / cell / f"{name}.json").read_text())
    saved = js["test"]                                                       # QW_SCREEN: the "test" field is VALID
    if (js.get("row"), js.get("arm"), js.get("seed")) != (row, arm.split("_gate")[0], seed) or js.get("valid") != saved:
        sys.exit(f"refuse: screen JSON identity/VALID mismatch ({cell}/{name}); no TEST was read")
    if not (math.isfinite(v) and math.isfinite(saved)) or abs(v - saved) > TOL:
        sys.exit(f"refuse: VALID replay {v:.8f} != saved {saved:.8f} ({cell}/{name}); no TEST was read")
    return {"cell": cell, "row": row, "arm": arm, "seed": seed, "valid_replay": v, "valid_saved": saved}


def score_test(rec: dict) -> dict:
    name, ck = load(rec["cell"], rec["row"], rec["arm"], rec["seed"])
    model, gate, _, route = diag_s33.rebuild(ck)                             # same deterministic rebuild as the replay
    tr = adrc.Data(rec["row"], "train")                                      # TRAIN statistics, as in training
    te = adrc.Data(rec["row"], "test", tr.stats)
    return {**rec, "test": diag_s33.valid_vrmse(model, gate, te, route), "n_test_windows": int(te.N)}


def journal(line: dict) -> None:
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(line) + "\n"); f.flush(); os.fsync(f.fileno())


def main() -> None:
    if OUT.exists() or CLAIM.exists() or JOURNAL.exists():
        sys.exit(f"refuse: {[str(p) for p in (OUT, CLAIM, JOURNAL) if p.exists()]} exists (one TEST pass only; recovery must be documented "
                 "in doc 35, never an implicit rerun)")
    lock_ok()
    # phase 1: every VALID replay and metadata check passes before any TEST file is opened
    reps = [replay_valid(c, r, a, s) for c, a in RUNS for r in ROWS for s in SEEDS]
    print(f"[s33t] VALID replay passed for {len(reps)} checkpoints", flush=True)
    lock_ok()
    # phase 2: exclusive claim, then one journalled TEST pass per checkpoint
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(CLAIM, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({"claimed": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid()}).encode()); os.fsync(fd)
    os.close(fd)
    recs = []
    for rec in reps:
        journal({"STARTED": [rec["cell"], rec["row"], rec["arm"], rec["seed"]], "t": time.time()})
        r = score_test(rec); recs.append(r)
        journal({"FINISHED": [rec["cell"], rec["row"], rec["arm"], rec["seed"]], "test": r["test"], "t": time.time()})
    get = {(x["cell"], x["row"], x["arm"], x["seed"]): x["test"] for x in recs}
    verdict = {}
    for r in ROWS:
        d = [get[("B2loss", r, "adrc_ryd_gate_rs_dyn", s)] - get[("A", r, "adrc_ryd_gate_rs_dyn", s)] for s in SEEDS]
        dm = [get[("B2loss", r, "adrc_mlp_gate_rs_dyn", s)] - get[("A", r, "adrc_mlp_gate_rs_dyn", s)] for s in SEEDS]
        verdict[r] = {"register_B2loss_minus_A": d, "register_lower_n": sum(x < 0 for x in d),
                      "mlp_B2loss_minus_A (descriptive)": dm}
    held = all(v["register_lower_n"] == len(SEEDS) for v in verdict.values())
    out = {"rule": "§33t primary: B2loss register below A register on 2/2 seeds on BOTH rows (TEST-informed)",
           "verdict": "promotion holds on TEST" if held else "promotion not confirmed on TEST",
           "rows": verdict, "scores": recs}
    fd = os.open(OUT, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps(out, indent=1).encode()); os.fsync(fd); os.close(fd)
    print("[s33t]", out["verdict"], json.dumps(verdict), flush=True)


if __name__ == "__main__":
    main()
