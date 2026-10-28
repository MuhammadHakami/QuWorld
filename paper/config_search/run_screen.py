"""Serialized exploratory configuration screen; no test access or scaling sweep.

Paired configurations are an initial screen, not a claim of independently
optimal classical tuning. Promotions require new seeds and confirmation data.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))
from config_search.core import RegisterConfig


def process_identity(pid):
    try:
        # /proc starttime identifies a process even if its PID is later reused.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def candidates():
    base = RegisterConfig()
    return [
        ("reference", base),
        ("encoded", replace(base, encoding_rms=.1)),
        ("bounded_static", replace(base, geometry="bounded_static", encoding_rms=.1)),
        ("bounded_dynamic", replace(base, geometry="bounded_dynamic", encoding_rms=.1)),
        ("conditioned_readout", replace(base, geometry="bounded_dynamic", encoding_rms=.1, readout="cond2")),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/config_search/trl_screen_v1")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--extend-steps", type=int, default=1024, help="Extend every candidate without pruning; 0 disables")
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--pause-idle-orchestrator", type=int, action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or (args.extend_steps and args.extend_steps <= args.steps):
        parser.error("Require positive steps and extension greater than initial steps")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "screen.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = []
    for name, cfg in candidates():
        path = args.out / f"{name}.json"
        expected = asdict(cfg)
        if path.exists() and json.loads(path.read_text()) != expected:
            raise ValueError(f"Existing config differs: {path}")
        path.write_text(json.dumps(expected, indent=2) + "\n")
        arms = ["vanilla", "mlp", "quantum"] if name == "reference" else ["mlp", "quantum"]
        for arm in arms:
            jobs.append(dict(name=name, arm=arm, config=str(path),
                             out=str(args.out / f"{name}_{arm}_s{args.seed}")))
    budgets = [args.steps] + ([args.extend_steps] if args.extend_steps else [])
    jobs = [{**job, "target_steps": budget} for budget in budgets for job in jobs]
    manifest = dict(kind="exploratory_configuration_screen", qubits=30, seed=args.seed,
                    steps=args.steps, budgets=budgets, jobs=jobs, test_access=False,
                    rule="No promotion from this one-seed screen; confirmation requires paired new seeds and separate validation trajectories")
    evidence = [ROOT / "26-configuration-validation-protocol.md", ROOT / "runs/config_search/mechanism_probe.json"]
    manifest["evidence_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(f"Prepared {len(jobs)} jobs, no GPU runs launched: {args.out}")
        return
    paused = []
    report = []
    child = None
    def log(message):
        with (args.out / "status").open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}\n")
        print(message, flush=True)
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for pid in args.pause_idle_orchestrator:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            if not any(name in command for name in ("physiq_orchestrator.sh", "wmb_orchestrator.sh")):
                raise ValueError(f"PID {pid} is not an expected idle video orchestrator")
            identity = process_identity(pid)
            os.kill(pid, signal.SIGSTOP)
            paused.append((pid, identity))
            log(f"PAUSED_IDLE_ORCHESTRATOR pid={pid} start={identity}")
        if args.wait_pid:
            identity = process_identity(args.wait_pid)
            log(f"WAIT_EXISTING_CAMPAIGN pid={args.wait_pid} start={identity}")
            while identity is not None and process_identity(args.wait_pid) == identity:
                time.sleep(30)
        for job in jobs:
            target_steps = job["target_steps"]
            # Do not infer completion from a status sentinel or log substring.
            dest = Path(job["out"])
            result = dest / "result.json"
            if result.exists():
                data = json.loads(result.read_text())
                if data["status"] == "complete" and data["steps"] >= target_steps:
                    log(f"EXISTING_RESULT {job['name']} {job['arm']}")
                    # Trainer validates provenance on resume; run it below too.
            while True:
                processes = subprocess.check_output([
                    "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
                ], text=True).strip()
                if not processes:
                    break
                log(f"WAIT_GPU pids={processes.replace(chr(10), ',')}")
                time.sleep(30)
            cmd = [sys.executable, str(ROOT / "paper/config_search/train_trial.py"),
                   "--arm", job["arm"], "--config", job["config"], "--out", job["out"],
                   "--steps", str(target_steps), "--seed", str(args.seed)]
            log(f"START {job['name']} {job['arm']} target_steps={target_steps}")
            with (args.out / f"{job['name']}_{job['arm']}.log").open("a") as output:
                child = subprocess.Popen(cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
                code = child.wait(); child = None
            if code != 0:
                raise RuntimeError(f"Trial exited {code}; resume checkpoint retained: {job['out']}")
            data = json.loads(result.read_text())
            if data["status"] != "complete" or data["steps"] < target_steps:
                raise RuntimeError("Trial did not reach fixed-step budget")
            log(f"RESULT {job['name']} {job['arm']} valid8={data['best_validation_vrmse8']:.6f}")
            snapshot = {**job, "result": data}
            snapshot_path = dest / f"result_at_{target_steps}.json"
            if not snapshot_path.exists():
                snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")
            saved = json.loads(snapshot_path.read_text())
            if saved["result"]["steps"] != target_steps:
                raise RuntimeError("Cannot reconstruct an earlier fidelity from a later result")
            report.append(saved)
        (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if len(budgets) == 2:
            from config_search.summarize_screen import summarize
            calibration = summarize(args.out)
            log(f"FIDELITY_GATE {calibration['short_fidelity_gate']} spearman={calibration['spearman']}")
        log("SCREEN_COMPLETE_NO_CONFIRMATORY_CLAIM")
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        for pid, identity in paused:
            if process_identity(pid) == identity:
                os.kill(pid, signal.SIGCONT)
                log(f"RESUMED_ORCHESTRATOR pid={pid}")


if __name__ == "__main__":
    main()
