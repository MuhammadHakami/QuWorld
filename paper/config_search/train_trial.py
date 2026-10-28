"""Resumable CUDA-only configuration trial on the OFFICIAL TRL2D backbone.

Search uses a declared subset of official validation trajectories; no test reads.
This is an exploratory screen, not a replacement for the completed campaign.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))
from config_search.core import RegisterConfig, build_official
from phase_b import data_ram


def atomic_save(obj, path):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def validation_ids(n, split):
    ids = [i for i in range(n) if (i % 3 == 2) == (split == "confirmation")]
    if not ids:
        raise ValueError("Not enough validation trajectories for disjoint subsets")
    return ids


@torch.no_grad()
def rollout_validation(model, valid, ids, horizon=8):
    model.eval()
    nh, channels = 4, valid.shape[2]
    starts = sorted(set(torch.linspace(nh - 1, valid.shape[1] - horizon - 1, 3).long().tolist()))
    errors = []
    for start in starts:
        history = valid[ids, start - nh + 1:start + 1].flatten(1, 2).to("cuda")
        for _ in range(horizon):
            pred = model(history)
            history = torch.cat([history[:, channels:], pred], 1)
        target = valid[ids, start + horizon].to("cuda")
        mse = (pred - target).square().mean((2, 3))
        variance = (target - target.mean((2, 3), keepdim=True)).square().mean((2, 3))
        errors.extend((mse / (variance + 1e-7)).sqrt().mean(1).tolist())
    model.train()
    return sum(errors) / len(errors)


def train_loss(model, train, generator, batch, k=4):
    ni = torch.randint(train.shape[0], (batch,), generator=generator)
    ti = torch.randint(3, train.shape[1] - k, (batch,), generator=generator)
    idx = ti[:, None] + torch.arange(-3, 1)
    history = train[ni[:, None], idx].flatten(1, 2).to("cuda")
    # Same stop-gradient pushforward objective as official phase-B TD training.
    for j in range(1, k + 1):
        with torch.set_grad_enabled(j == k):
            pred = model(history)
        if j < k:
            history = torch.cat([history[:, 4:], pred.detach()], 1)
    return (pred - train[ni, ti + k].to("cuda")).square().mean()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["vanilla", "mlp", "quantum"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=128)
    parser.add_argument("--max-wall-seconds", type=float, default=3600)
    parser.add_argument("--validation", choices=["search", "confirmation"], default="search")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/cnext_ckpt")
    parser.add_argument("--smoke", action="store_true", help="Two-step official-backbone pipeline check; not a benchmark result")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA required; no CPU fallback")
    if min(args.steps, args.batch, args.eval_every) <= 0 or args.lr <= 0 or args.max_wall_seconds <= 0:
        parser.error("Budgets and learning rate must be positive")
    cfg = RegisterConfig(**json.loads(args.config.read_text()))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(0.25)
    torch.backends.cudnn.benchmark = False
    run_start = time.perf_counter()
    train, valid = data_ram.load()
    if args.smoke:
        train, valid = train[:2, :12], valid[:3, :12]
        args.steps, args.batch, args.eval_every = 2, 1, 2
    ids = validation_ids(len(valid), args.validation)
    model, model_meta = build_official(args.arm, cfg, args.seed, args.model_dir)
    model = model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(args.seed)
    protocol = dict(version=1, arm=args.arm, config=asdict(cfg), seed=args.seed, smoke=args.smoke,
                    batch=args.batch, lr=args.lr, eval_every=args.eval_every,
                    validation=args.validation, validation_ids=ids, horizon=8, td_k=4,
                    train_shape=list(train.shape), validation_shape=list(valid.shape),
                    upstream_weights_sha256=sha(args.model_dir / "model.safetensors"),
                    upstream_config_sha256=sha(args.model_dir / "config.json"),
                    trainer_sha256=sha(__file__), core_sha256=sha(Path(__file__).with_name("core.py")),
                    qlayer_sha256=sha(ROOT / "paper/qlayer.py"),
                    model_sha256=sha(ROOT / "paper/model.py"),
                    data_stats_sha256=sha(data_ram.STATS))
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    latest = args.out / "latest.pt"
    best_path = args.out / "best.pt"
    step, best, prior_wall = 0, math.inf, 0.0
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["protocol_hash"] != protocol_hash:
            raise ValueError("Resume protocol mismatch; use a fresh output directory")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        step, best, prior_wall = state["step"], state["best"], state["wall_seconds"]
        gen.set_state(state["batch_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        del state
    elif any(args.out.glob("*.json")):
        raise ValueError("Existing metadata without a resumable checkpoint; use a fresh output directory")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    def check_and_save():
        nonlocal best
        score = rollout_validation(model, valid, ids)
        if not math.isfinite(score):
            raise ValueError("Nonfinite validation score")
        if score < best:
            best = score
            atomic_save(dict(model_state_dict=model.state_dict(), step=step,
                             validation=score, protocol_hash=protocol_hash), best_path)
        torch.cuda.synchronize()
        elapsed = prior_wall + time.perf_counter() - run_start
        atomic_save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                         step=step, best=best, protocol_hash=protocol_hash, batch_rng=gen.get_state(),
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                         wall_seconds=elapsed), latest)
        with (args.out / "curve.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=step, vrmse8=score, best=best, wall_seconds=elapsed)) + "\n")
        print(f"step={step} valid8={score:.6f} best={best:.6f} wall={elapsed:.1f}s", flush=True)

    if step == 0:
        check_and_save()
    while step < args.steps and time.perf_counter() - run_start < args.max_wall_seconds:
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss(model, train, gen, args.batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); step += 1
        if step % args.eval_every == 0:
            check_and_save()
    # Always checkpoint even on a wall-time cutoff; never label it complete.
    if step % args.eval_every or not (args.out / "result.json").exists():
        check_and_save()
    torch.cuda.synchronize()
    status = ("smoke_complete" if args.smoke else "complete") if step >= args.steps else "incomplete"
    result = dict(status=status, steps=step,
                  target_steps=args.steps, best_validation_vrmse8=best, protocol_hash=protocol_hash,
                  device="cuda", gpu=torch.cuda.get_device_name(),
                  wall_seconds=prior_wall + time.perf_counter() - run_start,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(), model=model_meta,
                  test_evaluated=False)
    tmp = args.out / "result.tmp"
    tmp.write_text(json.dumps(result, indent=2) + "\n"); tmp.replace(args.out / "result.json")
    if result["status"] == "incomplete":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
