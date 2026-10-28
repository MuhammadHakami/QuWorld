"""Phase-B ROUND 2: TD/pushforward fine-tune from the official checkpoint.

Separate file from finetune.py so the in-flight round-1 campaign is never
altered mid-arm. Difference: K-step pushforward loss (stop-grad inner steps,
backprop last step only — Brandstetter 2202.03376) instead of one-step MSE.
Wave-1.5 evidence: this objective unlocks the quantum bottleneck's edge
(interaction: quantum −23% vs control −10% vs baseline ~0%).

Usage: uv run paper/phase_b/finetune_td.py <variant> [seed] [budget_s] [lr] [target_steps] [k]
"""

import gc
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import data_ram  # noqa: E402
from cnext_hybrid import ROOT, build_hybrid  # noqa: E402
from finetune import atomic_save, free_gb, validate  # noqa: E402


@torch.no_grad()
def validate_rollout(model, valid, k: int = 8, stride_t: int = 8) -> float:
    """K-step autoregressive rollout VRMSE with sliding 4-frame history.

    THE selection metric for TD arms: the TD objective deliberately trades
    one-step accuracy for rollout stability, so best-on-one-step selection
    degenerates to the step-0 checkpoint (observed 2026-08-17 — the shipped
    'best' was bit-identical to the official weights). Selection must measure
    what the objective optimizes."""
    model.eval()
    n, t = valid.shape[0], valid.shape[1]
    nh = data_ram.N_HIST
    errs = []
    for t0 in range(nh - 1, t - 1 - k, stride_t):
        idx = torch.arange(t0 - nh + 1, t0 + 1)
        hist = valid[:, idx].reshape(n, -1, *valid.shape[-2:]).to(DEV)
        for _ in range(k):
            pred = model(hist)
            hist = torch.cat([hist[:, 4:], pred], dim=1)
        y = valid[:, t0 + k].to(DEV)
        mse = (pred - y).square().mean(dim=(2, 3))
        var = (y - y.mean(dim=(2, 3), keepdim=True)).square().mean(dim=(2, 3))
        errs.append((mse / (var + 1e-7)).sqrt().mean(dim=1).mean().item())
    model.train()
    return sum(errs) / len(errs)

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "vanilla"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1337
BUDGET_S = float(sys.argv[3]) if len(sys.argv) > 3 else 1500.0
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
TARGET_STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
K = int(sys.argv[6]) if len(sys.argv) > 6 else 4
BATCH = 8            # k-step unroll costs ~K forward passes; halve batch
N_HIST = data_ram.N_HIST
THROTTLE_S = float(os.environ.get("THROTTLE_MS", "0")) / 1000.0
DEV = "cuda"


def td_batch(data: torch.Tensor, gen: torch.Generator):
    """(x_hist16ch, [y_{t+1}..y_{t+K}]) windows for pushforward training."""
    n, t = data.shape[0], data.shape[1]
    ni = torch.randint(0, n, (BATCH,), generator=gen)
    ti = torch.randint(N_HIST - 1, t - K, (BATCH,), generator=gen)
    idx = ti.view(-1, 1) + torch.arange(-N_HIST + 1, 1)
    x = data[ni.view(-1, 1), idx]                      # (B, 4, 4, H, W)
    ys = [data[ni, ti + j].to(DEV) for j in range(1, K + 1)]
    return x.reshape(BATCH, -1, *x.shape[-2:]).to(DEV), ys


def push_loss(model, x, ys):
    """Autoregress with sliding 4-frame history; stop-grad inner rollout,
    backprop only the last step (memory ~= one-step training)."""
    hist = x                                            # (B, 16, H, W)
    for j, y in enumerate(ys):
        last = j == len(ys) - 1
        with torch.set_grad_enabled(last):
            pred = model(hist if last else hist.detach())
        if not last:
            hist = torch.cat([hist[:, 4:], pred.detach()], dim=1)
    return (pred - y).square().mean()


def main() -> None:
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.6)
    gen = torch.Generator().manual_seed(SEED)
    train, valid = data_ram.load()
    model = build_hybrid(VARIANT).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    out = ROOT / "runs" / "phase_b2"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{VARIANT}_s{SEED}.pt"
    latest = out / f"{VARIANT}_s{SEED}_latest.pt"

    steps = 0
    if latest.exists():
        st = torch.load(latest, weights_only=True, map_location=DEV)
        model.load_state_dict(st["model_state_dict"])
        opt.load_state_dict(st["optimizer_state_dict"])
        steps, best = st["step"], st["best"]
        print(f"val_check step={steps} vrmse={best:.4f} free={free_gb():.0f}G "
              f"(RESUMED)", flush=True)
    else:
        v1 = validate(model, valid)
        best = validate_rollout(model, valid)
        print(f"val_check step=0 vrmse1={v1:.4f} vrmse8={best:.4f} "
              f"free={free_gb():.0f}G (checkpoint start, td k={K}, "
              f"SELECTION=rollout)", flush=True)
        atomic_save({"model_state_dict": model.state_dict(), "val": best,
                     "step": 0}, ckpt)

    if steps >= TARGET_STEPS:
        print(f"target_reached steps={steps} best={best:.6f}", flush=True)

    t0, last_val = time.perf_counter(), time.perf_counter()
    while time.perf_counter() - t0 < BUDGET_S and steps < TARGET_STEPS:
        x, ys = td_batch(train, gen)
        loss = push_loss(model, x, ys)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        steps += 1
        if THROTTLE_S:
            torch.cuda.synchronize(); time.sleep(THROTTLE_S)
        if time.perf_counter() - last_val > 180.0:
            v1 = validate(model, valid)
            v = validate_rollout(model, valid)
            print(f"val_check step={steps} vrmse1={v1:.4f} vrmse8={v:.4f} "
                  f"loss={loss.item():.5f} free={free_gb():.0f}G", flush=True)
            if v < best:
                best = v
                atomic_save({"model_state_dict": model.state_dict(), "val": best,
                             "step": steps}, ckpt)
            atomic_save({"model_state_dict": model.state_dict(),
                         "optimizer_state_dict": opt.state_dict(),
                         "step": steps, "best": best}, latest)
            last_val = time.perf_counter()
            gc.collect(); torch.cuda.empty_cache()

    if steps >= TARGET_STEPS:
        print(f"target_reached steps={steps} best={best:.6f}", flush=True)
    v = validate_rollout(model, valid, stride_t=4)
    if v < best:
        best = v
        atomic_save({"model_state_dict": model.state_dict(), "val": best,
                     "step": steps}, ckpt)
    atomic_save({"model_state_dict": model.state_dict(),
                 "optimizer_state_dict": opt.state_dict(),
                 "step": steps, "best": best}, latest)
    print("---")
    print(f"variant:        {VARIANT}")
    print(f"td_k:           {K}")
    print(f"val_vrmse_8:    {best:.6f}  # SELECTION metric (rollout)")
    print(f"num_steps:      {steps}")
    print(f"train_seconds:  {time.perf_counter() - t0:.1f}")
    print(f"checkpoint:     {ckpt}")


if __name__ == "__main__":
    main()
