"""5-minute-budget sprint trainer (autoresearch protocol).

Usage: uv run paper/train.py <variant> [seed] [budget_s]
Variants: none | mlp | hea12 | hea16 | ryd12 | ryd16 (see model.build_model)

Fixed seed across variants for deterministic parity. Prints a summary block
parsed by the runner; val metrics computed AFTER the budget (excluded from it).
"""

import os
import sys
import pathlib
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from data import load, vrmse
from model import build_model

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "none"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1337
BUDGET_S = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
MAX_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
ROLLOUT_K = int(sys.argv[5]) if len(sys.argv) > 5 else 1  # pushforward/TD depth
BATCH = 16
LR = 1e-3
ROLLOUT_H = 8
DEV = "cuda"


def batches(x: torch.Tensor, gen: torch.Generator, k: int = 1):
    """Yield (x_t, [x_{t+1}..x_{t+k}]) for k-step pushforward/TD training."""
    n, t = x.shape[0], x.shape[1]
    while True:
        ti = torch.randint(0, t - k, (BATCH,), generator=gen)
        ni = torch.randint(0, n, (BATCH,), generator=gen)
        ys = [x[ni, ti + j].to(DEV, non_blocking=True) for j in range(1, k + 1)]
        yield x[ni, ti].to(DEV, non_blocking=True), ys


@torch.no_grad()
def evaluate(model, valid):
    model.eval()
    n, t = valid.shape[0], valid.shape[1]
    # one-step over the full valid set (strided to bound time)
    errs = []
    for ti in range(0, t - 1, 4):
        x = valid[:, ti].to(DEV)
        errs.append(vrmse(model(x), valid[:, ti + 1].to(DEV)).item())
    one = sum(errs) / len(errs)
    # 8-step autoregressive rollout from strided starts
    rerrs = []
    for t0 in range(0, t - 1 - ROLLOUT_H, 16):
        x = valid[:, t0].to(DEV)
        for _ in range(ROLLOUT_H):
            x = model(x)
        rerrs.append(vrmse(x, valid[:, t0 + ROLLOUT_H].to(DEV)).item())
    model.train()
    return one, sum(rerrs) / len(rerrs)


def main():
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    train, valid, _, _ = load()
    model = build_model(VARIANT, in_ch=train.shape[2]).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    def td_loss(x, ys):
        """K-step pushforward: autoregress, accumulate per-step MSE."""
        total, pred = 0.0, x
        for y in ys:
            pred = model(pred)
            total = total + (pred - y).square().mean()
        return total / len(ys)

    it = batches(train, gen, ROLLOUT_K)
    # warmup step excluded from budget (compile/alloc)
    x, ys = next(it)
    loss = td_loss(x, ys)
    loss.backward(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # wave G addendum: qubit-scarcity pressure. GATE_PENALTY=lambda adds lambda * mean participation
    # (drive gate) to the loss, so every qubit an input uses has a price; 0 (default) = no pressure.
    gate_lambda = float(os.environ.get("GATE_PENALTY", "0"))
    gated_regs = [r for r in getattr(getattr(getattr(model, "mid", None), "core", None), "regs", [])
                  if getattr(r, "gated", False)]

    def gate_cost(x):
        if gate_lambda == 0.0 or not gated_regs:
            return 0.0
        h = model.pre(model.lift(x)).mean(dim=(2, 3))
        return gate_lambda * torch.stack([torch.sigmoid(r.gate_lin(h)).mean() for r in gated_regs]).mean()

    t0, steps, last = time.perf_counter(), 0, 0.0
    while time.perf_counter() - t0 < BUDGET_S and steps < MAX_STEPS:
        x, ys = next(it)
        loss = td_loss(x, ys) + gate_cost(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        steps += 1; last = loss.item()
    torch.cuda.synchronize()
    train_s = time.perf_counter() - t0

    v1, v8 = evaluate(model, valid)
    print("---")
    print(f"variant:        {VARIANT}")
    print(f"rollout_k:      {ROLLOUT_K}")
    print(f"gate_penalty:   {float(os.environ.get('GATE_PENALTY', '0'))}")
    print(f"val_vrmse_1:    {v1:.6f}")
    print(f"val_vrmse_8:    {v8:.6f}")
    print(f"train_loss:     {last:.6f}")
    print(f"num_steps:      {steps}")
    print(f"num_params:     {n_params}")
    print(f"train_seconds:  {train_s:.1f}")
    print(f"peak_vram_mb:   {torch.cuda.max_memory_allocated() / 2**20:.1f}")
    if os.environ.get("ARENA_SAVE"):                              # doc 35 §23: checkpoint + Figure 2 dump (middle valid traj.)
        import numpy as np
        d = pathlib.Path(os.environ["ARENA_SAVE"]); d.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, d / f"{VARIANT}_s{SEED}.pt")
        model.eval()
        with torch.no_grad():
            i, t0 = valid.shape[0] // 2, (valid.shape[1] - 1 - ROLLOUT_H) // 2
            x = valid[i:i + 1, t0].to(DEV)
            for _ in range(ROLLOUT_H):
                x = model(x)
            y = valid[i:i + 1, t0 + ROLLOUT_H].to(DEV)
            np.savez(d / f"{VARIANT}_s{SEED}_fig.npz", truth=y[0, 0].cpu().numpy(), pred=x[0, 0].cpu().numpy(),
                     vrmse8=float(vrmse(x, y).item()), traj=i, t0=t0)
        model.train()
    part = participation(model, valid)
    if part is not None:
        print(f"participation:  {part:.4f}")   # wave G: mean drive gate over validation inputs (1 = every qubit in use)


@torch.no_grad()
def participation(model, valid) -> float | None:
    """Mean per-input qubit participation of a gated register (None for ungated / classical cores)."""
    core = getattr(getattr(model, "mid", None), "core", None)
    if core is None or not hasattr(core, "participation"):
        return None
    if not any(getattr(r, "gated", False) for r in getattr(core, "regs", [])):
        return None
    model.eval()
    vals = []
    for ti in range(0, valid.shape[1] - 1, 8):
        x = valid[:, ti].to(DEV)
        h = model.pre(model.lift(x)).mean(dim=(2, 3))     # exactly the Bottleneck's pooled latent
        vals.append(core.participation(h).mean().item())
    model.train()
    return sum(vals) / len(vals)


if __name__ == "__main__":
    main()
