"""Doc 35 §34: Flowers (Muser et al., arXiv:2603.04430; code github.com/t-muser/flowers, MIT, commit f1d2cad, vendored
at baseline/flowers) trained from scratch on The Well MHD_64 with the authors' 4->1 recipe, as a stronger classical
part for the routed co-processor. Same data pipeline, normalisation and validation VRMSE as our CNextU-net MHD campaign
(finetune_mhd.py / data_ram_mhd.py); TEST is never touched here.

Recipe (configs/train_4-to-1.yaml, configs/models/flower.yaml, flowers/trainer.py): Flower(lifting_dim 160, n_levels 4,
heads 40, groups 40), periodic boundaries, AdamW lr 5e-4 (§34c: the authors' MHD_64 rate; repo default 1e-3), weight decay 1e-2, no gradient clipping, learning rate set once
per EPOCH from the installed the_well LinearWarmupCosineAnnealingLR (5 warm-up epochs from 0.1 lr, cosine to 0.1 lr), MSE on normalised fields,
batch 32 (4 x 8 accumulation on one GB10), each epoch a shuffle WITHOUT replacement of the 77 TRAIN trajectories x 96
windows with the last partial batch dropped (231 steps). Deviations kept and documented (doc 35 §34a): normalisation
from the official stats.yaml (4 significant figures; Flowers' config has more digits); best checkpoint on our fixed
stride-2 VALID windows every 900 s instead of short shuffled VALID batches per epoch.

    uv run --no-sync python paper/phase_b/train_flowers_mhd.py <seed> <budget_s> <epochs> [tag]
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "baseline" / "flowers"))
import data_ram_mhd as data_ram  # noqa: E402
_argv, sys.argv = sys.argv, sys.argv[:1]           # finetune_mhd parses sys.argv at import time
from finetune_mhd import atomic_save, free_gb  # noqa: E402
sys.argv = _argv
from flower_standalone import Flower  # noqa: E402

ROOT = HERE.parents[1]
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1337
BUDGET_S = float(sys.argv[2]) if len(sys.argv) > 2 else 2400.0
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
TAG = sys.argv[4] if len(sys.argv) > 4 else ""
MICRO, ACCUM, LR, WD, WARM = 4, 8, 5e-4, 1e-2, 5   # §34c: authors' MHD_64 rate (arXiv:2603.04430 Table 5)
VAL_EVERY_S = 900.0
THROTTLE_S = float(os.environ.get("THROTTLE_MS", "0")) / 1000.0
DEV = "cuda"


def _lr_table() -> list[float]:
    """Per-epoch LR from the installed the_well LinearWarmupCosineAnnealingLR, configured as flowers/train.py:203-205
    (warmup_start_lr = eta_min = 0.1 lr, max_epochs = EPOCHS), so the schedule is the upstream one exactly."""
    from the_well.benchmark.optim.schedulers import LinearWarmupCosineAnnealingLR
    o = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=LR)
    sch = LinearWarmupCosineAnnealingLR(o, warmup_epochs=WARM, max_epochs=EPOCHS, warmup_start_lr=0.1 * LR, eta_min=0.1 * LR)
    out = []
    for _ in range(EPOCHS):
        out.append(o.param_groups[0]["lr"]); o.step(); sch.step()
    return out


LR_TABLE = _lr_table()


def lr_at(epoch: int) -> float:
    return LR_TABLE[min(epoch, EPOCHS - 1)]


@torch.no_grad()
def validate(model, valid, stride_t: int = 2, chunk: int = 5) -> float:
    """finetune_mhd.validate's metric (mean over VALID windows of the per-window field-mean VRMSE), in chunks of
    `chunk` trajectories to bound 3-D activation memory; equal-size chunks leave the mean unchanged."""
    model.eval(); errs = []
    for x, y in data_ram.valid_windows(valid, stride_t):
        for i in range(0, len(x), chunk):
            pred, yd = model(x[i:i + chunk].to(DEV)), y[i:i + chunk].to(DEV)
            mse = (pred - yd).square().mean(dim=(2, 3, 4))
            var = (yd - yd.mean(dim=(2, 3, 4), keepdim=True)).square().mean(dim=(2, 3, 4))
            errs.append((mse / (var + 1e-7)).sqrt().mean(dim=1))
    model.train(); torch.cuda.empty_cache()
    return torch.cat(errs).mean().item()


def epoch_order(n: int, t: int, epoch: int) -> torch.Tensor:
    """All (trajectory, start) windows, shuffled without replacement for this epoch (fixed by seed and epoch)."""
    g = torch.Generator().manual_seed(SEED * 1000 + epoch)
    ti = torch.arange(data_ram.N_HIST - 1, t - 1)
    pairs = torch.stack(torch.meshgrid(torch.arange(n), ti, indexing="ij"), -1).reshape(-1, 2)
    return pairs[torch.randperm(len(pairs), generator=g)]


def batch(data, pairs: torch.Tensor):
    ni, ti = pairs[:, 0], pairs[:, 1]
    if isinstance(data, data_ram.PreadSplit):         # pread-backed split: read the windows frame by frame
        fr = [data.frames(int(n), range(int(t) - data_ram.N_HIST + 1, int(t) + 2)) for n, t in zip(ni, ti)]
        x = torch.stack([f[:data_ram.N_HIST] for f in fr]); y = torch.stack([f[data_ram.N_HIST] for f in fr])
        return x.reshape(len(pairs), -1, *x.shape[-3:]).to(DEV), y.to(DEV)
    idx = ti.view(-1, 1) + torch.arange(-data_ram.N_HIST + 1, 1)
    x = data[ni.view(-1, 1), idx]                                    # (B, 4, 7, 64, 64, 64)
    return x.reshape(len(pairs), -1, *x.shape[-3:]).to(DEV), data[ni, ti + 1].to(DEV)


def main() -> None:
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("QW_MEM_FRAC", "0.35")))
    train, valid = data_ram.load()
    per_epoch = (train.shape[0] * (train.shape[1] - data_ram.N_HIST)) // (MICRO * ACCUM)     # drop_last: 231
    TARGET = EPOCHS * per_epoch
    model = Flower(dim_in=28, dim_out=7, n_spatial_dims=3, spatial_resolution=[64, 64, 64],
                   boundary_condition_types=["PERIODIC"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    out = ROOT / "runs" / "phase_b_mhd"; out.mkdir(parents=True, exist_ok=True)
    ckpt, latest = out / f"flower{TAG}_s{SEED}.pt", out / f"flower{TAG}_s{SEED}_latest.pt"
    steps, best = 0, float("inf")
    if latest.exists():
        st = torch.load(latest, weights_only=True, map_location=DEV)
        model.load_state_dict(st["model_state_dict"]); opt.load_state_dict(st["optimizer_state_dict"])
        steps, best = st["step"], st["best"]
        print(f"val_check step={steps} vrmse={best:.4f} free={free_gb():.0f}G (RESUMED)", flush=True)
    if steps >= TARGET:
        print(f"target_reached steps={steps} best={best:.6f}", flush=True); return
    t0 = last_val = time.perf_counter()
    while time.perf_counter() - t0 < BUDGET_S and steps < TARGET:
        epoch, k = divmod(steps, per_epoch)
        order = epoch_order(train.shape[0], train.shape[1], epoch)
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch)
        for a_ in range(ACCUM):
            j = (k * ACCUM + a_) * MICRO
            x, y = batch(train, order[j:j + MICRO])
            loss = (model(x) - y).square().mean() / ACCUM
            loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True); steps += 1                   # no clipping (authors' trainer)
        if THROTTLE_S:
            torch.cuda.synchronize(); time.sleep(THROTTLE_S)
        if time.perf_counter() - last_val > VAL_EVERY_S:
            v = validate(model, valid)
            print(f"val_check step={steps} epoch={steps // per_epoch} vrmse={v:.4f} loss={ACCUM * loss.item():.5f} "
                  f"lr={lr_at(steps // per_epoch):.2e} "
                  f"free={free_gb():.0f}G", flush=True)
            if v < best:
                best = v; atomic_save({"model_state_dict": model.state_dict(), "val": best, "step": steps}, ckpt)
            atomic_save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                         "step": steps, "best": best}, latest)
            last_val = time.perf_counter()
    v = validate(model, valid)
    if v < best:
        best = v; atomic_save({"model_state_dict": model.state_dict(), "val": best, "step": steps}, ckpt)
    atomic_save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                 "step": steps, "best": best}, latest)
    if steps >= TARGET:
        print(f"target_reached steps={steps} best={best:.6f}", flush=True)
    print(f"chunk done step={steps} valid={v:.6f} best={best:.6f} peak_vram_mb="
          f"{torch.cuda.max_memory_allocated() / 2**20:.0f} secs={time.perf_counter() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
