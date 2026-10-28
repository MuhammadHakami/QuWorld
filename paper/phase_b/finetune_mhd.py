"""Fine-tune (hybrid) CNextU-net on The Well MHD_64 from the official MHD
checkpoint — checkpoint-surgery port of finetune.py (TRL2D), freeze-resilient.

Faithful copy of finetune.py with the three touches documented in
README_mhd.md ("Port status") plus two campaign deltas:

1. data:     ``import data_ram`` -> ``import data_ram_mhd as data_ram``
             (same API: load() / train_batch() / valid_windows();
             memmap-backed, copy-on-write — see data_ram_mhd docstring).
2. model:    ``build_hybrid(VARIANT)`` ->
             ``build_hybrid(VARIANT, model_dir=data/cnext_mhd_ckpt)``
             (the default model_dir points at the TRL2D checkpoint).
3. validate: VRMSE reduces over ``dim=(2, 3, 4)`` (3D spatial; was (2, 3)).
4. output:   checkpoints under runs/phase_b_mhd/{variant}_s{seed}[.pt|_latest.pt]
             (TRL2D campaign writes to runs/phase_b/ — kept separate).
5. BATCH=8   (was 16). Memory estimate: TRL2D measured ~17.1 GB peak VRAM at
             batch 16 over 128x384 = 49,152 px/sample; MHD_64 is 64^3 =
             262,144 voxels/sample (5.33x) with the identical width schedule
             (init_features=42, stages=4, blocks_per_stage=2), so batch 16
             extrapolates to ~90 GB — over the 60% allocator cap
             (~72.6 GiB of 121 GiB unified) and it would squeeze the 53 GiB
             train memmap's page cache. Batch 8 -> ~45 GB predicted peak.

Usage: uv run paper/phase_b/finetune_mhd.py <variant> [seed] [budget_s] [lr] [target_steps]
Variants: vanilla | mlp30 | mr30hea | mr30ryd

Device-safety measures are inherited unchanged from finetune.py (single-process
in-RAM/memmap data, 60% CUDA allocator cap, gc + empty_cache after validation,
atomic fsync'd checkpoints, optional THROTTLE_MS, resumable via *_latest.pt,
``target_reached`` sentinel). Test set is NEVER touched here.
"""

import gc
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import data_ram_mhd as data_ram  # noqa: E402
from cnext_hybrid import ROOT, build_hybrid  # noqa: E402

MHD_CKPT = str(ROOT / "data" / "cnext_mhd_ckpt")

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "vanilla"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1337
BUDGET_S = float(sys.argv[3]) if len(sys.argv) > 3 else 1500.0
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
TARGET_STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
BATCH = 8  # was 16 on TRL2D — see docstring delta 5 for the 3D memory estimate
VAL_EVERY_S = 180.0
THROTTLE_S = float(os.environ.get("THROTTLE_MS", "0")) / 1000.0
DEV = "cuda"


def free_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return -1.0


def atomic_save(obj, path: pathlib.Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


@torch.no_grad()
def validate(model, valid, stride_t: int = 2) -> float:
    model.eval()
    errs = []
    for x, y in data_ram.valid_windows(valid, stride_t):
        pred = model(x.to(DEV))
        yd = y.to(DEV)
        mse = (pred - yd).square().mean(dim=(2, 3, 4))
        var = (yd - yd.mean(dim=(2, 3, 4), keepdim=True)).square().mean(dim=(2, 3, 4))
        errs.append((mse / (var + 1e-7)).sqrt().mean(dim=1).mean().item())
    model.train()
    del pred, yd
    gc.collect()
    torch.cuda.empty_cache()
    return sum(errs) / len(errs)


def main() -> None:
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.45)
    gen = torch.Generator().manual_seed(SEED)

    train, valid = data_ram.load()
    model = build_hybrid(VARIANT, model_dir=MHD_CKPT).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    out = ROOT / "runs" / "phase_b_mhd"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{VARIANT}_s{SEED}.pt"           # best-on-valid (weights only)
    latest = out / f"{VARIANT}_s{SEED}_latest.pt"  # crash-resume (weights+opt+step)

    steps = 0
    if latest.exists():
        st = torch.load(latest, weights_only=True, map_location=DEV)
        model.load_state_dict(st["model_state_dict"])
        opt.load_state_dict(st["optimizer_state_dict"])
        steps, best = st["step"], st["best"]
        print(f"val_check step={steps} vrmse={best:.4f} free={free_gb():.0f}G "
              f"(RESUMED)", flush=True)
    else:
        best = validate(model, valid)
        print(f"val_check step=0 vrmse={best:.4f} free={free_gb():.0f}G "
              f"(checkpoint start)", flush=True)
        atomic_save({"model_state_dict": model.state_dict(), "val": best,
                     "step": 0}, ckpt)

    if steps >= TARGET_STEPS:
        print(f"target_reached steps={steps} best={best:.6f}", flush=True)

    t0, last_val = time.perf_counter(), time.perf_counter()
    while time.perf_counter() - t0 < BUDGET_S and steps < TARGET_STEPS:
        x, y = data_ram.train_batch(train, BATCH, gen, DEV)
        loss = (model(x) - y).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        steps += 1
        if THROTTLE_S:
            torch.cuda.synchronize()
            time.sleep(THROTTLE_S)
        if time.perf_counter() - last_val > VAL_EVERY_S:
            v = validate(model, valid)
            print(f"val_check step={steps} vrmse={v:.4f} loss={loss.item():.5f} "
                  f"free={free_gb():.0f}G", flush=True)
            if v < best:
                best = v
                atomic_save({"model_state_dict": model.state_dict(), "val": best,
                             "step": steps}, ckpt)
            atomic_save({"model_state_dict": model.state_dict(),
                         "optimizer_state_dict": opt.state_dict(),
                         "step": steps, "best": best}, latest)
            last_val = time.perf_counter()

    if steps >= TARGET_STEPS:
        print(f"target_reached steps={steps} best={best:.6f}", flush=True)

    v = validate(model, valid, stride_t=1)
    if v < best:
        best = v
        atomic_save({"model_state_dict": model.state_dict(), "val": best,
                     "step": steps}, ckpt)
    atomic_save({"model_state_dict": model.state_dict(),
                 "optimizer_state_dict": opt.state_dict(),
                 "step": steps, "best": best}, latest)
    print("---")
    print(f"variant:        {VARIANT}")
    print(f"val_vrmse_1:    {best:.6f}")
    print(f"final_val:      {v:.6f}")
    print(f"num_steps:      {steps}")
    print(f"num_params:     {n_params}")
    print(f"train_seconds:  {time.perf_counter() - t0:.1f}")
    print(f"peak_vram_mb:   {torch.cuda.max_memory_allocated() / 2**20:.1f}")
    print(f"checkpoint:     {ckpt}")


if __name__ == "__main__":
    main()
