"""Fine-tune (hybrid) CNextU-net from the official checkpoint — official protocol,
freeze-resilient edition.

Usage: uv run paper/phase_b/finetune.py <variant> [seed] [budget_s] [lr] [target_steps]
Variants: vanilla | mlp30 | mr30hea | mr30ryd

Device-safety measures (the GB10 latched off twice under the streaming loader):
- single-process in-RAM data (data_ram.py, protocol-verified vs the_well)
- CUDA allocator capped at 60% of unified memory
- gc + empty_cache after every validation
- atomic, fsync'd checkpoints every VAL_EVERY_S (loses <=3 min to any kill)
- optional THROTTLE_MS env to shave sustained GPU load (board latch-off is
  load-accelerated)
- resumable: model+optimizer+step in *_latest.pt; prints `target_reached`
  sentinel when the step target is met. Test set is NEVER touched here.
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

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "vanilla"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1337
BUDGET_S = float(sys.argv[3]) if len(sys.argv) > 3 else 1500.0
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
TARGET_STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
BATCH = 16
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
        mse = (pred - yd).square().mean(dim=(2, 3))
        var = (yd - yd.mean(dim=(2, 3), keepdim=True)).square().mean(dim=(2, 3))
        errs.append((mse / (var + 1e-7)).sqrt().mean(dim=1).mean().item())
    model.train()
    del pred, yd
    gc.collect()
    torch.cuda.empty_cache()
    return sum(errs) / len(errs)


def main() -> None:
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    if os.environ.get("QW_GPU_GB"):                   # Sep 23: explicit GPU budget (paper/gpucap.py)
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); import gpucap; gpucap.cap_from_env()
    else:
        torch.cuda.set_per_process_memory_fraction(0.6)
    gen = torch.Generator().manual_seed(SEED)

    train, valid = data_ram.load()
    model = build_hybrid(VARIANT).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    out = pathlib.Path(os.environ.get("PHASEB_OUT", ROOT / "runs" / "phase_b"))   # pilots write elsewhere
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
