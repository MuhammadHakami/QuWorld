"""Fine-tune (hybrid) TFNO on a 2-D Well dataset from the official checkpoint —
port of finetune_mhd.py (same loop, AdamW lr 1e-4 wd 0.01, grad-clip 1.0,
best-on-valid checkpoint, crash-resume, fixed step target) with:
  data:  well2d.Split (direct HDF5 pread, no cache copy)
  model: well2d_backbone.build(variant, dataset, BACKBONE)  (BACKBONE=cnext default | tfno)
  valid: fixed deterministic subset (VAL_PER_TRAJ windows per valid trajectory)
  output: runs/well2d/<dataset>/<backbone>/{variant}_s{seed}[.pt|_latest.pt]
Usage: uv run paper/phase_b/finetune_well2d.py <dataset> <variant> [seed] [budget_s] [lr] [target_steps]
Env: BATCH (default 4), VAL_PER_TRAJ (default 8), VAL_EVERY_S (default 300), MEM_FRACTION (default 0.3)
Test set is NEVER touched here (eval_well2d.py does single-shot test evaluation).
"""

import gc
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import well2d  # noqa: E402
from well2d import vrmse_rows  # noqa: E402
from well2d_backbone import ROOT, build  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else "planetswe"
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "vanilla"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1337
BUDGET_S = float(sys.argv[4]) if len(sys.argv) > 4 else 3600.0
LR = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-4
TARGET_STEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 10**9
BATCH = int(os.environ.get("BATCH", "4"))
VAL_PER_TRAJ = int(os.environ.get("VAL_PER_TRAJ", "8"))
VAL_EVERY_S = float(os.environ.get("VAL_EVERY_S", "300"))
MEM_FRACTION = float(os.environ.get("MEM_FRACTION", "0.3"))
BACKBONE = os.environ.get("BACKBONE", "cnext")
DEV = "cuda"


def free_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return -1.0


def atomic_save(obj, path: pathlib.Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


@torch.no_grad()
def validate(model, valid, pairs, bsz: int = 8) -> float:
    model.eval(); scores = []
    for x, y in well2d.batched(valid, pairs, bsz):
        scores.append(vrmse_rows(model(x.to(DEV)), y.to(DEV)).cpu())
    model.train(); gc.collect(); torch.cuda.empty_cache()
    return torch.cat(scores).mean().item()


def main() -> None:
    torch.manual_seed(SEED); torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    gen = torch.Generator().manual_seed(SEED)
    train, valid = well2d.Split(DATASET, "train"), well2d.Split(DATASET, "valid")
    pairs = well2d.fixed_windows(valid, VAL_PER_TRAJ)
    model = build(VARIANT, DATASET, BACKBONE).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    out = ROOT / "runs" / "well2d" / DATASET / BACKBONE; out.mkdir(parents=True, exist_ok=True)
    ckpt, latest = out / f"{VARIANT}_s{SEED}.pt", out / f"{VARIANT}_s{SEED}_latest.pt"
    print(f"{DATASET} {BACKBONE} {VARIANT} s{SEED}: train {train.shape} valid {valid.shape} val-windows {len(pairs)} "
          f"batch {BATCH} lr {LR} target {TARGET_STEPS} params {n_params}", flush=True)

    steps = 0
    if latest.exists():
        st = torch.load(latest, weights_only=True, map_location=DEV)
        model.load_state_dict(st["model_state_dict"]); opt.load_state_dict(st["optimizer_state_dict"])
        steps, best = st["step"], st["best"]
        print(f"val_check step={steps} vrmse={best:.4f} free={free_gb():.0f}G (RESUMED)", flush=True)
    else:
        best = validate(model, valid, pairs)
        print(f"val_check step=0 vrmse={best:.4f} free={free_gb():.0f}G (checkpoint start)", flush=True)
        atomic_save({"model_state_dict": model.state_dict(), "val": best, "step": 0}, ckpt)

    t0, last_val, last = time.perf_counter(), time.perf_counter(), None
    while time.perf_counter() - t0 < BUDGET_S and steps < TARGET_STEPS:
        x, y = well2d.train_batch(train, BATCH, gen, DEV)
        loss = (model(x) - y).square().mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); steps += 1; last = loss.item()
        if time.perf_counter() - last_val > VAL_EVERY_S:
            v = validate(model, valid, pairs)
            print(f"val_check step={steps} vrmse={v:.4f} loss={last:.5f} free={free_gb():.0f}G "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
            if v < best:
                best = v; atomic_save({"model_state_dict": model.state_dict(), "val": best, "step": steps}, ckpt)
            atomic_save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                         "step": steps, "best": best}, latest)
            last_val = time.perf_counter()
    v = validate(model, valid, pairs)
    if v < best:
        best = v; atomic_save({"model_state_dict": model.state_dict(), "val": best, "step": steps}, ckpt)
    atomic_save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                 "step": steps, "best": best}, latest)
    if steps >= TARGET_STEPS:
        print(f"target_reached steps={steps} best={best:.6f}", flush=True)
    print("---")
    for k, val in (("dataset", DATASET), ("backbone", BACKBONE), ("variant", VARIANT), ("seed", SEED), ("val_vrmse_1", f"{best:.6f}"),
                   ("final_val", f"{v:.6f}"), ("num_steps", steps), ("num_params", n_params),
                   ("train_seconds", f"{time.perf_counter() - t0:.1f}"),
                   ("peak_vram_mb", f"{torch.cuda.max_memory_allocated() / 2**20:.1f}"), ("checkpoint", ckpt)):
        print(f"{k}:{' ' * max(1, 16 - len(k))}{val}")
    train.close(); valid.close()


if __name__ == "__main__":
    main()
