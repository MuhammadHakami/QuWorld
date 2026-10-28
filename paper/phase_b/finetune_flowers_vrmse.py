"""Doc 35 §37i: metric-aligned fine-tune of the final Flowers MHD_64 model (TRAIN fits, VALID selects, TEST once later).

Starts from runs/phase_b_mhd/flower_s1337.pt (100 epochs, MSE on normalised fields). The benchmark scores the per-sample,
per-channel VRMSE, which MSE weights differently (by global variance, not each window's own), so the loss here is the
metric itself: mean over windows and channels of sqrt(mse_c / var_c). Same data, windows, batch (4 x 8) and optimiser
(AdamW, wd 1e-2) as train_flowers_mhd.py; constant lr 5e-5 (the schedule's final rate); EPOCHS epochs; the stride-2
VALID score of train_flowers_mhd.validate every 900 s keeps the best checkpoint (runs/phase_b_mhd/flower_vft_s1337.pt).

    QW_GPU_GB=40 runs/capped.sh s37i 60 uv run --no-sync python paper/phase_b/finetune_flowers_vrmse.py <epochs> <budget_s>
"""
from __future__ import annotations

import pathlib
import sys
import time

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
BUDGET_S = float(sys.argv[2]) if len(sys.argv) > 2 else 2400.0
sys.argv = [sys.argv[0], "1337", "1", "100"]                      # train_flowers_mhd parses argv at import
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

import train_flowers_mhd as TF  # noqa: E402

LR = 5e-5
OUT = TF.ROOT / "runs" / "phase_b_mhd"


def vrmse_loss(pred, y):
    mse = (pred - y).square().mean(dim=(2, 3, 4))
    var = (y - y.mean(dim=(2, 3, 4), keepdim=True)).square().mean(dim=(2, 3, 4))
    return (mse / (var + 1e-7)).clamp_min(1e-12).sqrt().mean()


def main() -> None:
    torch.manual_seed(TF.SEED + 7)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(float(TF.os.environ.get("QW_MEM_FRAC", "0.35")))
    train, valid = TF.data_ram.load()
    per_epoch = (train.shape[0] * (train.shape[1] - TF.data_ram.N_HIST)) // (TF.MICRO * TF.ACCUM)
    target = EPOCHS * per_epoch
    model = TF.Flower(dim_in=28, dim_out=7, n_spatial_dims=3, spatial_resolution=[64, 64, 64],
                      boundary_condition_types=["PERIODIC"]).to(TF.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=TF.WD)
    ckpt, latest = OUT / "flower_vft_s1337.pt", OUT / "flower_vft_s1337_latest.pt"
    if latest.exists():
        st = torch.load(latest, weights_only=True, map_location=TF.DEV)
        model.load_state_dict(st["model_state_dict"]); opt.load_state_dict(st["optimizer_state_dict"])
        steps, best = st["step"], st["best"]
        print(f"val_check step={steps} vrmse={best:.4f} (RESUMED)", flush=True)
    else:
        st = torch.load(OUT / "flower_s1337.pt", weights_only=True, map_location=TF.DEV)
        model.load_state_dict(st["model_state_dict"]); steps = 0
        best = TF.validate(model, valid)
        print(f"val_check step=0 vrmse={best:.5f} (start = final Flowers, MSE-trained)", flush=True)
        TF.atomic_save({"model_state_dict": model.state_dict(), "val": best, "step": 0}, ckpt)
    if steps >= target:
        print(f"target_reached steps={steps} best={best:.6f}", flush=True); return
    t0 = last_val = time.perf_counter()
    while time.perf_counter() - t0 < BUDGET_S and steps < target:
        epoch, k = divmod(steps, per_epoch)
        order = TF.epoch_order(train.shape[0], train.shape[1], 1000 + epoch)
        for a_ in range(TF.ACCUM):
            j = (k * TF.ACCUM + a_) * TF.MICRO
            x, y = TF.batch(train, order[j:j + TF.MICRO])
            loss = vrmse_loss(model(x), y) / TF.ACCUM
            loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True); steps += 1
        if time.perf_counter() - last_val > TF.VAL_EVERY_S or steps == target:
            v = TF.validate(model, valid)
            print(f"val_check step={steps} epoch={steps / per_epoch:.2f} vrmse={v:.5f} loss={TF.ACCUM * loss.item():.5f}", flush=True)
            if v < best:
                best = v; TF.atomic_save({"model_state_dict": model.state_dict(), "val": best, "step": steps}, ckpt)
            TF.atomic_save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                            "step": steps, "best": best}, latest)
            last_val = time.perf_counter()
    TF.atomic_save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                    "step": steps, "best": best}, latest)
    if steps >= target:
        print(f"target_reached steps={steps} best={best:.6f}", flush=True)
    print(f"chunk done step={steps} best={best:.6f} secs={time.perf_counter() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
