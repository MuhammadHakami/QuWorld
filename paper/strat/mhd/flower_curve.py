"""Doc 35 §37, MHD_64: fit the §34 Flowers VALID curve (runs/phase_b_mhd/flower_s1337.log) and extrapolate to epoch 100.

VALID here is the training script's fixed stride-2 VALID set (480 windows, per-channel mean VRMSE), logged about every
900 s. Points before the end of warm-up (epoch 5) are dropped. Three forms, each fit on (a) every logged point after
warm-up and (b) the running minimum:
  steps : v = a + b * s^-c                        (power law in optimizer steps)
  lrsum : v = a + b * S1^-c, S1 = sum of lr         (power law in cumulative learning rate; cosine decay slows S1)
  pure  : v = b * s^-c                              (no floor, log-log linear; the most optimistic)
Uncertainty: residual bootstrap (2,000 resamples) per fit; model spread across forms. The cosine annealing bonus at
the end of training is not identifiable from this part of the curve (lr has only fallen from 5.0e-4 to 4.5e-4).

    uv run --no-sync python paper/strat/mhd/flower_curve.py
"""
from __future__ import annotations

import json
import math
import pathlib
import re

import numpy as np
from scipy.optimize import curve_fit

ROOT = pathlib.Path(__file__).resolve().parents[3]
LOG = ROOT / "runs/phase_b_mhd/flower_s1337.log"
OUT = ROOT / "runs/strat/mhd/flower_curve.json"
PER_EPOCH, EPOCHS, LR, WARM = 231, 100, 5e-4, 5


def lr_table() -> np.ndarray:
    """Per-epoch lr exactly as train_flowers_mhd._lr_table (the installed the_well LinearWarmupCosineAnnealingLR)."""
    import torch
    from the_well.benchmark.optim.schedulers import LinearWarmupCosineAnnealingLR
    o = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=LR)
    sch = LinearWarmupCosineAnnealingLR(o, warmup_epochs=WARM, max_epochs=EPOCHS, warmup_start_lr=0.1 * LR, eta_min=0.1 * LR)
    out = []
    for _ in range(EPOCHS):
        out.append(o.param_groups[0]["lr"]); o.step(); sch.step()
    return np.array(out)


LRS = lr_table()


def s1(steps: np.ndarray) -> np.ndarray:
    cum = np.concatenate([[0.0], np.cumsum(LRS * PER_EPOCH)])
    e = np.minimum(steps // PER_EPOCH, EPOCHS - 1).astype(int)
    return cum[e] + (steps - e * PER_EPOCH) * LRS[e]


def parse() -> tuple[np.ndarray, np.ndarray]:
    pts = {}
    for line in LOG.read_text().splitlines():
        m = re.match(r"val_check step=(\d+) epoch=\d+ vrmse=([0-9.]+)", line)
        if m:
            pts[int(m.group(1))] = float(m.group(2))
        m = re.match(r"chunk done step=(\d+) valid=([0-9.]+)", line)
        if m:
            pts[int(m.group(1))] = float(m.group(2))
    s = np.array(sorted(pts)); return s, np.array([pts[k] for k in s])


FORMS = {
    "steps": (lambda s, a, b, c: a + b * s ** (-c), lambda s: s, (0.05, 5.0, 0.5)),
    "lrsum": (lambda s, a, b, c: a + b * s ** (-c), s1, (0.05, 0.5, 0.5)),
    "pure": (lambda s, b, c: b * s ** (-c), lambda s: s, (5.0, 0.3)),
}


def fit(form: str, s: np.ndarray, v: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    f, tx, p0 = FORMS[form]
    x, xe = tx(s.astype(float)), tx(np.array([EPOCHS * PER_EPOCH], float))
    bounds = ([0.0, 0.0, 0.0], [0.2, 1e4, 5.0]) if len(p0) == 3 else ([0.0, 0.0], [1e4, 5.0])
    p, _ = curve_fit(f, x, v, p0=p0, bounds=bounds, maxfev=20000)
    pred, res = f(x, *p), v - f(x, *p)
    rng = np.random.default_rng(seed); ext = []
    for _ in range(n_boot):
        vb = pred + rng.choice(res, len(res), replace=True)
        try:
            pb, _ = curve_fit(f, x, vb, p0=p, bounds=bounds, maxfev=20000); ext.append(float(f(xe, *pb)[0]))
        except RuntimeError:
            continue
    ext = np.array(ext)
    return {"params": p.tolist(), "rmse": float(np.sqrt(np.mean(res ** 2))), "epoch100": float(f(xe, *p)[0]),
            "epoch100_ci95": [float(np.quantile(ext, 0.025)), float(np.quantile(ext, 0.975))], "n_boot_ok": len(ext),
            "epoch50": float(f(tx(np.array([50 * PER_EPOCH], float)), *p)[0])}


def main() -> None:
    s, v = parse()
    keep = s >= WARM * PER_EPOCH
    s, v = s[keep], v[keep]
    runmin = np.minimum.accumulate(v)
    res = {"log": str(LOG.relative_to(ROOT)), "n_points": int(len(s)), "last_step": int(s[-1]),
           "last_epoch": float(s[-1] / PER_EPOCH), "best": float(v.min()), "target_flowers_paper_test": 0.1165,
           "lr_now": float(LRS[min(int(s[-1] // PER_EPOCH), EPOCHS - 1)]), "fits": {}}
    for form in FORMS:
        for name, y in (("all", v), ("runmin", runmin)):
            r = fit(form, s, y)
            res["fits"][f"{form}_{name}"] = r
            print(f"{form:6s} {name:7s} epoch100 {r['epoch100']:.4f} CI95 [{r['epoch100_ci95'][0]:.4f}, "
                  f"{r['epoch100_ci95'][1]:.4f}] epoch50 {r['epoch50']:.4f} rmse {r['rmse']:.4f} params "
                  f"{np.round(r['params'], 4).tolist()}", flush=True)
    # recent slope: VRMSE drop per 10 epochs over the last 10 epochs of logged points (least squares on log v)
    last = s >= s[-1] - 10 * PER_EPOCH
    sl = np.polyfit(s[last] / PER_EPOCH, np.log(v[last]), 1)[0]
    res["recent_log_slope_per_10_epochs"] = float(10 * sl)
    print(f"recent slope: {100 * (math.exp(10 * sl) - 1):+.1f}% per 10 epochs (last 10 epochs, all points)")
    OUT.write_text(json.dumps(res, indent=1)); print("wrote", OUT)


if __name__ == "__main__":
    main()
