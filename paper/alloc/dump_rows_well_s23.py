"""Figure 2 rows for the Well benchmarks with the doc 35 §23 models: gated fine-tuned classical model, the same model with
the matched 3-D MLP and with the 3-D tethered register in the neck, on one TEST window fixed in advance (the middle one).
Writes paper/fig_cache/rows_s23/<dataset>.npz in the uniform row format read by fig_outputs_all.py.

    QW_GPU_GB=12 runs/capped.sh rows 40 .venv/bin/python paper/alloc/dump_rows_well_s23.py shear_flow rayleigh_benard ...
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import finetune_gated as FG  # noqa: E402
from ft_ensemble import load  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/fig_cache/rows_s23"
ARMS = (("classical", "vanilla"), ("mlp", "q30-geo=3d-mlp"), ("qubits", "q30-geo=3d"))
LABEL = {"shear_flow": "shear flow", "rayleigh_benard": "convection", "turbulent_radiative_layer_2D": "TRL2D"}


def tag(ds: str, v: str) -> str:
    for t in (f"{ds}_{v}_vrmse_gate_s1_pre", f"{ds}_{v}_vrmse_gate_s1"):
        if (FG.OUT / f"{t}.pt").exists():
            return t
    raise FileNotFoundError(f"no gated model for {ds} {v}")


@torch.no_grad()
def row(ds: str) -> None:
    out = {"label": LABEL[ds], "metric": "VRMSE", "kind": "field"}
    for name, v in ARMS:
        model, (_, test) = load(ds, tag(ds, v))
        pairs = FG.well2d.all_test_windows(test); w = len(pairs) // 2
        x, y = next(iter(FG.well2d.batched(test, [pairs[w]], 1)))
        p = model(x.to(FG.DEV))
        out[f"pred_{name}"] = p[0, 0].float().cpu().numpy(); out["truth"] = y[0, 0].float().cpu().numpy()
        out[f"m_{name}"] = float(FG.vrmse(p, y.to(FG.DEV)).mean()); out["sample"] = f"test window {w}"
        del model; torch.cuda.empty_cache()
        print(ds, name, round(out[f"m_{name}"], 5), flush=True)
    OUT.mkdir(parents=True, exist_ok=True); np.savez(OUT / f"{ds}.npz", **out)


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        row(ds)
