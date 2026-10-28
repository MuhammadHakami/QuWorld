"""Heat-plume (1HP) outputs for Figure 2: temperature (deg C) on the middle TEST sample, fixed in advance, from the
pretrained U-Net (classical) and the same model with the matched MLP / 30-atom register in its bottleneck (seed 1337).

    QW_GPU_GB=8 runs/capped.sh f1hp 16 .venv/bin/python paper/geothermal/dump_fields_1hp.py
"""
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_pelzer_unet as T  # noqa: E402

import os  # noqa: E402
S23 = bool(os.environ.get("S23"))                                   # doc 35 §23 set: fine-tuned classical, 3-D MLP, 3-D qubits
CKPT = ({"vanilla": T.OUT / "vanilla_s1337.pt", "mlp30": T.OUT / "q30-geo=3d-mlp_s1337.pt", "mr30ryd": T.OUT / "q30-geo=3d_s1337.pt"}
        if S23 else
        {"vanilla": T.OUT / "pretrain_lr0.001_s1337.pt", "mlp30": T.OUT / "mlp30_s1337.pt", "mr30ryd": T.OUT / "mr30ryd_s1337.pt"})
BUILD = {"vanilla": "vanilla", "mlp30": "q30-geo=3d-mlp" if S23 else "mlp30", "mr30ryd": "q30-geo=3d" if S23 else "mr30ryd"}


@torch.no_grad()
def main() -> None:
    x, y, info, tr, va, te = T.load_1000()
    i = int(te[len(te) // 2])
    out = {}
    for v, ck in CKPT.items():
        model = T.build(BUILD[v]).to(T.DEV)
        model.load_state_dict(torch.load(ck, weights_only=True, map_location=T.DEV)["model_state_dict"]); model.eval()
        p = model(x[i:i + 1].to(T.DEV)).cpu()
        pc, tc = T.D.denormalize_temperature(p, info), T.D.denormalize_temperature(y[i:i + 1], info)
        out[f"{v}_pred"] = pc[0, 0].numpy(); out["truth"] = tc[0, 0].numpy()
        out[f"{v}_mae"] = float((pc - tc).abs().mean())
        print(v, "MAE degC", round(out[f"{v}_mae"], 5))
    dst = HERE.parents[0] / "fig_cache" / ("fig_fields_1hp_s23.npz" if S23 else "fig_fields_1hp.npz")
    np.savez(dst, sample=i, **out); print("wrote", dst)


if __name__ == "__main__":
    main()
