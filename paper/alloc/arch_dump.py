"""Doc 35 / Appendix (app:arch): per-backbone module output shapes and parameter counts, traced from the code, for
the architecture appendix. CPU.   .venv/bin/python paper/alloc/arch_dump.py > runs/alloc/arch_dump.txt"""
from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def trace(name, model, *inputs, depth=1):
    rows = []
    def hook(n):
        def f(mod, i, o):
            o = o[0] if isinstance(o, (tuple, list)) else o
            rows.append((n, type(mod).__name__, tuple(o.shape) if torch.is_tensor(o) else "-", sum(p.numel() for p in mod.parameters())))
        return f
    hs = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n and n.count(".") < depth]
    with torch.no_grad():
        y = model(*inputs)
    for h in hs:
        h.remove()
    y = y[0] if isinstance(y, (tuple, list)) else y
    print(f"== {name}: params {sum(p.numel() for p in model.parameters()):,}; in {[tuple(i.shape) for i in inputs]} -> out {tuple(y.shape)}")
    for r in rows:
        print("   ", r)


def main() -> None:
    sys.path.insert(0, str(ROOT / "paper" / "phase_b"))
    from cnext_hybrid import load_official
    for tag, d in (("CNextU-net rayleigh_benard", "cnext_rb_ckpt"), ("CNextU-net shear_flow", "cnext_shear_flow_ckpt"),
                   ("CNextU-net MHD_64 (3-D)", "cnext_mhd_ckpt")):
        m, cfg = load_official(str(ROOT / "data" / d)); m.eval()
        trace(tag, m, torch.randn(1, cfg["dim_in"], *cfg["spatial_resolution"]))
    # FNO TRL2D arena: traced in .venv-well (neuralop 0.3.0): 19,011,716 params; lifting MLP 16->256->128, 4 FNO blocks
    # (128 ch, 16x16 modes, 1x1 skip), projection MLP 128->256->4.
    sys.path.insert(0, str(ROOT / "baseline" / "sophelio-starter"))
    from experiments_torch import UNetLite
    trace("Sophelio starter UNetLite", UNetLite(21).eval(), torch.randn(2, 21))
    sys.path.insert(0, str(ROOT / "paper" / "geothermal")); sys.path.insert(0, str(ROOT / "baseline" / "1HP_NN"))
    try:
        from processing.networks.unet import UNet
        trace("1HP heat plume U-Net", UNet(in_channels=4, out_channels=1, init_features=32, depth=3, kernel_size=5).eval(),
              torch.randn(1, 4, 256, 16), depth=2)
    except Exception as e:  # noqa: BLE001
        print("1HP U-Net:", repr(e)[:200])
    sys.path.insert(0, str(ROOT / "paper" / "sophelio"))
    import sota
    m = sota.Hybrid(427, 64, 2048, 2, 0.1, "none", None, 32, 4, None).eval()
    trace("Fusion MLP trunk (entry member config)", m, torch.randn(2, 427))


if __name__ == "__main__":
    main()
