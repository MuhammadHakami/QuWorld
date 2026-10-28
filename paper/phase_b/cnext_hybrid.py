"""Quantum bottleneck inside the official CNextU-net checkpoint.

Wraps `model.neck` (Stage at 672 ch, 8x24 spatial) with a zero-init gated
bottleneck: pool -> core -> residual. With alpha=0 at init, the hybrid is
numerically identical to the released checkpoint — fine-tuning comparisons
start from exactly the published model.

Variants: vanilla | mlp30 | mr30hea | mr30ryd  (cores from paper/qlayer,
parameter-matched at the core level, same as the sprint loop).
"""

import json
import pathlib
import sys

import torch
import torch.nn as nn

_PAPER = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PAPER))
from model import Bottleneck, MLPCore, _count  # noqa: E402
from qlayer import MultiRegister  # noqa: E402

ROOT = _PAPER.parent


class NeckWithBottleneck(nn.Module):
    def __init__(self, neck: nn.Module, bottleneck: nn.Module):
        super().__init__()
        self.neck = neck
        self.bottleneck = bottleneck

    def forward(self, x):
        return self.bottleneck(self.neck(x))


def load_official(model_dir: str):
    from safetensors.torch import load_file
    from the_well.benchmark.models import UNetConvNext

    cfg = json.loads(open(f"{model_dir}/config.json").read())
    model = UNetConvNext(**cfg)
    state = load_file(f"{model_dir}/model.safetensors")
    missing, _ = model.load_state_dict(state, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    return model, cfg


def build_hybrid(variant: str, model_dir: str = str(ROOT / "data/cnext_ckpt"),
                 q_layers: int = 4):
    model, cfg = load_official(model_dir)
    if variant == "vanilla":
        return model
    if variant.startswith("srr-"):
        # Spectral-Routed Register (paper/spectral/srr.py): srr-<core>[-rand], core in lin|mlp|ryd|ryd2d|gutz
        sys.path.insert(0, str(_PAPER / "spectral"))
        from srr import SRR
        parts = variant.split("-")
        return SRR(model, C=4, H=128, W=384, n_hist=4, core=parts[1], route="rand" if "rand" in parts[2:] else "trunk",
                   layers=q_layers)
    ch = cfg["init_features"] * 2 ** cfg["stages"]          # 672
    if variant == "mlp30":
        ref = MultiRegister(ch, 10, 3, q_layers, "hea")
        core, out = MLPCore(ch, 30, _count(ref)), 30
    elif variant in ("mr30hea", "mr30ryd", "gutz30"):
        # gutz30: Gutzwiller control — same registers, mean-field propagation
        ansatz = {"mr30hea": "hea", "mr30ryd": "rydberg",
                  "gutz30": "rydberg_mf"}[variant]
        core, out = MultiRegister(ch, 10, 3, q_layers, ansatz), 30
    elif variant.startswith("q"):                             # wave-G grammar, incl. doc 35 §23 'q30-geo=3d[-mlp]'
        from model import build_config_core, parse_config
        core, out = build_config_core(parse_config(variant), ch, q_layers)
    else:
        raise ValueError(f"unknown variant {variant}")
    model.neck = NeckWithBottleneck(model.neck, Bottleneck(ch, core, out))
    return model


def load_hybrid_checkpoint(variant: str, ckpt_path: str):
    model = build_hybrid(variant)
    state = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state
                          else state)
    return model


if __name__ == "__main__":
    for v in ["vanilla", "mlp30", "mr30hea", "mr30ryd"]:
        m = build_hybrid(v)
        p = sum(q.numel() for q in m.parameters())
        with torch.no_grad():
            y = m(torch.randn(1, 16, 128, 384))
        print(f"{v:8s} params={p:,}  out={tuple(y.shape)}")
