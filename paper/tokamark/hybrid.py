"""Quantum bottleneck inside the TokaMark CNN baseline (UKAEA/IBM/STFC `tokamark_baseline`
MultiConv_MLP): the multi-branch CNN encoders feed `encoder_mlp` whose output is the
context vector (B, D*bb_factor) consumed by the decoder MLP + CNN decoders. We wrap
`encoder_mlp` with the zero-init gated bottleneck on that vector (pool-free: the
Bottleneck accepts (B, ch) latents) — identical to the baseline at init.
Variants: vanilla | mlp30 | mr30ryd (cores parameter-matched, as in every other benchmark).
"""
import pathlib
import sys

import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))
from model import Bottleneck, MLPCore, _count  # noqa: E402
from qlayer import MultiRegister  # noqa: E402


def wrap(model: nn.Module, variant: str, q_layers: int = 4) -> nn.Module:
    if variant == "vanilla":
        return model
    dev = next(model.parameters()).device  # create_cnn_architecture already placed the model on the GPU
    ch = model.encoder_mlp[-1].out_features
    if variant == "mlp30":
        core, out = MLPCore(ch, 30, _count(MultiRegister(ch, 10, 3, q_layers, "hea"))), 30
    elif variant == "mr30ryd":
        core, out = MultiRegister(ch, 10, 3, q_layers, "rydberg"), 30
    elif variant.startswith("q"):                                # wave-G grammar, incl. doc 35 §23 q30-geo=3d[-mlp]
        from model import build_config_core, parse_config
        core, out = build_config_core(parse_config(variant), ch, q_layers)
    else:
        raise ValueError(f"unknown variant {variant}")
    model.encoder_mlp = nn.Sequential(model.encoder_mlp, Bottleneck(ch, core, out))
    return model.to(dev)
