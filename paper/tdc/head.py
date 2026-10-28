"""Arms for the TDC ADMET row: the frozen ChemBERTa embedding goes through a bottleneck, then one linear head.

Arms are identical outside the bottleneck and identical at initialisation (the bottleneck's residual gate is
zero-init), so `vanilla` is the trunk-plus-head baseline, `q30-mlpref` the parameter-matched classical control
and `q30...` the Rydberg register, exactly as on every other row.
"""
from __future__ import annotations

import pathlib
import sys

import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model import Bottleneck, MLPCore, _count, build_config_core, parse_config  # noqa: E402
from qlayer import MultiRegister  # noqa: E402

WIDTH = 384  # ChemBERTa-77M hidden size


def build(variant: str, q_layers: int = 4) -> nn.Module:
    """A (bottleneck, head) model over the pooled molecular embedding."""
    if variant == "vanilla":
        mid: nn.Module = nn.Identity()
    elif variant == "mlp30":
        mid = Bottleneck(WIDTH, MLPCore(WIDTH, 30, _count(MultiRegister(WIDTH, 10, 3, q_layers, "hea"))), 30)
    elif variant == "mr30ryd":
        mid = Bottleneck(WIDTH, MultiRegister(WIDTH, 10, 3, q_layers, "rydberg"), 30)
    elif variant == "mr50ryd":                                          # Fig. 2 (Oct 3): 50 atoms, 5 groups of 10
        mid = Bottleneck(WIDTH, MultiRegister(WIDTH, 10, 5, q_layers, "rydberg"), 50)
    elif parse_config(variant) is not None:
        core, out = build_config_core(parse_config(variant), WIDTH, q_layers)
        mid = Bottleneck(WIDTH, core, out)
    else:
        raise ValueError(f"unknown variant: {variant}")

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mid = mid
            self.head = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 1))

        def forward(self, x):
            return self.head(self.mid(x)).squeeze(-1)

    return Net()
