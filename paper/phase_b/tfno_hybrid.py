"""Quantum bottleneck inside the official The Well TFNO checkpoints
(polymathic-ai/TFNO-<dataset>): the SoTA single-step model on planetswe
(0.0853) and rayleigh_benard (0.6566).

The neuralop TFNO is lifting -> n_layers FNO blocks -> projection on
hidden_channels (128). We wrap `projection` with a zero-init gated
bottleneck on the hidden field (pool -> core -> residual), exactly as
cnext_hybrid wraps the CNextU-net neck: with alpha=0 at init the hybrid is
numerically identical to the released checkpoint.

Variants: vanilla | mlp30 | mr30ryd  (cores from paper/qlayer, parameter-
matched at the core level, same as every other benchmark arm).
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
CKPT = {"planetswe": ROOT / "data/tfno_planetswe_ckpt", "rayleigh_benard": ROOT / "data/tfno_rb_ckpt"}


class ProjWithBottleneck(nn.Module):
    def __init__(self, bottleneck: nn.Module, projection: nn.Module):
        super().__init__()
        self.bottleneck = bottleneck
        self.projection = projection

    def forward(self, x, *args, **kwargs):
        return self.projection(self.bottleneck(x), *args, **kwargs)


def load_official(model_dir: str):
    from safetensors.torch import load_file
    from the_well.benchmark.models import TFNO

    cfg = json.loads(open(f"{model_dir}/config.json").read())
    model = TFNO(**cfg)
    state = load_file(f"{model_dir}/model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model, cfg


def build_hybrid(variant: str, dataset: str, q_layers: int = 4):
    model, cfg = load_official(str(CKPT[dataset]))
    if variant == "vanilla":
        return model
    ch = cfg["hidden_channels"]  # 128
    if variant == "mlp30":
        ref = MultiRegister(ch, 10, 3, q_layers, "hea")
        core, out = MLPCore(ch, 30, _count(ref)), 30
    elif variant in ("mr30hea", "mr30ryd"):
        core, out = MultiRegister(ch, 10, 3, q_layers, {"mr30hea": "hea", "mr30ryd": "rydberg"}[variant]), 30
    else:
        raise ValueError(f"unknown variant {variant}")
    model.model.projection = ProjWithBottleneck(Bottleneck(ch, core, out), model.model.projection)
    return model


def load_hybrid_checkpoint(variant: str, dataset: str, ckpt_path: str, dev: str = "cuda"):
    model = build_hybrid(variant, dataset)
    st = torch.load(ckpt_path, weights_only=True, map_location=dev)
    model.load_state_dict(st["model_state_dict"])
    return model.to(dev)


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "planetswe"
    for v in ("vanilla", "mlp30", "mr30ryd"):
        m = build_hybrid(v, ds)
        print(f"{ds} {v}: params {_count(m)}")
