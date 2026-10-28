"""Backbone factory for the 2-D Well benchmark lane: the official checkpoint of a
dataset, with the quantum bottleneck inserted at the deepest point.

  cnext : the_well's own CNextU-net checkpoints (polymathic-ai/UNetConvNext-<dataset>),
          proven surgery path (cnext_hybrid: neck wrapped at 672 ch) — reproduces the
          published TRL2D number; The Well's original benchmark baselines (not the current
          state of the art): CNextU-net on shear_flow (0.8080; our evaluator scores the
          released checkpoint 0.9946) and rayleigh_benard (0.6699; reproduced at 0.6679).
  tfno  : neuraloperator 0.3.0 TFNO checkpoints (tfno_hybrid) — NOT reproducible on this
          stack as of Sep 19 2026 (FNO/TFNO TRL2D checkpoints give VRMSE 2.4-5.3 under
          both neuralop 0.3.0 and 2.0.0 while CNextU-net gives 0.19); kept for the record.
"""

import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
ROOT = _HERE.parent.parent

CNEXT_DIR = {
    "turbulent_radiative_layer_2D": ROOT / "data/cnext_ckpt",
    "rayleigh_benard": ROOT / "data/cnext_rb_ckpt",
    "planetswe": ROOT / "data/cnext_planetswe_ckpt",
    "shear_flow": ROOT / "data/cnext_shear_flow_ckpt",
    "active_matter": ROOT / "data/cnext_active_matter_ckpt",                          # doc 35 §31 prospective rows
    "gray_scott_reaction_diffusion": ROOT / "data/cnext_gray_scott_reaction_diffusion_ckpt",
}


def build(variant: str, dataset: str, backbone: str = "cnext"):
    if backbone == "cnext":
        from cnext_hybrid import build_hybrid
        return build_hybrid(variant, model_dir=str(CNEXT_DIR[dataset]))
    if backbone == "tfno":
        from tfno_hybrid import build_hybrid
        return build_hybrid(variant, dataset)
    raise ValueError(f"unknown backbone {backbone}")


def load_checkpoint(variant: str, dataset: str, ckpt_path: str, backbone: str = "cnext", dev: str = "cuda"):
    model = build(variant, dataset, backbone)
    st = torch.load(ckpt_path, weights_only=True, map_location=dev)
    model.load_state_dict(st["model_state_dict"])
    return model.to(dev)
