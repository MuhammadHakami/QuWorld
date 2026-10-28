"""Reproduce the official CNextU-net test number (published: 0.1956 one-step
mean-VRMSE on turbulent_radiative_layer_2D), and emit per-window VRMSE for
paired statistics.

Usage: uv run paper/phase_b/eval_official.py [model_dir] [out_csv]
Defaults: data/cnext_ckpt -> runs/eval_official_cnext.csv

Protocol (mirrors the_well.benchmark): WellDataset test split, n_steps_input=4,
ZScore normalization on inputs; model predicts the normalized next state;
VRMSE is std-normalized so per-channel affine normalization cancels and
evaluation in normalized space is exact.
"""

import csv
import pathlib
import sys

import torch
from einops import rearrange

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import os

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/cnext_ckpt")
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "runs/eval_official_cnext.csv")
DATASET = os.environ.get("WELL_DATASET", "turbulent_radiative_layer_2D")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = int(os.environ.get("EVAL_BATCH", "16"))


def load_model(model_dir: str):
    if model_dir.endswith(".pt"):
        # hybrid fine-tune checkpoint: runs/phase_b/<variant>_s<seed>.pt
        from cnext_hybrid import load_hybrid_checkpoint

        variant = pathlib.Path(model_dir).stem.rsplit("_s", 1)[0]
        print(f"loading hybrid checkpoint, variant={variant}")
        return load_hybrid_checkpoint(variant, model_dir)

    import json

    from safetensors.torch import load_file
    from the_well.benchmark.models import UNetConvNext

    cfg = json.loads(open(f"{model_dir}/config.json").read())
    model = UNetConvNext(**cfg)
    state = load_file(f"{model_dir}/model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    return model


def vrmse_per_window(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """pred/true: (B, *spatial, C) -> (B,) mean-over-fields VRMSE (eps=1e-7).
    Spatial dims are flattened so 2D and 3D fields share one code path."""
    p = pred.flatten(1, -2)
    t = true.flatten(1, -2)
    mse = (p - t).square().mean(dim=1)
    var = (t - t.mean(dim=1, keepdim=True)).square().mean(dim=1)
    return (mse / (var + 1e-7)).sqrt().mean(dim=-1)


@torch.no_grad()
def main() -> None:
    from the_well.data import WellDataset
    from the_well.data.normalization import ZScoreNormalization

    ds = WellDataset(
        well_base_path=str(ROOT / "data/well"),
        well_dataset_name=DATASET,
        well_split_name="test",
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, num_workers=4)
    # "a.pt+b.pt+c.pt": equal-weight ensemble of fine-tuned checkpoints (mean of normalised next-state predictions)
    models = [load_model(m).to(DEV).eval() for m in MODEL_DIR.split("+")]
    model = lambda x: torch.stack([m(x) for m in models]).mean(0)
    n_params = sum(p.numel() for m in models for p in m.parameters())

    scores = []
    for batch in loader:
        xin = batch["input_fields"]
        if xin.dim() == 5:                                      # 2D: (b,t,h,w,c)
            x = rearrange(xin, "b t h w c -> b (t c) h w").to(DEV)
            pred = rearrange(model(torch.nan_to_num(x)), "b c h w -> b h w c")
        else:                                                   # 3D: (b,t,d,h,w,c)
            x = rearrange(xin, "b t d h w c -> b (t c) d h w").to(DEV)
            pred = rearrange(model(torch.nan_to_num(x)), "b c d h w -> b d h w c")
        y = batch["output_fields"][:, 0].to(DEV)
        scores.extend(vrmse_per_window(pred, torch.nan_to_num(y)).tolist())

    scores_t = torch.tensor(scores)
    pathlib.Path(OUT_CSV).parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window", "vrmse"])
        w.writerows(enumerate(scores))

    print(f"model_dir:      {MODEL_DIR}")
    print(f"num_params:     {n_params}")
    print(f"test_windows:   {len(scores)}")
    print(f"mean_vrmse:     {scores_t.mean():.4f}   (published CNextU-net: 0.1956)")
    print(f"median_vrmse:   {scores_t.median():.4f}")
    print(f"per_window csv: {OUT_CSV}")


if __name__ == "__main__":
    main()
