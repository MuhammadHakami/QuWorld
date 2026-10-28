"""Loader for the Pelzer 1HP geothermal heat-plume benchmark (DaRUS darus-3653).

Prepared dataset `benchmark_dataset_2d_100datapoints_5years` (1HP_NN format):
  Inputs/RUN_i-1.pt          (4, 256, 16) float32 in [0,1], channel order "pksi"
                             [Liquid Pressure, Permeability X, SDF, Material ID]
  Labels/RUN_i-1.pt          (1, 256, 16) float32, Temperature [C] rescaled to [0,1]
  InputParams/RUN_i.pt       (4,)  [hp_x_m=50, hp_y_m=120, permeability_m2, pressure_grad]
  LearnableParams/RUN_i-1.pt (4,)  [iso1K_length_m, iso1K_width_m, T_max_C, T_max_flat_idx]
                             (length/width are NaN when the plume exceeds the domain)

Grid: (flow, transverse) = (256, 16) cells at 5 m -> 1280 m x 80 m; pump at cell (23, 7).
Normalization stats live in info-1.yaml ("Rescale" to [0,1]; SDF unnormalized).
Frozen 70/20/10 split (1HP_NN ratio) seeded 20260816, persisted in splits.json.
"""

import json
import pathlib

import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2] / "data" / "geothermal"
PREPARED = ROOT / "darus-3653" / "benchmark_dataset_2d_100datapoints_5years"
SPLIT_FILE = pathlib.Path(__file__).resolve().parent / "splits.json"

SEED = 20260816
N_SAMPLES = 100
N_TRAIN, N_VAL, N_TEST = 70, 20, 10  # 1HP_NN convention [0.7, 0.2, 0.1]
CELL_M = 5.0
GRID = (256, 16)  # (flow, transverse)
CHANNELS = ("Liquid Pressure [Pa]", "Permeability X [m^2]", "SDF", "Material ID")
INPUT_PARAMS = ("hp_x_m", "hp_y_m", "permeability_m2", "pressure_gradient")
LEARNABLE_PARAMS = ("iso1K_length_m", "iso1K_width_m", "T_max_C", "T_max_flat_idx")


def load_info() -> dict:
    """Per-channel normalization stats (min/max/mean/std/norm) from info-1.yaml."""
    path = PREPARED / "info-1.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run the darus fetch script to complete darus-3653"
        )
    return yaml.safe_load(path.read_text())


def _load_stack(subdir: str, suffix: str) -> torch.Tensor:
    files = [PREPARED / subdir / f"RUN_{i}{suffix}.pt" for i in range(N_SAMPLES)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} files missing in {PREPARED / subdir} "
            f"(e.g. {missing[:3]}) - dataset incomplete"
        )
    return torch.stack(
        [torch.load(f, map_location="cpu", weights_only=True) for f in files]
    )


def load_fields() -> tuple[torch.Tensor, torch.Tensor]:
    """Full-field task: X (100, 4, 256, 16) in [0,1], Y (100, 1, 256, 16) in [0,1]."""
    x = _load_stack("Inputs", "-1")
    y = _load_stack("Labels", "-1")
    if x.shape != (N_SAMPLES, 4, *GRID) or y.shape != (N_SAMPLES, 1, *GRID):
        raise ValueError(f"unexpected shapes: inputs {x.shape}, labels {y.shape}")
    return x, y


def load_scalars() -> tuple[torch.Tensor, torch.Tensor]:
    """Scalar task (QCNN comparator): params (100, 4), targets (100, 4).

    Targets contain NaN where the 1K-isoline exceeds the domain (18 lengths,
    32 widths); filter with `~torch.isnan(targets).any(dim=1)` for regression.
    """
    return _load_stack("InputParams", ""), _load_stack("LearnableParams", "-1")


def denormalize_temperature(y: torch.Tensor, info: dict | None = None) -> torch.Tensor:
    """Map normalized labels/predictions back to degrees Celsius."""
    stats = (info or load_info())["Labels"]["Temperature [C]"]
    if stats.get("norm") != "Rescale":
        raise ValueError(f"unexpected label norm: {stats.get('norm')}")
    return y * (stats["max"] - stats["min"]) + stats["min"]


def _derive_split(seed: int = SEED) -> dict:
    perm = torch.randperm(N_SAMPLES, generator=torch.Generator().manual_seed(seed))
    return {
        "seed": seed,
        "n_samples": N_SAMPLES,
        "train": sorted(perm[:N_TRAIN].tolist()),
        "val": sorted(perm[N_TRAIN : N_TRAIN + N_VAL].tolist()),
        "test": sorted(perm[N_TRAIN + N_VAL :].tolist()),
    }


def split() -> dict:
    """Frozen train/val/test indices. splits.json is authoritative; the seeded
    derivation must reproduce it exactly (guards against silent drift)."""
    derived = _derive_split()
    if SPLIT_FILE.exists():
        frozen = json.loads(SPLIT_FILE.read_text())
        if frozen != derived:
            raise RuntimeError(
                f"{SPLIT_FILE} disagrees with seeded derivation - do not retrain; "
                "resolve the discrepancy first"
            )
        return frozen
    return derived


def freeze_split() -> pathlib.Path:
    """Write the frozen split proposal to splits.json (idempotent)."""
    SPLIT_FILE.write_text(json.dumps(_derive_split()) + "\n")
    return SPLIT_FILE


def mae_celsius(pred: torch.Tensor, true: torch.Tensor, info: dict | None = None) -> torch.Tensor:
    """MAE in degrees C from normalized tensors (B, 1, H, W). 1HP_NN headline metric."""
    info = info or load_info()
    err = denormalize_temperature(pred, info) - denormalize_temperature(true, info)
    return err.abs().mean()


def pat(pred_c: torch.Tensor, true_c: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """PAT: fraction of cells with |error| above `threshold` degrees C (LGCNN metric,
    lower is better). Inputs already in degrees C."""
    return ((pred_c - true_c).abs() > threshold).float().mean()


def max_error_ok_fraction(
    pred_c: torch.Tensor, true_c: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Fraction of samples whose max cell error is below `threshold` degrees C
    ("good" samples in Davis et al. 2302.08199). Inputs (B, 1, H, W) in degrees C."""
    per_sample = (pred_c - true_c).abs().amax(dim=(1, 2, 3))
    return (per_sample < threshold).float().mean()


if __name__ == "__main__":
    print(f"frozen split -> {freeze_split()}")
    x, y = load_fields()
    params, targets = load_scalars()
    info = load_info()
    idx = split()
    y_c = denormalize_temperature(y, info)
    print(f"fields: X{tuple(x.shape)} Y{tuple(y.shape)}; T in "
          f"[{y_c.min():.2f}, {y_c.max():.2f}] C")
    print(f"scalars: params{tuple(params.shape)} targets{tuple(targets.shape)}; "
          f"{int(torch.isnan(targets).any(dim=1).sum())}/100 samples have NaN targets")
    print(f"split sizes: {len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])} "
          f"(seed {idx['seed']})")
