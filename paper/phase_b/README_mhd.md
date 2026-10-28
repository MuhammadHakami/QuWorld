# Phase-B on The Well MHD_64 — anchor eval + 3-arm surgery commands

Port of the TRL2D Phase-B pipeline to `MHD_64` (3D compressible MHD turbulence,
64^3, 7 channels). Data loader: `paper/phase_b/data_ram_mhd.py` (memmap-backed,
protocol-verified vs `WellDataset`; see its docstring for the RAM decision).

## Dataset facts (verified from the HDF5s, 2026-08-17)

- Splits: train 77 trajectories (9 files x 8 + 1 file x 5), valid 10, test 10;
  T=100 steps each. `MHD_64.yaml`'s "n_trajectories_per_file: 1" is stale.
- HDF5 groups per file: `t0_fields/density` (n,100,64,64,64),
  `t1_fields/magnetic_field` and `t1_fields/velocity` (n,100,64,64,64,3), fp32.
- the_well channel order (t0 then t1, `attrs["field_names"]` order, vector
  components x,y,z): **[density, Bx, By, Bz, vx, vy, vz]**.
  Channels-first input flattens t-major: 4 frames x 7 ch = 28 input channels
  (matches the official checkpoint `dim_in=28, dim_out=7`).
- Normalization stats: `data/well/MHD_64/stats.yaml` (= `data/mhd64/stats.yaml`,
  the file `WellDataset` resolves — NOT `data/mhd64/data/stats.yaml`).

## Prerequisite: dataset symlink (already created)

```bash
ln -sfn ../mhd64 data/well/MHD_64   # so WellDataset finds well_base_path/MHD_64
```

## Anchor eval (do NOT run while the GPU campaign is live)

`eval_official.py` already handles 3D inputs and the `WELL_DATASET` env var:

```bash
WELL_DATASET=MHD_64 EVAL_BATCH=8 uv run paper/phase_b/eval_official.py \
    data/cnext_mhd_ckpt runs/eval_official_cnext_mhd.csv
```

- Expected: `mean_vrmse` close to **0.1633** (The Well paper, CNextU-net
  one-step test VRMSE on MHD_64; the HF model card's 0.1487 is best-epoch
  *validation* VRMSE). The script's printed `(published CNextU-net: 0.1956)`
  suffix is the TRL2D anchor — ignore it for MHD.
- Test windows: 10 trajectories x 96 = 960 one-step windows.
- Each 3D input sample is 28x64^3 fp32 (~29 MB); `EVAL_BATCH=8` keeps
  activations modest on the GB10.

## Fine-tune: 3-arm checkpoint surgery (vanilla | mlp30 | mr30ryd)

Intended commands, mirroring the TRL2D campaign (seed x arm grid, 1500 s
budget, lr 1e-4 — MHD lr may need its own sweep; official MHD training used
5e-3):

```bash
uv run paper/phase_b/finetune.py vanilla 1337 1500 1e-4   # frozen-arch control
uv run paper/phase_b/finetune.py mlp30   1337 1500 1e-4   # classical param-matched core
uv run paper/phase_b/finetune.py mr30ryd 1337 1500 1e-4   # quantum MultiRegister (rydberg)
# repeat for seeds 1338, 1339
```

### Port status — finetune.py needs three one-line touches before running on MHD

`finetune.py` is still wired to TRL2D. The surgery itself transfers unchanged:
`Bottleneck` pools over `dims 2..N` (dimension-agnostic) and the MHD checkpoint
has the same neck width (`init_features=42, stages=4` -> 672 ch; neck spatial is
4^3 instead of 8x24). Required touches (or a thin `finetune_mhd.py` wrapper):

1. `import data_ram` -> `import data_ram_mhd as data_ram` (same API:
   `load()`, `train_batch()`, `valid_windows()`).
2. `build_hybrid(VARIANT)` -> `build_hybrid(VARIANT, model_dir=str(ROOT / "data/cnext_mhd_ckpt"))`
   (default points at the TRL2D checkpoint `data/cnext_ckpt`).
3. `validate()` reduces VRMSE over `dim=(2, 3)` (2D spatial); for 3D use
   `dim=(2, 3, 4)` — or flatten spatial dims as in
   `eval_official.vrmse_per_window`.
   Likewise `eval_official.load_model` rebuilds hybrids from `.pt` via
   `build_hybrid(variant)` — it needs the same `model_dir` override for MHD
   hybrid checkpoints.

Memory note for training: batch 16 -> the CPU-side window gather is
16 x 5 frames x 7 x 64^3 fp32 ~ 590 MB per step read through the page cache;
`BATCH` may need lowering on the GB10 alongside the CUDA allocator cap.

## ⚠ ANCHOR FINDING (2026-08-17)
The released HF checkpoint does NOT reproduce its card/paper: our pipeline
(validated to 1.1% on TRL2D) measures test 0.5447 per-channel / 0.6644
field-grouped and valid 0.6614, vs claimed 0.1487 valid / 0.1633 test.
ZScore confirmed correct (RMS worse: 0.976). Likely a wrong-epoch upload.
CONSEQUENCE: Gate B anchor = the released artifact's own score (all arms
paired against it); the published 0.1633 is cited as unreproduced-by-artifact
context. "Beat 0.1633" only claimable if our arms actually cross it.
TODO(user decision): report upstream issue to PolymathicAI.
