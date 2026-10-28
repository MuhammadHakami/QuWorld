# TokaMark starter kit — Group G2 magnetics forecasting

Working notes for targeting the TokaMark benchmark (Group 2, magnetics
dynamics) with a frozen modern backbone + our quantum adapter, replacing the
published CNN+LSTM baseline. Everything below is sourced; items marked
**[verified]** were checked directly against the live S3 store or the
official repo files on 2026-08-17 from this machine (DGX Spark, aarch64).

## Sources

| What | Where |
|---|---|
| Benchmark paper (v3, 2026-06-11) | arXiv:2602.10132 — "TokaMark: A Comprehensive Benchmark for MAST Tokamak Plasma Models" (UKAEA + IBM Research Europe + STFC Hartree) — <https://arxiv.org/abs/2602.10132> |
| Benchmark repo | <https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark> (license: OpenMDW-1.0) |
| Baseline repo (CNN / CNN+LSTM) | <https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark_baseline> |
| Foundation-model repo (TokaMind, arXiv:2602.15084) | <https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamind> |
| Underlying data system | FAIR-MAST (SoftwareX 2024) — catalog at <https://mastapp.site>; data license CC BY 4.0 |
| Robustness follow-up (useful baselines context) | arXiv:2607.11915 — "Benchmarking Sensor Robustness in Plasma Diagnostic Models" |

## 1. Dataset & S3 access **[verified]**

- 11,573 MAST shots, one **Zarr v2** store per shot, **consolidated
  metadata** (`.zmetadata`), 11 diagnostic groups per shot
  (`magnetics, pf_active, summary, equilibrium, thomson_scattering,
  charge_exchange, gas_injection, interferometer, pulse_schedule,
  soft_x_rays, spectrometer_visible`).
- Endpoint (from `src/MAST_tools/utils/store_utils.py` in the tokamark repo):
  - `s3_endpoint_url = "https://s3.echo.stfc.ac.uk"` (STFC Echo, Ceph RGW)
  - dataset path `/mast/tokamark/v1` → bucket `mast`, prefix `tokamark/v1`
  - per-shot store: `s3://mast/tokamark/v1/{shot_id}.zarr`
  - access is **anonymous** (`anon=True` in the official code; plain
    unsigned HTTPS GET works too) — no agreements, no credentials.
- **[verified]** `https://s3.echo.stfc.ac.uk/mast/tokamark/v1/30421.zarr/.zmetadata`
  returns HTTP 200 anonymously; anonymous `ListObjectsV2`
  (`https://s3.echo.stfc.ac.uk/mast?list-type=2&prefix=tokamark/v1/30421.zarr/`)
  works as well.
- **[verified]** compression: blosc (inner codec **lz4**, byte-shuffle,
  typesize 8, dtype `<f8` everywhere we looked).
- **[verified]** size of shot 30421: 236.4 MB total (993 objects);
  task 2-1 signals only: **2.43 MB/shot**; +0.09 MB for task 2-2 (LCFS),
  +3.80 MB for task 2-3 (psi map). Extrapolated mirror sizes (one-shot
  sample, so ±50%): task 2-1 ≈ **30 GB**, all of G2 ≈ 75 GB, full dataset
  ≈ 2.7 TB.

Official access pattern (from tokamark README / `store_utils.py`):

```python
store_manager_settings = {
    "s3_endpoint_url": "https://s3.echo.stfc.ac.uk",
    "s3_mast_dataset_path": "/mast/tokamark/v1",
    "base_fsspec_protocol": "simplecache",   # disk cache of remote reads
    "target_fsspec_protocol": "s3",
}
# internally: s3fs.S3FileSystem(anon=True, endpoint_url=...), store per shot:
# posixpath.join("/mast/tokamark/v1", f"{shot_id}.zarr")
```

## 2. G2 task definition **[verified — copied from repo task YAMLs]**

Group 2 "magnetics dynamics" = tasks 2-1, 2-2, 2-3. All three are
`task_type: markovian` **forecasting** tasks with identical inputs/actuators
and windowing (`src/tokamark/tasks_configs/group_2_magnetics_dynamics/`):

- **input window**: 0.005 s (5 ms) of history
- **output window**: 0.025 s (25 ms) forecast, `delta: 0` (forecast starts
  immediately after the input window)
- **stride**: 0.001 s (1 ms) between windows
- **no resampling**: paper v3 states "signals are used at their original
  sampling rates, and no resampling or imputation is performed, with
  evaluation performed using masked metrics on available ground truth."

**Inputs** (8 signals; channel counts from shot 30421 **[verified]**):

| signal | shape (ch × t) | time axis |
|---|---|---|
| `magnetics/flux_loop_flux` | 15 × 3633 | `time` |
| `magnetics/b_field_pol_probe_ccbv_field` | 40 × 3633 | `time` |
| `magnetics/b_field_pol_probe_obr_field` | 18 × 3633 | `time` |
| `magnetics/b_field_pol_probe_obv_field` | 18 × 3633 | `time` |
| `magnetics/b_field_tor_probe_saddle_voltage` | 12 × 36321 | `time_saddle` (10× faster) |
| `pf_active/coil_current` | 10 × 2906 | `time` |
| `pf_active/solenoid_current` | 2906 | `time` |
| `summary/ip` | 2906 | `time` |

→ 115 input channels on 3 different clocks (magnetics ~3633 samples/shot,
pf_active/summary ~2906, saddle ~36321).

**Actuators** (2 signals, 5 channels): `pf_active/coil_voltage` (4 ch),
`summary/power_nbi` (1 ch).

**Outputs**:

- **Task 2-1** (the headline "magnetics forecasting" task): 18 signals /
  21 channels — `pf_active/coil_current` (10), `pf_active/solenoid_current`,
  `summary/ip`, and 15 equilibrium scalars: `elongation, elongation_axis,
  triangularity_upper, triangularity_lower, x_point_r (2), x_point_z (2),
  minor_radius, magnetic_axis_r, magnetic_axis_z, q95, beta_tor, beta_pol,
  beta_normal, bvac_rmag, bphi_rmag`. Equilibrium signals live on the slow
  EFIT clock (~146 samples/shot → ~5 ms cadence: ~1 input sample, ~5
  output samples per window).
- **Task 2-2**: plasma boundary `equilibrium/lcfs_r`, `lcfs_z` (139 × 146).
- **Task 2-3**: flux map `equilibrium/psi` (65 × 65 × 146).

## 3. Splits **[verified — repo CSV]**

`src/tokamark/metadata/TokaMark_data_splits.csv` (11,573 rows):
**train 8,963 / val 1,115 / test 1,110** shots (~80/10/10, shot-level,
random split), plus 5 CV fold columns and `*_ip_negative` flag columns
(308/38/39 negative-Ip shots). A separate **temporal split**
(`TokaMark_temporal_data_splits.csv`) orders by campaign/time — the paper
reports both. Per-task usable shots differ by signal availability (paper
v3, Table B.1): task 2-1 = 8,268/1,028/1,020, task 2-2 = 8,382/1,047/1,031,
task 2-3 = 8,403/1,050/1,034 (train/val/test, random split).

## 4. Metric — NRMSE **[verified — `src/tokamark/evaluator.py`]**

Paper formula: `NRMSE_k = RMSE_k / σ_k` (σ_k = empirical std of signal k;
NRMSE = 1 ⇔ predicting the mean). Aggregation in `evaluator.py`:

1. per window & feature: `RMSE = sqrt(nanmean(diff²))` (masked metrics —
   NaNs excluded);
2. windows → shot: mean of MSE (RMSE² averaged, then sqrt);
3. normalize: `NRMSE = RMSE / std`, with `std` per signal from the
   **train-split stats metadata** (`random_split_signals_stats.yaml`);
4. shots → signal row: equal-weight mean across shots;
5. signals → task: equal-weight mean; tasks → group: equal-weight mean of
   task means. (Checked: mean(0.2197, 0.0568, 0.1263) = 0.1343 exactly.)

Inputs are z-scored with the same train-split stats (`use_std_scaling=True`
in `data.py`); outlier shots and bad-EFIT-rating segments are removed by
default (`remove_outliers=True`, `remove_bad_efit_rating=True`, metadata
YAMLs shipped in the repo).

## 5. Published baselines (paper v3, Table 3, Group 2 NRMSE)

| model | random split | temporal split |
|---|---|---|
| Mean predictor | 0.9106 | 0.9393 |
| CNN (multi-branch conv enc/dec, MLP backbone) | 0.1590 | 0.7068 |
| **CNN+LSTM (LSTM backbone)** | **0.1343** | 0.2928 |

Per-task, random split: task 2-1 CNN 0.2751 / CNN+LSTM **0.2197**;
task 2-2 0.0662 / 0.0568; task 2-3 0.1358 / 0.1263. The persistence model
was not evaluated on G2 ("—" in Table 3). **Our headline target: beat
0.1343 group-2 NRMSE (random split), and ideally 0.2928 on the harder
temporal split, where the CNN+LSTM degrades 2.2×.**

Baseline architecture (tokamark_baseline repo): per-modality Conv1D/2D/3D
encoders to a fixed latent dim (default 16), shared backbone (3-layer MLP,
dropout 0.2 — or seq2seq LSTM over chunk-level tokens), task-specific conv
decoders. Adam + weight decay 1e-4, early stopping on val loss. Paper
hardware: **single NVIDIA A100-SXM4-80GB, CUDA 12.8** (no GPU-hours given).

## 6. This starter kit **[verified]**

- `dataset.py` — streaming-first task-2-1 loader (zarr 3 + s3fs anon, Zarr v2
  consolidated): official windowing/outlier/EFIT/z-score semantics ported from
  the repo (`TokaMark_dataset.py`, `MAST_dataset.py`, transforms), local LRU
  npz cache in `data/tokamark_cache/` (5 GiB cap, oldest-mtime eviction), exact
  S3 byte accounting. Verified 2026-08-17: 3 train shots → 1,678 windows, all
  shapes checked, 6.32 MB moved, cold ~13 s/shot, warm ~0.01 s.
- `task_meta.py` — loaders for the verbatim official metadata copies in
  `metadata/` (task 2-1 YAML, train-split stats, splits CSV, outlier map).
- `verify_stream.py` — end-to-end 3-shot verification (shapes, window counts,
  timings, transfer budget <1 GB): `uv run --no-sync python
  paper/tokamark/verify_stream.py`.
- `baseline_plan.md` — CNN+LSTM 0.1343 repro steps vs our frozen-backbone+30q
  plan, GB10 GPU-hours, and the pre-registered repro-vs-cite decision rule.
- `fetch_sample.py` — streams ONE shot's G2 signals anonymously over plain
  HTTPS (stdlib `urllib` + numpy only), prints shape/chunk/dtype/compressor
  for every task 2-1/2-2/2-3 signal, and fully decodes a few small 1-D
  arrays (time axes, Ip, q95) via a built-in pure-Python blosc/LZ4
  split-block decoder to sanity-check actual values. Run:

  ```bash
  uv run --no-sync python paper/tokamark/fetch_sample.py            # shot 30421
  uv run --no-sync python paper/tokamark/fetch_sample.py 21719      # any shot id
  ```

  Verified run (2026-08-17, shot 30421): all 28 G2 signals present with the
  shapes in the table above, and decoded values are physically sensible —
  `summary/time` −0.061→0.665 s @ 4 kHz, `magnetics/time` @ 5 kHz,
  `summary/ip` peak 6.49e5 A (0.65 MA), `q95` ∈ [2.79, 7.80]. Exit 0.
  Note: S3 latency from this site is variable (sub-second to ~minutes per
  object on bad days) — another argument for the phase-1 local mirror.

### Dependency gap (repo venv, aarch64 — DO NOT `uv add` without approval)

Present: `torch 2.13.0+cu130`, `numpy 2.4.4`, `fsspec 2024.10.0`,
`h5py 3.16.0`, `requests`, `pandas`. **Missing for the official TokaMark
stack**: `zarr`, `s3fs`, `xarray`, `numcodecs`, `aiohttp` (fsspec's http/s3
backends need these), plus optionally `tensordict` (baseline repo).
All have aarch64 manylinux wheels; the eventual install is
`zarr s3fs xarray numcodecs` (one `uv add`, pending approval).
`fetch_sample.py` deliberately avoids all of them.

## 7. Baseline-repro & adapter plan (GB10, 121 GiB unified)

GB10 vs the paper's A100-80GB: ~3–6× slower for this workload (memory-
bandwidth-bound: 273 GB/s vs 2 TB/s; the models themselves are small).
Windows/epoch is the cost driver: ~8.3k train shots × ~0.5 s usable ×
1 ms stride ≈ **~4M windows/epoch** for task 2-1.

| phase | what | compute (est.) |
|---|---|---|
| 0 | deps: `uv add zarr s3fs xarray numcodecs` (needs approval), `pip install -e` the tokamark repo clone (outside our pyproject) | CPU only |
| 1 | local mirror of task 2-1 signals for all 11,573 shots (`s5cmd`/`rclone` anonymous, ~30 GB; add psi/LCFS later if we attack 2-2/2-3) | network, hours; no GPU |
| 2 | sanity anchors: mean predictor (expect ≈0.91) + our own persistence baseline via `scripts/persistence.py`; validates our evaluator wiring against Table 3 before any training | ≤1 GPU-h |
| 3 | CNN+LSTM repro on task 2-1, random split, official configs; pilot on 10% shots first (~2–4 GPU-h), then full run. A100 full-run guess 8–20 h → GB10 **~30–90 GPU-h**; budget 3 seeds only if the first matches 0.22 ± 0.01 task-2-1 NRMSE | 40–120 GPU-h |
| 4 | ours: frozen modern backbone (pretrained TS/world-model encoder) + quantum adapter; only adapter + heads train → smaller step cost, same data cost. Pilot 10% (~4 GPU-h), full **~15–40 GPU-h/seed × 3 seeds** | 50–130 GPU-h |
| 5 | eval on official test split with repo `evaluator.py`; report random + temporal splits; paired stats across shots (equal-weight shot rows make bootstrap-over-shots the natural test) | ≤2 GPU-h |

Risks: (a) I/O-bound epochs if we stream from S3 — mirror first, keep the
`simplecache` path as fallback; (b) multi-clock signals (saddle at 10×
rate) need the repo's own windowing — reuse `TokaMarkDataset` rather than
reimplementing; (c) paper gives no baseline training time — calibrate the
GB10 estimate from the phase-3 pilot before committing to seeds; (d) shot
0.5 s usable duration varies — recompute windows/epoch after mirroring.

Total budget: ~100–250 GB10 GPU-hours end-to-end, front-loadable with
CPU/network phases 0–2 while other runs occupy the GPU.
