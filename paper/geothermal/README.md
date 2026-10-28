# Geothermal Heat-Plume Benchmark (Pelzer 1HP)

Second-domain evidence pillar: quantum bottleneck adapters in a **frozen U-Net**
surrogate for open-loop groundwater heat-pump (GWHP) plume prediction. Chosen
because (a) it is a real PDE surrogate task with published classical baselines,
and (b) a 2026 QCNN paper (arXiv:2606.23411) already uses it, giving us a direct
quantum comparator with a weaker architecture prior.

## Data on disk (fetched + md5-verified against the DaRUS API, 2026-08-17)

| Dataset | DOI | Contents | Status |
|---|---|---|---|
| `data/geothermal/darus-3650/` | 10.18419/darus-3650 | Raw 1HP PFLOTRAN, 1000 runs (`RUNs.zip`, 188 MB, zip-integrity OK) + inputs | **COMPLETE** 15/15 files |
| `data/geothermal/darus-3653/` | 10.18419/darus-3653 | Prepared `.pt`, 100 datapoints, 4 subfolders | **COMPLETE** 402/402 files |
| `darus-3652` | 10.18419/darus-3652 | Two-heat-pump raw, 4.3 GB | **DEFERRED** (not fetched) |

darus-3653 has colliding filenames across its subfolders (`Inputs/RUN_0-1.pt`,
`Labels/RUN_0-1.pt`, ...), so files **must** live in the dataverse directory
structure — a flat download silently loses ~2/3 of the dataset (this happened;
the fetch script now restores the canonical layout and verifies md5s).

## Prepared format (`benchmark_dataset_2d_100datapoints_5years`, inspected)

Grid: (flow, transverse) = **(256, 16)** cells at 5 m → 1280 m x 80 m
(raw simulation is 20 x 256; preparation crops transverse 20 → 16).
Heat pump fixed at cell (23, 7) for all 100 samples; only permeability and
pressure gradient vary. Labels are at **5 years** (not steady state).

| File | Shape | Content |
|---|---|---|
| `Inputs/RUN_i-1.pt` | (4, 256, 16) f32 | ch0 Liquid Pressure [Pa], ch1 Permeability X [m^2] (constant/sample), ch2 SDF, ch3 Material ID one-hot — "pksi"; ch0/1/3 Rescaled to [0,1], SDF raw |
| `Labels/RUN_i-1.pt` | (1, 256, 16) f32 | Temperature [C] Rescaled to [0,1]; denorm range [10.60, 15.60] C |
| `InputParams/RUN_i.pt` | (4,) | [hp_x=50 m, hp_y=120 m, permeability 1.03e-11..4.37e-9 m^2, pressure grad -0.0035..-0.0015] |
| `LearnableParams/RUN_i-1.pt` | (4,) | [1K-isoline length (m), width (m), T_max (C), T_max location as **flattened cell index** (23*16+7=375)] |

Normalization stats: `info-1.yaml` (min/max/mean/std per channel, "Rescale"
convention from 1HP_NN `NormalizeTransform`). **39/100 samples have NaN**
learnable params (plume exceeds domain: 18 lengths, 32 widths NaN) — filter
before scalar regression; the QCNN paper drops the same edge cases.

## Protocol (ours, frozen)

- Loader: `paper/geothermal/data.py`. Split: **70/20/10** (1HP_NN's hardcoded
  `[0.7, 0.2, 0.1]` ratio) with **seed 20260816**, frozen in `splits.json`;
  `split()` cross-checks the file against the seeded derivation and refuses to
  run on drift. 1HP_NN itself uses `random_split` seed 1 — we keep their ratio,
  not their seed, and report our split for reproducibility.
- Two tracks:
  1. **Field track** (primary): Inputs → Labels, loss MSE on normalized fields
     (1HP_NN convention), report denormalized **MAE [C]**, **PAT@0.1C**
     (LGCNN metric: fraction of cells with |err| > 0.1 C), per-sample max
     error, and fraction of samples with max error < 0.5 C.
  2. **Scalar track** (QCNN comparator): InputParams → LearnableParams
     (4 scalars), NaN-filtered, MSE. arXiv:2606.23411 uses 264 train / 90 test
     of 352 samples on the 1000-dp prepared variant; with only the 100-dp set
     on disk we mirror their *protocol* (scalar outputs, MinMax+Standard
     scaling, batch 16) at our split, and scale to 1000 dp later by running
     1HP_NN `prepare_dataset.py` on darus-3650's `RUNs.zip`.
- Test set is 10 samples at 100 dp — report mean ± std over **3 seeds** and
  add 5-fold CV before making claims; the 1000-dp preparation removes this
  fragility and is the path to the headline table.

## Numbers to beat

| Baseline | Setting | Number | Source |
|---|---|---|---|
| **Pelzer & Schulte 2024 two-stage U-Net** | this exact domain, 1000 dp, 16x256, 100 test dp | **MAE 0.046 C** (worst cell ~0.21 C at injection point) | cited in arXiv:2606.23411; Geoenergy Sci. Eng. 2024 |
| 1HP_NN reference model | `UNet(init_features=32, depth=3, kernel=5)`, inputs "gksi", MSE loss | architecture + protocol reference | github.com/JuliaPelzer/1HP_NN |
| TurbNet-Geo-32 (Davis et al.) | **different dataset**: 65x65 cells @ 2 m, 650/150 split | 16.0% of test samples with max err < 0.5 C; conditional median pixel err < 0.4 C (among pixels with err > 0.2 C) | arXiv:2302.08199 |
| TurbNet (Leiteritz et al.) | lab-scale 64x64, 128 mm domain | 0.68% relative error | arXiv:2203.14961 |
| LGCNN / U-Net (Pelzer et al. 2025) | large domain 2560^2 @ 5 m | U-Net MAE 0.0416 C, PAT 11.69% (73 sims); LGCNN 0.0905 C, PAT 28.92% (1 sim) | arXiv:2507.06062 |
| QCNN (Maheshwari et al.) | scalar track, IQP encoding + quantum conv/pool, ibm_kyiv | **no numeric MSE published** — classical CNN beat all quantum variants; loss-curve ordering only | arXiv:2606.23411 |

Primary target: **beat MAE 0.046 C on the 1000-dp field task** with a quantum
bottleneck adapter matching (or shrinking) the classical parameter budget — and
beat the QCNN comparator's own classical CNN on the scalar track, which their
quantum model does not.

Caveats to phrase carefully when citing (verified against sources): the
"<0.4 C median" is conditional on pixels already erring > 0.2 C; the "~16%"
figure is TurbNet-Geo-32 on the 2 m-resolution Davis dataset, not this one;
the QCNN split is 264/90 (its "352 total" vs 264+90 is the paper's own
inconsistency); the QCNN paper states no explicit qubit count (symbolic Nq;
127-qubit Kyiv is the backend).

## Planned method: quantum bottleneck in a frozen U-Net

Mirror the TRL2D pipeline (`paper/{model,qlayer,train}.py`):

1. **Pretrain** the 1HP_NN reference U-Net (depth 3, init_features 32, MSE) on
   the field track to reproduce ~0.05 C MAE classically.
2. **Freeze** encoder + decoder; insert the adapter at the bottleneck
   (256-channel, 32x2 spatial): project to n_q qubits (8-12), apply the
   `qlayer.py` variational circuit (statevector sim), project back, residual
   gate initialized to zero.
3. **Fair pairing**: matched-parameter classical bottleneck adapter (MLP/conv)
   trained identically — same budget, same schedule, 3 seeds each (the
   wave-1.5 fair-pairing discipline).
4. **Scalar track**: same frozen encoder, small head on the bottleneck for the
   4 learnable params — a strictly stronger prior than the QCNN's from-scratch
   circuit, isolating "quantum layer quality" from "architecture quality".

## GPU-hour estimate (GB10, 121 GiB unified)

| Stage | Per run | Runs | Subtotal |
|---|---|---|---|
| U-Net pretrain, 100 dp (fields are 4x256x16 — tiny) | ~0.5 h | 3 seeds | 1.5 h |
| U-Net pretrain, 1000 dp (after darus-3650 preparation, CPU preprocessing ~1 h) | ~1.5 h | 3 seeds | 4.5 h |
| Quantum adapter finetune (<=12 qubits statevector, batched) | ~2 h | 3 seeds x 2 configs | 12 h |
| Matched classical adapter control | ~1 h | 3 seeds x 2 configs | 6 h |
| Scalar-track head (both variants) | ~0.5 h | 6 | 3 h |
| **Total** | | | **~27 GPU-h** (+ margin → budget 35) |

Fits in two overnight windows alongside phase-B; no multi-GPU needed.

## Reproduce the data fetch

Fetch/verify script (resumable, md5-checked, skips darus-3652):
scratchpad `fetch_darus.py` — re-run any time; exits nonzero if any file
fails verification. To regenerate the frozen split: `python paper/geothermal/data.py`.
