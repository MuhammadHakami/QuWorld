---
title: Benchmarks — Credible Evidence Plan (verified datasets)
date: 2026-07-26
project: quworld
tags: [benchmarks, datasets, evaluation, pre-registration]
status: active
---

# Benchmark & Evidence Plan

Every dataset fact below was web-verified on 2026-07-26 ([[06-verification-log]] §2). Design goal: on single-box compute, **pre-registration + parameter-matching + mandatory dequantization controls** are what convert "small study" into "credible study".

## Datasets (Track A — latent physics rollouts)

| Dataset | Verified size | License | Loader | Role |
|---|---|---|---|---|
| The Well `turbulent_radiative_layer_2D` | **6.9 GB** | CC-BY-4.0 | `the_well` | Primary turbulence testbed (headline configs) |
| The Well `MHD_64` | **71.6 GB** | CC-BY-4.0 | `the_well` | Fusion-adjacent MHD testbed |
| AirfRANS | 10 GB preproc (71.3 GB raw) | data **ODbL-1.0** (share-alike; fine for academic use — flag in paper), loader MIT | `airfrans` | Aerodynamics / solid-geometry testbed |
| ~~The Well `shear_flow`~~ | 547 GB | — | — | **Dropped: exceeds disk budget** |
| ~~PDEBench `ns_incom_inhom_2d_512`~~ | **2.3 TB total** | — | — | **Demoted to optional**: selective per-file download from DaRUS only if a reviewer-facing gap appears; The Well covers NS-class dynamics |
| JHTDB (appendix only) | cutout slices | token required | `giverny` | 2–3 canonical isotropic-turbulence slices for a qualitative figure |

Preprocessing: z-score per channel on train split; (u, v, scalar) → 3-channel frames at backbone-native resolution; encode once with frozen VAE; cache latents ([[03-pipeline]] P0a); split by trajectory, never by frame; context 33 frames, predict 1 chunk, autoregressive rollout at eval.

## Track B — video physics (generation-space)

- **WorldModelBench** — verified: 350 instances (7 domains × 50), official `evaluate.py` with VLM judge. Run **all 350 × 3 models** (vanilla, LoRA, HQ-SSA) on Cosmos-Predict2.5; ~2 days/model overnight.
- **Physics-IQ** — verified: **66 scenarios total** (396 videos = 66 × 3 angles × 2 takes). The old "stratified 100-scenario subset" is moot — run the **full benchmark** with the official scorer, take-1 videos.
- VideoPhy-2: dropped (unchanged decision).

## Benchmark suite

| ID | Comparison | Verdict it supports |
|---|---|---|
| B1 | HQ-SSA vs vanilla backbone | adapter helps at all |
| B2 | HQ-SSA vs LoRA (param-matched) | beats the default PEFT |
| B3 | HQ-SSA vs Cayley-orthogonal (param-matched) | *unitarity alone* doesn't explain the gain |
| B4 | HQ-SSA vs MPO / χ-sweep MPS surrogate of **our own circuit** | gain survives dequantization — **mandatory headline** |
| B5 | HQ-SSA vs product-state (χ=1) | entanglement is load-bearing — **mandatory headline** |
| B6 | HQ-SSA vs FNO-class specialist | honest distance to specialist SOTA |
| B7 | Injection site ablation (output-latent vs mid-network) | tap-point choice measured, not assumed |
| B8 | n / D / L scaling + finite-shot (128–8192) + gradient-variance vs n | bottleneck & trainability measured |
| B9 | **Pareto**: trainable params at matched quality (rollout MSE within 1σ of LoRA) | ~500-param quantum core headline; immune to "you under-tuned LoRA" (identical tuning budget by construction) |

## Metrics

Rollout PI-MSE at horizons {10, 50, 100} · invariant drift `|I_k(x̂_T) − I_k(x_0)| / |I_k(x_0)|` for mass, momentum, kinetic energy, enstrophy · `‖∇·û‖²` residual · log-spectral distance · valid-prediction-time (first NRMSE > 0.4; Lyapunov-time units where chaotic) · WorldModelBench + Physics-IQ official scores · wall-clock and parameter overhead per adapter.

## Statistics & pre-registration

- **Pre-register before P1 starts**: hypotheses, dataset subsets, seeds, metrics, gate decision rules — committed to the repo, timestamped hash.
- H1: HQ-SSA < LoRA on invariant drift (primary endpoint, `turbulent_radiative_layer_2D`, horizon 100).
- H2: entangling > product-state (B5). H3: full circuit > low-χ surrogate (B4).
- Seeds: **5 seeds** for headline comparisons at n=20; **3 seeds** at n=24 (throughput-limited, [[03-pipeline]]); 2–3 seeds on grid points. Wilcoxon paired across seeds, Holm correction, effect sizes reported regardless of sign.
- No silent caps: anything bounded (seeds, horizons, dropped datasets) is stated in the paper with the reason.
