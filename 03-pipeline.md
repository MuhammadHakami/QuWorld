---
title: Pipeline — Three-Phase Decoupled Training on One DGX Spark
date: 2026-07-26
project: quworld
tags: [pipeline, training, caching, engineering]
status: active
---

# Three-Phase Decoupled Pipeline (caching-first)

Binding constraints (measured, [[01-hardware-budget]]): one GPU, ~273 GB/s bandwidth-bound unified memory, box must run 24/7 with checkpoint/resume. The backbone never sits in the training loop.

## P0 — offline caching (backbone runs, adapter doesn't)

One-time overnight jobs writing to the 4 TB NVMe:

- **P0a** VAE-encode all datasets ([[04-benchmarks]]) to latents once (Wan2.1 VAE for the Cosmos-Predict2.5 track; LingBot's own VAE for its track).
- **P0b** Run the frozen backbone teacher-forced over every training clip; store per step *t*: pooled tap latent `h_t`, backbone's own next-chunk prediction `Ẑ⁰_{t+1}`, ground truth `Z*_{t+1}`.
- **P0c** Fit a **linear invariant probe** `P: latent → (mass, energy, enstrophy)` on cached pairs; report its R² in the paper. The probe replaces per-step VAE decoding in P1's `L_inv`.

Backbone in bf16 (2B ≈ 4 GB, 14B ≈ 28 GB weights + VAE + KV — fits). If a checkpoint unexpectedly exceeds ~20 B: torchao FP8 weight-only (aarch64-safe choice) and say so in the paper.

## P1 — backbone-free adapter training (the workhorse, ~5 GB resident)

Injection at the output latent for this phase: `Ẑ_{t+1} = Ẑ⁰_{t+1} + α ⊙ W_out o(h_t; θ)`. Training touches only cached tensors, the torch statevector propagator, and two small MLPs.

**Measured P1 throughput** (torch complex64 propagator, L=6, fwd+bwd per sample, [bench/torch_sv_bench.py](bench/torch_sv_bench.py)):

| n | fwd+bwd / sample | batch-8 step | steps/hour | 10k-step config |
|---|---|---|---|---|
| 16 | 35 ms | ~0.3 s | ~12k | ~1 h |
| 20 | 114 ms | ~0.9 s | ~4k | ~2.5 h |
| 24 (ckpt) | 2.65 s | ~21 s | ~170 | ~2.5 days |

→ **Grid at n ∈ {12, 16, 20} is cheap; n=24 is expensive as-is.** Propagator-v2 roadmap (in order of expected payoff): (1) batched state tensor `(B, 2^n)`; (2) pair-qubit 4×4 gate fusion (halves full-state passes); (3) `torch.compile` over `layer_fn`; (4) precomputed diagonal masks kept resident. Target ≥3× → n=24 config ≤ 1 day. G1 gate re-scoped accordingly ([[05-schedule-gates]]).

Optimizer AdamW, lr 1e-3 (adapter) / 3e-4 (projections), cosine decay, 500-step warmup, grad clip 1.0, batch 8–16, seeds per [[04-benchmarks]] §stats.

## P2 — limited closed-loop refinement + honest evaluation (~75 GB mode)

- **P2a** Refine best configs with scheduled sampling, backprop through only the final few backbone blocks (early blocks frozen, never materialized in the graph), batch 1–2 + grad accumulation + activation checkpointing. Days, not weeks — starts from P1 weights.
- **P2b** ALL reported results come from full closed-loop rollouts through the real frozen pipeline. The injection-site question (output-latent vs mid-network tap) is a **reported ablation**.

## Engineering constraints discovered on-box (2026-07-26)

1. **`cudaq` and `torch` cannot share a process** on this stack (LLVM CommandLine registry clash → hard abort). All CUDA-Q work (expectation cross-checks, `tensornet-mps` χ-sweeps, finite-shot sampling) runs as subprocess scripts exchanging JSON — pattern established in [bench/cudaq_check.py](bench/cudaq_check.py). The P1 loop is pure PyTorch.
2. Torch propagator correctness is pinned to CUDA-Q: expectation agreement ≤ 1e-5 required at every n before any training run (measured 1.0e-6 at n=12).
3. Job queue: cron + flock (or a simple Python queue-runner) with per-job checkpoint/resume; every script idempotent; W&B (or local CSV) logging.
4. `qiskit-aer-gpu` has **no aarch64 wheels** — not a fallback on this box. PennyLane `lightning.gpu` has aarch64 wheels (CUDA-12 stack) — the only external adjoint fallback if propagator-v2 misses its target.

## Repository layout (implementation, to be created under `impl/`)

`adapters/{pqc_torch.py, lora.py, cayley.py, mpo.py, product.py}` · `backbones/{cosmos25_hook.py, lingbot_hook.py}` · `data/{well.py, pdebench.py, airfrans.py}` · `eval/{invariants.py, spectra.py, wmb_runner.py, physiq_runner.py}` · `quantum_ref/{cudaq_check.py, mps_sweep.py, shots.py}` (subprocess-isolated) · `hardware/pulser_export.py` · Hydra configs · `REPRODUCE.md` mapping every paper table to one command.
