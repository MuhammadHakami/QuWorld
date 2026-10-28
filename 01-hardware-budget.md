---
title: Hardware Budget — DGX Spark (GB10), measured
date: 2026-07-26
project: quworld
tags: [hardware, budget, cuda-q, measured]
status: measured-on-box
---

# Hardware Budget (measured, not estimated)

All numbers below were **measured on the actual target machine** on 2026-07-26 — this box *is* the DGX Spark. See [[03-pipeline]] for how the budget is spent and [[05-schedule-gates]] for the gates that consume it.

## Machine (verified via nvidia-smi / free / df)

| Component | Verified value |
|---|---|
| GPU | NVIDIA GB10 (Grace-Blackwell), single GPU, no MIG |
| Unified memory | 121 GiB usable (LPDDR5x, ~273 GB/s) + 15 GiB swap |
| CPU | 20 Grace cores (aarch64) |
| CUDA | 13.0, driver 580.159.03 |
| Storage | 3.7 TB NVMe, 2.4 TB free |
| Python stack | Python 3.12.3, uv 0.11.2, CUDA-Q 0.15.0 (cu13 wheels, aarch64) |

## G0 gate result: PASSED (2026-07-26)

CUDA-Q 0.15.0 installs from PyPI with **cu13 aarch64 wheels** and the `nvidia` GPU statevector target runs on GB10. Benchmark: 6-layer hardware-efficient ansatz (RY·RZ + CZ ring — same shape as our adapter PQC), sum-of-Z observable, fp32/complex64. Script: [bench/g0_bench.py](bench/g0_bench.py).

| n qubits | Statevector | PQC params (L=6) | Forward observe, measured | Torch-sim grad step est. (~3× fwd) |
|---|---|---|---|---|
| 12 | 32 KB | 144 | 1.70 ms | ~5 ms |
| 16 | 0.5 MB | 192 | 2.24 ms | ~7 ms |
| 20 | 8 MB | 240 | 3.40 ms | ~10 ms |
| 24 | 128 MB | 288 | 30.0 ms | ~90 ms |
| 26 | 512 MB | 312 | 130 ms | ~0.4 s |
| 28 | 2.1 GB | 336 | 560 ms | ~1.7 s |
| 30 | 8.6 GB | — | inference-only | — |
| 32 | 34 GB | — | inference-only | — |
| 33 | 69 GB | — | showcase ceiling | — |
| 34 | 137 GB | — | **does not fit** | — |

`tensornet-mps` target: **works** (20q observe in 1.7 s) → the χ-sweep dequantization ablation is feasible on-box.

## Consequence 1 — qubit budget (unchanged from draft, now evidence-backed)

- **Main sweep: n ∈ {12, 16, 20}** — gradient steps are millisecond-scale; the full ablation grid is cheap.
- **Headline config: n = 24** — ~0.1 s/sample-gradient → batch-8 step ≈ 1 s → ~3.5k steps/hour in the backbone-free P1 mode. Meets G1 (≥1k steps/hour) with margin.
- **One n = 28 run** — ~1.7 s/sample-gradient; short schedule only.
- **n = 30–33: inference-only** (expressivity scaling, entanglement entropy, MPS cross-check). 33q is the hard ceiling; never write 34q anywhere.

## Consequence 2 — gradient strategy REVISED (G0 finding)

CUDA-Q 0.15.0's Python API exposes only finite-difference and `ParameterShift` gradients — **no adjoint differentiation**. Parameter-shift on our ~500 params at 24q ≈ 2 × 500 × 30 ms ≈ 30 s per gradient: unusable for training.

Revised stack:
1. **Primary (training): hand-rolled PyTorch complex64 statevector propagator.** Gates as batched einsum/tensor contractions on a (batch, 2^n) tensor; exact gradients via autograd (adjoint-equivalent cost, ~3× forward); integrates natively with the adapter's `W_in`/`W_out` graph — no custom `autograd.Function` bridge needed at all. Memory: n=24 with ~40 gate intermediates ≈ 5 GB — fits the P1 5 GB envelope. For n=28 use gate-checkpointing (recompute in backward).
2. **Validation + ablations: CUDA-Q.** Cross-check torch-sim expectation values against `cudaq.observe` (tolerance 1e-5) at every n; use `tensornet-mps` for the χ-sweep; use CUDA-Q sampling for the finite-shot ablation.
3. **Fallback: PennyLane `lightning.gpu` adjoint** if the torch propagator underperforms (aarch64 wheel status: see [[06-verification-log]]).

This is a *simplification*: the original plan needed a custom torch↔CUDA-Q autograd bridge; now training is pure PyTorch and CUDA-Q is the independent referee.

## Consequence 3 — memory envelopes per run mode

| Mode | Resident | Notes |
|---|---|---|
| P0 caching (backbone fwd, bf16) | ~30–60 GB | 14B-class ≈ 28 GB weights + VAE + KV; write latents to NVMe |
| P1 adapter training (backbone-free) | ~5 GB | cached tensors + torch statevector + 2 small MLPs |
| P2 closed-loop refinement | ~75 GB | last-blocks backprop, batch 1–2, grad accumulation, activation ckpt |
| 33q inference showcase | ~70 GB | single statevector + workspace |

Disk: cached latents for 3 datasets + teacher-forced predictions budgeted ≤ 1.5 TB (of 2.4 TB free); re-measure after P0 on the first dataset.
