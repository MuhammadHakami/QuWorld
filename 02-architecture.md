---
title: Architecture — Hardware-Native Quantum Adapter (HQ-SSA)
date: 2026-07-26
project: quworld
tags: [architecture, ansatz, rydberg, adapter]
status: stable-pending-backbone-verification
---

# Architecture Specification

One crisp claim: a **~500-parameter, hardware-native quantum adapter** injected as a zero-initialized gated residual into a **frozen** world foundation model reduces physical-invariant drift over long rollouts, evaluated against parameter-matched classical controls. Budget lives in [[01-hardware-budget]]; training flow in [[03-pipeline]]; evidence plan in [[04-benchmarks]].

## Notation

Backbone latent stream `Z_t ∈ ℝ^(N_tok × d)` per frame/chunk; `n` qubits; `L` ansatz layers; trainables `θ = {θ_rot, Ω, δ, τ, r_atoms}` plus classical projections `W_in`, `W_out`, gate `α`.

## Hook points (final module names pend [[06-verification-log]])

- **Primary backbone (Cosmos-class, NVIDIA):** tap the latent stream at the boundary between the reasoning/context blocks and the generation (denoising) blocks — exact module pinned by a forward trace in step 1, since the released architecture may differ from the draft's assumption.
- **Secondary backbone (open interactive world model, LingBot-World-class):** tap the causal state at the KV-cache chunk boundary of the chunked autoregressive rollout — one adapter call per chunk (~1 per 18 frames), not per token.
- In **P1** (backbone-free training, see [[03-pipeline]]) injection moves to the output latent: `Ẑ_{t+1} = Ẑ⁰_{t+1} + α ⊙ W_out o_t`. Injection site (output-latent vs mid-network) is a **reported ablation**, not a hidden assumption.

## Physics register (the stated information bottleneck)

Attention-pool `Z_t → h_t ∈ ℝ^d`, then `φ_t = π·tanh(W_in h_t) ∈ [−π, π]^(2nD)` for `D` data-re-uploading rounds. Only ~3n expectation values come back out — poly(n) readout is a bottleneck **by construction**; we report performance scaling vs `n` and `D` as evidence we measured it.

## Ansatz — Rydberg-native learned generator

Each layer ℓ: trainable single-qubit `R_Z R_Y` rotations, then an analog block `U_ℓ = exp(−i H_Ryd(ℓ) τ_ℓ)` with

```
H_Ryd/ħ = Σᵢ (Ω_ℓ/2)Xᵢ − Σᵢ δ_{ℓ,i} nᵢ + Σ_{i<j} (C₆/|rᵢ−rⱼ|⁶) nᵢnⱼ ,   nᵢ = (I−Zᵢ)/2
```

Trainables: global Rabi `Ω_ℓ`, local detunings `δ_{ℓ,i}`, evolution times `τ_ℓ`, and the 2D atom coordinates `rᵢ` themselves (≥4 μm spacing, ≤100 μm field). **Geometry-as-parameter** makes the trained object a valid neutral-atom program → Phase-2 export to Pulser ([[05-schedule-gates]] G4).

Why this defends the physics claim: the adapter is unitary evolution under a learned Hamiltonian generator; conserved observables of that generator are tied to physical invariants via an explicit loss term — a testable mechanism, **not** a claim that unitarity "natively preserves" physical conservation laws (it conserves only the latent L2 norm; say so in the paper).

## Readout and injection

Measure `O = {⟨Zᵢ⟩, ⟨Xᵢ⟩, ⟨ZᵢZⱼ⟩_NN}` (≈3n exact expectation values in simulation) → `W_out` MLP → ℝ^d → zero-init gated residual `Z'_t = Z_t + α ⊙ broadcast(W_out o_t)`, `α` init 0 (provably non-destructive at step 0).

## Objective

```
L = L_pred + λ₁·L_inv + λ₂·L_spec + λ₃·‖α‖₁
```

- `L_pred`: latent MSE + β·decoded MSE (β ≈ 0.1) — in P1 the decoded term uses the **invariant linear probe** (see [[03-pipeline]] P0c), not per-step VAE decoding.
- `L_inv`: relative drift of mass ∫ρ, momentum ∫ρu, kinetic energy ∫½ρ|u|², enstrophy (2D), plus `‖∇·û‖²` where incompressible.
- `L_spec`: log-spectral distance over radially binned energy spectra.
- `‖α‖₁` keeps the residual gate sparse so gains are attributable.
- Sweep λ₁ ∈ {0.1, 1, 10}, λ₂ ∈ {0.01, 0.1} once on dataset 1, then freeze.

## Parameter budget (n = 24, L = 6, D = 2, d = 4096 example)

| Component | Shape | Params |
|---|---|---|
| Attention-pool query | d | 4,096 |
| W_in | d × 2nD | ~393k |
| **PQC: θ_rot, δ, Ω, τ, coords** | — | **~530** |
| W_out MLP | 3n → 256 → d | ~1.07M |
| Gate α | d | 4,096 |
| **Total per hook** | — | **~1.5M** |

Every baseline is built to the same ~1.5M total **including** `W_in`/`W_out`. The ~500-parameter quantum core is the efficiency headline and what makes the entanglement ablation meaningful.

## Baseline zoo (identical `h_t → residual` interface, parameter-matched)

1. LoRA (rank set to hit ~1.5M),
2. classical Cayley-orthogonal adapter,
3. MPO/tensor-train adapter,
4. product-state (χ=1) adapter,
5. one specialist neural operator (FNO-class) as the non-adapter reference.

## Gradient path (REVISED after measured G0 — see [[01-hardware-budget]])

Training uses a pure-PyTorch complex64 statevector propagator (per-layer activation checkpointing at n≥24); CUDA-Q is the independent referee (expectation cross-check ≤1e-5, `tensornet-mps` χ-sweep, finite-shot sampling ablation). No torch↔CUDA-Q autograd bridge exists in the training loop anymore.
