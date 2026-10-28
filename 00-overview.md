---
title: quworld — Overview & Novelty Position
date: 2026-07-26
project: quworld
tags: [overview, novelty, strategy]
status: active
---

# quworld — Physics-Consistency Quantum Adapter for Frozen World Models

**One-sentence claim:** a ~500-parameter, hardware-native (Rydberg-generator) quantum adapter, injected as a zero-init gated residual into a *frozen* video world foundation model, reduces physical-invariant drift (mass / energy / enstrophy / divergence) over long rollouts — demonstrated with parameter-matched classical controls and a full dequantization ablation, trained end-to-end on **one DGX Spark (126 GB)**.

Framing: **"quantum-ready, honestly controlled"** — never quantum advantage. The de facto reviewer standard is Schuld & Killoran, PRX Quantum 3, 030101; we write to it.

## Novelty position (verified against arXiv/OpenReview as of 2026-07-26)

The scoop scan ([[06-verification-log]] §3) confirms the **full conjunction still stands**: no existing work combines (a) a quantum/unitary adapter with (b) a frozen *video world model*, (c) conservation-law / rollout-drift objectives, and (d) a hardware-native Rydberg ansatz. But the *generic* mechanism claim is taken, so the paper's novelty statement is **domain + objective + ansatz + controls**, never "first unitary adapter":

- **Cayley Unitary Adapters** (arXiv:2605.05914, May 2026, Multiverse/Orus) — unitary adapters in frozen LLMs, ~6k params, 156-qubit IBM hardware. *Closest work; cite prominently; we differ in domain (video world models), objective (physical invariants), ansatz (analog Rydberg generator with geometry-as-parameter), and controls.* Note: this paper is REAL — an earlier draft wrongly called it hallucinated.
- **QuIC** (arXiv:2502.06916) and **Quantum-PEFT** (arXiv:2503.05431, ICLR 2025) — quantum-inspired/unitary PEFT prior art; cite as mechanism precedent.
- **MQAdapter** (arXiv:2607.12418, Jul 2026) — quantum adapters entering multimodal VLMs; the window is closing from the quantum-PEFT side. **Ship fast.**
- **Physically Native World Models** (arXiv:2605.00412) — classical Hamiltonian latent dynamics for world models; owns the adjacent *classical* framing; engage in related work, position ours as the parameter-efficient adapter route with a quantum-hardware exit.

## Backbones (verified checkpoints, [[06-verification-log]] §2)

| Role | Model | Why |
|---|---|---|
| Primary training backbone | **Cosmos-Predict2.5-2B** (flow-matching DiT in Wan2.1 VAE latent space, Cosmos-Reason1 cross-attn; arXiv:2511.00062) | Clean documented latent tap; comfortable on the Spark |
| Headline transfer | **Cosmos-Predict2.5-14B** | Same architecture family, P0-cached passes fit |
| Architecture-generality track | **LingBot-World v2 14B causal-fast** (HF `robbyant/lingbot-world-v2-14b-causal-fast`, CC BY-NC-SA 4.0, rel. 2026-07-09) | Chunk-boundary tap: `--chunk_size` = **4 latent frames** (frame_num = 4n+1), `--local_attn_size 18`, `--sink_size 6`; 480×832 config |
| Eval-only stretch | **Cosmos3-Nano 16B** (MoT omnimodel, OpenMDW-1.1) | "Works on the newest omnimodel" appendix; MoT two-tower AR+diffusion — *no* clean unidirectional tap, so eval-only |

Licensing verified: OpenMDW-1.1 / NVIDIA Open Model License both permit releasing adapter weights (attribution + guardrail terms); LingBot CC BY-NC-SA fits a non-commercial paper.

## The three evidence pillars

1. **Parameter-matched classical controls** — LoRA, Cayley-orthogonal, MPO/tensor-train, product-state (χ=1), all ~1.5M total including projections, identical `h_t → residual` interface ([[02-architecture]]).
2. **Dequantization suite as mandatory headline** — χ-sweep (CUDA-Q `tensornet-mps`, verified working on-box), entanglement-entropy vs gain, finite-shot gradients, barren-plateau variance scan; addresses Cerezo et al. (Nat. Commun. 16:7907) head-on.
3. **Invariant-drift benchmarks** on verified datasets ([[04-benchmarks]]) + full WorldModelBench (350 instances) + full Physics-IQ (66 scenarios — the "100-scenario subset" idea was moot; 66 is all of it).

## Deadline reality (corrected — see [[05-schedule-gates]])

**ICLR 2027 dates are NOT announced** (the Sep 19/24 dates in earlier drafts were ICLR 2026's, shifted a year). Working assumption: late Sep 2026, monitor iclr.cc. AAAI-27 abstracts **already closed** (Jul 21). QTML 2026 submissions **closed** (Jul 7). The live intermediate target: **NeurIPS 2026 workshops (Sydney), contribution deadline ~Aug 29, 2026**.

## File map

[[01-hardware-budget]] — measured GB10 numbers, qubit budget, gradient strategy ·
[[02-architecture]] — ansatz, hooks, objective, parameter budget ·
[[03-pipeline]] — P0/P1/P2 caching-first training ·
[[04-benchmarks]] — datasets, baselines, metrics, pre-registration ·
[[05-schedule-gates]] — timeline, gates G0–G4, risks ·
[[06-verification-log]] — every claim fact-checked 2026-07-26 ·
[bench/](bench/) — G0/G1 scripts + measured results
