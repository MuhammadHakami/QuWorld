---
title: Industrial Re-Strategy — energy-domain benchmarks and technique adoption
date: 2026-08-15
project: quworld
tags: [strategy, industrial, energy, benchmarks, competitors]
status: adopted-2026-08-15
---

# Industrial Re-Strategy (post-sweep, 2026-08-15)

Framing (user goal): fault-tolerant large-qubit machines are decades out → the near-term claim is **hybrid qubit+world-model > world-model alone for analog-like physical simulation, today, with a native analog-hardware export path**. Competitor landscape + technique trends: [notebooks/04_competitors.ipynb](notebooks/04_competitors.ipynb). Full sourced intel: workflow `wf_bcf7b16c-448`.

## 1. Competitive reality check

- **Interactive world-model lane** (Cosmos 3, Genie 3, LingBot-World 2.0, Matrix-Game): **zero architectural physics mechanisms** — physics is eval-level or emergent. Our C5 slot (quantum module in a frozen world model) remains uncontested.
- **PDE lane:** official TRL2D 0.1956 (CNextU-net) survives the billion-param FMs (PhysiX 0.2098, PDE-FM 0.2321; Walrus/Poseidon numbers are median-protocol).
- **⚠ WaveLiT threat (arXiv:2605.25949):** claims TRL2D **0.1308** (L) / 0.1346 (B, 9M) under a *claimed* Well-consistent protocol (workshop paper, 2D-only, no MHD). **Response:** (a) pre-submission task: verify their protocol from their appendix/code; (b) paper wording: our goal is the strongest *official* baseline under the original mean-VRMSE protocol, reproduced by us at 0.1934, with WaveLiT cited as a concurrent non-official entry and, if protocol-comparable, acknowledged as the stronger community number; (c) our contribution is checkpoint-surgery + paired significance, which no entry (WaveLiT included) provides. Do **not** stake the paper on "we hold SOTA".
- "Physics Transformer" TRL2D 0.15: internally inconsistent baselines (their FNO 0.29 vs official 0.5001) → cite as non-comparable.

## 2. Technique adoption (steal list → our pipeline)

| Adopt | Where |
|---|---|
| **Delta-prediction + per-field normalization** (Walrus-credited) | Phase-B fine-tune arms: predict Δu — near-zero-cost change, test in next pilot round |
| **Train-time jitter** (harmonic noise injection) | Phase-B round 2 ablation |
| **Operator+generative split framing** | Position the quantum bottleneck as the **high-frequency correction pathway** on a frozen deterministic backbone (arXiv:2507.02106 line) — this is *the* on-trend architectural story for C2 |
| **Rollout-aware eval always** | Never report one-step alone (The Well authors note the CNextU-net edge "quickly dissipates"): report 6:12 + spectral error in every table |
| **Test-time BoN + physics verifier** (Round-Trip Consistency 2608.00675) | Cheap H2-track addition for the world-model prong |

Avoid (measured failures): soft PINN penalties, global spectral-norm constraints, discrete tokenizers on fields, scale-as-physics-fix.

## 3. Energy-domain benchmark ladder (all single-GPU, published baselines)

| Priority | Domain | Benchmark | Beat | Size | Status |
|---|---|---|---|---|---|
| **P1 (now)** | Turbulent mixing (baseline arena) | The Well TRL2D | 0.1934 (our anchor) | 6.9 GB ✓ local | Overnight campaign |
| **P2 (next, same harness)** | **Fusion** — plasma/blanket MHD class | The Well **MHD_64** | **CNextU-net 0.1633** (rollout 6:12 ≥1.24 all) | 71.6 GB | Identical loader/VRMSE/ckpt surgery — lowest-friction second pillar; stretch add-on: TokaMark Task 4-4 (real MAST data, ~25 GB) |
| **P3** | **Nuclear molten-salt** — natural-circulation convection proxy | The Well **rayleigh_benard** (Pr/Ra slice of 342 GB) | TFNO 0.6566 | slice ~50-80 GB | State limitation openly: no internal heating (fueled-core Nusselt deviates ~70%, arXiv:2507.08515); baselines are weak here → biggest headroom |
| **P4** | **Nuclear LWR/HWR thermal-hydraulics** | **BubbleML 2.0** boiling subsets | published UNet/FNO unstable at 800+-step rollouts | tens of GB | Different loader — post-deadline; NEA CHF (tabular) as credibility add-on |
| **P5 (stretch)** | **Water waves** | The Well **planetswe** | TFNO 0.0853 (hard) | 185.8 GB | Honest stretch goal; CNextU-net is weak here (0.3724) so ckpt-surgery story changes |
| — | **Geothermal** | none exists | — | — | Future-work paragraph only (open-DARTS benchmark = follow-up paper opportunity) |

Disk check before P2/P3 downloads: ~2.3 TB free minus LingBot 81 GB — MHD_64 (71.6) + rayleigh_benard slice fits comfortably.

## 4. Quantum-energy positioning (credible citations only)

Use: Pasqal–EDF/GENCI hardware runs; Pasqal–Aramco 200q QCaaS (May 2026); **ORNL/IBM FLiBe molten-salt blanket chemistry within 0.7 kcal/mol of FCI (arXiv:2606.30402)** — the same molten-salt system our P3 proxy models thermally; fault-tolerant-era quantum-PDE honesty via Joseph et al. (Phys. Plasmas 30:010501); MPS/turbulence interscale-correlation motif (PRR 7:013112). Avoid: market-size figures, "demonstrated quantum advantage" in grid/CFD (none exists), vendor timelines.

**The pitch sentence:** utilities already run the neutral-atom hardware family we emulate (EDF, Aramco); quantum chemistry already handles molten-salt electronic structure (ORNL); our hybrid closes the missing middle — the *thermal-hydraulic/MHD field dynamics* — with a ~500-parameter analog-native module inside today's classical surrogates, benchmarked under pre-registered kill rules.

## 5. Sequencing vs the Aug 29 deadline

Workshop paper = P1 result (whatever the verdict) + C5 world-model prong + this positioning. P2 (MHD_64) starts the moment the overnight campaign's GPU frees — it reuses every script (`build_hybrid` on the MHD CNextU-net checkpoint `polymathic-ai/UNetConvNext-MHD_64`). P3-P5 + BubbleML/TokaMark = the ICLR 2027 full paper's evidence spine.
