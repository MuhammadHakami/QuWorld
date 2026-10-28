---
title: Verification Log — every load-bearing claim fact-checked
date: 2026-07-26
project: quworld
tags: [verification, citations, fact-check]
status: snapshot-2026-07-26
---

# Verification Log (2026-07-26)

Method: 8 parallel web-research agents (arXiv/OpenReview/HF/GitHub/official sites, primary sources only) + on-box measurement for everything hardware. Full raw results: workflow `wf_1a1918ce-19c` journal. Nothing below is from memory — every line has a checked source.

## §1 Corrections to earlier drafts

| Earlier draft said | Verified reality |
|---|---|
| ICLR 2027 abstracts Sep 19 / papers Sep 24, 2026, Brazil | **Those are ICLR 2026 dates shifted a year.** ICLR 2027 pages don't exist yet; dates/location TBA |
| AAAI-27 abstracts "early August" | **Jul 21, 2026 — already passed** (papers Jul 28; conf. Feb 16–23 2027, Montreal) |
| QTML = fallback venue | QTML 2026 submissions **closed Jul 7**; Dec 6–11 Stellenbosch, overlaps NeurIPS week |
| "Cayley Unitary Adapters" hallucinated | **REAL**: arXiv:2605.05914 (May 7, 2026, Multiverse/Orus) — closest prior work; must be cited prominently |
| "QENO = quantum-enhanced neural operator" | arXiv:2603.29407 is a hybrid **3D cloud-field forecaster** (its no-advantage disclaimer is real & verbatim) |
| Cosmos 3 "64B unconfirmed"; reasoning→generation tap | Cosmos 3 verified: **Super 64B / Nano 16B / Edge 4B**, OpenMDW-1.1, GTC Taipei Jun 1 2026 — but it's a **two-tower MoT (AR + diffusion), no clean unidirectional tap**; the clean tap lives in **Cosmos-Predict2.5** (flow-matching DiT, Wan2.1 VAE, Reason1 cross-attn; arXiv:2511.00062) |
| LingBot chunk = ~18 frames | `--chunk_size` = **4 latent frames** (frame_num = 4n+1); 18 = `--local_attn_size`, `--sink_size 6`; 480×832 is the 480P config (default 1280×720) |
| CUDA-Q adjoint differentiation | **Not shipped** (issue #4119); Python API: finite-diff + ParameterShift only → torch propagator is the training path ([[01-hardware-budget]]) |
| Qiskit Aer GPU fallback | **x86_64 only** — no aarch64 GPU wheels; source-build only |
| Physics-IQ "100-scenario subset" | Benchmark has **66 scenarios** (396 videos = 66×3 angles×2 takes) → run it in full |
| AirfRANS "permissive" | Data is **ODbL-1.0** (share-alike); loader MIT |
| PDEBench incompressible NS as a main dataset | `ns_incom_inhom_2d_512` = **2.3 TB** → demoted to optional selective download |

## §2 Confirmed load-bearing facts

- **Cosmos-Predict2.5-2B & 14B** downloadable on HF; adapter-weight release permitted (OpenMDW-1.1 / NVIDIA Open Model License, attribution + guardrails). 2B docs quote ~33 GB min VRAM (measure on P0 day 1).
- **LingBot-World v2 14B causal-fast**: HF `robbyant/lingbot-world-v2-14b-causal-fast`, CC BY-NC-SA 4.0, released 2026-07-09; 1.3B + causal-pretrain unreleased as of 2026-07-26.
- **CUDA-Q 0.15.0**: cu13 aarch64 wheels; `nvidia` + `tensornet` + `tensornet-mps` (χ via `CUDAQ_MPS_MAX_BOND`, GPU-only) — all verified **running on this GB10**.
- **PennyLane lightning.gpu 0.45.0**: manylinux_2_28_aarch64 wheels, CUDA-12 stack (adjoint fallback).
- **DGX Spark**: 128 GB LPDDR5x @ 273 GB/s, ConnectX-7 200 Gbps two-Spark pairing (official).
- **Pulser**: DMM local detuning + QutipEmulator classical emulation. **QuEra Aquila via Braket**: local detuning is experimental per-task — fixed spatial pattern, **detuning ≤ 0**, degraded T2 → constrain the export or target Pasqal cloud.
- **The Well**: `turbulent_radiative_layer_2D` 6.9 GB, `MHD_64` 71.6 GB, CC-BY-4.0, loader `the_well`.
- **WorldModelBench**: 350 instances (7×50), official `evaluate.py`, VLM judge.
- **JHTDB**: access via `giverny`, auth token, `getCutout` 2D slices standard.
- **NeurIPS 2026**: Sydney Dec 6–12; workshop contribution deadline ~Aug 29, notifications Sep 29. ML4PS 2026 not yet announced.
- Journals: PRX Quantum, npj QI, QST, MLST, QMI all take hybrid-QML; Schuld & Killoran PRX Quantum 3, 030101 is the de facto claims standard.

## §3 Scoop scan (arXiv/OpenReview through 2026-07-26)

**Verdict: full novelty conjunction STANDS** — no quantum/unitary adapter in a frozen *video world model*, no conservation-drift objective for a quantum adapter, no Rydberg-ansatz adapter anywhere (~40 "quantum adapter" + 3 "unitary adapter" full-text arXiv hits all reviewed). Generic mechanism claim is taken; frame as domain+objective+ansatz+controls.

| Risk | Work | Why it matters |
|---|---|---|
| MEDIUM | Cayley Unitary Adapters — arXiv:2605.05914 (2026-05-07) | Unitary adapters in **frozen LLMs**, ~6k params, IBM 156-qubit hardware. Closest work; cite §1 of related work |
| MEDIUM | QuIC — arXiv:2502.06916 | Quantum-inspired orthogonal adapters for LLaMA/ViT PEFT, 44× compression vs LoRA; mechanism prior art |
| MEDIUM | MQAdapter — arXiv:2607.12418 (2026-07-14) | Quantum adapters entering **multimodal VLMs**; window closing |
| MEDIUM | Physically Native World Models — arXiv:2605.00412 | **Classical** Hamiltonian latent dynamics for world models; owns the adjacent framing |
| LOW | Quantum-PEFT — arXiv:2503.05431 (ICLR 2025) | Unitary parameterization as PEFT precedent |
| LOW | QENO/cloud-fields — arXiv:2603.29407 | Hybrid forecaster, not an adapter; cite for honest-disclaimer framing |
| LOW | arXiv:2605.02798 | Energy-to-solution QFT benchmarking; different substance |
| LOW | Sci. Adv. aec5049 (2026) | Quantum-informed chaos prediction; context citation |

## §4 Citation ledger (verified real)

Quantum-PEFT (ICLR 2025, arXiv:2503.05431) · QuanTA (NeurIPS 2024) · QAA (CIKM 2025) · QTHA/QPA · Cerezo et al., Nat. Commun. **16:7907** (barren-plateau ⇒ classical simulability) · "Role of Quantum in HQC NNs: A Realistic Assessment" (Jan 2026) · Pasqal logical-qubit DE demo (May 2026) · QRC Lorenz on IBM hardware, Comms. Phys. (May 2026) · RF-QRC (Imperial) · VQ-VAE+LBM quantum generative CFD (Dec 2025) · QLBM boundary conditions (May 31, 2026) · Schuld & Killoran, PRX Quantum 3, 030101. **Never cite:** "QCHFT" (still unverifiable — does not exist).

## §5 On-box measurements (this session)

G0 PASSED and G1 partially passed with measured numbers — see [[01-hardware-budget]] and [bench/](bench/): CUDA-Q observe 1.7 ms (12q) → 560 ms (28q); torch propagator fwd+bwd 35 ms (16q) / 114 ms (20q) / 2.65 s (24q, checkpointed, 4.25 GiB peak); torch↔CUDA-Q agreement 1.0e-6; cudaq/torch LLVM process clash → subprocess isolation mandated.
