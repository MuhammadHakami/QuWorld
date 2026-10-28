---
title: Schedule, Gates & Risk Register
date: 2026-07-26
project: quworld
tags: [schedule, gates, risks, venues]
status: active
---

# Schedule, Gates & Risks

## Venue reality (verified 2026-07-26 — supersedes all earlier drafts)

| Venue | Verified status | Action |
|---|---|---|
| **ICLR 2027** | Dates **NOT announced** (iclr.cc/2027 is 404; earlier "Sep 19/24" were ICLR 2026 dates shifted) | Primary target; plan to a **late-Sep 2026 working assumption**; check iclr.cc weekly |
| AAAI-27 | Abstracts closed Jul 21, 2026; papers Jul 28 | **Dead** for us |
| QTML 2026 (Stellenbosch, Dec 6–11) | Submissions closed Jul 7, 2026 | **Dead** as fallback; attend/poster only if late-breaking track appears |
| **NeurIPS 2026 workshops** (Sydney, Dec 6–12) | Contribution deadline **~Aug 29, 2026** (notif. Sep 29) | Live intermediate target: 4-page workshop version of pillar-2 (dequantization study) |
| ML4PS 2026 | Not yet announced | Re-check Aug 2026 |
| Journal ladder (extended version) | PRX Quantum / npj QI / QST / MLST / QMI all accept hybrid-QML; Schuld & Killoran PRX Quantum 3, 030101 is the de facto claims standard | Unchanged |

**G3 pivot rule updated:** QTML is closed, so the pivot target on a null result is **NeurIPS 2026 workshop (Aug 29) + ML4PS when announced**, centering the ablation science.

## Gates

| Gate | Test | Status / rule |
|---|---|---|
| **G0** | CUDA-Q runs on GB10 aarch64; gradients cross-checkable | ✅ **PASSED 2026-07-26, measured** — CUDA-Q 0.15.0 cu13 wheels work; `nvidia` + `tensornet-mps` targets OK; torch↔CUDA-Q agreement 1.0e-6; *no adjoint in CUDA-Q* → torch propagator is the training path ([[01-hardware-budget]]) |
| **G1** (week 1) | P1 throughput | ✅ at n≤20 (measured ~4k steps/h). **Re-scoped for n=24:** propagator-v2 must reach ≤1 s/sample fwd+bwd (≥3× current 2.65 s); else headline drops to n=20 and n=24 becomes a 1-dataset, 3-seed secondary |
| **G2** (week 3) | LingBot-World v2 status | Partially resolved: 14B causal-fast **released** (Jul 9, CC BY-NC-SA); 1.3B + causal-pretrain still unreleased → 14B is the generality track now; 1.3B becomes fast-iteration model *if* it lands |
| **G3** (week 4) | H1 on `turbulent_radiative_layer_2D` | If LoRA ≥ HQ-SSA everywhere on drift → pivot per venue table above |
| **G4** (week 6) | Pulser export fidelity > 0.99/layer | Verified feasible: Pulser DMM supports local detuning + QutipEmulator; **Aquila constraint discovered:** Braket local detuning is an experimental per-task capability with **fixed spatial pattern, detuning ≤ 0, degraded T2** — the exported program must respect the sign constraint or target Pasqal cloud instead; paper stands without any hardware run |

## Wall-clock budget (box runs 24/7; re-measured where marked ✓)

| Job | Estimate |
|---|---|
| P0 caching (3 datasets + teacher-forced pass + probe) | 3–4 days |
| P1 quantum configs, n≤20 grid + 5-seed headline ✓measured | **~2 days** (was 5–6 — torch propagator at 20q is fast) |
| P1 n=24 headline (3 seeds, 1 dataset) | 2.5–7.5 days, gated on propagator-v2 |
| P1 classical zoo (all baselines × datasets × seeds) | 2–3 days |
| P2 closed-loop refinement (best 6 configs) | 2–3 days |
| Track B: WorldModelBench 350×3 + Physics-IQ full 66 | 7–9 days |
| Dequantization suite (χ-sweep, shots, entropy, variance) | 2 days |
| 30–33q inference showcase | 0.5 day |
| **Total box-time** | **~22–31 days** — fits a late-Sep deadline only if P0 starts within days |

## Risk register

1. **ICLR 2027 dates unknown** — could land earlier than assumed; mitigation: NeurIPS-workshop version by Aug 29 forces the core results forward anyway.
2. **Propagator-v2 misses 3×** — fallback: PennyLane `lightning.gpu` adjoint (aarch64 wheels verified, CUDA-12 stack alongside cu13 needs its own venv + subprocess isolation), else n=20 headline.
3. **Cosmos-Predict2.5 docs quote ~33 GB min for 2B** — likely includes text encoder + no offload; P0 measures true footprint day 1; 14B stays P0-cached regardless.
4. **cudaq/torch process clash** — solved by subprocess isolation ([[03-pipeline]]); do not regress by importing both anywhere.
5. **MQAdapter-line convergence** (arXiv:2607.12418) — quantum adapters are moving toward multimodal; every idle week raises scoop risk on the video-world-model slot. Pre-registration commit also timestamps priority.
6. **AirfRANS ODbL share-alike** — derived *data* products must stay ODbL; adapter weights are fine; note in release checklist.
7. Open-loop→closed-loop generalization gap — P2b measures and reports it rather than assuming it away.

## Two honest flags to carry into the writing (unchanged)

The pooled register is a hard information bottleneck (report n/D scaling as evidence it was measured), and hook placement is provisional until the step-1 forward trace of each released checkpoint.
