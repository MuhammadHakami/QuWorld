---
title: Duplication Audit — verdict matrix and required reframing
date: 2026-08-15
project: quworld
tags: [novelty, audit, citations, actions]
status: verdict-2026-08-15
---

# Duplication Audit (2026-08-15)

Method: 7 finder agents (distinct modalities: exact-system, Rydberg-ansatz, quantum-bottleneck autoencoders, world models, qubit-scaling claims, OpenReview/citation-graph, QRC-for-PDE) → adversarial deep-reads of the 8 closest works → per-claim verdict. Full per-work axis scoring: workflow `wf_8bc5529f-734` journal. **Overall: GO** — nothing occupies the Phase-B slot; speed is risk mitigation (adjacent space moving ~1 paper/month).

## Verdict matrix

| Claim | Verdict | Closest work | Required framing |
|---|---|---|---|
| C1 Rydberg block (input-cond. detunings + trainable geometry + drive) | **PARTIAL** | ResQ (2506.21537: input-cond. detunings, fixed geometry); PulserDiff (2505.16744: differentiable positions, tooling) | Claim only the **composite** (gradient-trained geometry + encoder-driven detunings via statevector autodiff inside a classical predictive net, Pulser-exportable); cite the collage (also ReCon 2408.13389, QGARS 2605.06727); rebut 2504.06702's R⁻⁶ gradient objection (bounded geometry, exact autodiff) |
| C2 Quantum neck in released SOTA + beat published VRMSE | **PARTIAL** | QuFeX/Qu-Net (2501.13165); HQF-Net (2604.06715: frozen DINOv3 + quantum neck); PHQFNO (2507.08746) | "Quantum U-Net bottleneck" exists from-scratch; **zero quantum work on The Well or any released SciML checkpoint**. Frame as **checkpoint-surgery methodology + result** (official ckpt, published 0.1956 target, paired per-window stats) — never as architecture novelty. Add novelty-delta table vs QuFeX/HQ-UNet/PHQFNO/HQ-LP-FNO |
| C3 "Adding qubits improves accuracy" | **DUPLICATED as headline** | IonQ (2504.08732: monotone qubit-scaling beats param-matched classical on pretrained model); QuEra (2407.02553); IonQ (2605.02798) | Retire the sentence as a novelty claim. Re-scope: **first pre-registered, parameter- AND tuning-matched, kill-ruled, significance-tested qubit-scaling study on a public PDE benchmark**; note QuEra's scaling confounds qubit count with input dimension |
| C4 30q = blockade-decoupled 3×10 registers | **PARTIAL** | Patch QGAN (2010.06201); circuit cutting (2412.04844, 2608.05595); QuEra ran ~6 decoupled chains on one array | **Demote to implementation detail** (one method sentence), acknowledge prior art |
| C5 Quantum adapter at KV-cache chunk boundary of frozen video world model + official judges | **NOVEL** (zero hits) | QSWM (2608.05371) is a title collision only (quantum-inspired cellular automata, no quantum module); MQAdapter (2607.12418) is frozen-vision, not generative | **Promote to lead contribution.** No quantum module in any video/world model; WorldModelBench & Physics-IQ have zero quantum entries. Distinguish QSWM prominently |
| C6 Pre-registered kill rules + dequantization suite | **PARTIAL** (components exist, packaging + pre-registration unoccupied) | Bowles/Schuld (2403.07059); χ-sweep theory (2307.06937); tuning-matched null (2607.24065) | Claim the **operationalized two-prong protocol** bound to C2/C5, citing every component. **Most easily scooped — timestamp the pre-registration publicly ASAP** |

## Must-cite additions (by claim)

- **C1:** 2408.13389 (ReCon) · 2506.21537 (ResQ) · 2505.16744 (PulserDiff) · 2605.06727 (QGARS) · 2606.11000 (QA-AEGNN) · 2504.06702 (geometry-gradient critique) · 2107.03247 (QEK) · 2401.09915 (Qadence)
- **C2:** 2501.13165 (QuFeX/Qu-Net) · 2604.27206 (HQ-UNet) · 2604.06715 (HQF-Net) · 2507.08746 (PHQFNO) · 1912.08278 (Mari transfer learning)
- **C3:** 2504.08732 (IonQ FT scaling) · 2605.02798 · 2604.06007 (scaling laws) · 2601.04732 (Freinberger: Wilcoxon, quantum hurts) · 2408.12739 (QCNN simulable)
- **C4:** 2010.06201 · 2412.04844 · 2608.05595 · 2504.10592 · 2605.04991
- **C5:** 2608.05371 (QSWM) · 2605.16404 (Q-MambaVision) · 2410.09846 (QPA) · 2504.05336 (QASA)
- **C6:** 2307.06937 · 2403.07059 · 2306.00061 + 2206.11740 (classical surrogates/shadows) · 2409.01394 (finite-shot QRC) · 2506.20658 (falsifiability) · 2512.02066 (prior paired-Wilcoxon hybrid win) · 2607.24065

## Actions (from audit; status)

1. Retire C3 headline; re-scope → manuscript edit (this session).
2. Reframe C2 as methodology+result; novelty-delta table → manuscript.
3. C1 composite-only claim + 2504.06702 rebuttal → manuscript §method.
4. Demote C4 → manuscript.
5. Promote C5 to lead → manuscript title + contribution 1.
6. **Public timestamp of pre-registration — needs user decision (push repo public / other channel).**
7. Pre-submission follow-ups: Qadence/PulserDiff pass; manual OpenReview sweep; re-check citers of 2509.04006/2604.04828/2603.29407/2605.05914 the week of submission.
8. No experiment redesign required — matched-control suite, kill rules, 12→16→30 ladder stand.
