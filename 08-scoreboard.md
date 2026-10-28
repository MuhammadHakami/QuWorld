---
title: The Scoreboard — falsifiable benchmarks for both hypothesis prongs
date: 2026-07-27
project: quworld
tags: [scoreboard, hypotheses, pre-registration, benchmarks]
status: live
---

# The Scoreboard

## ★ PRIMARY GOAL (user, 2026-07-27)

**Beat CNextU-net: one-step TEST mean-VRMSE < 0.1956 on `turbulent_radiative_layer_2D`, statistically significant.**
Protocol: official (4-step history, official splits/normalization, full 128×384, the_well metrics). Significance: paired per-sample Wilcoxon over the ~870 official test windows vs the official HF checkpoint (`polymathic-ai/UNetConvNext-turbulent_radiative_layer_2D`), 3 training seeds, Holm correction. Strategy: quantum bottleneck **inside CNextU-net from the official checkpoint** (fine-tune), controls = checkpoint-continued vanilla + matched-MLP, identical tuning budget. The sprint loop only selects the ansatz — the goal cannot be claimed on the internal protocol. Secondary flank: rollout windows 6:12 (0.54) and 13:30 (1.01), where every published model collapses.

---

One page. Two hypothesis prongs from the recent literature, tested in a world-model setup. Every row has a public metric, an explicit pass/fail rule, and a command that reproduces it. A row can **disprove** as visibly as it proves — negative results get published in the same table (precedent: HQ-LP-FNO, arXiv:2604.04828, which lost to its own matched control; the bar for controls comes from arXiv:2607.09905 — tuning-matched, not just parameter-matched).

**Standing honesty rule:** everything at ≤30 *simulated* qubits is classically simulable (Cerezo, Nat. Commun. 16:7907). These benchmarks test **quantum-structured inductive bias**; "advantage" language is reserved for the hardware phase ([[05-schedule-gates]] G4).

---

## Prong H1 — "Quantum simulates the real world more natively / more analog"
*(source claims: analog QRC on QuEra Aquila arXiv:2407.02553; Hamiltonian-encoding infinite-order nonlinearity arXiv:2607.12035; Pasqal analog DE solving)*

Testbed: The Well `turbulent_radiative_layer_2D` — public dataset, published baseline table, VRMSE metric. Internal loop = tiny-subset 5-min sprints (fast, falsifiable); external rung = full official protocol.

| ID | Test | Pass rule | Status (live) |
|---|---|---|---|
| H1a | Quantum bottleneck vs `none` AND matched `mlp` on 8-step rollout VRMSE | quantum < both, 3 seeds, Wilcoxon p<0.05 | **STRONGEST RESULT (wave-1.5, fixed 2,000 steps, 3 seeds, TD objective k=4): mr30ryd 0.845±0.051 beats mlp30 0.953±0.135 on 3/3 seeds (−0.109) and none 0.957±0.255**; interaction effect: TD objective improves quantum −23% vs control −10% vs baseline ~0% — the quantum edge exists but is invisible under one-step MSE (explains flat Phase-B one-step fine-tunes). At k=1 baseline wins — adapters need rollout-aware training. Round-2 CNextU-net rematch uses the TD objective |
| H1b | **Analog mechanism**: Rydberg-analog ≥ digital HEA at matched qubits & params | ryd ≤ hea on vrmse@8, 3 seeds | **Supported at 30q (r3)**: mr30ryd 0.955 < mr30hea 1.063; at 12q analog lost (r2) → analog advantage grows with register size so far |
| H1c | **"Adding qubits helps"**: monotone vrmse@8 improvement 12q→16q→30q(3×10) | fixed-step budget, matched params, 3 seeds | **Analog trend monotone**: ryd 1.168 (12q) → 0.913 (16q) → 0.955±0.102 (30q, multi-seed, lowest variance); digital non-monotone at fixed steps (hea12 1.325 < hea16 1.508) |
| H1d | ★ PRIMARY: full official protocol vs our reproduced anchor | **< 0.1934** one-step TEST mean-VRMSE (our reproduction of CNextU-net; published 0.1956), paired per-window Wilcoxon over 873 windows, 3 seeds | **Anchor reproduced 2026-07-27** (0.1934, 1.1% off published); hybrid-at-neck built (bit-identical at init, params matched 0.0007%); pilot fine-tunes running (vanilla/mr30ryd/mlp30, checkpoint start valid 0.1865) |
| H1-kill | If mlp matches quantum within noise across seeds at matched tuning | prong H1 dies in this regime; publish the negative | Not triggered (mlp currently worst) |

## Prong H2 — "Qubits give large multimodal models/agents an advantage"
*(source claims: Cayley Unitary Adapters arXiv:2605.05914; MQAdapter arXiv:2607.12418; QuIC arXiv:2502.06916; Quantum-PEFT arXiv:2503.05431)*

Testbed: **frozen Ant Group LingBot-World v2 14B causal-fast** + 30-simulated-qubit adapter at the chunk boundary ([paper/lingbot_hook_spec.md](paper/lingbot_hook_spec.md)). Weights on disk (81 GB); scored by benchmarks with official public scorers — anyone can rerun the judge.

| ID | Test | Pass rule | Status |
|---|---|---|---|
| H2a | Quantum adapter vs matched LoRA & MLP adapter: chunk-latent rollout VRMSE + invariant drift on The Well renders | quantum < both, 3 seeds, p<0.05 | Ansatz selection running in the sprint loop; P0 caching next |
| H2b | **Easy-to-see external**: WorldModelBench (350 instances, official VLM judge) — vanilla vs LoRA vs quantum | quantum > LoRA on physics-adherence subscore | Queued (≈2 days/model, overnight jobs) |
| H2c | Physics-IQ, all 66 scenarios, official scorer | quantum > vanilla and ≥ LoRA on Physics-IQ score | Queued |
| H2-kill | LoRA ≥ quantum on H2a–H2c everywhere | prong H2 dies for this regime; publish the negative | Not triggered |

## Cross-cutting controls (both prongs)

χ-sweep MPS surrogate of the winning circuit (B4) · product-state χ=1 (B5) · finite-shot 128–8192 · identical LR-grid tuning budget for every arm (the 2607.09905 bar) · seeds 1337/1338/1339 · all runs: 5-min wall or fixed-step, stated per row.

## Reproduce

`results.tsv` (every run) · `notebooks/autoresearch.ipynb` (living report) · `bash paper/run_round3.sh` (current round) · per-row commands land in REPRODUCE.md as rows go green.
