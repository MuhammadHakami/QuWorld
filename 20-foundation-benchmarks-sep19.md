---
title: Foundation world-model benchmarks — verified SoTA (Sep 19 2026)
date: 2026-09-19
project: quworld
tags: [benchmarks, foundation-models, physics-iq, worldmodelbench, iclr]
status: reference for Phase C (ICLR primary area: foundation or frontier models)
---

Verified live on 2026-09-19 by a research agent (sources at bottom). Comparable class for our frozen LingBot-World v2 14B (built on Wan2.2) = **single-sample i2v**.

| Benchmark | Metric (dir.) | Top (comparable class) | Open-weights baselines with official numbers | Cost / scorer |
|---|---|---|---|---|
| **Physics-IQ (original)** arXiv 2501.09038, WACV 2026 | 0–100 (↑), spatial/spatiotemporal/weighted IoU + MSE vs real | i2v non-BoN: **Cosmos3-Super 43.8** (2026-05); Sora 2 42.3; Cosmos3-Nano 40.2. (BoN/v2v tops 64.5 are not comparable) | **Wan2.2 A14B I2V 38.3** (high run variance: 36.6 / 29.6 in re-runs), CogVideoX-5B 32.3, Cosmos3-Edge 34.1; Cosmos-Predict2.x and LingBot: none | 198 videos/run; 4 runs (mean±SD) only to claim SOTA, 1 run accepted; open scorer emits original + verified |
| **Physics-IQ Verified** arXiv 2606.18943 | same, refined GT/prompts, sample-equal weighting (↑) | **i2v Cosmos3-Super 42.7 ± 0.8** (2026-09-04); MiniMax H3 39.8; Cosmos3-Nano 37.3 | **Wan2.2 14B 32.2 ± 0.6**; Wan2.2 5B 27.7; CogVideoX-5B 31.8; Sora 2 26.5 | same script; official config = best-practice prompts + verified GT (needs HF gated GT) |
| **WorldModelBench** arXiv 2502.20694 | VLM judge 0–10 (↑): instruction 0–3 + commonsense 0–1 + temporal 0–1 + 5 physics laws | KLING 9.10 (board frozen since 2024-11); best open OpenSoraPlan-T2V 8.04, CogVideoX-I2V 7.08 | none for Wan2.x / Cosmos-Predict2.x / LingBot | 350 image+text instances, i2v-native; open fine-tuned VILA-2B judge, local evaluate.py |
| VideoPhy-2 arXiv 2503.06800 (ICLR 2026) | human SA/PC joint % (↑) | Wan2.2 27B-A14B 55.4 full / 47.7 hard (T2V protocol) | Wan2.2 official; others none | 3,940 text prompts (1,200 hard); human-scored board; open auto-evaluator (Pearson 0.42) — T2V, not i2v |
| PhyGenBench arXiv 2410.05363 (ICML 2025) | PCA 0–1 (↑), GPT-4o-based pipeline | Gen-3 0.51, Kling 0.49, CogVideoX-5B 0.45 (frozen since 2024-10) | none | 160 T2V prompts; closed-source path needs gpt-4o-0513 |
| WorldScore arXiv 2504.00983 (ICCV 2025) | 0–100 static/dynamic (↑) | dynamic: CogVideoX-I2V 59.12; video-model static: CogVideoX-I2V 62.15 (3D methods dominate static) | Wan2.1 I2V 57.56 static / 52.85 dynamic | 3,000 examples; pip scorer; heavy |

**LingBot-World v2 facts:** repo robbyant/lingbot-world-v2, tech report arXiv 2607.07534 (no quantitative benchmark table); variants 14b-causal-fast / 14b-causal-pretrain / 14b-bid / 1.3b-causal-fast (last three released 2026-09-10); **license CC BY-NC-SA 4.0** (v1 was Apache 2.0 — constrains any derivative-weight release). No entry on any of the six boards; v1 third-party: WorldOlympiad rank 1 (0.683), WBench.

**Implications for Phase C:** (1) Physics-IQ and WorldModelBench are the only i2v-native, self-scorable boards → primary/secondary as planned; (2) report our own 1-run scores per arm with the paired 198-row test — never quote a single Wan2.2 number as "the" baseline (run variance ~±4); (3) any absolute claim must be against Cosmos3-Super 43.8 / 42.7 (i2v), not the BoN 48.9/64.5 numbers; (4) VideoPhy-2/PhyGenBench are T2V protocols — camera-ready only if at all.

Sources: github.com/google-deepmind/physics-IQ-benchmark (README, PR #80), arXiv 2606.18943, arXiv 2502.20694 + WorldModelBench-Team leaderboard JSON, arXiv 2503.06800 + videophy2.github.io, github.com/OpenGVLab/PhyGenBench, arXiv 2504.00983 + HF worldscore_leaderboard_submission, github.com/robbyant/lingbot-world-v2, arXiv 2607.07534, arXiv 2511.00062 (Cosmos-Predict2.5), arXiv 2603.13770 (PhysAlign re-run).
