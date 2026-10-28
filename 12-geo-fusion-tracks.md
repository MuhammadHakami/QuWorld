---
title: Fusion & Geothermal Tracks — deep-dive verdicts and task specs
date: 2026-08-15
project: quworld
tags: [fusion, geothermal, benchmarks, torax, tokamark, simulators]
status: adopted-2026-08-15
---

# Fusion & Geothermal Tracks (deep-dive `wf_de830ff6-161`)

User directive: find real benchmarks for geothermal and fusion; where none exist, generate them from SOTA simulators. Verdict: **fusion has real benchmarks** (first sweep missed them); geothermal has a semi-benchmark + a build-verified simulator path. **Execute fusion first** (verified platform, best world-model fit, true public baselines, dual-evidence ladder).

## FUSION (execute first) — notebooks `11_tokamark.ipynb`, `12_torax_bench.ipynb`

### F-A: TokaMark — real-machine public benchmark (headline)
- 11,573 real MAST tokamak shots, 39 signals, per-shot Zarr, **CC BY 4.0, public S3, no agreements** (arXiv:2602.10132, UKAEA/IBM/STFC)
- Targets: **G2 magnetics forecasting CNN+LSTM NRMSE 0.1343** and **G3 kinetic profiles 0.3428** — forecasting tasks that map directly onto our frozen-backbone + quantum-adapter design and VRMSE/paired-stats machinery
- ~2–4 days: reproduce a baseline group, freeze it, insert the adapter, matched controls, paired stats
- Follow-on comparison points: TokaMind foundation-model papers

### F-B: TORAX-Bench — self-generated simulator benchmark (a genuine gap: none exists)
- **TORAX** (DeepMind, Apache-2.0, JAX 1D tokamak core transport; verified vs RAPTOR NRMSD <5%; used for SPARC pulse design)
- **Feasibility VERIFIED on this box this session**: torax 1.4.3 + jax installs clean on aarch64 (feasibility agent additionally confirmed `jax[cuda13]` GPU wheels); measured **5.0 s of physics in 0.09 s wall-clock on CPU** after ~30 s one-time JIT → thousands of discharges/hour on Grace cores, **GPU stays free for training**
- Task spec: actuator-conditioned profile rollout. Sample 10–20-dim scenario space (Ip ramp/flattop, ECRH/NBI power waveforms + deposition, density target/gas puff, pedestal, Zeff; ITER CHEASE geometry, optionally SPARC EQDSK) via Sobol/LHS. State: [Te, Ti, ne, q/ψ] × ~50 radial cells × ~400 saved steps + actuator channels. Tasks: one-step VRMSE (identical protocol to TRL2D) + autoregressive rollout. ~1–5 GB dataset, weekend-scale generation, released with splits + baseline suite (U-Net/FNO/persistence + QLKNN-line citations for credibility)
- Prior-art anchors: QLKNN (QuaLiKiz-NN canonical surrogate line), RAPTOR-NN, DeepMind TCV control
- API note: `torax.run_simulation(cfg)` returns a tuple in v1.4.3

### F-C (optional, external validation): Sophelio Fusion Equilibrium Challenge (NeurIPS 2026)
- 9,121 DIII-D+MAST discharges, 133 GB, CC BY 4.0 on HF, **live Codabench leaderboard** — ψ(R,Z) reconstruction; an entry = third-party-scored evidence

## GEOTHERMAL (fast-follow second domain) — notebook `13_geothermal_pelzer.ipynb`

### G-A: Pelzer/Stuttgart heat-plume family (workshop-feasible, days)
- PFLOTRAN-simulated groundwater heat-pump plumes, DaRUS, CC BY 4.0: 1HP raw 1,000 samples 20×256 (doi:10.18419/darus-3650), 2HP 1,000 @ 250×512 (darus-3652), Munich real-permeability variant (darus-5065); code JuliaPelzer/1HP_NN — megabytes-to-small-GB, minutes to train
- Published classical baselines: TurbNet-Geo/U-Net (arXiv:2203.14961, 2302.08199; median pixel error <0.4 °C, only ~16% of test under the 0.5 °C max-error bar — beatable), LGCNN SOTA (arXiv:2507.06062)
- **Decisive asset: arXiv:2606.23411 (Jun 2026) — a QCNN paper on this exact data that retreats to 4 scalar outputs and LOSES to classical CNNs.** The only published quantum comparator in geothermal ML: our full-field quantum-bottleneck result would be a direct, pointed head-to-head
- Gap to own: no frozen community split → we freeze and release ours (stated openly)
- Framing caveat (main risk): steady-state field regression, not rollout — off the world-model message; position as breadth evidence

### G-B: DUGS doublet benchmark via open-DARTS (full-paper third pillar, optional)
- open-DARTS **source build already completed on this box** (aarch64 .so artifacts in scratchpad; wheel-packaging step needs a pip-enabled venv) — but single-threaded per sim (proprietary x86 bos_solvers unavailable) → throughput via ~20-way process parallelism: ~1,000 × 30-yr 3D doublet runs in 2–4 days, 10–40 GB
- Task per Chen/Voskov/Daniilidis DUGS protocol: heterogeneous perm/poro realizations + operational params → temperature-field evolution, production-T curve, thermal-breakthrough time; metrics: field RMSE/SSIM, 0.5/1.0 °C max-pixel thresholds, production-T RMSE/R², paired Wilcoxon; split by geological realization
- Published bars: production-T R² ~0.9995 / RMSE 0.035 °C (in-design DNNs), SSIM ~0.993
- Fallback engine: JutulDarcy.jl + Fimbul.jl (MIT, Tier-1 aarch64, ships `egg_geothermal_doublet()`)
- ⚠ License ambiguity: repo Apache-2.0 vs PyPI GPL-3.0 metadata — verify before shipping derived code; generated data unencumbered

## Sequencing (vs deadlines)
1. **Now→Aug 18 (CPU, parallel to overnight GPU campaign):** TORAX generator pilot (~50 discharges, task-spec validation) — notebook 12 scaffold
2. **Aug 18–23:** TokaMark G2 baseline repro + adapter insertion (GPU, after Gate A/B jobs) — notebook 11
3. **Workshop paper (Aug 29):** fusion track as third pillar if Gates A–C leave room; else ICLR
4. **Sep (ICLR):** full TORAX-Bench release + TokaMark results + Pelzer geothermal head-to-head vs the QCNN paper + optional Sophelio entry + optional DUGS
