---
title: Master Plan — end-to-end proof program to submission
date: 2026-08-15
project: quworld
tags: [master-plan, todo, submission, notebooks]
status: live-checklist
---

# Master Plan: prove the general hybrid system, then submit

## ⚡ Sprint board (2026-08-17, deadline-ordered, sub-agent fan-out)
| # | Task | Deadline served | Status |
|---|---|---|---|
| A1 | WaveLiT protocol verification (0.1308 claim → paper wording) | Gate A Aug 19 | agent |
| A2 | WM-PAI CFP mechanics (page limit, template, OpenReview fields) + ICLR 2027 date check | Aug 29 | agent |
| A3 | TokaMark G2 starter: data access, loader, baseline-repro plan | Aug 23–26 | agent |
| A4 | MHD_64 3D data pipeline (data_ram port + equivalence assert; anchor cmd ready) | Gate B Aug 23 | agent |
| A5 | TORAX-Bench v1: widened scenario space, 1k discharges, frozen splits, CPU baselines | Aug 29 freeze | agent |
| A6 | LingBot P0 prep: field→video renderer + VAE-encode + chunk-cache scripts (GPU-gap-ready) | Aug 29 (H2 min) | agent |
| A7 | Sophelio rules/starter-kit intel (no 133 GB download yet) | Oct 18 | agent |
| A8 | Pelzer loader + baseline script + QCNN protocol details | ICLR | agent |
| — | TRL2D round-1 test evals + paired stats | Aug 19 | auto (campaign) |
| — | Round-2 TD surgery | Aug 19–23 | auto (chained) |
| — | MHD anchor eval + Pelzer/TokaMark GPU runs | Aug 20+ | GPU gaps |
Rule: one foundation system everywhere — agents build harnesses/data, never per-challenge architectures.

**Hypothesis to prove:** a ~500–2k-parameter input-conditioned Rydberg quantum module, inserted through one fixed interface into frozen classical backbones, improves physical simulation across domains and backbones over parameter- and tuning-matched classical controls — with pre-registered kill rules making failure equally publishable.

**Rule of engagement:** every challenge lives in its own executed notebook in `notebooks/` so results are verifiable by re-running. One notebook per task/leaderboard. No claim enters the manuscript without its notebook.

**Deadlines:** WM-PAI workshop (OpenReview open) **Aug 29** · ICLR 2027 full paper (dates TBA, ~late Sep).

---

## Phase A — TRL2D challenge (leaderboard #1) — notebook `05_trl2d_challenge.ipynb`
- [x] Anchor reproduced (official CNextU-net = 0.1934 / 873 windows)
- [x] Hybrid-at-neck built, bit-identical at init; params matched 0.0007%
- [x] Pilot: control arm done (3k steps, 0.18985 — converged, +0.0004 only)
- [x] Pilot: all arms done 2026-08-15 19:59 — quantum 0.18977 < vanilla 0.18985 < mlp 0.19009 (hypothesis ordering, within noise; pipeline + resume machinery validated incl. one real silent-kill recovery)
- [x] Overnight campaign COMPLETE 2026-08-17: 10 arms @20k; R1 one-step — all arms beat anchor (quantum 0.19315, p 6e-12) indistinguishably (kill fired as pre-registered)
- [x] Single-shot evals + paired Wilcoxon DONE (R1 + R2); rollout anchors reproduced (0.5482/1.0132 vs published 0.54/1.01)
- [x] Round 2 TD COMPLETE 2026-08-18: **GATE A PASS — quantum uniquely improves official long-horizon test rollout 0.9528±0.0432 vs controls 1.030/1.032, 3/3 seeds paired vs both**
- [x] WaveLiT verified partially-comparable/unverifiable; wording in manuscript
- **Gate A:** quantum < both controls with p<0.05 → headline result. Kill: controls match → negative-result framing, paper pivots to protocol+C5. **Go/no-go: Aug 19.**

## Phase B — Fusion pillar: MHD_64 (leaderboard #2) — notebook `06_mhd64_challenge.ipynb`
- [ ] Download MHD_64 (71.6 GB) + official `UNetConvNext-MHD_64` checkpoint
- [ ] Reproduce their test number ourselves (published 0.1633) — anchor #2
- [ ] data_ram port (3D fields, 7 ch — check memory: 64³ grid is small) + equivalence assert
- [ ] 3 arms × 3 seeds ckpt-surgery fine-tune (same scripts, new paths) + paired stats
- **Gate B:** same as Gate A. Start when overnight GPU frees (~Aug 18). **Go/no-go: Aug 23.**

## Phase C — Foundation-model prong (ICLR 2027 PRIMARY AREA: foundation or frontier models) — notebooks `07_worldmodel_adapter.ipynb`, `08_physics_judges.ipynb`
**Decision (user, 2026-09-19):** submit ICLR under *foundation or frontier models*; justify it with adapter results inside the FROZEN LingBot-World v2 14B causal-fast video world model (81 GB local), small trainable params, all on the DGX Spark. Paper due **Sep 25 AoE**. GPU is serialized: one DiT instance at a time (~55–65 GB each).

### Sprint schedule (RE-PLANNED 2026-09-19 11:50 on measured throughput: GPU pinned at 650 MHz / 15 W ≈ 25 TFLOP/s, ~47 s per DiT chunk; root-level clock check pending with the user)
| when (local) | stage | GPU-h | sentinel |
|---|---|---|---|
| Sep 19 11:01–11:50 | teacher pilot: 6 configs on train_000 → physics prompt wins every shift; best shift 10/physics rmse 0.611 (shift 3/5 ≈ 0.615) | 0.8 | pilot.log |
| Sep 19 ~11:55–17:00 | teacher-forced cache, PRE-REGISTERED SUBSET 36 train + 9 valid (runs/phase_c/teacher_subset.txt), chosen config, resumable per trajectory | ~5 | (internal) |
| Sep 19 ~17:00–20:00 | P1 adapters on cached chunks: mr30ryd / lora / mlp30 × seeds 1337/38/39 × LR {3e-4,1e-3,3e-3}, TD k=4, 2k steps, 420 s/unit cap; selection = chunk-latent rollout VRMSE on valid | ≤3.2 | `P1_DONE` then `TEACHER_DONE` |
| Sep 19 ~20:00 → Sep 20 ~15:00 | Physics-IQ vanilla: T5 cache (CPU gap) → 3-scenario pilot → 198 videos (≈330 s each) → trim → verify → score | ~19 | `physiq_vanilla_DONE` |
| Sep 20 ~15:00 → Sep 21 ~10:00 | Physics-IQ quantum (best P1 seed, hook at final-step x0) | ~19 | `physiq_quantum_DONE` |
| Sep 21 ~10:00 → Sep 22 ~05:00 | Physics-IQ LoRA — auto-skipped if not started by Sep 22 08:00 | ~19 | `physiq_lora_DONE` / `_SKIPPED` |
| Sep 22 | P2 closed-loop latent rollouts (3 seeds) + paired 198-row Physics-IQ stats + figures; Section 5 text handed to the ICLR draft | 2–3 | `P2_DONE` |
| Sep 23–24 | integration, audits, freeze | — | commit |
Chain runner: runs/phase_c/teacher_p1_chain.sh (waits for pilot → cache → P1 → sentinels); Physics-IQ orchestrator: runs/phase_c/physiq_orchestrator.sh (gated on TEACHER_DONE + free GPU). Memory rule learned 11:29: never run T5/scorer CPU jobs while a DiT is resident (114/121 GB → earlyoom risk).

### SoTA benchmarks for this prong (numbers live in 15-challenge-ladder.md)
- [ ] **Physics-IQ** (Motamed et al. 2025; 66 scenarios × 3 views, official scorer, original + verified scores) — PRIMARY. Comparable class = single-sample i2v: top Cosmos3-Super 43.8 original / 42.7±0.8 verified; Sora 2 42.3; backbone family Wan2.2 A14B 38.3 original / 32.2±0.6 verified (run variance ±4 — report our own per-arm scores, paired 198-row test); BoN/v2v tops (48.9/64.5) are NOT comparable. LingBot-World v2 has no published score on any board (license CC BY-NC-SA 4.0 — no derivative-weight release). Verified table: 20-foundation-benchmarks-sep19.md. Report vanilla / LoRA / quantum with paired 198-row Wilcoxon.
- [ ] **WorldModelBench** (Li et al. 2025; 350 instances, VLM judge) — SECONDARY: pre-registered 60-instance stratified subset × 3 arms if GPU time remains by Sep 22; full 350 for camera-ready.
- [ ] **VideoPhy-2 / PhyGenBench** are T2V protocols (not i2v-native) and **WorldScore** is dominated by 3D methods — camera-ready candidates only, if at all (details in 20-foundation-benchmarks-sep19.md).
- [ ] Cosmos-Predict2.5-2B transfer of the winning adapter — camera-ready (generality evidence).

### Gates
- **Gate C (pre-registered):** quantum adapter < rank-matched LoRA AND < matched-MLP on chunk-latent rollout VRMSE (P1/P2, 3 seeds), and quantum ≥ both on Physics-IQ paired rows; kill rule: controls match → report the null and keep the TRL2D/MHD results as the paper's evidence spine. No claim enters the ICLR draft without its notebook.

## Phase D — Molten-salt proxy: rayleigh_benard slice (leaderboard #3) — notebook `09_rayleigh_benard.ipynb`
- [ ] Stratified slice download (~50–80 GB of 342 GB), stats + loaders
- [ ] Beat TFNO 0.6566 with ckpt surgery (weakest official baselines = headroom)
- [ ] Internal-heating limitation section (arXiv:2507.08515)
- **Post-deadline (ICLR evidence spine); start only after Gates A-C settle.**

## Phase E — Mechanism & dequantization suite (crosses all wins) — notebook `10_dequantization.ipynb`
- [ ] χ-sweep MPS surrogate of the winning circuit (CUDA-Q subprocess referee)
- [x] Product-state (χ=1) + no-entangler ablations — inference-side: INERT (≤2e-5, Aug 23); training-side (pre-registered, 3 seeds): noent 13:30 = 1.0156±0.0433 vs entangled 0.9528±0.0432 → **entanglement necessary during optimization**
- [ ] Finite-shot gradients 128–8192; gradient-variance vs n (barren-plateau check)
- [ ] Qubit ladder 12→16→30 on the winning task (pre-registered scaling protocol, C3-rescoped)
- [ ] Entanglement-entropy vs gain correlation
- [ ] Pulser export of winning generator + emulator fidelity (>0.99/layer) — the hardware-path artifact
- **Runs interleaved on the 5 GB P1 mode whenever the big GPU jobs sleep.**

## Phase G — Fusion track (deep-dive verdict: execute before Phase D) — notebooks `11_tokamark.ipynb`, `12_torax_bench.ipynb`
- [x] TORAX generator pilot DONE 2026-08-16: 50 varied discharges in 22s CPU (0.2-0.6s each, no re-JIT across configs), profiles [Te,Ti,ne,q]x27cellsx24steps + actuators — full 1k-discharge bench = minutes
- [x] **TORAX-Bench v1 DONE 2026-08-17**: 1000 discharges, 12-dim time-varying actuators, frozen 800/100/100 splits, card, baselines (persistence@8 mean VRMSE 0.0923 = number to beat) — releasable artifact
- [ ] TokaMark G2 magnetics forecasting: reproduce CNN+LSTM 0.1343 NRMSE baseline → freeze → quantum adapter vs matched controls, paired stats
- [ ] **COMMITTED (user 2026-08-16): Sophelio NeurIPS 2026 equilibrium challenge entry** — notebook `14_sophelio.ipynb`: download (133 GB, HF), verify competition timeline/rules, build world-model-class backbone (frozen encoder + our 30q adapter — NOT CNN+LSTM) for ψ(R,Z) reconstruction, Codabench submission
- Framing rule (user): every fusion benchmark is attacked with the world-model + 30-qubit system on modern frozen backbones; published CNN/LSTM baselines are targets, not templates
- Details: [[12-geo-fusion-tracks]]

## Phase H — Geothermal track (fast-follow) — notebook `13_geothermal_pelzer.ipynb`
- [ ] Pelzer 1HP/2HP heat-plume: freeze+release split, U-Net baselines, quantum bottleneck, head-to-head vs the QCNN prior art (arXiv:2606.23411 — the only quantum result in geothermal ML, and it loses to classical)
- [ ] Optional third pillar: DUGS doublet benchmark via open-DARTS (source-built on-box) or JutulDarcy fallback
- Details: [[12-geo-fusion-tracks]]

## Phase F — Paper & submission
- [ ] Results tables auto-generated from notebooks into manuscript (REPRODUCE.md: table → command)
- [ ] Bib verification pass (audit-added entries carry unverified authors)
- [ ] Week-of-submission scoop re-sweep (citers of 2509.04006 / 2604.04828 / 2605.05914 / QSWM; OpenReview manual)
- [ ] **USER DECISION PENDING: public pre-registration timestamp** (repo push / snapshot) — C6 is the easiest claim to scoop
- [ ] WM-PAI paper (**8 pages excl. refs**, single track): TRL2D story complete + artifacts + roadmap; OpenReview `NeurIPS.cc/2026/Workshop/WM_PAI`, **ENFORCED due Aug 29 13:59 UTC**
- [ ] Camera-ready author block (currently anonymized placeholder)
- [ ] ICLR 2027 full version (**abstract Sep 18, paper Sep 25 AoE — CONFIRMED**): + Phases B/D, BubbleML, planetswe, TokaMark, WorldModelBench full
- **Submission gate: no notebook, no claim.**

## Standing risks & mitigations
Board latch-off (RMA-path) → everything detached/resumable/cron-@reboot; WaveLiT 0.1308 → official-baseline + paired-stats framing; C6 scoop → timestamp decision with user; field velocity (~1 paper/month) → re-sweep before submission; GPU is single → strict queue priority A → C → B → E.
