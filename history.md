---
title: Project history — quworld / QUOTA (Jul 26 → Sep 19, 2026)
date: 2026-09-19
project: quworld
tags: [history, timeline, decisions, results, incidents]
status: written 2026-09-19 23:35 from the commit log (168 commits), the numbered planning docs, the ladder, the sprint todo and the session memory; extend at the bottom
---

# Project history

This file is the narrative record of the program: what was decided, when, by whom, what was measured, and what broke. Numbers are copied from the commit that first recorded them or from `15-challenge-ladder.md`; when the two disagree the ladder wins because it is the living scoreboard. Section 1 is the timeline, section 2 the results ledger as of the writing time, section 3 the standing decisions, section 4 the incidents and the lessons they left, section 5 the infrastructure that exists because of them, section 6 the open front.

Conventions: times are local (Asia/Riyadh, UTC+3). "Arms" are the three matched models entered into every benchmark: `vanilla` (trunk continued as-is), `mlp30` (parameter-matched classical bottleneck), `mr30ryd` (the quantum register: three 10-qubit Rydberg-Hamiltonian registers, input-conditioned detunings, trainable geometry, zero-initialised gated residual). "3×3" means three arms × three seeds (1337, 1338, 1339).

---

## 1. Timeline

### 1.1 Planning day (Jul 26, before the repository)

- The program starts as a paper idea: a ~500-parameter Rydberg-generator quantum adapter injected as a zero-init gated residual into a frozen video world model, judged on physical-invariant drift over long rollouts. Framing fixed from day one: "quantum-ready, honestly controlled", never quantum advantage (Schuld & Killoran as the reviewer standard).
- Everything claimed was fact-checked the same day (`06-verification-log.md`): the ICLR 2027 dates in early drafts were ICLR 2026 dates shifted a year; AAAI-27 and QTML 2026 had already closed; "Cayley Unitary Adapters" (arXiv 2605.05914) turned out to be real and is the closest prior work; CUDA-Q 0.15 works on the GB10 but ships no adjoint gradient, so training uses a pure-PyTorch complex64 statevector propagator (validated against CUDA-Q to 1e-6); cudaq and torch cannot share a process. Physics-IQ has 66 scenarios, not a "100-scenario subset".
- Backbones verified: Cosmos-Predict2.5 2B/14B, LingBot-World v2 14B causal-fast (CC BY-NC-SA, released Jul 9), Cosmos3-Nano as eval-only. Three evidence pillars declared: parameter-matched classical controls, a dequantization suite, invariant-drift benchmarks. Gates G0–G4 written; G0 (on-box feasibility) passed.
- Hardware reality recorded (`01-hardware-budget.md`, memory `dgx-spark-gb10-hardware`): one DGX Spark GB10, 121 GiB unified memory at ~273 GB/s, single GPU, aarch64. Every later design choice (caching-first pipeline, ≤30 simulated qubits, one job at a time) follows from this.

### 1.2 The pivot and the first three rounds (Jul 27)

- **User pivot:** instead of only measuring drift inside frozen world models, challenge open simulation benchmarks head-on and prove that "adding qubits to a parameter-matched classical surrogate improves simulation". The world-model track becomes the transfer story. Same day the user fixed the headline system: the frozen Ant Group LingBot-World v2 14B causal-fast model plus 30 simulated qubits (hook spec traced from the code: chunk = 4 latent frames, 480×832).
- Repository created (commits `1c7a1e8`, `d88bc4e`, `6f4408c`): plan files, a round-1 harness (FNO trunk + input-conditioned HEA/Rydberg quantum bottleneck + matched MLP control), baseline repos cloned, an autoresearch loop on a stratified tiny subset of The Well `turbulent_radiative_layer_2D` (TRL2D). Advance rule: a quantum branch survives only if it beats both the no-bottleneck trunk and the matched MLP on 8-step rollout VRMSE.
- **Round 1** (`787501b`): the 12-qubit HEA block cut rollout VRMSE 23% versus the trunk; the MLP control was discarded. Fused per-qubit gate application (`d3902d7`) and the 30-simulated-qubit `MultiRegister` (3×10q) with its `mlp30` control landed (`f51725b`).
- **Round 2** (`5fb988f`): per-step quantum rollout win of 37% at fixed steps; HEA qubit scaling non-monotone; run-to-run variance measured. Three executed analysis notebooks found the persistence floor: only quantum variants beat it at 8 steps (`b3d8773`).
- **Round 3** (`adbd678`): `mr30ryd` 0.955 ± 0.102 was best, with the ordering analog > digital > baseline > control; the analog trend was monotone in qubits (12q 1.168 → 16q 0.913 → 30q 0.955). The official CNextU-net anchor was reproduced at one-step test VRMSE 0.1934 (published 0.1956), and the goal "beat CNextU-net with statistical significance" was pinned (memory `goal-beat-cnextunet`).
- The WM-PAI workshop manuscript (NeurIPS 2026 template) was started the same afternoon with 20 verified references (`7c5e6e1`); the Phase-B evaluator with per-window VRMSE for paired statistics, the paired Wilcoxon tool and an overnight 3×3 queue with single-shot test discipline followed (`0771869`, `878f29a`).

### 1.3 Fighting the box (Jul 28 → Aug 4)

- Jul 28: Phase-B live notebook section (anchor distribution 0.1934 ± 0.069 over 873 windows, paired-test power analysis).
- The Spark froze repeatedly during fine-tuning. Aug 3: memory-hardened finetune (batch 16, unpinned workers), earlyoom preferring python (`b6715f9`). Aug 4: step-targeted resumable finetune with optimizer state and a session-surviving detached runner (`297aa70`).
- Forensics over seven boots (memory `spark-hang-hardening`, `spark-freeze-root-cause`): five unclean shutdowns with zero kernel OOM/hung-task/Xid records; three crashes on an idle box. Verdict: board/power-level latch-off, a hardware class acknowledged by NVIDIA (forum thread 374413), not memory. A hardening kit (`~/spark-hardening`: sysctl watermarks, earlyoom, SBSA watchdog, blackbox recorder) covers the separate OOM class. The only software lever is a clock lock, which needs root.

### 1.4 Campaign machinery and strategy (Aug 15 → Aug 16)

- Aug 15: freeze-resilient campaign (in-RAM protocol-verified data, cgroup memory cap, fsync'd 3-minute checkpoints, GPU throttle, `@reboot` auto-resume) and the overnight campaign runner (3×3+1 arms at 20k steps, single-shot test evals, paired-stats report) (`d98bf39`, `ba3ed18`).
- **Duplication audit** with a 16-agent workflow (`1063891`, `09-novelty-audit.md`): C5 (quantum adapter inside a frozen video world model) is novel and became the lead contribution; C3 ("adding qubits improves") is duplicated as a headline (IonQ arXiv 2504.08732) and survives only as the pre-registered, tuning-matched, kill-ruled instantiation; C2 is reframed as checkpoint-surgery methodology.
- **Industrial re-strategy** (`8c119a1`, `10-industrial-benchmarks.md`): an energy benchmark ladder (TRL2D → MHD_64 as the fusion pillar → rayleigh_benard as a molten-salt proxy → BubbleML → planetswe), a technique steal list (delta prediction, train-time jitter), and the WaveLiT threat response (its 0.1308 TRL2D claim is under an unverifiable protocol; the official CNextU-net stays the target).
- **Master plan** (`75e95d1`, `11-master-plan.md`): one notebook per challenge, no claim enters the manuscript without its notebook. Phases A (TRL2D), B (MHD_64), C (foundation-model prong), D (rayleigh_benard), E (mechanism/dequantization), F (paper), G (fusion: TokaMark, TORAX-Bench), H (geothermal: Pelzer).
- Pilot complete Aug 15 19:59 (`c4dfcd9`): quantum 0.18977 < vanilla 0.18985 < mlp 0.19009, the hypothesised ordering but within noise; the machinery (including one real silent-kill recovery) was validated and the overnight 10-arm campaign auto-launched.
- Fusion and geothermal deep dive (`a60e4ab`, `12-geo-fusion-tracks.md`): TokaMark found as a real MAST benchmark; TORAX-Bench specified as a self-generated tokamak transport benchmark (feasibility verified on-box); Pelzer 1HP geothermal chosen with a QCNN comparator.
- Aug 16 01:05: the user committed the program to the **Sophelio NeurIPS 2026 Fusion Equilibrium Challenge** and set the standing rule that the system is always a world model plus 30 simulated qubits on every fusion benchmark; CNN+LSTM baselines are targets, not templates (`ee0cf48`).
- **Adaptive geometry program** (`007d704`, `13-adaptive-geometry.md`), from the user's directive that flat fine-tune results came from inflexible qubit use: make the register structure learnable and input-conditioned (neutral atoms can be re-arranged per simulation), add a temporal-difference (TD) continual-learning objective, constrain to exact Pasqal 2D/3D geometries. Pieces: `ConditionedGeometryBlock`, `KMixtureRegister`, K-step pushforward/TD objective. A structure-search bake-off (evolution with a QD archive, e-prop three-factor learning, gradient, SPSA) was designed (`14-structure-bakeoff.md`).
- Wave 1 (`a5b6d16`): 3D conditioned geometry best (0.888, 20% under the fair 2D pair); the TD objective also un-starves the classical control (kill-rule data); k-mixture needs hard routing. **Wave 1.5** (`6557a16`): quantum + TD wins 3/3 seeds versus the matched control, 0.845 ± 0.051 vs 0.953 (interaction effect: quantum −23%, control −10%). This is the sprint-arena result that stayed in every paper draft.
- TORAX-Bench pilot generated (50 discharges, 22 s CPU). MHD_64 and Pelzer downloads started.
- First infrastructure lesson: after an OOM reboot the cron `@reboot` PATH lacked `~/.local/bin`, every `uv` invocation died, and a false `OVERNIGHT_DONE` was written after a 20-attempt burn (`92a7d33`). PATH is now exported in all runners.

### 1.5 Deadlines, fan-out, and the first official verdicts (Aug 17 → Aug 18)

- Aug 17 02:33: deadlines confirmed from OpenReview and iclr.cc (`175c0e4`): WM-PAI workshop enforced deadline Aug 29 13:59 UTC (8 pages excluding references), ICLR 2027 announced (abstract Sep 18, paper Sep 25 AoE, 9 pages; the workshop-to-ICLR path is explicitly allowed). An 8-agent sprint board was run the same night.
- Delivered in that fan-out: Sophelio rules and scoring recon (Challenge 2 zero-shot MAST is the open prize); WaveLiT wording; the self-improvement foundation notebook (16) with a live QD evolution demo; the LingBot P0 pipeline (renderer, VAE encoder, teacher-cache skeleton); a verified TokaMark starter (anonymous STFC S3 access with a pure-Python blosc decoder); **TORAX-Bench v1 complete** (1,000 discharges, 12-dimensional time-varying actuators, all four transport equations evolved, frozen 800/100/100 splits, baselines persistence@8 0.0923 and climatology 0.5546) (`dbe7b67`); the **living challenge ladder** `15-challenge-ladder.md` created as the canonical scoreboard (`cc0168f`); MHD fine-tune harness; results scaffold with pending-macro gating and `REPRODUCE.md` (14 claims → commands); the Sophelio entry scaffold with a locally verified official scorer (S = 1.0 self-test) and the vacuum-prior floor S = 0.3895 (Consistency = 0 on smooth maps, so ψ sharpness is the objective); the TokaMark streaming loader (official pipeline port, verified on 3 shots / 1,678 windows); the Physics-IQ harness (198 switch-frames local, 6-stage resumable runner, the chunk-size trap defused).
- Wave 1.6 (`57c06eb`): k-mixture hard routing validated (1.005 ± 0.091, was 1.575); the earlier "3D −20%" was a step-count artifact; adapters pay only under TD.
- **Aug 17 10:54, MILESTONE (round-1 official verdict, `4b3448f`):** every arm including quantum beats the anchor and the published number at one step (quantum 0.19315, p = 6e-12 on the best seed) but vanilla matches, so attribution is null under the one-step protocol and the pre-registered kill rule fired. The one-step flank closed honestly; the rollout flank opened.
- Aug 17 11:44: the released UNetConvNext-MHD_64 checkpoint does not reproduce its card or paper (0.66 measured vs 0.1487/0.1633); Gate B was reframed to artifact-as-anchor (`56e7cc9`).
- Aug 17 13:19: a TD selection bug (selection on one-step loss degenerated to the step-0 checkpoint, so the evaluated arm was bit-identical to the anchor) was found and fixed; zero information had leaked (`54131e8`).
- Aug 17 15:06: the official test-rollout evaluator reproduced the published rollout anchors (0.5482 at window 6:12, 1.0132 at 13:30 vs 0.54/1.01); the TD-vanilla arm beat the published 6:12 rollout (0.5228) while trading away the long window (`8fc59bc`). The seed-1337 TD trio then showed the thesis signature: quantum uniquely improved the long-horizon test rollout (13:30 = 0.9927 < anchor 1.0132) while both classical arms traded it away (`b34a7a4`).
- **Aug 18 05:27, MILESTONE (TD campaign complete, `00ecd61`):** the quantum arm uniquely improves the official long-horizon test rollout, 0.9528 ± 0.0432 vs controls 1.030 (vanilla) and 1.032 (MLP), anchor 1.0132, seed-paired 3/3 versus both controls. This became the headline of every paper draft.
- Aug 18 09:31: a third MHD OOM was root-caused (53 GiB memmap residency plus CUDA unified allocations) and fixed with a pread + fadvise loader, batch 6, allocator fraction 0.45; the git repository itself had been corrupted by the crash and was repaired (`cdfdf05`).

### 1.6 Outage, mechanism, and the workshop paper (Aug 18 → Sep 7)

- Aug 18 → Aug 23: a five-day board-off outage. The `@reboot` resume absorbed it (MHD arm 1 continued at step 6.2k); the ladder row damaged by the git corruption was repaired; Gate B was re-scoped (seed-1337 trio for the workshop, 3×3 for ICLR) (`6025913`).
- **Aug 23, Phase E (`83ff101`):** the fine-tuned quantum adapter is inference-inert: product-state, χ = 1, 128-shot and readout-zeroed surrogates all reproduce the intact model to within 2e-5, entanglement entropy 0.03–0.09 bits. The headline is therefore a training-dynamics effect. A no-entanglement training ablation was pre-registered (product state from step 0, 3 seeds, TD, rollout selection) and run the same day: the no-entanglement arms lose the effect (13:30 test 1.0156 ± 0.0433, only 1/3 seeds below the anchor; selection worse on 3/3 seeds) (`cfd12ee`). Verdict, relabelled Sep 6 after a reviewer-style critique: blockade **interactions** are necessary during optimisation (zeroing V_ij removes entanglement and the mean-field nonlinearity together).
- Aug 23: the user set the **core-hypothesis framing**: every practical simulation is part discrete, part analog; the scarce qubits are allocated to the hardest analog sub-process and the frozen world model carries the rest (`0aba896`). Hardware-feasibility citations for geometry-as-parameter (Barredo 2018, Manetsch 2025) were verified and added.
- Aug 24: MHD arm 1 (vanilla s1337) finished at 0.1771 valid at 20k steps, leaving the broken 0.66 artifact far behind (`bdcb975`).
- **Sep 6, the workshop deadline day** (the WM-PAI deadline had moved to Sep 6 13:59 UTC): the 4-page draft became FREEZE v2, then v3 and v3.1, within a few hours:
  - Qubit ladder result: performance **saturates below 5 qubits** (5q 0.8427 ± 0.0709 ≈ 30q 0.8447 ± 0.0511; the fit degenerates to a step at plateau 0.849) (`5db4bc0`).
  - MHD seed-1337 trio done: vanilla 0.1771, quantum 0.1772, MLP 0.1771 valid; every arm recovers the broken artifact, no quantum-vs-control claim at one seed.
  - Five rigor figures (register graphs, training curves, ground-truth vs prediction field grids for TRL2D and MHD with interface-error analysis, scaling, multistart), reviewer-critique fixes (per-trajectory Wilcoxon at n = 9 is underpowered and is reported as such; sample-SD convention; "interactions, not entanglement"; sequential pre-registration disclosed).
  - **Gutzwiller (mean-field) control** pre-registered (`18-gutzwiller-control-plan.md`): interactions kept, entanglement forbidden. Arena pilot matched the entangled arm on all three seeds (0.839 ± 0.046 vs 0.845 ± 0.051). The decisive official-test arms came out **ambiguous** under the fixed rule: 13:30 = 1.0495 ± 0.1219 (1.1882 / 0.9592 / 1.0011), 2/3 below the anchor, point estimate at control level (`8ae0953`). Entanglement is not separable from interaction nonlinearity at three seeds.
  - **Sliding-start power protocol** pre-registered (`19-multistart-power-plan.md`, 9 starts × 9 trajectories = 81 paired rollouts per model) and run on the existing checkpoints: all-start 13:30 VRMSE quantum 0.9486 ± 0.0231 < no-entanglement 0.9664 < anchor 0.9707 < vanilla 0.9867 < MLP 1.0041; Holm-significant versus vanilla 2/3 seeds, versus MLP 2/3, versus anchor 0/3, versus no-entanglement 1/3 (`5543e38`).
  - Official ICLR 2027 style files, a compile-tested skeleton, AI-use and reproducibility statements, and ICLR draft v0/v0.1 (title at the time: "Rollout Accuracy...") (`c229971`, `5e28262`, `aedd96f`). The workshop PDF and the OpenReview fields were handed to the user for upload.
  - The user fixed the **stance**: utilitarian utilisation-maximisation, never advantage.
- Sep 7 → Sep 18: the MHD 3×3 grid completed on Sep 15 (official test, 960 windows, single-shot): vanilla 0.1855 ± 0.0146, MLP 0.1938 ± 0.0145, quantum 0.1855 ± 0.0143. A null versus vanilla (seed-paired 0/3), bimodal basins (~0.177 vs ~0.202) by seed, every arm beats the broken artifact at 100% win rate. Recorded in the ladder on Sep 19 (`5abbc7e`).

### 1.7 Sep 19: the ICLR sprint, the benchmark program, and the reset

The busiest day of the program (59 commits). Chronologically:

- **09:57–10:32, title and abstract.** Six iterations under user direction ended at "QUOTA: A Few Configurable Qubits Improve Long-Horizon Rollouts of Physical World Models" for both the workshop and ICLR drafts, with abstracts of ~200–255 words that state the headline gain relatively (7.5–7.7% lower test VRMSE than both controls at the official start) and carry only citable, evidence-based sentences (`4cec1d4` → `f8b2b2d`).
- **10:46–11:47, Phase C plan.** The foundation-model prong was planned as a GPU-serialised sprint for the ICLR primary area "foundation or frontier models": P0 (renders + VAE latents, 81 trajectories, pre-existing from Aug 17) → teacher cache (frozen LingBot-World v2 14B causal-fast at shift 10 with a physics prompt, 45 trajectories) → P1 adapters (quantum, LoRA, matched MLP on the chunk latents) → Physics-IQ arms → P2. Verified foundation-benchmark SoTA numbers were recorded (`5ddff91`). The Physics-IQ harness v2 (generate/trim/verify/score/compare/project, SDPA fallback, paired 198-row statistics, 20 tests) landed (`8901490`). Throughput was measured at ~47 s per DiT chunk with the GPU pinned at 650 MHz / 15 W, and the timeline re-planned around it; a root-level clock check was requested from the user (`90fa5d4`). P0_DONE 11:00:59; bulk Physics-IQ ground truth (5.97 GB) downloaded.
- **12:35–12:54, ICLR sprint re-baseline** (`21-iclr-sprint-todo-sep19.md`): a ten-benchmark status table, a big TODO split into a GPU lane (Phase C), a CPU lane (TORAX, Pelzer, TokaMark), a writing lane and post-deadline items; the industrial-ladder table and macros were wired into the ICLR draft, and the MHD text moved to the 3×3 null.
- **12:55–13:00, user directives that re-ordered everything:** results on the 10+ benchmarks above all else, the paper later; GPU only, no CPU training or inference; for every benchmark clone the SoTA or baseline repository and start from its pretrained or official model and protocol, never re-implement the classical part, only the quantum register is ours; always cross-compare with the SoTA under its own protocol; no subagents. (memory `goal-benchmarks-over-paper`).
- **13:26–13:48, GPU-only benchmark lane** (`d0dd8a3`, `4d037e2`): TORAX and Pelzer trainers moved to CUDA; the Pelzer 1,000-run set prepared with 1HP_NN's own pipeline; a The Well 2-D chain; the `Bottleneck` made vector-safe for TokaMark (`9df46fa`). **TORAX-Bench 3×3 done 13:48:** test rollout VRMSE@8 vanilla 0.0321 ± 0.0018, MLP 0.0295 ± 0.0033, quantum 0.0292 ± 0.0023; both adapters beat the trunk (quantum < vanilla on 3/3 seeds, −9%), quantum ≈ MLP; paper macros filled (`3a76b52`).
- **14:29, checkpoint finding:** The Well's neuraloperator FNO/TFNO checkpoints do not reproduce on this stack, even on TRL2D and even under neuralop 0.3.0 with era tensorly, CPU = GPU; CNextU-net checkpoints do. The TFNO track was parked and the Well rows moved to CNextU-net backbones. planetswe was dropped (its re-uploaded HF data matches no released checkpoint; persistence beats every published model) and replaced by shear_flow (`4f10365`).
- **14:36–15:39, benchmark harnesses as minimal patches of the SoTA repos:** TokaMark train/eval with `--variant` wrapping `encoder_mlp` (`3207f9d`); Sophelio arms on the starter kit's UNetLite pipeline with the vendored official scorer (`4290b13`); the rayleigh_benard CNextU-net anchor reproduced on the full test split (0.6679 vs 0.6699 published; median 0.0345, heavy-tailed) (`763c2aa`); the first memory guard (`runs/memguard.sh`); Pelzer 1HP_NN U-Net learning-rate sweep (best 1e-3, val MAE 0.0497 °C vs published 0.046, within 8%) (`a6ca162`); TokaMark full-run runner queued behind the shot download; shear_flow local and verified; the VILA judge environment built; TokaMark smoke trains end to end; TokaMark rows and published baselines (CNN 0.2091 TaskNRMSE) in the ladder.
- **16:17–16:31, Sophelio and WorldModelBench.** Sophelio vanilla arm S = 0.7838 on 40 held-out DIII-D shots with the official scorer (R²ψ 0.9819, scalar R² 0.4721, D_LCFS 0.0639, Consistency 0.3966) (`8c4dd44`). The scorer's fork workers had copied a 10 GB parent eight times and pushed the box toward OOM, so the workers moved to spawn and the memory guard's patterns were anchored on the python interpreter (`7ab25c2`). Concurrent sessions on the same worktree were given explicit lane ownership (`2693855`). The WorldModelBench judge became operational (VILA ewm-qwen2-1.5b on CUDA, ~8 s per question), a 60-instance domain-stratified subset was pre-registered (seed 20260919), a judging chain and an i2v generation driver on the Physics-IQ pipeline were wired and queued behind the Physics-IQ arms (`07af6a9`, `fb8666e`, `144cdd1`, `fbacd7b`).
- **17:13, SoTA sweep** (`22-sota-research-sep19.md`, 56 new PDFs, 4 repositories): TokaMark SoTA is TokaMind FT-Base 0.150 (weights released); Physics-IQ i2v top is StrucPhysVideo-TI2V 45.52; Sophelio's cross-machine award needs S(DIII-D) ≥ 0.85; Walrus reports median-protocol Well numbers; a QCNN exists on reduced Pelzer targets and loses to CNNs; ranked future-work items FW-A … FW-H entered the ladder. Follow-ups the same hour: an opt-in `--thomson` feature flag for a Sophelio round-2 trunk (`f4ffde2`), the TokaMind hybrid driver with their warm-start protocol and the register on the CLS latent (`d56f26a`), the FW-A readout families (`QREADOUT = z | diag | cond2`, identical to ⟨Z⟩ at init, 23 tests) (`8531df8`).
- **17:29–17:47, the memory cascade.** With the teacher resident at ~110 GB and ~16 GB of VS Code server processes from restarted sessions, MemAvailable fell to 4–8 GB and the guard shed the Pelzer U-Net arms mlp30/mr30ryd for seed 1338, the seed-1339 pretrain and the Sophelio arm. The Pelzer runner fell through and wrote a false `PELZER_UNET_DONE`; a redo script was written; the TokaMark downloader was rebuilt without the s3fs listings cache (RSS 3.4 GB → 0.6 GB) (`e616f7c`, `1c5b619`).
- **18:2x, network outage.** DNS resolution for the STFC S3 endpoint failed for a period; 2,232 TokaMark shot fetches failed and the downloader's log filled with asyncio tracebacks (8.3 GB before it was rotated).
- **18:25–19:00, Sophelio cascade.** All eight queued Sophelio arms were killed by the guard within ~90 s of loading (each arm's ~10 GB footprint plus the teacher's growth dropped the box under 9 GB), and the runner wrote a false `SOPHELIO_DONE`. The peer redo of Pelzer seed 1339 wrote a false `PELZER_UNET_S1339_DONE` at 18:21 for the same reason (its pretrain had been killed at 17:47).
- **20:39, teacher cache complete:** 45 trajectories in 481 min. **21:00–22:56, P1 units:** mr30ryd s1337/1338/1339, LoRA s1337/1338/1339, mlp30 s1337/1338 (three learning rates each, 420 s per unit).
- **22:58, hardware reset.** The journal ends at 22:54 with earlyoom reporting 53% memory available and no shutdown record: the latch-off class, not OOM. Every detached runner, the guard, the P1 chain and the downloader died.
- **23:03–23:21, recovery.** Uptime 5 min at 23:03 with 99 GB free. This session relaunched the guard (23:07) and fixed the Sophelio arm script (raw shots and full arrays dropped after slicing, only the 40 test shots kept, ~4 GB less RSS), smoke-tested it end to end and relaunched the runner at 23:10 (`b6ae8d4`). The peer session relaunched the rest at 23:11 through `runs/relaunch_sep19_reboot.sh` (P1 resume → Pelzer redo with a now-conditional sentinel → Well chain v2), the TokaMark downloader (9,260 of 11,573 shots complete, log rotated and silenced, runner gate now requires `done: ok N fail 0`), the TokaMind and FW-A runners, and both Phase-C orchestrators (`4fd9a4d`). P1 finished at 23:19:53 and `P1_DONE` / `TEACHER_DONE` were written: valid chunk-latent rollout VRMSE vanilla backbone 0.6996, quantum 0.5838 ± 0.0052, LoRA 0.5592 ± 0.0011, MLP 0.5631 ± 0.0030, so **H2a is false at P1** (quantum loses to both controls on 3/3 seeds), with the stated confound that units are time-budgeted and the register runs 510 ms/step versus 140–160 for the controls (about a quarter of the optimisation) (`12c786c`). The first relaunched Sophelio arm landed at 23:20: mlp30 s1337 S = 0.7973 (R²ψ 0.9928, Consistency 0.4304). Tailscale was found logged out after the reset (user re-login needed).

---

## 2. Results ledger (as of Sep 19, 23:35)

| Benchmark (metric) | SoTA / anchor | Ours | Reading |
|---|---|---|---|
| The Well TRL2D, one-step test VRMSE ↓ | CNextU-net 0.1956 (repro 0.1934) | quantum 0.19315; vanilla matches | at SoTA; attribution null (kill fired) |
| TRL2D rollout 13:30 VRMSE ↓ | 1.01 published (repro 1.0132) | quantum 0.9528 ± 0.0542; vanilla 1.030; MLP 1.032 | quantum only arm below anchor, 3/3 seeds; sliding-start Holm 2/3 vs each control |
| TRL2D sprint arena VRMSE@8 ↓ (internal) | best classical 0.953 ± 0.135 | quantum + TD 0.845 ± 0.051; 5q ≈ 30q | saturates below 5 qubits |
| The Well MHD_64 VRMSE ↓ | paper 0.1633; released ckpt 0.5447 | vanilla 0.1855 ± 0.0146; MLP 0.1938; quantum 0.1855 ± 0.0143 | null vs vanilla; all beat the artifact |
| Mechanism (Phase E/E2) | — | inference-inert (≤2e-5); no-interaction training loses the effect (1.0156, 1/3); Gutzwiller ambiguous (1.0495 ± 0.1219) | training-dynamics effect; interactions necessary |
| TORAX-Bench VRMSE@8 ↓ (ours) | persistence 0.0923 | vanilla 0.0321 ± 0.0018; MLP 0.0295 ± 0.0033; quantum 0.0292 ± 0.0023 | adapters beat trunk 3/3; quantum ≈ MLP |
| Pelzer 1HP test MAE °C ↓ | their U-Net 0.046 | their trunk + pipeline: vanilla 0.0343, MLP 0.0339, quantum 0.0335 (2 seeds each) | below SoTA on our split of their 1,000 runs; seed 1339 re-running |
| Pelzer 100-dp FNO track MAE °C ↓ | — | none 0.0675 ± 0.0040; MLP 0.0767 ± 0.0189; quantum 0.0729 ± 0.0097 | null, n_test = 10 |
| rayleigh_benard test VRMSE ↓ | TFNO 0.6566; CNextU-net 0.6699 (anchor 0.6679) | 0/9 arms; vanilla s1337 at step 5,798/6,000 | resumed after the reset |
| shear_flow test VRMSE ↓ | CNextU-net 0.8080 | data local; anchor pending | queued after RB |
| TokaMark task 1-1 TaskNRMSE ↓ | TokaMind 0.150; CNN 0.2091 | none; 9,260/11,573 shots local | CNN-trunk and TokaMind-trunk grids queued |
| Sophelio S ↑ | award ≥ 0.85 | vanilla 0.7838; mlp30 s1337 0.7973; 7 arms running | consistency and scalar terms are the headroom |
| Physics-IQ score ↑ | StrucPhysVideo 45.5; Wan2.2-14B 32.2 | none; teacher cache and P1 done (H2a false at P1, confounded) | ≈19 h per arm of generation next |
| WorldModelBench judge ↑ | KLING 9.10; best open 8.04 | none; judge operational, subset frozen | generation queued after Physics-IQ |

Paper state: workshop PDF FREEZE v3.1 (Sep 6) handed over for upload; ICLR draft v0.1 titled QUOTA with the industrial-ladder table wired in; TORAX cells filled, Pelzer and Physics-IQ cells pending; due Sep 25 AoE.

---

## 3. Standing decisions and directives (dated)

- Jul 26: never claim quantum advantage; write to the Schuld–Killoran standard; every claim fact-checked before it enters a draft.
- Jul 27: challenge open simulation benchmarks directly; headline system = frozen LingBot-World v2 14B causal-fast + 30 simulated qubits; advance rule = beat both the trunk and the matched MLP.
- Aug 15: duplication audit outcome (C5 lead, C3 only as the pre-registered instantiation, C2 as methodology); energy benchmark ladder.
- Aug 16: participate in Sophelio; world model + 30 qubits on every fusion benchmark; register structure learnable and input-conditioned with TD continual learning under Pasqal geometry constraints.
- Aug 17: one foundation adapter stack entered into all challenges with the same structure; per-benchmark adaptation is register configuration only; agents build harnesses and data, never per-challenge architectures.
- Aug 23: core hypothesis = division of labour between scarce qubits (hardest analog sub-process) and the frozen world model.
- Sep 6: stance = utilitarian utilisation-maximisation; sample-SD convention; pre-registration disclosed sequentially; page-limit gate checked by locating the References page.
- Sep 19: short evidence-based abstracts (QUOTA title); results on the 10+ benchmarks above all else, paper later; GPU only; clone each SoTA repository and start from its official model and protocol; cross-compare under the SoTA's own protocol; no subagents; keep the ladder current with every number; add future work.
- Standing: double-blind (repository link withheld, "QuWorld" never in blind text); matched controls, fixed protocol, three seeds, single-shot official test evaluations.

---

## 4. Incidents and the lessons they left

| When | What happened | Root cause | What changed |
|---|---|---|---|
| Jul 28 → Aug 4 | repeated hard freezes during fine-tuning | board/power latch-off (hardware class; NVIDIA thread 374413) | hardening kit, detached resumable runners, checkpoints every 3 min, `@reboot` resume |
| Aug 16 | false `OVERNIGHT_DONE`, 20 failed attempts | cron `@reboot` PATH lacked `~/.local/bin` after an OOM reboot | PATH exported in every runner |
| Aug 17 | TD arm bit-identical to the anchor | selection on one-step loss picked the step-0 checkpoint | selection on rollout VRMSE@8, both metrics logged |
| Aug 17 | MHD_64 released checkpoint at 0.66 vs 0.1633 claimed | bad upstream artifact (ZScore confirmed) | artifact-as-anchor framing; upstream report left to the user |
| Aug 18 | third MHD OOM and a corrupted git repository | 53 GiB memmap residency + unified CUDA allocations | pread/fadvise loader, batch 6, allocator cap; repository repaired |
| Aug 18 → 23 | five-day board-off outage | hardware | absorbed by resume; Gate B re-scoped |
| Sep 19 | neuraloperator FNO/TFNO checkpoints do not reproduce | unknown (identical across neuralop versions, CPU = GPU) | CNextU-net backbones for all Well rows; TFNO parked |
| Sep 19 | planetswe matches no released checkpoint | HF data re-uploaded 2025-03-20 with swapped velocity components | planetswe replaced by shear_flow |
| Sep 19 | `pkill -f` killed the operator's own shell twice; memguard v1 killed a tool shell | pattern matched the caller's command line | bracket patterns, kill by PID, guard anchored on the python interpreter |
| Sep 19 | Sophelio scorer drove the box to 6.5 GB | fork workers copied a 10 GB parent ×8 | spawn context, 4 workers, only test shots kept in memory |
| Sep 19 | HF downloads crashed at 91% and the judge weights stalled | unauthenticated HF API limit of 500 requests per 5 min | 4-worker retry loops; `HF_TOKEN` requested from the user |
| Sep 19 17:29–19:00 | memguard cascade: five Pelzer arms and all eight Sophelio arms killed; two false DONE sentinels | teacher at ~110 GB + ~16 GB of restarted VS Code sessions + arm footprints | conditional sentinels, higher gates, session-sprawl note; idle sessions to be closed by the user |
| Sep 19 18:2x | 2,232 TokaMark shot fetches failed, 8.3 GB log | DNS outage to the STFC S3 endpoint | resumable downloader, asyncio logging silenced, runner gate on `fail 0` |
| Sep 19 22:58 | hard reset, every runner lost | latch-off class (no shutdown record) | relaunch sequencer with one GPU co-runner, resume from result files, Tailscale re-login needed |

Lessons in one line each: a runner may write DONE only from the result files it produced; never grep-kill a literal present in your own command line; gate every GPU launch on free memory and keep one co-runner next to a resident teacher; verify every downloaded checkpoint against its published number before building on it; sessions on one worktree need explicit lane ownership and cost host memory.

---

## 5. Infrastructure that exists because of the above

- **Model code:** `paper/qlayer.py` (Rydberg / HEA / mean-field ansätze, `MultiRegister`, FW-A readout families), `paper/model.py` (`Bottleneck`: pool → core → zero-init gated residual; vector- and field-safe), matched `MLPCore` via parameter counting.
- **Benchmark harnesses as patches of SoTA repositories:** `paper/phase_b/` (The Well: `well2d.py` bit-exact loader, CNextU-net backbone factory, fine-tune and streaming test evaluators, multistart evaluator and statistics), `paper/torax_bench/`, `paper/geothermal/` (1HP_NN pipeline + U-Net track), `paper/tokamark/` (CNN baseline and TokaMind drivers), `paper/sophelio/` (starter pipeline + vendored official scorer), `paper/phase_c/` (Physics-IQ harness v2, T5 cache, teacher cache, P1 adapters), `paper/wmb/` (subset, generation, judge).
- **Runners:** self-chaining detached scripts under `runs/` with sentinels (`TORAX` → `PELZER_UNET` → `WELL2D_<ds>`; `TOKAMARK_DONE` → TokaMind → FW-A; `P0` → teacher → `P1_DONE`/`TEACHER_DONE` → Physics-IQ arms → WMB generation → judging), `runs/memguard.sh` (shed order, 9 GB), `runs/relaunch_sep19_reboot.sh` (post-reset sequencer).
- **Environments:** main `.venv` (uv), `.venv-well` (neuralop 0.3.0, parked), `.venv-tokamark` (zarr 3, s3fs), `.venv-vila` (py3.10, VILA judge). SoTA repositories under `baseline/`: 1HP_NN, tokamark, tokamark_baseline, tokamind, WorldModelBench, the_well, neuraloperator, physics-iq, sophelio-starter, lingbot-world-v2, ANO_SR.
- **Data on the box:** The Well TRL2D, MHD_64, rayleigh_benard (370 GB), shear_flow (439 GB); Sophelio (103.9 GB, 9,123 parquet); Pelzer 1,000-run prepared set; TokaMark shots (MAST zarr, in progress); Physics-IQ ground truth; Phase-C renders, latents and the 45-trajectory teacher cache.
- **Documents:** `00`–`22` numbered planning and verdict files; `15-challenge-ladder.md` is the living scoreboard; `21-iclr-sprint-todo-sep19.md` the sprint list; `REPRODUCE.md` maps claims to commands; manuscripts in `paper/manuscript/` (workshop `main4p.tex`, ICLR `iclr2027/main_iclr.tex`).

---

## 6. The open front (as of Sep 19, 23:35)

1. Sophelio: seven arms in flight (≈10 min each now that the teacher is gone), then the round-2 shared-trunk lift toward S ≥ 0.85, then the Oct 18 dev submission.
2. Pelzer: seed-1339 pretrain and three arms (peer redo), then the ladder and the `\LadPelz*` macros.
3. rayleigh_benard 3×3 and the shear_flow anchor plus 3×3 (Well chain v2).
4. TokaMark: finish the shot download, CNN-trunk 3 arms, TokaMind-trunk 3×3 against 0.150.
5. Physics-IQ: vanilla → quantum → LoRA generation (≈19 h each) and scoring; a fixed-step re-run of the quantum P1 units is needed before H2a is called.
6. WorldModelBench: 60-instance generation per arm, judging (≈1.5 h per arm).
7. FW-A readout arena, FW-B register-configuration rule, and the rest of the ranked future-work list.
8. Paper: ICLR 2027 due Sep 25 AoE (deprioritised by the user behind the benchmark results); Pelzer, Physics-IQ and remaining ladder cells to fill.
9. User-side: close idle sessions, provide `HF_TOKEN`, root GPU clock check, Tailscale re-login.

_Append new entries below this line in the same style (date, what, numbers, commit)._

### September 20: benchmark and manuscript continuation

- Read the complete historical record and audited live runners before continuing. Sophelio and Rayleigh–Bénard are active on CUDA; the TokaMark download and downstream queues remain in progress. No duplicate training jobs launched.
- Closed the Pelzer U-Net evidence update from nine result JSONs: vanilla 0.036363 ± 0.004407, MLP 0.036152 ± 0.004554, quantum 0.035881 ± 0.004608 °C (sample SD). Quantum wins 2/3 versus vanilla and 1/3 versus MLP. No quantum-specific or verified same-split SoTA claim. Added a source-hashed result audit and executed geothermal notebook; no repeat test inference.
- Corrected the ICLR ladder's stale planetswe, TokaMark and Sophelio entries; filled Pelzer macros and added its null-attribution discussion. Clarified shared trainable geometry versus per-input dynamic detunings, arena-only conditioned geometry, state reset, and the difference between distance clamping and physical atom-spacing constraints.
- Extended the named program to eleven distinct tasks with BubbleML 2.0, explicitly pending. Cloned official Bubbleformer at 8d1dade4c34b69f6048c6bcba04b62d63117efe5; checkpoint/scorer/data equivalence remains the next reproduction gate. `23-dynamic-benchmark-continuation.md` records provenance and the proposed fixed-step dynamic-configuration comparison. Existing FW-A time-budgeted outputs must stay exploratory under that rule.
- Changes are local and uncommitted; existing model edits and running configurations were preserved.
- Verification: Pelzer audit and notebook execution passed; `git diff --check` passed. ICLR PDF rebuilt (16 total pages): main text ends on page 9, References starts page 10. Tectonic emits an existing bibliography rerun-limit/internal-consistency warning, so this is a compiled working draft, not a submission-ready certification. Rayleigh–Bénard vanilla s1337 official test completed at mean VRMSE 0.4303 / median 0.0325 over 34,300 windows; matched grid still in progress.

### September 20: Lean and joint classical–quantum scaling

- User requested Lean and a scaling law for classical plus quantum models. Added `24-lean-hybrid-scaling.md`: a signed-gain, saturating configuration-aware candidate with independent classical-size/data/optimizer-budget axes. This is a falsifiable proposal, not a fitted or universal law.
- Installed Lean 4.19.0 through elan without modifying shell PATH, pinned mathlib v4.19.0 in `formal/`, and compiled 15 theorems: utilization bounds/monotonicity, classical limits, signed gain identity, positivity, conditional improvement/no-benefit, saturation increment, an inverse-resource allocation bound, and independent-register amplitude accounting. `lake build` and the axiom audit passed; no unfinished proof or new project axiom. Standard Lean/mathlib axioms only. Provenance hashes: `runs/scaling/lean_verification.json`.
- Audited 18 existing ladder/control logs without GPU training/inference. All have 2,000 steps/TD depth 4; parameter counts vary slightly. Positive-qubit leave-one-count-out RMSE: constant 0.159402, log trend 0.183490, exponential saturation 0.170113. Flat null predicts best among these candidates. Joint scaling exponents remain unidentifiable at one classical size, data budget, and step budget.
- Added reproducible audit script, source-hashed CSV/JSON, standalone plot and executed `notebooks/17_hybrid_scaling.ipynb`. Corrected exact-total-parameter claims in the active ICLR draft; added a Lean/scaling appendix. PDF rebuilt: 17 pages total, main text ends page 9, References start page 10. Existing bibliography rerun warning remains. Existing benchmark queues/configurations were preserved.

### September 20: input/scenario configuration takes priority over scaling

- User clarified JEPA means V-JEPA 2.1/world-model approaches, with no mandated methodology; optimize configuration first within thirty qubits and fit scaling afterward. Added `25-input-conditioned-configuration-first.md` with primary-source research, native-latent/hypernetwork → outer search → learned selector/planning gates and all eleven task contexts. Accuracy at matched steps is not yet compute optimality; no guaranteed global optimum or new SoTA claim.
- Inspected official V-JEPA 2.1 code, cloned commit 204698b45b3712590f06245fbfba32d3be539812. No checkpoint weights or JEPA GPU experiment. Its hub source points to localhost at this revision; official direct checkpoint verification is required before use.
- Added CUDA-only `paper/config_search/`: official TRL2D CNext backbone, five 3×10-qubit candidates (reference, encoding, bounded static/dynamic geometry, conditioned readout), paired capacity-matched MLPs plus vanilla. Geometry has true pairwise-separation bounds; no full hardware-feasibility claim. Screen is eleven fixed candidate jobs, seed1337, 256 steps, not an implemented evolution/BO/RL optimizer. Controller-matched classical ablations and multi-seed confirmation remain required.
- Twelve contract tests passed in 2.37s. Two-step official-backbone CUDA smoke and strict resume passed (`runs/config_search/smoke_bounded_dynamic_v2`); smoke scores are not evidence of benchmark gains. Search/confirmation trajectory subsets are disjoint for this search but come from historically used validation. No official test access.
- Repaired strict diagnostic checkpoint loading and zero-feature rank handling. Old `register_diag_trained_s1337.json` could silently ignore official CNext weights when loaded into an arena FNO; preserved artifact and added an explicit INVALID note. Its recommendations are not accepted evidence.
- Queued detached screen launcher PID667083 after existing Well chain PID42174. Verified Physics-IQ/WMB orchestrators35074/35077 had only sleep children before holding them; runner restores them on normal exit/error/SIGTERM. Existing Well GPU training is unchanged. Status/provenance/recovery details are in REPRODUCE. Improvements remain unmeasured until actual screen and confirmation results exist. Changes local/uncommitted.

### September 20: strategy-first testing revision

- User explicitly requested testing after first establishing a viable evidence-based strategy. Stopped the pending v1 screen before any trial and verified both idle video orchestrators resumed. Added `26-configuration-validation-protocol.md` with prespecified mechanism thresholds, 256→1024 fidelity calibration, required training-time conditioning controls, five new paired confirmation seeds, practical accuracy and full compute gates. No literature result is treated as proof on our tasks.
- Actually ran Stage A on CUDA: strictly loaded all three official TRL2D quantum checkpoints and intervened on six fixed training trajectories. Context permutation relative prediction L2 = 2.896e-5 / 2.959e-5 / 2.679e-5; readout zeroing = 4.118e-5 / 4.349e-5 / 4.252e-5. Repeated normal prediction difference zero. Detectable but small; neither useful inference-time adaptation nor accuracy improvement established. Retain training-time controller as first hypothesis. Artifact includes checkpoint hashes, gates and feature variation.
- Updated screen runner to extend all eleven trials from256 to1024 without pruning and preserve fidelity snapshots. Added automatic rank-agreement report with conservative tie handling; three positive/reversed/tied fidelity tests pass. Twelve existing configuration contract tests also pass (2.85s). No new official test evaluation.
- Queued strategy-v2 launcher821003 behind existing Well42174, with idle video orchestrators35074/35077 held and restored on exit. Strategy and Stage-A evidence hashes recorded in manifest. Old v1 has zero performance trials. Stage B remains pending GPU availability; later matched conditioning controls, confirmation and scaling remain pending. Local changes uncommitted.

### September 20: causal signal routing prototype and measured difficulty prediction

- Continued the active flexible-qubit allocation objective. Verified the previous turn made progress and the current strategy-v2 queue is live (launcher821003), waiting on live Well42174; active GPU work is shear-flow seed1339 evaluation. No queue restart or change to its frozen candidate code.
- Identified that global pooling prevents distinct signal-region allocation. Added isolated `signal_router.py`: causal per-location difficulty score, distinct sparse register dispatch, actual0..30 allocated qubits, bounded input-conditioned geometry, local residual, exact classical fallback, explicit training warm-up, architecture-matched classical core. Per-register width fixed; generic vector tokenization remains incomplete.
- Ran actual CUDA difficulty-prediction experiment on official CNext training trajectories:12 fitting/12 held-out from scorer fitting, one window,200 optimizer steps. AUROC0.867594, BCE0.534805 vs fitting-prior0.693147; selected locations exceed the classical-error threshold88.89% vs23.70% overall. Threshold accuracy75.69% is below all-negative76.30%, and every input activates30 qubits. Therefore promising error ranking, no demonstrated calibration, adaptive resource savings or quantum accuracy gain.
- Preserved JSON, predictor weights and matching source archives under `runs/config_search/signal_routing_probe/`. Added `27-causal-signal-routing.md` with interpretation and the next counterfactual allocation/control tests. This does not redefine completion: full robust cross-task marginal-benefit allocation and empirical optimization remain unverified.

### September 20: actual paired allocation training and rollout pilot

- Verified prior turn was progress and strategy-v2/Well processes remain live; no restart. Added explicit scalar/per-input integer register caps to isolated SignalRouter with actual skipped execution and exact zero-budget fallback. Eight CUDA routing contract checks pass (2.76s).
- Registered and executed a64-step paired quantum/classical routing pilot on CUDA, batch2 TD4 lr1e-4, identical randomized0..3-register budgets and official initialization. Difficulty scorer frozen. Training trajectories0..47; allocation probes48..59 were excluded from this pilot's training and scorer fit. No official validation/test access.
- Full rollout8 probe at0/10/20/30q: quantum mean VRMSE .51726135/.51728417/.51724555/.51727779; classical .51688056/.51686881/.51689655/.51686925. Quantum direct mean paired relative effects versus its own zero-budget output: -.00409%, +.00345%, -.00351%. Slightly better classical results at all caps; no useful adaptive allocation or optimum demonstrated.
- Added `summarize_allocation.py`: checks baseline equality, shared training budgets, complete per-trajectory/step traces, capacity bounds and separates direct effects from between-trained-backbone differences. Saved result/audit/checkpoints plus hash-verified source and preregistration archives in `runs/config_search/allocation_pilot_v1/`.
- Decision: do not fit RL/benefit selector to tiny pilot differences or promote20q. Next investigate representation scaling and correction-path effectiveness with matched classical controls; longer Stage-B calibration remains pending. Goal active; robust cross-task optimization unachieved.

### September 20: correction-scale diagnosis and controlled remedy test

- Revalidated live Well42174 and strategy-v2 launcher821003. Prior turn was progress; no queued campaign restart or frozen-screen source changes.
- Added `diagnose_routing.py`; actual checkpoint/training-data audit showed quantum readout variation0.068–0.088 with zero encoding saturation, but correction/latent RMS1.45e-7 and geometry gradients~1e-11. This localized a scale/gradient issue rather than proving input-information collapse.
- Added RelativeSignalRouter with a norm-bounded local residual, exact zero-gate initialization and identical classical transformation. Registered and ran a matched64-step follow-up with the same budgets/seed/data. Nine routing tests pass (3.29s).
- Scale-aware quantum correction increased30.96×; geometry gradients11.6–31.1×. Nonetheless quantum0/10/20/30q VRMSE8 .51683671/.51684412/.51686331/.51684874 remained worse than classical .51598421/.51601116/.51598405/.51596582. Direct allocation effects stay tiny; no candidate/budget promoted. Archived result/audit/checkpoints/protocol/sources and mechanism comparison under allocation_relative_v1.
- Reused probe data remain explicitly exploratory. Next prioritize longer learning curves and correction utility before a learned benefit selector/RL. Full robust configurable-system objective stays active and unverified.

### September 20: longer controlled routing campaign implemented and queued

- Prior turn produced progress (diagnosis, actual paired scale-aware pilot and negative result). Revalidated live queue821003/Well42174 before making the next change; no failed-handle inference or duplicate training.
- Preregistered `28-long-routing-comparison.md`: seven arms (vanilla; quantum/classical × adaptive/static/permuted locations), seed1337,1024 steps, TD4 batch2, random per-input0..3-register training caps, five evaluation checkpoints and all four inference caps. Official search validation only; confirmation and tests untouched. Permutation is a fixed within-input location bijection, preserving batch-independent causal inference; not mislabeled cross-example shuffled-context training.
- Added ControlledSignalRouter and `routing_trial.py`, with strict source/protocol resume, model/optimizer/RNG recovery, partial checkpoints every128 steps, complete allocation traces and no automatic promotion. Ten routing tests passed (3.05s). CUDA official-backbone two-step permuted-quantum smoke and successful resume passed; a mismatched static-routing resume was rejected and checkpoint hash stayed unchanged.
- Replaced only the waiting coordinator (old Python821006 terminated gracefully; it restored video orchestrators), after verifying it had not launched a configuration job. New live coordinator launcher834208 (`run_research_chain.py`) holds verified idle video orchestrators35074/35077, waits for active Well42174, runs the unchanged22-stage configuration screen, then seven routing trials. Active Well GPU training uninterrupted. Queue at `runs/config_search/research_chain_v1/`; source hashes and immutable manifest guard queued changes. SIGTERM/failure restores idle video orchestrators; SIGKILL requires identity-checked recovery as documented.
- Full training remains pending GPU availability. No new performance win claimed; goal active. Next actual evidence must come from longer learning curves and matched location controls rather than additional64-step gate tuning.

### September 20: strict analysis prepared for longer routing evidence

- Previous goal turn was progress (resumable controlled campaign implemented/queued). Verified research coordinator834208 and Well42174 remain live; no routing result exists yet and no restart was performed.
- Added independent `audit_long_routing.py`, verifying all seven completed arms, five curve checkpoints, metadata/hash/checkpoint/snapshot agreement, shared protocol and sampled budget counts, paired trajectory identities, finite errors and complete bounded allocation traces. Separates adaptive location benefit, classical-core comparison and own-backbone allocation effects. Correlated windows do not become independent seeds; no automatic promotion.
- Eight audit contract tests passed in0.01s, including malformed/missing data and pseudoreplication rejection. Ran the actual audit: correctly reports `pending` with all seven expected result files missing. No new performance result or completion claim. Registered trainers/queue unchanged; goal active.

### September 20: Lean bounds for the dynamic routing correction

- Verified waiting research coordinator834208 and active shear-flow training824207 remain live; no restart. Previous goal turn made progress by adding the paired evidence audit.
- Added five compiled Lean theorems for relative correction magnitude, zero gate/allocation, per-token squared-energy bound and30-qubit cap. Theorems explicitly assume bounded gate/direction/probability and the RMS identity; they do not prove PyTorch equivalence, floating-point behavior, downstream stability, optimal allocation or accuracy benefit.
- `lake build` and full20-theorem axiom audit passed in the pinned Lean4.19.0/mathlib environment; standard axioms only, no sorry/new axiom. Saved exact audit output and source hashes in `runs/scaling/lean_axioms_routing.txt` and `lean_routing_verification.json`. Registered experiment sources and queue unchanged; empirical optimization remains pending, goal active.

### September 20: repaired final evaluation reporting and recovered saved evidence

- Verified active Well training824207 and research coordinator834208. Previous turn produced formal-proof progress. Training remains active; no restart or duplicate model evaluation.
- Traced shear-flow evaluation failures to `well2d.PUBLISHED[dataset]` after all CSV predictions had been written. Fixed optional reference formatting in `eval_well2d.py` so the pending final evaluation can exit normally; no change to prediction/scoring logic, checkpoint selection or source-hashed configuration/routing trainers. Python compilation and diff checks pass.
- Added `audit_well2d_csv.py`, reading HDF5 metadata only to verify exact trajectory/start identities. Nine available saved CSVs each contain all21,952 unique finite windows; no invalid files. Quantum seed1339 CSV still missing because training is live. Recovered three-seed vanilla .26855169±.04022814, MLP .25390004±.02931800; quantum available seeds1337 .22969811 and1338 .22363578, no incomplete-seed aggregate. Official local anchor .99461164; external protocol discrepancy remains unresolved.
- Saved `runs/well2d/shear_flow/cnext/csv_audit.json` with source/artifact hashes, complete-arm summaries and explicit partial-grid status. No repeat test inference or SoTA claim. Old shell/logs preserved; documented that its DONE marker and grep recovery are not sufficient proof and must not trigger repeated tests. Goal active; routing campaign still queued.

### September 20, 17:36: completed shear-flow evidence and research handoff

- Verified actual training completion6000 steps and final live evaluator887411, then its normal exit after21,952 test windows. Full CSV audit now reports complete_csv_grid: ten files (official anchor plus3×3), no missing/invalid/duplicate windows. No repeat test inference.
- Three-seed means±sample SD: vanilla .26855169±.04022814, MLP .25390004±.02931800, quantum .26594996±.06810766. Q wins2/3 paired seeds against each but loses mean to MLP; seed1339 .34451598 reverses the optimistic partial-grid picture. No robust quantum-specific or published-SoTA claim. Anchor discrepancy remains. Updated living ladder and continuation table.
- Actual handoff confirmed: Well42174 exited; research coordinator834208 launched configuration screen895418 at17:36:17 and reference-vanilla trainer895441 at17:36:18. First trial logged step128. Idle video orchestrators remain held by the coordinator; no restart or protocol modification. Longer routing tests follow the screen. Goal active; optimization evidence still pending.

### September 20: fixed-step configuration reporting audit

- Revalidated live research coordinator834208/screen895418; first four256-step stages completed and encoded quantum training active. Reference endpoint VRMSE8: vanilla .56857352, MLP .56968702, quantum .56793671. Approximately0.11% quantum gain versus vanilla at one short-budget seed is exploratory, not promotion evidence.
- Found existing fidelity summary uses best-so-far scores, whereas protocol26 also requires fixed-step comparison. Added independent `audit_screen_endpoints.py` without changing source-hashed running trainers/coordinator. Reads immutable stage snapshots plus exact-step curve entries; rejects invalid/conflicting endpoints and keeps best-so-far separate. Complete endpoint fidelity gate is withheld while stages remain pending. Use this endpoint audit alongside the historical best-score report when screen finishes; do not infer endpoint rankings from best checkpoints.
- Five regression tests passed, including reversed endpoint rankings despite favorable best-score rankings, incomplete campaigns and invalid/contradictory endpoint records. Actual audit saved at `runs/config_search/trl_strategy_v2/endpoint_audit.json`: four complete stages, eighteen pending. No official test inference or performance promotion; all candidates still receive1024 steps, followed by seven controlled routing trials. Goal remains active.

### September 20, 17:52: complete short-budget configuration screen

- Verified all eleven256-step trials completed normally, and live trainer905634 began the vanilla1024-step extension under unchanged coordinator834208/screen895418. No pruning, restarts or official test evaluation. Endpoint audit refreshed:11 complete/11 pending stages; full fidelity gate withheld.
- Endpoint VRMSE8 (quantum / matched MLP): reference .56793671/.56968702; encoded .56665479/.56948101; bounded static .56962409/.57031164; bounded dynamic .56885926/.56789831; conditioned readout .57038409/.56904376. Vanilla .56857352. Encoded quantum leads this exploratory short-budget screen, while dynamic geometry and conditioned readout lose to their matched MLP controls. No configuration promoted; one historical seed and256 updates do not establish robust accuracy/compute benefit.
- Next decision remains dependent on all1024-step outcomes and seven subsequent controlled routing trials. Keep full cross-task configurable-system goal active; these are candidate-screen results, not the requested final optimization or scaling claim.

### September 20, 23:28 +03 — requested detailed history and eleven-benchmark position

This entry supersedes earlier live/pending descriptions where the artifacts below demonstrate completion. User requested an updated us-versus-SoTA table and detailed history. Re-read the living ladder, primary benchmark sources and saved results; do not interpret this as eleven SoTA wins. Seven tasks have completed hybrid/control grids; four lack a final benchmark score. The sprint arena, configuration search, mechanism ablations and two TokaMark backbones are not additional benchmark tasks.

#### Benchmark snapshot

Numbers below are three-seed means unless otherwise stated. V = our fine-tuned vanilla backbone; M = our classical MLP adapter; Q = our hybrid. Lower is better except Sophelio and video scores. Published references are not automatically current global SoTA or equal-compute comparisons.

| Task | External reference / relevant anchor | Our Q | Our V / M | Evidence-based position |
|---|---|---:|---:|---|
| TRL2D, rollout steps13:30, fixed start | Published CNext1.01; local released-checkpoint anchor1.0132 | .9528 ± .0542 | 1.0301 / 1.0322 | Strongest matched-control result: Q wins3/3 seeds against both. Approximately6% below local anchor; not universal/compute-optimality evidence. |
| MHD_64, one-step VRMSE | Published CNext.1633; released-checkpoint reproduction mismatch | .1855 ± .0143 | .1855 / .1938 | Null against vanilla; has not reached published.1633. |
| TORAX-Bench, rollout8 | Own generated-data benchmark; matched-protocol persistence.129 | .0292 ± .0023 | .0321 / .0295 | Adapter gain over vanilla; Q beats M in only1/3 seeds. No external SoTA ranking. |
| Pelzer1HP, MAE °C | Recorded published U-Net.046; exact split parity unresolved | .035881 ± .004608 | .036363 / .036152 | Numerically below reference; no matched external SoTA claim. Q winsM1/3, V2/3. |
| Rayleigh–Bénard, mean one-step VRMSE | Published TFNO.6566 / CNext.6699; local CNext.667859 | .422075 ± .059008 | .456296 / .527535 | Lower mean than published baselines and local controls; Q winsM3/3 but V1/3. Additional fine-tuning budget and seed variance matter; not a global SoTA claim. |
| shear_flow, mean one-step VRMSE | Published CNext.8080; local anchor.994612, discrepancy unresolved | .265950 ± .068108 | .268552 / .253900 | MLP has best mean. Q wins2/3 paired seeds but one large regression; no robust Q-specific win. |
| TokaMark task1-1, TaskNRMSE | Freshly checked TokaMind paper Table2: FT-Base.1373, CNN.1882 | No full score | No full score | Dataset-store failures; CNN/TokaMind are one task. Earlier.150/.2091 references are not this freshly checked table. |
| Sophelio DIII-D, S | Live challenge rank not verified; local40-shot protocol is not leaderboard | .785608 ± .003813 | .779008 / .786834 | MLP has highest mean. No zero-shot MAST result or leaderboard claim; below recorded.85 eligibility target. |
| Physics-IQ Verified, i2v | Current official merged board: Cosmos3-Super42.7 ± .8 | No benchmark score | No benchmark score | Latent P1 losses are not Physics-IQ scores. Generation/cache pipeline still incomplete. |
| WorldModelBench, total score | Official board JSON: KLING9.10; open OpenSoraPlan-T2V8.04 | No benchmark score | No benchmark score |60-instance plan is not full350 leaderboard protocol. Judge smoke is not a completed benchmark. |
| BubbleML2.0 | Official Bubbleformer reference; checkpoint/horizon/scoring parity pending | No score | No reproduced score | Repo cloned; dataset/checkpoint reproduction remains pending. |

TRL2D sliding-start results are a separate protocol: Q.9486±.0231, V.9867, M1.0041, local anchor.9707, interaction-preserving mean-field.9782. Do not substitute these numbers into the fixed-start published comparison. Primary Holm comparisons did not establish an anchor win across seeds; quantum versus mean-field also remained unresolved. One-step Q.19315 versus published.1956 is not Q-specific because vanilla matches it.

Rayleigh–Bénard was independently re-audited from existing CSVs using `paper/phase_b/audit_well2d_csv.py`: all ten files contain34300 unique finite windows (175 trajectories ×196 starts), status `complete_csv_grid`; no new test inference. Aggregates above are from `runs/well2d/rayleigh_benard/cnext/csv_audit.json`. Shear-flow uses the existing analogous audit (21952 windows/file). Sophelio means/sampleSD were recomputed directly from all nine `runs/sophelio/*_s133*.json` official.S fields. Pelzer source remains `runs/pelzer_unet/summary_sep20.json`. TRL2D/MHD/TORAX figures retain the completed campaigns recorded earlier in this history and the living ladder; no new test evaluation or retuning on test data.

Primary-source refresh: [The Well Tables2–3](https://arxiv.org/html/2412.00568), [RB official card](https://huggingface.co/datasets/polymathic-ai/rayleigh_benard), [TokaMind Table2](https://arxiv.org/html/2602.15084v1), [Physics-IQ official board](https://github.com/google-deepmind/physics-IQ-benchmark#physics-iq-verified-leaderboard), [WorldModelBench](https://worldmodelbench-team.github.io/) and its `leaderboard_data.json`, [Bubbleformer](https://github.com/HPCForge/Bubbleformer). The Well paper explicitly describes compute-limited baselines rather than peak modern performance. Newer foundation-model statistics must be matched for mean/median, horizon, split and training allowance before ranking. Pelzer.046 remains a previously recorded reference, not freshly re-established split equivalence.

Corrections to older competitor notes: the currently retrieved Physics-IQ merged table has Cosmos3-Super42.7±.8 as top i2v; the earlier StrucPhysVideo45.52 claim is not supported by that table and is not used here. TokaMind Table2 currently gives.1373 for FT-Base task1-1, not.150; its CNN is.1882, not the separately recorded benchmark CNN.2091. Preserve older notes as history, not current verified standings.

#### Configuration campaign finished; endpoint calibration failed

The11 candidates all completed256 steps and were all resumed to1024 without pruning. Source/protocol hashes and model/optimizer/RNG state remained enforced; exploratory seed1337, official backbone, batch4 TD4, search-validation only. All22 stage snapshots exist. Screen finished18:38:50. No confirmation/test evaluation and no candidate promotion.

| Configuration | Arm |256-step endpoint |1024-step endpoint | Best through1024 |
|---|---|---:|---:|---:|
| Reference | Vanilla |.568574|.589847|.550069|
| Reference | MLP |.569687|.595301|.551765|
| Reference | Q |.567937|.593854|.550643|
| Encoded | MLP |.569481|.591887|.551464|
| Encoded | Q |.566655|.589376|.548994|
| Bounded static geometry | MLP |.570312|.593572|.550990|
| Bounded static geometry | Q |.569624|.593042|.551888|
| Bounded dynamic geometry | MLP |.567898|.590491|.551236|
| Bounded dynamic geometry | Q |.568859|.592835|.553312|
| Conditioned readout | MLP |.569044|.588161|.553649|
| Conditioned readout | Q |.570384|.588595|.552021|

`fidelity_report.json` ranks best-so-far scores: rho=.7 and early-top-two recovery pass. This is NOT the registered fixed-step ranking. Independent `endpoint_audit.json` ranks exact endpoints: rho=-.3, early top two encoded/reference miss the late quantum winner conditioned_readout; fixed-step short-fidelity gate FALSE. Therefore do not use256-step pruning for endpoint-based future searches. Best-checkpoint and endpoint results answer different questions and must both remain visible. Overall endpoint winner is the conditioned-readout MLP, not quantum. Encoded quantum's best-checkpoint win is not automatic confirmation or compute optimality. Previous five endpoint-audit regression checks passed; the actual complete audit now has no pending stages.

#### Seven-arm long-routing campaign finished and audited

Chain automatically ran all seven registered1024-step arms18:38:51–19:02:31. This campaign uses first48 training trajectories, batch2 TD4, seed1337, frozen difficulty scorer, random per-input caps0/1/2/3 registers, relative local correction bound10%, search-validation rollout8 at five saved steps. It differs from the batch4 global-configuration screen; cross-campaign raw numbers are not matched comparisons.

`paper/config_search/audit_long_routing.py` executed successfully: `complete_exploratory_audit`, one independent seed, promotion false. Checkpoint/snapshot/protocol/budget-count and allocation-trace consistency checked. Report: `runs/config_search/long_routing_v1/audit.json`.

| Trained arm | Routing | Endpoint with budget0 | Endpoint with cap3 | Mean active qubits at cap3 (Q only) |
|---|---|---:|---:|---:|
| Vanilla | — |.627966|—|0|
| Quantum | Adaptive |.619982|.619959|30|
| Quantum | Static locations |.624500|.624520|15.2778|
| Quantum | Permuted locations |.626209|.626204|30|
| Classical | Adaptive |.638364|.638344|0 (classical slots only)|
| Classical | Static locations |.626875|.626879|0 (classical slots only)|
| Classical | Permuted locations |.621538|.621552|0 (classical slots only)|

The raw audit labels classical slot-equivalent counts `mean_active_qubits`; those are NOT physical/simulated quantum executions for the classical arm. Preserve the raw artifact but interpret classical resources as matched slot budgets. Actual total parameters are18767377(Q),18765334(classical),18575344(vanilla): report the2043-parameter Q/classical gap, not exact matching.

Adaptive Q at cap3 beats adaptive classical (~2.88%), permuted Q (~1.00%) and static Q (~0.73%) at this endpoint. However, strongest classical routed endpoint is permuted classical.621552, only~.26% above adaptive Q. The adaptive-Q own-backbone difference from budget0 to30 qubits is only.00002325 VRMSE (~.00375% relative aggregate improvement). Almost all between-trained-model difference persists at zero inference allocation. Adaptive/permuted Q always use the full available cap on these validation inputs; no learned demand-dependent compute savings demonstrated. Static cap3 averages fewer active registers, so that comparison is not equal executed quantum cost. Earlier learning curves also change ordering. These are exploratory training-dynamics signals, not robust causal benefit from difficult-signal allocation, not proof of entanglement necessity, and not a reason to fit RL to tiny allocation labels.

#### Runtime state, limitations and next work

- Research coordinator completed normally19:02:31 and restored idle Physics-IQ/WMB orchestrators35074/35077. Coordinator834208, screen895418 and the final trainer have exited; do not keep describing the screen or routing trials as running. The virtualenv activation requested by user was verified as `/home/mohhakami/Documents/paper/quworld/.venv/bin/python`; uv remains the dependency/run tool.
- Both video orchestrators were live on inspection around23:25. Physics-IQ log repeats T5 caching with7 cached/56 missing; its latest cache log raises `PHYSIQ_T5_DEVICE=cuda but CUDA is unavailable (CUDA_VISIBLE_DEVICES?)`. A live retry loop is not progress toward a scored video result. This needs diagnosis in the next execution turn; no CPU fallback or new score claimed. WMB has no completed score artifact found.
- TokaMark CNN logs contain unreadable29902.zarr; TokaMind smoke contains missing27534.zarr. Earlier downloader/DONE markers do not establish a usable dataset or full task result.
- Lean's20 conditional/resource theorems remain compiled/audited evidence for mathematical bounds only. Neither a universal optimal controller nor a supported empirical hybrid scaling law exists. Three independent10-qubit registers do not equal an entangled30-qubit state.
- Next scientific work: preserve these results; reject endpoint-based short pruning; diagnose learning-curve/optimizer and correction-path utility; use architecture/context-matched controls and fresh seeds before extending a claimed benefit. Reserve confirmation and official tests under document26. Quantify useful per-input marginal benefit and full search/training/inference costs before a learned selector, JEPA/world-model/RL expansion or scaling. Cross-task transfer across all eleven remains unproven. Goal stays active.

### September 21, 00:01: OOM reset during Physics-IQ generation; NVFP4 directive

- **What happened.** The Physics-IQ vanilla arm started at 23:35 on Sep 20 (after the T5 cache stage was fixed: it had been hiding CUDA unconditionally and the orchestrator retried every 2 min). With nothing else on the GPU, the generator reported `peak_alloc 69.6 GB, system used 123.7 GB` on its second video; MemAvailable sat at 5% from 23:40 while earlyoom terminated daemons (Xorg, uvicorn, dbus, gdm3) that it could see, never the python inside the `MemoryMax=80G` systemd scope; the journal ends 23:44:42 and the box came back at 00:01. Two of 198 videos survived.
- **Root cause (measured, not inferred).** Before the first video: 37.7 GB CUDA reserved, 2 GB RSS, 88.5 GB system used. The gap is the page cache holding the eight ~9 GB fp32 safetensors shards (~65 GB) that `from_pretrained` streamed through; on unified memory the kernel could not reclaim them fast enough against 36 GB of pinned CUDA working memory per video. memguard's shed list did not include the Phase-C generators. The Sep 19 teacher-cache pilot had the same profile at 101 GB and survived only because the arithmetic was 20 GB kinder.
- **Fixes (commit a19893c).** Runner: `posix_fadvise(DONTNEED)` on every weight file after pipeline load; T5 stage honours `PHYSIQ_T5_DEVICE=cuda`. Orchestrator: a MemAvailable floor tripwire (12 GB, two 15 s samples) SIGTERMs the resumable generator instead of losing the box. memguard: `physiq_runner.py` and `wmb/generate.py` appended to the shed order.
- **User directive (00:15): "ideally always go with NVFP4 as much as possible throughout."** Saved as memory `nvfp4-everywhere`. Feasibility on the GB10 the same hour: torch 2.13+cu130 on compute capability 12.1 exposes `float4_e2m1fn_x2` and `F.scaled_mm` with the NVFP4 recipe (BlockWise1x16 e4m3 scales in the 32×4×4 swizzle + TensorWise fp32 global scale); torchao 0.14–0.18 and modelopt have no aarch64 wheels. Wrote `paper/nvfp4.py` (dependency-free `NVFP4Linear`, `quantize_linears_`, `fidelity_check`; e2m1 rounding verified exhaustively, ties-to-even). Layer-level result at the DiT's 5120×13824 shape: rel L2 0.139 vs bf16 on Gaussian inputs (both operands 4-bit), 27–46 ms vs 8–9 ms per call (activation quantization is pure-PyTorch elementwise; a fused kernel is future work). Wired `PHYSIQ_DIT_QUANT=nvfp4` into `build_pipeline` and a `--stage quanttest` (one scenario bf16 then NVFP4: chunk-0 / last-chunk x0 rel L2, video PSNR, time, peak system memory) — running at 00:25 under an 80 GB scope with a 12 GB floor watchdog. Rule from the memory note: every arm at the same precision, fidelity vs bf16 recorded in the ladder; the P1 adapters were trained on bf16 teacher latents.
- **TokaMark unblocked.** The 01:52 `TOKAMARK_DONE` was false: all three CNN arms and the TokaMind smoke crashed on `GroupNotFoundError` for shots 29902/27534. Cause: a resumed fsspec `get(recursive=True)` into an existing destination nests the copy as `<id>.zarr/<id>.zarr/`, leaving stale partial groups without a root `.zgroup` at the outer level; 1,282 of 11,573 shots were like this while the downloader reported `ok 11573 fail 0`. `runs/tokamark/repair_nested_zarr.py` swapped the complete nested copies into place (leftovers under `data/tokamark/zarr_partial_leftovers`, 38 GB, reversible). A zarr open audit then found 205 shots with no `equilibrium` group; S3 confirms they lack it upstream and the official split contains 204 of them, so the baseline tolerates it. Sentinel removed; `runs/tokamark_relaunch_sep21.sh` relaunches the CNN lane and the TokaMind chain once the quanttest releases the GPU.
- **State at 00:40.** GPU: quanttest. Sophelio: 3×3 complete since Sep 20 (mean S 0.7856 quantum / 0.7790 vanilla / 0.7868 MLP). Physics-IQ: 2/198 vanilla videos; will resume under whichever precision the quanttest justifies. WMB: generation still queued behind Physics-IQ. Tailscale: logged out since the Sep 19 reset (user re-login pending).
- **00:26 NVFP4 verdict (`runs/phase_c/physiq_quanttest.json`, commit below).** Scoped to the 400 transformer-block linears (the fp32-asserting time-embedding path and other conditioning linears stay native): weights 28.1 → 7.9 GB, peak CUDA 69.6 → 49.4 GB, peak system 113.1 → 107.5 GB; but chunk-0 x0 rel L2 0.342, last-chunk 0.491, PSNR 20.3 dB vs bf16 and 518 s vs 232 s per video. Fails the 0.05 fidelity gate → Physics-IQ arms resume in **bf16** (the page-cache drop alone brought the bf16 peak from 124 GB to 113 GB with a working 12 GB tripwire). NVFP4 stays a tracked future-work item (FW-I in the ladder): fused activation kernel, outlier smoothing, fp8 rung, smaller trunks first.
- **00:26–00:30.** TokaMark CNN lane took the GPU first (relauncher fired between quanttest attempts): vanilla s23 trains on the full repaired store, step 1000 val loss 0.169, no zarr errors. Physics-IQ orchestrator relaunched and correctly waiting for a python-free GPU; WMB orchestrator relaunched behind it.

### September 21, 10:20–10:50: priorities reset, Optuna campaign, notebook, two new rows

- **User (10:20):** park the three long video/world-model rows (a B300 may take them later); improve fusion/fission/geothermal/turbulent; add two industry benchmarks (materials/drug + logistics/traffic/scheduling); Optuna tuning on train/val with a single test evaluation. Physics-IQ / WMB orchestrators stopped (resumable). Memory `goal-priorities-sep21-hpo`.
- **Optuna layer (commit 6ea2838, 7aef9e1):** `paper/hpo/driver.py` (per-arm SQLite studies, TPE, same trial budget per arm, selection on VALID only, `final` = selected config × 3 seeds × one TEST run) + bench modules for TORAX (16 trials/arm), Pelzer U-Net (12) and Sophelio (10; official scorer on the 20 validation shots via `--score-split val`). Grammar: `-L=<depth>`, `-mlpref` (control pinned to the reference register's budget while Q tunes depth/topology/spacing/link). Runner `runs/hpo_runner.sh` started 10:21 (TORAX none → mlp → q → finals → Pelzer → Sophelio).
- **Wave G-k (budget selector):** k=3 chosen 1.080 vs random-k 1.082 vs MLP 1.105 → 9 of 30 atoms is control-level; k=5, k=7 running.
- **Notebook (commit 86d329a):** `notebooks/18_method_and_benchmarks.ipynb` executed (28 cells, 14 figures, 0 errors): method, drawn model diagram + trained register graph, topology families, scoreboard from result files, radar and pie charts vs SoTA, live model outputs for TRL2D (one-step + rollout curve), RB, shear_flow, Pelzer, TORAX (fresh 1,000-step arm: 0.0280), TokaMark per-feature (both trunks), Sophelio score terms, Wave G bars, live HPO cell.
- **New rows chosen (10:45):** traffic forecasting METR-LA/PEMS-BAY (BasicTS trunk; SoTA 60-min MAE ≈ 3.08 / 1.63) and TDC ADMET (22 endpoints, scaffold split, 5 seeds); Matbench Discovery rejected for this box (257k relaxations). `runs/newrows/setup.sh` clones BasicTS + TDC, builds `.venv-tdc`, downloads the ADMET group and attempts the BasicTS traffic data.
- **Standing at 10:50:** TokaMind quantum arm epoch ~12/20; HPO TORAX `none` searching; G-k k=5 running.

### September 21 evening → 22 01:06: Optuna campaign, first three rows complete

| Row (tuned on VALID, 3 seeds once on TEST) | vanilla | matched MLP | register (selected config) | published |
|---|---|---|---|---|
| TORAX VRMSE@8 ↓ | **0.0158 ± 0.0006** | 0.0176 ± 0.0042 | 0.0164 ± 0.0056 (L=6, shared, dense grid) | — (fixed protocol was 0.029–0.032) |
| Pelzer MAE °C ↓ | **0.0291 ± 0.0017** | 0.0352 ± 0.0069 | 0.0348 ± 0.0065 (L=3, ring) | 0.046 |
| Sophelio S ↑ | **0.7905 ± 0.012** | 0.7782 ± 0.013 | 0.7830 ± 0.018 (L=2, ring, dense) | award bar 0.85 |

Reading: tuning moves every arm a lot (TORAX halved, Pelzer −20%, Sophelio +0.012), and on every row the adapters win the single-seed validation selection and then lose to the trunk on test with 4–10× the trunk's seed spread. Adapter selection needs a multi-seed objective; the driver gets a `--search-seeds` option next. TokaMind 3×3: seeds 1337/1338 all tie at 0.150–0.151; last arm (Q s1339) running. Video rows parked; SVRPBench / TDC ADMET / Matbench harnesses queued (`21-…` §E).

## Sep 22 21:40 — Optuna round 2 (multi-seed selection) and the completed TokaMind grid

Round 2 re-tuned the two adapter arms with the objective changed from one seed's validation metric to the **mean validation metric over seeds 1337+1338**, at the same trial budget per arm, followed by the same single 3-seed test evaluation.

| Row (tuned on VALID, 3 seeds once on TEST) | trunk (round 1) | matched MLP | register | published |
|---|---|---|---|---|
| TORAX VRMSE@8 ↓ | 0.0158 ± 0.0006 | **0.0141 ± 0.0011** | 0.0168 ± 0.0055 (L=6, ring, shared) | — |
| Pelzer MAE °C ↓ | 0.0291 ± 0.0017 | 0.0292 ± 0.0016 | **0.0288 ± 0.0022** (L=3, grid, sp 1.5) | 0.046 |
| Sophelio S ↑ | 0.7905 ± 0.012 | 0.7848 ± 0.003 | **0.7929 ± 0.014** (L=3, ring, shared) | bar 0.85 |

The round-1 conclusion ("adapters win validation, then diverge on test") was a **selection artefact, not an adapter property**. With two seeds in the objective the TPE picks a markedly lower learning rate (Pelzer 2.4e-5 vs 1e-4) and five of the six divergent test seeds disappear: Pelzer MLP 0.0352→0.0292, Pelzer Q 0.0348→0.0288, TORAX MLP 0.0176→0.0141, Sophelio both arms up. The one survivor is the TORAX register, which still throws a 0.0232 seed at lr 2.5e-3 — on that row the register is the arm whose tuned configuration does not stabilise.

The adapters now lead two of three rows, but every lead is a fraction of one seed SD (Pelzer +0.0004 on ±0.0022; Sophelio +0.0024 on ±0.014), so the honest reading is a **three-way tie on Pelzer and Sophelio and an MLP win on TORAX**. Two caveats are recorded in the ladder rather than papered over: (1) the trunk numbers above were selected under the round-1 single-seed objective, so the comparison currently spans two protocols — `runs/hpo_none_ms_runner.sh` is re-selecting the trunk under the identical two-seed objective and the rows are marked PROVISIONAL until it lands (~6 h); (2) `q30-…-link=shared` with a ring geometry was selected independently on TORAX and Sophelio, which is the first sign that the Wave-G promotion transfers as a *search prior* even though its fixed-protocol transfer was null.

TokaMind finished its 3×3: vanilla 0.1509 ± 0.0005, MLP 0.1504 ± 0.0002, register **0.1501 ± 0.0009** against the published FT-Base 0.150. All three arms reproduce the SoTA and the paired register-vs-trunk difference is +0.0008 ± 0.0011 (t = 1.18, p ≈ 0.36, n = 3) — not significant. Combined with the CNN-trunk row (0.2118 → 0.2033, −4%), the pattern across both trunks of the same task is that the adapter's headroom is whatever the trunk leaves on the table.

## Sep 22 22:20 — The judge: quantitative and qualitative verdict, three self-caught errors, two new rows

**Trunk re-selection (round 2 completion).** TORAX's trunk arm re-tuned under the same two-seed objective the adapters got: 16 fresh trials, and the TPE converged to the round-1 point, so the test result is identical (0.015822 ± 0.000586). Pelzer and Sophelio trunks are running; until they land those tuned rows are flagged "mixed protocol".

**Quantitative judge** (`paper/judge/judge.py` → `runs/judge/scorecard.md`) reads every per-seed value from the result files, not the ladder. It fixes one primary row per benchmark a priori and reports, per row, seed-paired gains, paired t, the sign test, dz, Glass's delta and the minimum detectable effect. Across rows it reports a tie-aware sign test, a mean-over-rows t-interval, and a DerSimonian–Laird pool with leave-one-out. Verdict (`30-campaign-judgment-sep22.md`):
- Q vs trunk: 5 better / 1 worse / 2 ties (p = 0.22), mean over rows +2.0% [−2.5, +6.5] → **null**.
- Q vs matched MLP: 4 / 2 / 2 (p = 0.69) → **null**.
- MLP vs trunk: 2 / 2 / 4 → **null**.
- In every multi-seed row where we "beat the published number", the vanilla trunk beats it by the same margin.
- Power: n = 3 resolves ≤2% only on TokaMind, Pelzer and Sophelio; four rows have 20–40% seed variance.
- Defensible claim: the register is a drop-in, no-regret component across eight SoTA-class industrial trunks.

**Errors the judge caught**, each fixed before it reached a claim:
1. **MHD compared arms of unequal training length.** 4 of 9 runs stopped at ~7.5k of 20k steps, and seed 1338's MLP alone produced "Q beats MLP 3/3". At matched length the arms agree to 4 decimals and the gates never opened (|α| ≈ 4e-5). The row now uses length-matched seeds, and the unequal grid stays as a flagged secondary.
2. **shear_flow's published anchor is not reproduced** by our evaluator (official checkpoint 0.9946 vs 0.808).
3. **An inverse-variance pool is hijacked by rows whose arms are identical**, which is why the mean-over-rows interval is now reported as well.

My first reading of Q vs MLP ("6/8, +2.3%") was inflated by error 1 and is withdrawn in the document.

**Pre-registration** `31-powered-replication-prereg.md` was committed before any run: Pelzer 12 new units (3 trunks × 4 adapter seeds), Sophelio 8 new seeds, TokaMind 4 new seeds. The primary endpoint is H1, Q vs MLP pooled over new units only. The analysis path was verified on synthetic data (planted +2% → +2.08% [1.85, 2.31]; planted null → −0.07% [−0.30, +0.15]).

**Logistics row: SVRPBench → RouteFinder.** SVRPBench's public release failed validity checks: the train and test parquet files are byte-identical, 20% of instances contain a customer whose demand exceeds every vehicle's capacity, the TWVRP rows carry no time windows, and there is no scorer or code. RouteFinder (TMLR 2025 → ICML 2026) is a pretrained VRP foundation model with released checkpoints and data. **RF-TE is reproduced to +0.002% objective (max 0.012%) on all 16 n = 100 variants** (mean gap 3.126% vs published 3.124%), the tightest anchor in the campaign; this needed a torchrl legacy-spec alias shim. The arms harness runs their own EAL fine-tuning protocol plus our bottleneck on the encoder's node embeddings; the smoke test is running.

**Drug row: TDC ADMET** harness uses TDC's own protocol verbatim with a frozen ChemBERTa-77M trunk; 22 endpoints × 3 arms × 5 official seeds are queued.

**Queue:** HPO trunk (Pelzer → Sophelio) → TDC → replication (Pelzer → Sophelio → TokaMind) → RouteFinder EAL.

## Sep 22 23:10 — Deep research: per-input register configuration (x, y, z, connectivity, tethering, Jev), FFT/wavelet evidence, Lean

Report: `32-dynamic-register-configuration.md`.

**Survey.** Per-task parameterized geometry is established (PulserDiff trainable coordinates, trainable-frequency models). Per-input geometry exists only for graph inputs (QEK, attributed-graph kernels) and abstract per-token couplings (HyperQ, 21 Sep 2026). No published work routes the content a classical trunk fails on to atoms placed by that content.

**Hardware.** On real machines the cheap per-input lever is the occupancy of a fixed 3-D site grid, since every shot rearranges atoms anyway. Continuous positions change per task (SLM kHz, 3-D AOD 4.2 m/s). z matters for non-planar graphs and for signed or zero dipolar couplings. Tethers are quantum wires, messenger qubits or relay atoms.

**Lean.** `formal/QuotaScaling/SpectralRouting.lean`: 18 theorems, standard axioms only.
- The pooled bottleneck sees only the DC mode.
- Parseval band split, and the exact residual after removing K coefficients (the ceiling for any K-tile corrector).
- Top-K optimality of budgeted band assignment.
- L-layer spectrum lies in [−L, L].
- vdW geometry acts as frequency modulation.
- Occupancy patterns on a lattice grid are hardware-valid by construction.
- Dipolar sign structure needs z.

**Evidence on TRL2D:**
- The trunk misses ~45% of the change at |k| ≥ 33 but 8–11% at |k| ≤ 8, and the top 1% of wavelet tiles hold 92% of detail error energy.
- The register's only measurable effect is at k = 0 (−0.49% ± 0.21, 3/3 seeds), a band holding 2.9% of error energy: a mechanistic explanation of the campaign nulls.
- The top-30 tiles hold 32.9% of detail error energy (11× the DC ceiling). The zero-parameter router (trunk's predicted change) gets 9.4% (~1,650× random).
- Learned routers lose to it at K ≤ 300 on VALID and TEST and win only at K = 3000, so with scarce qubits routing should be non-parametric.
- The hard tiles form small fronts chaining across scales, which motivates a 3-D pyramid placement (x, y ← tile, z ← scale).

**Jev** (TypeSafe, launched 15 Sep 2026) is the right *pattern* (typed options, calibrated, abstain), implemented in-graph. The product stays out of the model path: text-only, API-only, 70–500 ms, not differentiable.

## Sep 23 02:10 — Goal switch to "reach and improve SoTA"; Sophelio full-data pipeline + joint NAS; TDC trunk switch; RealPDE added

**Triage of gaps.** Sophelio is the largest reachable gap. Our starter-kit pipeline scores 0.80 on DIII-D against the Challenge-1 leaderboard's 0.9932; the 0.85 figure was only the Ch2 award gate. The starter used 200 of the 7,041 local shots and a ridge scalar head frozen at R² 0.472.
- Built a full-data extractor: coil currents at 14 time lags, Thomson summaries and profiles; 1.56M frames.
- Built a PCA-ψ residual MLP with learned scalar and auxiliary heads. Smoke test: R²qb 0.897.
- Built a **joint classical+quantum NAS** following Hyb-HANAS (DAC 2026) and HQ-NAS (2605.18345), on Optuna 5.0 (multi-objective TPE, `trial.set_constraint`), with successive-halving pruning and LLM-proposed seeds. The couplings include the spectral-routed register, which writes only the high-order ψ band.
- Running unattended.

**TDC.** A 2026 audit found only MapLight, MapLight+GNN and CaliciBoost reproduce among the leaders. The ChemBERTa run was stopped before it started; CheMeleon (Chemprop 2.2 foundation model) is next.

**RealPDE** (user request) is added as a paper benchmark, with the competition rules waived by the user and the licence kept. The protocol is RealPDEBench's public foil splits; the target is Rel L2 0.0145 (U-Net, real-world finetuning). A possible test leak in the competition release (`3750_0` is an out-dist test trajectory) is being checked.

The replication was deferred before any unit ran.

## Sep 23 03:10 — Two OOM incidents and their cause; RealPDE trunk corrected; Sophelio feature bug; TDC classical lift

**OOM, twice (02:26 reboot; 02:43 kernel killed the user systemd manager and the tmux session).**
- The kernel OOM dump names the cause: my RealPDE smoke test held ~100 GB of unified GPU memory.
  - Three 3-D neural operators ran at batch 64, and on the GB10 CUDA memory is system RAM.
  - Anonymous memory was only ~13 GB.
- Jobs started from a Claude shell inherit `oom_score_adj -1000`, so earlyoom and the kernel killed Xorg, dbus, tmux and systemd instead.
- Guards now in place:
  - `runs/capped.sh`: OOM priority +800, plus a cgroup cap when available.
  - `QW_GPU_GB` (`paper/gpucap.py`): per-process CUDA budget. It later turned a Transolver overrun into an in-process error.
  - `runs/memguard.sh 14`: kills the largest GPU job below 14 GB free. It later caught CatBoost reserving 95% of free memory.
- Rule: smoke-test peak memory on a small batch before any full run.

**Sophelio v1 had a silent feature bug.**
- Float32 standardisation overflowed on Thomson n_e and p_e (~1e20), so all 84 density and pressure features were zeroed.
- v1's VALID S 0.9822 therefore used Thomson temperatures only.
- Fixed with float64 moments. The joint NAS was restarted, seeded with the v1 trunk ± coupling (none / latent / srr).

**RealPDE: the competition checkpoints are off-distribution.**
- The competition release and RealPDEBench label the same experiments but process them differently: u mean 0.046 vs 0.280 on (3750, 0), and no frame matches.
- The competition FNO scores rel-L2 0.147 on RealPDEBench windows.
- RealPDEBench publishes its own checkpoints, and on 32 VALID windows the FNO-finetune one gives **0.0144**, at the published best of 0.0145. That is the trunk.
- Loader and metric are verified identical to theirs; the full evaluation is running.
- The "leak" is by flow condition, not data; reported both ways.

**TDC.**
- CheMeleon alone is below the reproducible leaders (caco2 0.343 vs 0.276; bioavailability 0.61 vs ~0.74).
- MapLight is being reproduced on GPU CatBoost, then ensembled with the CheMeleon arms.

**SRR pilot** (pre-declared, non-decisive) is running on TRL2D.

## Sep 23 04:45 — RealPDE: the paper's Table 1 is not reproducible on the current data release; the anchor is the released checkpoints

- **Released FNO-finetune checkpoint, RealPDEBench's own code and metric, public HF data v2.0.1:**
  - TEST (5,140 windows) rel-L2 **0.0141**, RMSE 0.0031.
  - By partition: excluding the 2 competition flow conditions 0.0143; in-dist 0.0132; out-dist 0.0146; seen 0.0143.
  - Relative high-frequency error 0.658.
- **Paper Table 1 (arXiv 2601.01829), real-world finetuning:**
  - FNO: rel-L2 0.0206, RMSE 0.0120.
  - U-Net (best): 0.0145 / 0.0094.
- **Why they differ:**
  - The implied target scale (RMSE / rel-L2) is ~0.22 on the current release against ~0.58 in the paper.
  - The HF data were re-converted on 2026-09-18 (v2.0.1, `lazy_slicing_v2`). Its `version.json` asks for code ≥ 0.2.0, which is not public; upstream `main` is 0.1.0.
- **Loader is consistent with the checkpoints:** it equals `FoilHFDataset.__getitem__`, and the resolution-bound CNO and U-Net checkpoints work at 64×128.
- **Consequence:**
  - The anchor for this row is the set of released checkpoints (U-Net / FNO / CNO / Transolver) evaluated with their code on the current release.
  - Table 1 is cited alongside but is not claimed as the reference number.

## Sep 23 05:30 — Sophelio: where the remaining gap is

All numbers are VALID official S on the NAS's fixed 40-shot subset. The leaderboard reference is 0.9932.

- **NAS stage 1 so far:**
  - The v1 config with the n_e / p_e fix: **0.9831**.
  - γ = 0.5 (detail up-weighting): 0.9754.
  - k = 256 with width 2048: 0.9732.
  - Bigger or detail-weighted trunks are worse. R²ψ is always 0.9996; the spread is all in Consistency (0.887–0.931).
- **The PCA basis is not the limit.** True ψ projected onto k components scores S 0.9991 / 0.9994 / 0.9996 (k = 64 / 128 / 256), with Consistency ≥ 0.998. The gap is coefficient-prediction error.
- **Post-processing, using model outputs only:**
  - Per-shot Gaussian smoothing, σ = 1 frame: 0.9831 → 0.9843. Wider windows hurt.
  - Equal-weight ensemble of the 3 stage-1 trunks: 0.9848, or **0.9855** with σ = 1 (Consistency 0.943).
- **What is left:**
  - Boundary shape: elongation 0.887, upper triangularity 0.928, volume 0.952, li 0.950.
  - The direct scalar heads are better than the ψ-derived axis (R_axis 0.976 vs 0.958). Axis alignment would give only about +0.001 S.
  - Next research item: a shape-aware ψ objective. Linearise the scorer's shape quantities with respect to the PCA coefficients and weight the loss along those directions. Shape labels for TRAIN come from the k = 256 reconstruction, which is near-exact.

## Sep 23 05:25 — TokaMark 1-1: 3-seed ensembles beat the published TokaMind 0.150; shot-level register signal

Each arm's ensemble is the mean of its own 3 fine-tunes. It is scored by TokaMind's own metric code through a record/replay hook, and the replay was verified exact.

| Arm | Ensemble NRMSE |
|---|---|
| Vanilla | 0.14625 |
| MLP | 0.14601 |
| **Register (mr30ryd)** | **0.14544** |

All three beat TokaMind's published 0.150; the register ensemble by 3.0%. Single seeds averaged 0.1509 / 0.1504 / 0.1501.

**Shot-level paired tests** (1,020 test shots; `paper/tokamark/paired_ens.py`):

| Comparison | Gain [bootstrap 95% CI] | Paired t | Wilcoxon |
|---|---|---|---|
| Register vs MLP | +0.39% [+0.01, +0.78] | p = 0.047 | p = 0.18 |
| Register vs trunk | +0.55% [+0.14, +0.98] | p = 0.012 | p = 0.006 |
| MLP vs trunk | +0.17% [−0.18, +0.51] | n.s. | n.s. |

**Reading:**
- The tests condition on the 3 trained models per arm. The seed-level (3 vs 3) differences stay non-significant.
- Register vs MLP is marginal: three comparisons were made, and Wilcoxon does not agree with the t-test.
- Suggestive, not established. It is the strongest register signal so far, on a row where every arm beats SoTA.

**TRL2D one-step TEST, 3-seed ensembles:** vanilla 0.1929 / MLP 0.1930 / register 0.1930 (published 0.1956). The ensemble lift is ~0.0003 and there is no arm difference.

## Sep 23 06:15 — Sophelio: diverse-trunk ensembles are the lever (full VALID)

On all VALID shots, not the NAS's 40-shot subset:

| Candidate | S | Consistency |
|---|---|---|
| NAS trial 5 (width 512, k 64, no lags, no Thomson) | 0.9848 | 0.939 |
| Trial 0 (v1 config, full features) | 0.9842 (0.9850 with σ = 1) | 0.937 |
| **Trials 0 + 5, equal weight** | **0.9880** | **0.953** |
| 4 trunks | 0.9872 | 0.950 |

- The two trunks are complementary. Trial 5 is better on triangularity (top / bottom 0.976 / 0.971 vs 0.951 / 0.936); trial 0 is better on R_axis and volume.
- The fast proxy and the 40-shot official score both rank candidates unreliably.
- Plan: when stage 1 ends, greedy forward ensemble selection (Caruana) over all saved trunks on full VALID, then a single TEST.
- Gap to the Ch1 leaderboard (0.9932): 0.005.

## Sep 23 07:05 — RealPDE: where the error is (VALID, 5,133 windows; `paper/realpde/error_bands.py`)

Error-energy share by thirds of the (t, h, w) wavenumber:

| Trunk | Low | Mid | High | Total error energy |
|---|---|---|---|---|
| U-Net | 77% | 17% | 6% | 1.27e9 (best) |
| FNO | 88% | 9% | 3% | 2.63e9 |
| CNO | 88% | 9% | 3% | 2.97e9 |

- The target's energy is essentially all in the low third (mean flow).
- An equal-weight 3-trunk ensemble (1.66e9) is worse than U-Net alone, so any ensemble must be VALID-weighted.
- **Cross-model error coherence:** low band 0.62–0.65; mid and high bands 0.80–0.93. Three unrelated architectures share their high-k error, which is the signature of PIV measurement noise in the targets. Routing to high-k tiles (SRR) would chase noise here.
- **What is learnable is model-specific and low-k.** That is the reverse of TRL2D (high-k error, DC-blind register), so RealPDE is the row where the pooled (DC / low-k) register insertion is aimed at the right band. It is a clean contrast test of the mechanism.

## Sep 23 09:15 — Status checkpoint

- **TokaMark 1-1:** above SoTA. 3-seed ensembles: register 0.1454 / MLP 0.1460 / vanilla 0.1462, against TokaMind's 0.150. Shot-level, register vs trunk p = 0.006.
- **RealPDE:** released-checkpoint anchor on TEST: U-Net 0.0095 / FNO 0.0141 / CNO 0.0151.
  - The paper's Table 1 (best 0.0145) is not reproducible on data v2.0.1.
  - High-k error is shared across models (PIV noise floor). The learnable error is low-k.
  - Low-mode corrector arms are built. The "none" arm reproduces the trunk exactly.
  - The FNO arm run was guard-killed with 21 GB of anonymous memory; an instrumented diagnosis is running.
  - VALID-weighted ensemble relaunched (coarser grid).
- **Sophelio (VALID):**
  - Best single trunk 0.9864 (40 shots); trials 0 + 5 ensemble 0.9880 on full VALID. Leaderboard 0.9932.
  - NAS stage 1 was guard-killed at 08:10; it resumes when memory frees, then stage 2 (couplings).
  - Greedy ensemble selector ready (`paper/sophelio/ens_select.py`).
- **TDC:**
  - CheMeleon vanilla done on all 22 endpoints; it is below the leaders on most.
  - MapLight on GPU CatBoost gave caco2 0.2904 with GPU defaults (published 0.276). CPU-default tree settings are now pinned; Exact leaves crash on GPU.
  - `runs/tdc_maplight_runner.sh` re-checks caco2 and runs all 22 only if MAE ≤ 0.285.
  - CheMeleon mlp30 / mr30ryd pass is running. `paper/tdc/ensemble.py` scores MapLight + CheMeleon ensembles.
- **TRL2D:** 3-seed ensembles, all arms ~0.1930. The SRR pilot waits for NAS stage 1.
- **RouteFinder EAL:** parked. It diverges to NaN under both fp16 and bf16.

## Sep 23 11:10 — TDC: exact MapLight anchor, verified leaderboard, validation-selected stack

**Leaderboard verified from source.** All 22 TDC ADMET pages were scraped on 2026-09-23 into `runs/tdc_ensemble/leaderboard_2026-09-23.json`.

- The caco2 leader is now CaliciBoost at 0.256, not MapLight's 0.276.
- The bioavailability page repeats the Pgp table (identical top-4 values). The MapLight paper's Fig. 1 shows bioavailability near 0.74, so that page's ranks are not meaningful.

**MapLight anchor (CPU CatBoost, user exception):**
- Reproduced within 2 SD (or 0.005) on 18 of 22 endpoints.
- Bioavailability cannot be checked against the site.
- The other three misses are in our favour: solubility 0.786 vs 0.792, BBB 0.921 vs 0.916, CYP2C9 inhibition 0.788 vs 0.783.
- caco2 is 0.282 vs 0.276, inside 2 SD.

**Fine-tuned CheMeleon (vanilla) alone:**
- #1 on lipophilicity (0.451 vs MiniMol 0.456) and LD50 (0.548 vs 0.552). Both margins are within about 1 SD.
- Below MapLight on 15 endpoints.

**Stack, `paper/tdc/stack.py`.** The protocol is in the docstring and was declared before any blended TEST score existed.

- One blend weight per endpoint over MapLight (train-only refit) × CheMeleon (best-on-valid checkpoint), chosen on TDC's seed-specific valid split. TEST is scored once.
- Validation predictions are recovered by matching each seed's saved TEST predictions to a checkpoint. All checkpoints matched; there were zero misses.
- Primary classical entry: MapLight+CheMeleon features × CheMeleon-vanilla.
- Hybrid comparison: × mr30ryd vs the matched × mlp30.

## Sep 23 11:15 — Sophelio: the gap is prediction error in consistency, not the PCA basis

**PCA ceiling** (ground-truth ψ projected onto k components, true q95/betaN, official scorer, fixed 40 VALID shots):

| k | S | Consistency |
|---|---|---|
| 32 | 0.9977 | 0.993 |
| 64 | 0.9991 | 0.998 |
| 128 | 0.9994 | 0.9985 |
| 256 | 0.9996 | 0.999 |

- A 64-coefficient decoder is not the bottleneck (the leader is 0.9932).

**Consistency diagnostic** (t0 + t5 ensemble, 7,758 frames, `paper/sophelio/cons_diag.py`):
- Zero derivation or LCFS failures.
- The residual is heavy-tailed: the worst 1 % of frames hold 23–36 % of SS_res for six scalars and 55 % for li.
- κ R² is 0.892. Closing the gap means fixing a small set of frames.

## Sep 23 12:05 — Sophelio shot-edge ablation (VALID, 40 fixed shots, trial-5 config; `paper/sophelio/edge_ablation.py`)

| Arm | Seed 1 | Seed 2 | Mean |
|---|---|---|---|
| base (trial-5 config) | 0.9762 | 0.9837 | 0.9800 |
| + edge inputs (time to end, elapsed fraction) | 0.9849 | 0.9828 | 0.9839 |
| **+ edge weighting (ψ loss ×3 on the first/last 10 % of each shot)** | **0.9858** | **0.9856** | **0.9857** |
| + both | 0.9863 | *(pending)* | — |

- **Seed noise is large.** The same config gives 0.9864 at seed 0 (rerun reproduces the NAS value exactly, so the code path is verified), 0.9762 at seed 1 and 0.9837 at seed 2. NAS rankings at a spread of ≈0.01 are mostly noise.
- **Edge weighting is the consistent lever:** +0.0057 on the mean, and the seed spread drops from 0.0075 to 0.0002.
- **Plan (`runs/sophelio_ens_runner2.sh`):**
  - EMA arms first.
  - Then edge-weighted seeds of the top stage-1 configs: t5 (seeds 3–6), t19 (seeds 1–2), t7 (seed 1).
  - Then greedy selection over every saved trunk.
- NAS stage 1 finished with best trial 5 (0.9864, tied with trial 19). The coupling stage has started.

## Sep 23 12:00 — TDC: MapLight+GNN reproduced via a PyTorch GIN port (`paper/tdc/gin.py`)

- DGL cannot sit beside torch 2.14 here: the install would replace nvidia-cusparselt, and its build is CPU-only. dgllife's pretrained `gin_supervised_masking` was therefore ported to plain PyTorch on GPU, compatible with the released weights:
  - pretrain atom/bond featurizers, with self-loops as bond type 4;
  - 5 GIN layers of width 300, mean-pooled.
- **Check:** MapLight+GIN on CYP2C9, seed 1, gives AUPRC 0.8600. Published MapLight+GNN is 0.859 ± 0.001, with the same 2,863 features.
- **Queued (CPU CatBoost):** MapLight+GIN and MapLight+GIN+CheMeleon on all 22 endpoints.
- **Stack:** now chooses the MapLight feature set per endpoint on VALID, from 4 sets. This amendment is logged in the stack.py docstring and was made before any blended TEST score existed.
- **MapLight + CheMeleon features is mixed:** caco2 0.276 (better) and HIA 0.988 (better); bioavailability, lipophilicity, Pgp and BBB worse. On CYP2C9 it gives only 0.798, so CheMeleon does not supply the GIN's CYP signal.

## Sep 23 12:36 — RealPDE: low-mode correctors beat the released U-Net (TEST, 5,140 windows, RealPDEBench metric, 3 seeds)

| Arm (frozen U-Net-finetune + corrector) | VALID rel-L2 | TEST rel-L2 (all) | in-dist | out-dist | Core params |
|---|---|---|---|---|---|
| none (released trunk) | 0.00947 | 0.00948 | 0.00866 | 0.00997 | 0 |
| **lin (linear map on the low-mode block)** | **0.00936** | **0.00937 ± 0.000001** | 0.00857 | 0.00984 | 0 |
| mlp30 (matched MLP core) | 0.00938 | 0.00939 ± 0.000001 | 0.00858 | 0.00986 | 69,927 |
| mr30ryd (3 × 10-atom Rydberg register) | 0.00938 | 0.00939 ± 0.000001 | 0.00858 | 0.00986 | 70,196 |

- **Below SoTA:** every corrector beats the released U-Net (0.00948), which is itself the best published foundation-model number on this data. The best is the classical linear corrector, 1.2 % lower rel-L2, with identical results across seeds.
- **Register vs MLP is a null:** 0.00939 each, on the same insertion and with matched parameters. On this row the gain is classical, and the register is exactly at parity with its matched classical control.
- **Queued (`runs/realpde_stack_runner.sh`):** corrected U-Net predictions (lin, seed 1, pre-declared), mixed with U-Net-real and FNO. The ensemble weights are chosen on VALID and TEST is scored once.

## Sep 23 13:45 — Sophelio correction and stage-2 progress

**Edge weighting over 6 seeds** (VALID, 40 shots): 0.9858 / 0.9856 / 0.9856 / 0.9843 / 0.9792 / 0.9852, mean **0.9843**.
- Base over 3 seeds: 0.9864 / 0.9762 / 0.9837, mean 0.9821. The gain is +0.002, not the +0.006 the first two seeds suggested.
- Edge weighting also has bad seeds (seed 5).
- EMA of the weights (decay 0.999) is neutral on both seeds tested.

**Stage-2 coupling search** (frozen trunk t5 at 0.9864; 40 VALID shots; single seed; no matched control yet):
- The two best are 20-atom SRR registers: ring, 3 layers, 0.9871; chain, 4 layers, 0.9876 (+0.0012).
- Latent-register couplings sit level with the trunk (0.9864–0.9866).

**Next (`runs/sophelio_confirm_runner.sh`):**
- `paper/nas/sophelio_confirm.py` re-runs the top 3 × {register, matched MLP `-mlp`} × 3 seeds, paired by seed, and saves every model.
- Then greedy ensemble selection twice: classical pool plus register-coupled members, and classical pool plus MLP-coupled members. That keeps the final classical+quantum ensemble comparison matched.

## Sep 23 15:00 — Sophelio stage 2 confirmed (`paper/nas/sophelio_confirm.py`; VALID, 40 shots; frozen trunk t5 at 0.9864)

| Coupling | Register (3 seeds) | Matched MLP (3 seeds) | Δ | Paired p |
|---|---|---|---|---|
| Trial 22: SRR, 10-atom ring (2 × 5), 3 layers, lr 2.4e-3 | 0.9878 | 0.9876 | +0.0002 | 0.43 |
| Trial 20: same register, lr 1.6e-3 | 0.9878 | 0.9868 | +0.0010 | 0.030 (0.09 after Bonferroni ×3) |
| Trial 13: SRR, 5 atoms, 3 layers | 0.9879 | 0.9876 | +0.0002 | 0.41 |

**Reading:**
- **The robust gain is classical.** SRR detail-band routing (the coupling writes only PCA components k ≥ 8) lifts the frozen trunk by +0.0012–0.0015 with either core.
- **The register ties its matched MLP.** Its mean Δ is ≥ 0 on every config, but nothing survives multiple-comparison correction.
- **Trial 20 is confounded.** Its nominal p = 0.03 comes from a learning rate the search tuned for the register arm; at trial 22's rate the MLP twin recovers (0.9876).
- **No register advantage is claimed.** A fair claim would need the MLP twin tuned on the same budget.

All 18 coupled models are saved (`coupled_c*-{reg,mlp}_seed*.pt`) for the two matched ensemble selections.

## Sep 23 15:20 — TDC: MapLight+GNN reproduced on 20/22 endpoints (GIN port, CPU CatBoost)

**Within 2 SD (or 0.005) of the published MapLight+GNN on 20 of 22.**
- The misses are bioavailability (the TDC page is a copy of Pgp) and CYP2D6-substrate (0.715 vs 0.720).
- At or just above the current leader: Pgp 0.939, CYP2C9 0.860, half-life 0.577 (CFA 0.576), microsomal clearance 0.634 and hERG 0.883. These are the same model, so they count as ties, not new results.

**Runner deadlock.** The first `all` run hung inside CatBoost: every thread in futex wait, 0 CPU for 1 h 40 min. The runner now uses one process per endpoint under a 20-minute timeout with one retry (`runs/tdc_stack_runner.sh` v3). In a fresh process the same endpoint took 91 s.

## Sep 23 16:45 — RealPDE headline: 0.00931 TEST rel-L2 vs 0.00948 for the released U-Net (−1.8 %)

| Entry (weights chosen on VALID; TEST scored once) | VALID | TEST all | in-dist | out-dist |
|---|---|---|---|---|
| Released U-Net-finetune (anchor) | 0.00947 | 0.00948 | 0.00866 | 0.00997 |
| + linear low-mode corrector (3 seeds) | 0.00936 | 0.00937 | 0.00857 | 0.00984 |
| 0.8 U-Net + 0.2 U-Net-real (FNO weight 0) | 0.00940 | 0.00941 | 0.00859 | 0.00991 |
| **0.8 corrected U-Net + 0.2 U-Net-real (FNO weight 0)** | **0.00930** | **0.00931** | **0.00850** | **0.00977** |

- This is a classical lift: a corrector plus a labelled ensemble. Register vs MLP on this row is a null (0.00939 each).
- **Bug found and fixed.** `arms.py --save-pred` wrote to `pred_seen_*` because the TEST-subset loop shadowed `tag`. The first 3-way ensemble therefore mixed zero-filled placeholders; `ensemble.py`'s `open_pred` creates missing files as zeros. The files were renamed, the loop variable fixed, and the ensemble rerun.

## Sep 23 17:50 — RealPDE FNO correctors; Sophelio classical ensemble at full-VALID 0.9912

**RealPDE FNO-finetune + low-mode corrector** (TEST rel-L2, 3 seeds each):

| Arm | TEST rel-L2 |
|---|---|
| Trunk | 0.01413 |
| lin | 0.01292 |
| mlp30 | 0.01267 |
| mr30ryd | 0.01267 |

- The corrector cuts FNO error by 10 %, and here capacity helps (MLP beats lin).
- The register is again exactly at its matched MLP, now on two trunks (U-Net and FNO).

**Sophelio greedy ensemble** (100-shot selection set, then all VALID shots):
- Members: t19-ew2 s2, t19-ew2 s1, t5-ew2 s3, t7-ew2 s1, t5-edge-ew2 s2, t5, t19.
- **Full VALID S 0.9912** (σ = 0), up from 0.9880: consistency 0.967, ψ R² 0.99963, qb 0.9964, D_LCFS 0.0148.
- The leader is 0.9932, so the gap is 0.002.
- Edge-weighted trunks hold 6 of the top 7 single scores on the 100-shot set, which is independent of the 40-shot NAS set.

**TEST rule** (`paper/sophelio/score_test.py`, committed before any TEST score):
- The classical, +register-coupled and +MLP-coupled selections are each scored once on all TEST shots.
- The entry is the one with the best full-VALID S.

## Sep 23 21:40 — Allocation-strategy study registered and running (doc 33)

User thesis: the register should handle the analog part that the classical world model gets wrong, and the trunk keeps the rest. The job is to find the best input-dependent allocation strategy across 10+ benchmarks.

**Registered before any result:** `33-allocation-strategy-study.md`.
- **Setup:** a frozen SoTA trunk plus a routed co-processor on the trunk's residual, in an orthonormal Haar basis.
- **Strategies:** pooled / static / proxy / learned / gain / random, plus an oracle diagnostic.
- **Cores:** lin / mlp / ryd, plus gutz / ryd2d / mlpX4 controls.
- **Budget:** K = 30 atoms, with a K sweep over {10, 100, 300}.
- **Metric:** exact from coefficient energies.
- **Code:** `paper/alloc/`; runners `runs/alloc_{cache_runner,cache_runner2,grid_runner}.sh`.

**Lean** (`formal/QuotaScaling/Utilization.lean`, standard axioms only):
- `gain_le_ceiling`: any correction on a component set removes at most the error energy there, so captured ≤ U for every core.
- `ceiling_mono`: nested allocations have nested ceilings.
- `pooled_gain_le_dc`: a broadcast correction gains at most the DC energy, whatever core or number of atoms produced it.

**Ceilings at K = 30 (TEST share of weighted error energy):**

| Row | DC | Oracle | Static | Proxy | Random |
|---|---|---|---|---|---|
| TRL2D | 0.056 | 0.445 | 0.238 | 0.172 | 0.0006 |
| RealPDE (U-Net) | 0.0067 | 0.054 | 0.017 | 0.008 | — |
| RealPDE (FNO) | 0.074 | 0.109 | 0.085 | 0.045 | — |
| Pelzer (scratch test) | 0.047 | 0.647 | 0.459 | 0.321 | — |

**v0 finding (archived in `runs/alloc/results_v0/`):**
- With tile-local features, tile co-processors captured ≈ 0 (TRL2D, RealPDE U-Net) whatever the core or learning rate.
- Pooled captured 0.0022 of its 0.0067 DC ceiling on RealPDE.
- **Amendment 1** (documented in doc 33): tile co-processors also get the pooled global context, for every arm. The grid was restarted from scratch.

**Sep 23 22:40: TRL2D and RealPDE (U-Net) phase A complete (v1, 3 seeds, K = 30)**

TRL2D:
- **Tile strategies capture 0.** This holds for every core, including the oracle allocation (ceiling U = 0.445).
- **Capacity doesn't help:** a 64× MLP (20k parameters, tagged `_cap`) also captures 0 on static tiles and −0.28 % on oracle tiles.
- **Pooled:** +0.06 % (captured 0.0014 of the 0.056 DC ceiling).
- **Leak test (pipeline check):** with the true residual added as a feature, the co-processor captures 0.4434 of U 0.4448 and VRMSE falls from 0.1936 to 0.1497 (−23 %). The ceiling is reachable and the machinery is correct. The fine-scale residual of the released CNextU-net is simply unpredictable from causal features.

RealPDE (U-Net):
- **Pooled:** +0.12 % (captured 0.0022 of the 0.0067 DC ceiling, about a third).
- **Static / learned / gain:** +0.01–0.02 % (captured 0.0004 of 0.009–0.017).
- **Proxy and random:** 0.

The register equals the MLP in every cell of both rows.

## Sep 23 22:07 — Sophelio single TEST scoring (rule committed before any TEST score)

| Selection (373 TEST shots) | Full-VALID S | TEST S |
|---|---|---|
| classical, 7 members | 0.9912 | 0.9910 |
| +register coupling, 4 members | 0.9910 | 0.9905 |
| **+MLP coupling, 7 members (entry: best full-VALID S)** | **0.9913** | **0.9910** |

- The TEST entry is **S = 0.9910**. The leader is 0.9932, so this is below SoTA by 0.0022.
- The register-coupled selection scores lowest on TEST.
- The paper's `\SophTEST` now reads 0.9910 (appendix SoTA table).

**Sep 23 22:57: first rows with signal, K sweep, and sub-study B (analog vs digital probe)**

**Pelzer (1HP_NN U-Net, frozen; study metric VRMSE; 3 seeds), TEST gain (lin / mlp / ryd):**

| Strategy | Ceiling U | Gain |
|---|---|---|
| static | 0.46 | +8.4 / +9.4 / +9.2 % |
| learned | 0.46 | +6.9 / +7.9 / +8.5 % |
| gain | 0.35 | +6.7 / +7.0 / +7.2 % |
| proxy | 0.32 | +2.5 / +3.3 / +2.7 % |
| pooled | 0.047 | +1.3 / +1.4 / +1.45 % |
| random | — | ≈ 0 |
| oracle (1 seed) | — | +15 % |

- Register vs MLP: all p > 0.5.
- **K sweep:**
  - pooled captured share is flat from K = 10 to 300 (0.015–0.020, ceiling 0.047);
  - proxy captured share rises 0.02–0.03 → 0.05–0.06 → 0.11–0.14 at K = 10 → 30 → 100.
  - This is utilization over qubits, in practice.

**RealPDE (FNO), seed 1:** pooled captures 0.072 of 0.074 (+3.3 %), static 0.081 of 0.085 (+3.7 %). The error is systematic, so the gain is close to the ceiling.

**Sub-study B probe (registered; TEST R², 3 seeds):**

| ω | Register | MLP (75 p) | mlp+ (93 p) | 4× MLP |
|---|---|---|---|---|
| 0.5 | 0.999 | 0.9997 | — | — |
| 1 | 0.998 | 0.9995 | — | — |
| 2 | 0.989–0.991 (wins 3/3 vs MLP, 2/3 vs mlp+) | 0.973–0.983 | — | — |
| 4 | 0.87–0.88 (loses) | 0.94–0.97 | — | 0.98–0.995 |
| 8 | pending | ≈ 0 | ≈ 0 | ≈ 0 |

- **Digital:** MLP 0.32–0.94 (unstable).
- **Parity:** every core ≈ 0.07. The target is not identifiable for permutation-symmetric group cores.
- **B2 (exploratory, post hoc):** `rydlin` (linear angle encoding) vs `mlpsin` (sine-MLP) at ω = 4, 8, 16, launched.

**Sep 23 23:45: rayleigh_benard (uniform windows, amendment 3), MHD_64, FNO seed 1, and the finished probe**

**rayleigh_benard, seed 1, TEST (1,200 uniform windows; trunk mean 0.640, full split 0.668):**

| Strategy (30 atoms) | lin | mlp | ryd | Captured / U |
|---|---|---|---|---|
| pooled | +2.6 % | +6.2 % | +6.7 % | 0.072 of 0.083 |
| static | +28.4 % | +33.0 % | +33.3 % | 0.138 of 0.182 |
| proxy | +29.1 % | +36.7 % | — | — |

Decomposition for static with the register:
- Early-transient windows (t0 < 10, 50 windows): 7.11 → 4.75.
- Remaining windows: 0.359 → 0.239.
- Median: 0.0369 → 0.0353. 94 % of windows improve.
- **Reading:** the mean-VRMSE gain concentrates in the convection-onset transient the released CNextU-net gets badly wrong. Report the mean (the benchmark's metric) and the median together.

**MHD_64:** pooled captures its whole DC ceiling (0.0020) for +0.11 %. Tile ceilings are tiny (2.5 % oracle at K = 30, over 262k tiles).

**RealPDE (FNO), seed 1:** every causal strategy lands at +3.8 %, capturing 0.084 of U ≈ 0.09 (92 % efficiency). The cores are identical.

**Probe, final (registered part):**
- Analog-edge rule NOT met: the register beats the matched MLP on 3/3 seeds only at ω = 2.
- **Digital (threshold) target:** register R² 0.067 and mean-field 0.23, against 0.72 (MLP), 0.92 (mlp+) and 0.95 (4× MLP). The register's function class is smooth and cannot draw thresholds, which supports "digital → classical", not "analog → quantum wins".
- **B2 (exploratory):** the linear-encoding register equals the tanh register at ω = 4 (0.874). The sine-MLP scores 0.988 there. Everything fails at ω ≥ 8.

**Sep 23 23:55: SRR pilot (TRL2D, fine-tuned, seed 1337, 5k steps, VALID) finished.**

| Arm | VALID VRMSE |
|---|---|
| vanilla | 0.189819 |
| srr-ryd | 0.189811 |
| srr-lin | 0.189885 |
| srr-mlp | 0.189961 |

- **Pre-registered rule** (doc 32 addendum): run full Stage B only if srr-ryd beats vanilla by more than 0.5 % and is not worse than srr-mlp.
- **Outcome:** srr-ryd beats vanilla by 0.004 %. The rule fails, so the SRR line on TRL2D stops and is reported as a null.
- This is consistent with the frozen-trunk study, where TRL2D's fine-scale residual is unpredictable even under oracle placement.

### Sep 24 21:30: persistence audit, ADR-C v1 results, v2 and gate registered

**ADR-C v1 (3 seeds; mean TEST relative gain):**

| Row | Best arms | Register vs classical |
|---|---|---|
| pelzer | register 18.3 %, MLP 18.9 %, gutz 20.1 % | GNN 11.3 %; static 6.1 %; register seed 3 falls to 9 % |
| rayleigh_benard | static_lin 55.8 % | register 55.2 %, GNN 55.6 % |
| shear_flow | static_lin 90.3 % | register 88.3 %, GNN 88.0 % |

- The seed-1 register lead does not hold. The gains come from routing and components, not from the register.

**Why RB and shear gain so much.** The component features let a linear head move the trunk toward persistence, and full-TEST persistence beats both published SoTA and our trunks:

| Row | Persistence (full TEST) | Published SoTA | Our trunks | Per-window oracle min(trunk, persistence) |
|---|---|---|---|---|
| rayleigh_benard | **0.2024** | TFNO 0.6566 | 0.356–0.571 | 0.044–0.048 |
| shear_flow | **0.0646** | CNextU-net 0.8080 | 0.224–0.345 | 0.036–0.039 |

- The trunks beat persistence on 84 % (RB) and 54–63 % (shear) of windows, but lose the mean in a heavy tail.
- On TRL2D, persistence scores 0.576, so the models win there.
- Evidence: `runs/persistence/`.

**Registered next (before any full run):**
- **Doc 35 §8 (ADR-C v2).** Frequency-localized components: band-wise POD within dyadic FFT shells. The per-band chain captured_b ≤ span_b ≤ full_b is recorded for every run.
- **Doc 36 (band-gated anchor plus exact per-window gate).** The gate is classical: a digital router between persistence and the trunk, per field and band. It is exact from cached band statistics and shared by all arms, with arm `none` isolating it.
- **Smoke (48 windows):** on RB the trunk goes 0.339 → blend 0.150 → gate 0.050. On shear the blend takes 0.370 → 0.056.
- **Fitted blend coefficients:** the fine-tuned trunks' velocity fields are worse than persistence (α ≈ 0). Their high-k bands are down-weighted (α 0.74 → 0.12 at the top octave).

**Infrastructure fixes:**
- `runs/capped.sh` unit name gains `-$$`. Same-second launches collided and crashed 5 RB seed-3 arms; they have been rerun.
- A stale grid runner waiting for the excluded planetswe row was killed.
- The fine-tuned-trunk runner now waits on a PID instead of a self-matching pgrep.

### Sep 24 23:55: gate leak found and corrected (doc 36 §8)

- **The leak.** The first per-window gate normalized its input features by the target frame's variance (non-causal).
- **Void numbers.** Every gated number reported between 22:30 and 23:50 is void: shear 0.0273 / 0.0223, RB 0.0422 / 0.0413, TRL2D 0.1923 / 0.1921. 116 JSONs are archived in `runs/alloc/adrc_results_leaky_gate/`.
- **The fix.** The features are now normalized by var(x_t), and the runs are relaunched (`runs/alloc_gate2_runner.sh`, `runs/alloc_gate2_caches.sh`).
- **Numbers that stand.** The TRAIN-fitted band blend uses no per-window input: shear 0.2297 → 0.0451 (full TEST); RB 0.432 → 0.142 (stratified TEST); TRL2D 0.1934 → 0.1926.

**Also:**
- **Stratified iteration (doc 36 §7).** k windows per trajectory × 8 time bins. It is validated against the full-TEST CSVs: RB within ~2 %, where the old uniform 1,200-window sample was 16 % off.
- **Utilization-supervised router (doc 35 §9).** It moved the RB qubit slots from the top octave (92 %) to the band holding the reachable error (80 % in b0). Its numbers are under rerun because they came from the leaky gate.

### Sep 25 01:20: second agent, causal-gate reruns, route-supervised router, end-to-end fine-tuning lanes

**Second agent.** A Gemini 3.8 Flash session (tmux `codex`) is connected through `.bridge/`, a file mailbox that is git-excluded.
- Its independent leakage audit of adrc_cache.py, adrc.py and finetune_gated.py found no further leaks.
- Its strategy critique was accepted in part:
  - Stop frozen-trunk iteration on RB and shear.
  - Trim the 8-arm grids to 3 cores plus the gate plus a linear reference.
  - The fix for the Well trunks is metric-aligned training.
- Rejected parts: an MHD band gate "< 0.160" (measured: 0.1735), and a TokaMark "persistence head" (it would feed previous EFIT targets as inputs to a reconstruction task, a leak).
- Its band-constrained routing patch was reviewed, applied and registered (doc 35 §10).
- It now runs the Sophelio post-processing sweep, VALID only. Stages 1–3 give no gain over the entry (0.9913).

**Causal gate reruns** (partial seeds; `runs/alloc/adrc_summary.md`):

| Row | Gate alone | Route-supervised co-processors |
|---|---|---|
| shear, full TEST | 0.0272 | register 0.0202, GNN 0.0205, MLP 0.0199 |
| RB, stratified TEST | 0.0420 | register 0.0413, GNN 0.0418, MLP 0.0415 |

The route-supervised router lowers every core. The register ties.

**End-to-end fine-tuning** (doc 37, `paper/alloc/finetune_gated.py`):
- Metric-aligned VRMSE loss plus band gate, with exact gate pre-training on the frozen trunk.
- First RB VALID points (joint schedule): register-neck 0.1427 → 0.1008, vanilla 0.1441 → 0.0898.
- **Kills:** memguard killed one cell at 13 GB MemAvailable, so the low-value lanes were paused.
- **Queued:**
  - shear and TRL2D, three trunks each, with exact pre-training;
  - MHD after the RB lane;
  - the TokaMark Scratch+FT-Base equal-weight ensemble, registered before any Scratch TEST score.

### Sep 25 07:50 — fine-tuning lanes and the TokaMind-Scratch reproduction

**Two-stage gated fine-tuning** (doc 37): exact gate pre-training on the frozen trunk, then joint training at lr 2e-5. Seed 1, stratified TEST:

| Row | Register neck | Vanilla | MLP neck |
|---|---|---|---|
| RB | 0.0425 | 0.0441 | 0.0438 |
| shear | 0.0271 | 0.0298 | requeued (memguard kill) |
| TRL2D | 0.1921 | 0.1921 | 0.1922 (published 0.1956) |
| MHD_64 | 0.1728 | 0.1727 | — |

- Gate pre-training captures almost all of the gain; the joint stage adds ≤ 0.001.
- Without pre-training, the joint RB schedule reaches only 0.066–0.085. MSE loss gives 0.1415 and no gate gives 0.4331. The gate plus a metric-aligned loss is what matters.
- The trunks tie at the matched slot; the register is not credited.

**TokaMark 1-1, TokaMind-Scratch recipe** (our run, seed 1337):
- 0.1379, reproducing the published 0.140.
- The registered E4 (Scratch plus three FT-Base vanilla seeds, equal weight) gives 0.1400, worse than Scratch alone. We report it because it was registered.
- The register and MLP Scratch arms are still training.
- Paper: the app:sota TokaMark row is updated.

**Registered Sep 25 07:56, before any numbers: TokaMark β diagnostic (TEST-derived, not fitted).**
- **Data:** a ground-truth dump of the Scratch vanilla eval (`TOKAMIND_DUMP_TRUTH`).
- **Measures, per β channel:**
  - R = Σ_bins n_b μ_b² / Σ e², over bins of shot-length quartile × relative time in shot (deciles);
  - the jump-variance ratio Var(Δy_true) / Var(Δy_pred) along each shot.
- **Rule:**
  - Build an Ip-conditioned co-processor (register/MLP/GNN at the same slot, fit on TRAIN) only if R ≥ 0.25 in the short-shot quartile AND the jump ratio ≤ 2.
  - Otherwise, treat it as EFIT label noise at termination, keep Scratch as is, and report the diagnostic.
- **Result, 08:00:** R ≤ 0.006 on every β channel and every shot-length quartile, and the jump ratio is 1.5–9.6. This is label noise. Scratch stays at 0.1379, no co-processor is built, and the diagnostic is in app:sota (`runs/tokamark/results/beta_diag.json`).

### Sep 25 09:40 — the E-ft ensemble (registered) on RB
- **Stratified VALID:** E-ft 0.0400 vs the best single trunk 0.0422 (register neck); E-ft is selected.
- **Full TEST, one run (34,300 windows):** **0.0403**. For comparison: persistence 0.2024, published TFNO 0.6566, trunk 0.43.
- **Residual correlations, descriptive:** 0.59–0.64 and symmetric. This is ordinary variance reduction, so there is no register credit.
- **Shear MLP-neck two-stage run:** stratified TEST 0.0280 (register neck 0.0271, vanilla 0.0298).
- **MHD_64 E-ft:** VALID tie (0.17262–0.17269; residual correlation 0.999), so the vanilla single trunk is selected. Full TEST 0.1725 (released ckpt 0.5447; published 0.1633 is not reproducible).
- **Shear E-ft:** VALID 0.0256 vs 0.0274 for the best single model; E-ft is selected. Full TEST 0.0252 (n=21,952). The frozen-trunk gate + route-supervised co-processor pipeline is still lower at ~0.020 (full TEST, cores tie).
- **TRL2D E-ft:** VALID 0.19171 vs 0.19179 for the best single model; residual correlations 0.997–0.999. Full TEST **0.1909** (n=873), below the published 0.1956 and the earlier plain 3-seed ensemble (0.1929). The app:sota row is updated.
- **TokaMind-Scratch matched necks (seed 1337):** vanilla 0.1379, MLP neck 0.1391, register neck 0.1404. Neither neck helps; the register is slightly worse. This goes in app:sota. The registered E4 ensembles for mr30ryd and mlp30 are running (tokamind_scratch_ens.sh).
- **Registered E4 ensembles (Scratch + 3 FT-Base, equal weight):** vanilla 0.14000, register neck 0.13997, MLP neck 0.14016. They tie. TokaMark is closed: Scratch vanilla reproduces SoTA at 0.1379, and neither neck helps.

### Sep 25 11:15 — finite-shot readout (doc 35 §11, registered before any number)
- **Setup:** the route-supervised register, retrained as in §9 (seeds 1–3), evaluated with binomial readout.
- **Mean ΔVRMSE at N = 100 / 1,000 / 10,000:**
  - shear: 7.6e-5 / 8e-6 / 1e-6 (threshold 5.5e-5, half the register–MLP gap);
  - RB: 3.2e-4 / 3.4e-5 / 4e-6 (threshold 1.8e-4).
- **Outcome:** shot-robust from N = 1,000 on both rows, not at N = 100.
- **Cost:** 3N shots per input, or up to 9N when tethers make the groups sequential (2 of 3 shear seeds).
- **Paper:** the paragraph is in the app:dynamic hardware section, stating the shot counts only; no repetition rate is claimed.
- **Correction:** my first pass applied max(0.5·gap, 1e-4) on shear. The registered wording is "0.5·gap, or 1e-4 if the gap is < 1e-4", so shear fails at N = 100. Fixed before reporting.

### Sep 25 15:30 — transform diagnosis → z-domain band filter (doc 38 addendum)
**Failure analysis** (`paper/alloc/failure_cases.py`, `spectral_diag.py`, `runs/failure/`):
- TRL2D and RB sit at the per-octave Wiener floor, and their poles match the truth.
- The high octaves have a change pole of ≈ −½, the signature of frame-to-frame white content.
- TRL2D error is on the mixing interface: 98% of the error energy lies in 5% of the Haar coefficients.
- RB's worst windows are the convection-onset wall layer.
- Shear's model misses the −½ pole in its high octaves.

**ZBF** (two-tap per octave, fitted on TRAIN; see `runs/failure/*_zbf.json`):
- TRL2D: VALID +0.12%, dropped.
- Shear: VALID +16.0%, kept. Stratified TEST 0.02515 → **0.02019** (n=1,792; 72% of windows improved).
- Shear coefficients: pressure b0 α 0.31, β 0.72 (a continued pressure trend); top pressure octaves α ≈ 0, β ≈ −½.

**TRL2D E-ft vs the released checkpoint, per trajectory:** 6/9 lower, Wilcoxon p = 0.049, bootstrap CI [−0.0046, −0.0006].

**Replication (doc 38):** stopped for time before any evaluation.

**Future-work appendix:** drafted by Gemini; overclaims removed; transform-based long-horizon metrics added.
- **ZBF shear by target-variance quintile** (every 7th test-sample window, n=256): quintiles 0–2 improve by +29/+29/+33%; quintiles 3–4 worsen by −15/−30%. It's a metric-aligned correction for quiet windows (Gemini's point 2 was right). A divergence check was attempted but the operator couldn't be validated on the truth (ratio 1.0), so the paper says incompressibility is untested.

### Sep 25 15:40 — calibration noise (doc 35 §12, registered)
- **Setup:** route-supervised register, 3 seeds × 3 repeats.
- **Jitter (1%, 5% of the spacing) and 1% detuning:** ΔVRMSE ≤ 7e-6 on both rows; robust.
- **5% detuning:** RB 1.4e-4, robust (threshold 1.8e-4); shear 7.3e-5, NOT robust (threshold 5.5e-5).
- **Reading:** placement errors are tolerated; detuning must be calibrated to ~1%.
- **Output:** `runs/alloc/noise_summary.json`. The paper's hardware appendix is updated.

### Sep 25 16:25 — hybrid scaling with the qubit budget (doc 35 §13, registered)
**Setup:** classical part fixed (trunk + gate + classical head). The register is fully input-dependent: x, y, z, detunings, routing and, new, the tether strengths. K = 0/5/10/20/30.

**Hybrid VRMSE relative to the classical part, K = 30:**
- shear: register 0.857, MLP 0.885
- RB: ~0.96 for both
- TRL2D and MHD: 1.000

**Energy-space view:**
- η = qubit capture / U(K): shear 0.42 → 0.72, RB 0.50 → 0.66.
- The coverage U(K) is computed from the data and fits the closed form U∞[1 − (1 + K/K0)^(1−a)] within 0.015 (0.006 or less except TRL2D), with a = 1.8–2.6.
- U saturates near 200 qubits because the POD basis spans only that share of the residual. The limit is the basis, not the qubit count.

**Register vs MLP on shear (exploratory):** better in 11/12 cells, +2.6%; K=30 p = 0.016 uncorrected. RB ties. This is not the §6 test (no GNN arm), so no claim.

**Gemini's research critique, tested against the files:**
- "Subband coding gain" is refuted: the classical head's capture falls as K grows.
- "Kolmogorov cutoff" is refuted: the saturation is the basis span.
- Its closed-form fit is adopted.

**Where it goes:** the paper gets `app:hscale` plus a figure, and one main-text sentence in §5.5.

## Sep 25 ~20:00: RouteFinder routed-register crash diagnosed (environment bug, not the register)
**Symptom:** route10/20/30 register arms died at training step 3 with NaN sampling probabilities. route5 and every MLP arm finished.
**Cause:**
- In RouteFinder's multi-depot environment, when a remaining customer cannot be served from any depot, each depot deadlock marks the current depot unavailable. Once all depots are marked, the action mask is all False (3 of 12,800 rows).
- Reproduced with the register's outputs and gradients finite throughout.
- route10/20/30 share one CPU RNG draw at init (one 10-atom core), so they trained on the same generated instances, one of them unservable.
**Fix (doc 35 §16 amendment, commit c65d5c6):**
- A dead-end guard that acts only where the unguarded code crashes, so finished runs are unchanged. Guarded rows are counted per run.
- Common random numbers: every arm now trains on vanilla's instances.
- Queue v2 (4 lanes) runs the full protocol first. The 6 earlier non-vanilla pilots trained on other instances and stay exploratory.
**Guard check:** on the instances route10 originally trained on (the bad stream), the fit that crashed at step 3 now completes, with 100 row-steps guarded. On vanilla's instances the count is 0.
**Pilot numbers so far (1 epoch × 2k, seed 1, mean gap over 16 md variants, lower is better; not common random numbers):** vanilla 24.48, pooled q30 23.82 / q30-mlp 24.01, route30-mlp 23.45, route10-mlp 24.41, route20-mlp 24.19, route5 24.15 / route5-mlp 25.77.

## Sep 25 ~21:00: WeatherNext 3 adapted: noise-conditioned input-dependent configuration (doc 35 §17)
**User request:** adapt WeatherNext 3 (arXiv 2609.03582) to the classical+qubit system, with the focus on dynamic, input-dependent qubit configuration.
**Checked against the PDF:**
- The LinkedIn summary's numbers are not in the paper: T850 1.32 K, SSR 0.98, "#1 OWB", k^−5/3, "<60 s", and "bypasses 4D-Var" (the model ingests satellite data *in addition to* analyses).
- What is in the paper:
  - FGN members come from one low-dimensional noise vector in the conditional normalisation layers;
  - training uses the fair CRPS with 2 members;
  - epistemic dropout shares one mask across members in training;
  - the continuous query head works as described;
  - §4.6 notes that marginal CRPS "cheats" on covariance.
**Design (registered before runs):**
- ξ ∈ ℝ³² conditions only the co-processor's input-dependent configuration: core-input FiLM, the per-atom offset net and per-input tethers.
- Loss: the fair energy score, exact in the POD coefficient space, where a point forecast's energy score equals its VRMSE.
- Register vs matched MLP, K = 30, 4 rows × 3 seeds.
**Gemini:** agrees on propriety, on keeping noise out of the classical head, and that Born-rule shots can't be trained through. Its claim "SSR bounded by U(K)" is refuted by the smoke test (SSR 1.04 on shear).
**Found along the way:** the deterministic hybrid is not bit-reproducible at a fixed seed. Shear, MLP, seed 1 gives 0.0201, 0.0205 and 0.0206 on three runs (~3%), so single-seed gaps below that are noise.

## Sep 25 20:56: first RouteFinder pilots with common random numbers (1 epoch × 2k, seed 1)
- **30-qubit routed register:** 23.91% (vanilla 24.48%), better on 13/16 variants.
- **30-qubit routed MLP:** 25.05% (4/16).
- **Register vs MLP:** register better on 14/16 variants.
- **Checks:** zero-shot validation is identical across arms; 0 guarded dead ends.
- **Instance noise:** the MLP arm moved 23.45 → 25.05 when only the training instances changed, so pilot gaps of ~1–1.5 points are within instance noise. Exploratory; the full protocol (3 seeds) is the test.

## Sep 25 21:10–21:30: vanilla-trunk scaling registered; memory incident
- **User:** "do the qubits scaling on vanilla first as this one the paper can't go without."
  - §13's fixed classical part used register-necked mr30ryd trunks on shear, RB and TRL2D.
  - Doc 35 §19 reruns the full K ladder on vanilla trunks, seeds 1–3 on all four rows (shear's vanilla gate cache is being built).
  - It adds the full-protocol RouteFinder ladder K = 5/10/20 at seed 1.
  - All of this runs ahead of the WeatherNext arms (§17/§18).
- **Memory incident:**
  - After I stopped VS Code (user-approved), 35 GB was free. Starting the Well lanes, the cache build and the editor's reconnect then drove it to 13 GB.
  - Memguard v3 killed the vanilla full run, seed 1, at 76 min.
  - Fix: memguard v4 kills cheap restartable jobs first; vanilla seed 1 is re-queued first; the Well side runs one lane.
- **Gemini:** the Pinson & Tastu bib entry it proposed was fabricated. The real source is a DTU technical report (TR 2013-15); corrected.
- **21:47, CRN pilot route10:** 24.31% (11/16 variants better than vanilla).
  - Pilot ladder so far (register): K = 0 24.48, K = 10 24.31, K = 30 23.91; MLP K = 30 25.05.
  - Vanilla full seed 1 restarted at 21:47.

## Sep 25 23:10: first full-protocol RouteFinder result (EAL, multi-depot, 10 epochs × 10k, CRN)
- **route30 (routed register), seed 1:** mean test gap **8.59%** over the 16 md variants (2.84 h).
  - The guard fired on 600 decode steps in training, so vanilla's instance stream itself contains unservable dead ends. Without the guard the original code would crash; the Sep 23 "16-bit NaN" may have been this.
- **Not comparable to the 3.126% anchor:** that is RF-TE on the 16 *training* variants.
- **Published EAL multi-depot numbers** (RouteFinder Table 3, 10 epochs × 10k): gaps of about 11–19% on the MD mixed-backhaul variants. Our md per-variant gaps range 5.6–12.8%.
- **The matched comparison** is vanilla EAL on the same instances (running, ETA ~01:00) and route30-mlp (running).
- **CRN pilot route10-mlp:** 24.98% (2/16 better than vanilla).
  - Pilot ladder: vanilla 24.48; register K10 24.31 / K30 23.91; MLP K10 24.98 / K30 25.05.
  - The register is ahead of the MLP at both K in the pilot (one seed).

## Sep 26 00:05: vanilla-trunk hybrid scaling complete (doc 35 §19, 108 runs)
The fixed classical part is now purely classical: vanilla CNextU-net, band gate and classical head.

| Row | Register at 30 qubits (vs classical part) | MLP at 30 qubits | Register < MLP cells | U∞ |
|---|---|---|---|---|
| shear_flow | 0.877 (−12.3%) | 0.900 | 10/12 (K30 p = 0.30) | 0.416 |
| rayleigh_benard | 0.949 (−5.1%) | 0.954 | 11/12 (K30 3/3, p = 0.19) | 0.047 |
| TRL2D | 1.000 | 1.000 | tie | 0.555 |
| MHD_64 | 1.000 | 1.000 | tie | 0.019 |

- **Same shape as the mr30ryd trunk**, so the scaling law does not rest on a register-trained trunk.
- **Paper updated:**
  - abstract 14% → 12%; §5.5 now reads 12% / 5%;
  - app:hscale numbers, with the mr30ryd version kept as a sensitivity check;
  - both figures regenerated.
- Main text still ends on page 9.

## Sep 25 23:54: RouteFinder full protocol, seed 1: register ties its matched MLP
- **Results:** route30 8.587% vs route30-mlp 8.560% mean md gap. The register is better on 7/16 variants; both arms hit 600 guarded dead-end steps (same instances).
- **Pilot vs full budget:** the pilot-budget register lead (23.91 vs 25.05) does not survive the full budget. This is the paper's general finding (a matched classical core does the register's job at convergence), now on routing.
- **CRN pilot route20:** 24.36% (11/16 better than vanilla). Pilot register ladder: 24.48 (K0) / 24.31 / 24.36 / 23.91 (K30), non-monotone at one seed.
- **Next:** vanilla full seed 1 (ETA ~01:00) decides whether routing helps at all at the full budget.

## Sep 26 00:47: RouteFinder full protocol seed 1 complete: no routed gain over vanilla
- **Results:** vanilla 8.528%, route30-mlp 8.560% (8/16 variants better than vanilla), route30 register 8.587% (6/16). Zero-shot is identical across arms; the same 600 guarded dead-end steps in each.
- **Pilot vs full budget:** the pilot-budget lead of the register (23.91 vs 24.48 vanilla vs 25.05 MLP) does not survive the full budget.
- **Paper:** app:routing says so.
- **CRN pilots, 20 qubits:** register 24.36, MLP 23.61. Pilot register-vs-MLP flips sign across K (K10 −0.67, K20 +0.76, K30 −1.14), which is noise at one seed.
- **Queue reorder:** the §18 FGN pair (configuration diversity inside the existing best-of-8 selection, equal compute) now runs at seed 1 right after the K = 20/10 ladder, before seeds 2–3. It is the RouteFinder design with a mechanism for a gain at equal candidates.
- **Gemini audit (00:56), checked:**
  - Correct: the RouteFinder numbers.
  - Rejected:
    - "Tokamak Radiative Layer" (TRL2D is the turbulent radiative layer);
    - the plume/vortex narrative and "FGN rewards sharp plumes" (our spread lives in 30 POD modes);
    - MHD "monopoles diffused by the solver" (no solver in the loop);
    - RB "gate filters 95.3%" (U∞ = 0.047 is basis coverage of the post-gate residual);
    - "atoms only see (x, y)" (detunings come from embeddings that carry demand and time windows);
    - "add a residual gate" (the write-back is already zero-initialised).
  - Kept as future work: variance inflation for TRL2D's under-dispersed FGN ensemble (SSR 0.57); it needs retraining.

## Sep 26 01:20: debate with Gemini on the FGN claim; dressing control on TRL2D (registered)
**Control:** tuned Gaussian dressing of the deterministic hybrid (TRAIN σ, s on VALID, TEST once), 3 seeds each.

| TRL2D energy score | Register | MLP |
|---|---|---|
| Deterministic (= VRMSE) | 0.1923 | 0.1922 |
| Dressing, routed modes | 0.1754 | 0.1755 |
| Dressing, all modes | 0.1629 | 0.1634 |
| FGN | **0.1593** | **0.1612** |

- **Reading:** about 85–90% of the energy-score gain is available from tuned spread alone. FGN adds a small, consistent 1–2% beyond the all-mode dressing (3/3 seeds, both arms) and ~9% beyond same-support dressing.
- **Gemini's framing rejected:** "an artifact of the rule's variance requirement". A tuned dressing is a legitimate probabilistic forecast, and the score is proper.
- **Gemini's point adopted:** the VALID-σ variant (c) is now running.

## Sep 26 01:45: monitor debate rounds 1–2 with Gemini (standing rule, user)
- **Round 1:**
  - Accepted: the pilot falsifies the "open-route register penalty". The register beats the MLP on open and closed routes at the pilot budget (23.11/24.71 vs 24.18/25.92).
  - Rejected: re-routing MHD to b3/b4. Those octaves hold 88.5% of the error, but the POD span there is ≤ 0.001 and oracle U saturates at 0.019.
  - Adopted as §17 (d): FGN with a VALID-tuned spread scale τ, symmetric with the dressing control.
- **Qualitative, fig_spectral_fill (vanilla):** on shear, the added qubits mainly take b0 over from the classical head (qubits 0.14 → 0.40, head 0.35 → 0.13, combined 0.49 → 0.52); ~0.23 of b0 is left. On RB, b1–b4 (~60% of the error) are untouched.
- **Round 2 (01:44):**
  - Accepted: the head's b0 share falls because the code hands routed modes from the classical head to the co-processor (`adrc.py` scatter). It is architectural, not a finding.
  - Rejected: `--additive_neck` (the routed modes already carry the linear `W_q`); Gemini's cross-K bound argument (compares q30 with U(5)).
  - Adopted as §20: a measured K = 50 point on shear and RB (3 seeds, both arms), with the fan's prediction registered first (shear q50 ∈ [0.249, 0.380], RB ∈ [0.027, 0.040]).
  - The figure legend now says "qubit-routed modes": capture on routed modes includes their linear map `W_q`.
- **route5 CRN pilot:** 23.66% (14/16 better than vanilla). Pilot register ladder: 24.48 / 23.66 / 24.31 / 24.36 / 23.91 for K = 0/5/10/20/30, non-monotone at one seed.
- **Round 3:**
  - Accepted: Gemini's mechanism (pooled energy vs mean-of-sqrt per window) for why η and VRMSE rank register and MLP differently; VRMSE stays primary.
  - Rejected: its VRMSE formula (‖y−ŷ‖/‖y‖; the Well's is per-field √(MSE/var)) and its "Hamiltonian regularises / MLP overfits bursts" narrative.
- **Control fixes (02:15):**
  - The stationary-σ dressing breaks on shear (0.077–0.093 vs deterministic 0.021, grid edge).
  - Variant (e) scales the noise per window by the causal std of x_t. Scratch smoke on shear MLP s1: routed 0.0191, all-mode 0.0186, deterministic 0.0205, with interior scales.
  - Now the primary control (24 cells); the VALID-σ variant stopped at 10 files.
- **Early FGN + τ (d), TRL2D seed 1:** ES 0.1506 (register), 0.1583 (MLP) vs best dressing 0.163.
- **Round 4 (02:17):**
  - Accepted: std(x_t) as the causal amplitude normaliser for control (e); ‖p − x_t‖ would collapse the noise on unpredicted transitions. Framing adopted: the matched MLP with the same noise paths and tuned τ is the primary control; dressing is the lower bar.
  - Verified: Gemini's p = 0.055 for register < MLP on TRL2D FGN at τ = 1 (3/3 seeds: 0.1601/0.1589/0.1587 vs 0.1616/0.1618/0.1603).
  - Rejected: "the register's 4.8% lead under τ is a major finding". It was seed 1 only. Seed 2 gives 0.1518 vs 0.1542 (1.6%), paired p = 0.31 at 2 seeds. The "entangled manifold" explanation is unmeasured, and one row is not the §6 standard (GNN arm, majority of rows).

## Sep 26 02:40: route20 full (worse than vanilla), first K = 50 point (inside the fan), round 5
- **RouteFinder route20 full, seed 1:** 8.870%, worse than vanilla (8.528%) on all 16 variants (+0.15 to +0.51, mean +0.34). Zero-shot is identical, the same 600 guarded steps, so the difference arises in training.
  - Full-budget ladder at seed 1: K0 8.53 / K20 8.87 / K30 8.59 (MLP K30 8.56).
  - **Per-variant counts are not independent evidence:** one trained policy scores all 16 variants, so a single run-level fluctuation moves them together. This applies equally to the pilot's "13/16". Vanilla seed 2 (running) gives the first run-to-run spread at full budget.
  - app:routing updated.
- **§20, K = 50 on shear, seed 1:** register routed-mode capture q50 = 0.2547, inside the registered interval [0.249, 0.380] and just above the lower edge.
  - MLP q50 = 0.268. Hybrid VRMSE s1: register 0.0205, MLP 0.0204 (vs K30 s1 0.0211).
  - **Gemini's blind prediction** ([0.242, 0.247], "sub-fan saturation") is wrong on this seed.

## Sep 26 03:00: paper debate with Gemini (numbers, sentences, acceptance odds)
- **Applied** (length-neutral; still 9 pages, AI statement p10):
  - The "≤30 atoms" claim contradicted the 300-atom pooled sweep; it now reads "simulated exactly in groups of at most ten atoms".
  - The abstract's TD-rollout claim now says "relative to both continued classical controls, though not to the released checkpoint" (0/3 primary vs the anchor).
  - The heat-plume allocation sentence adds "and the matched MLP does as well".
  - Removed the repeated "Where … what …" parallelism in §5.4.
- **Rejected:**
  - Putting K = 50 into the abstract (one seed).
  - Gemini's rewritten sentences ("scale monotonically", "mathematically bounded"): less accurate and more machine-like.
  - Gemini's acceptance case (35–45%): it rested on a fabricated precedent ("QRC-Baselines, ICLR 2025 Spotlight, arXiv:2410.08985"; that ID is an unrelated knowledge-graph paper by Ni et al.), a mis-numbered Bowles et al. (real: arXiv 2403.07059), and PDE-Refiner given as ICLR 2024 (it is NeurIPS 2023).
- **Base rate (verified):** ICLR 2026 accepted 5,355/19,525 (27.4%); 2025 was 32.1%. Our estimate stays 15–25%.
- **Framing (03:05, user: simulation-only quantum papers are accepted):**
  - Verified precedent: QPA (Liu, Yang, Goan, Hsieh; ICLR 2025 poster; arXiv 2410.09846), which is simulation-based with a training-only quantum part.
  - The simulability sentence now says simulation is what makes matched controls possible (cites QPA).
  - The abstract's inference-inert register "needs no quantum hardware at deployment"; the abstract closes with "Registers are simulated exactly, so every claim faces a matched classical control; none is a claim of quantum advantage."
  - Gemini's A/B/C rewrites rejected: they call FGN "fractional noise dressing" (dressing is the control), present the FGN gain as register-specific (the MLP gets it too), say "deployed models run entirely on classical hardware" (false for routed co-processors), drop the anchor and registered-after scopes, and assert an unmeasured "training regularizer".

## Sep 26 03:15 — K=50 confirmed on shear (3/3 inside registered fan); control (e) beats FGN on TRL2D
- §20 on shear: the register's q50 was 0.255, 0.253 and 0.269, all inside [0.249, 0.380] (η = 0.68).
  - The MLP captured 0.268 and 0.263, so it does as well.
  - Paper: app:hscale results, the regenerated figure (K = 50), and one main-text clause in §5.5 ("a 50-qubit capture on shear_flow, predicted before it ran, fell in range on all three seeds").
- §17 (e), the window-scaled dressing (primary control), gives 0.147 on TRL2D and beats FGN + τ (0.152 register / 0.158 MLP).
  - On shear s1, FGN wins (0.017 vs 0.019).
  - app:fgn was rewritten honestly, and fgn_table has two new columns (tuned τ, dressed (e)).
- Gemini errors caught:
  - its K=50 claim that the register beats the MLP by 5% at 50 atoms (in fact the MLP captures more);
  - "FGN decisively beats dressing" (true only against the weak control (a));
  - a C6/R^6 "entangled manifold" explanation of the τ gap (unmeasured).
- Duplicate liu2025qpa bib entry removed (BibTeX error). Two VS Code server processes killed (user-authorized) so the RB K=50 runs could start.
- 03:40 Paper polish, reviewer read of pp. 1–9:
  - Figure 1 overlaps fixed.
  - Abstract gains "a 50-atom capture predicted before it ran held on three seeds". To pay for it, "The core does not matter at this scale:" became "At every allocation…", the abstract QPA cite was dropped (kept in §1), and "state-of-the-art" became "SoTA".
  - The TDC ADMET sentence moved to app:alloc.
  - The Figure 4 caption fit and the Fusion Equilibrium clause were tightened, with their caveats kept.
  - AI statement still starts p10.
- 03:40 Acceptance debate (user request).
  - Estimates: mine 25% (18–32); Gemini 30% (22–40). The base rate is ICLR 2026's 27.4%.
  - Verified from the ICLR 2027 guidelines and blog:
    - full paper and supplementary share one deadline, Sep 25 23:59 AoE (Sep 26 14:59 local);
    - abstracts may be revised if they stay close to the original;
    - missing reciprocal reviewer = desk reject; in 2026, no-reviewer papers were accepted at about half the rate.
  - Strategies agreed: primary area "applications to physical sciences" (user); reciprocal reviewer (user); anonymized supplementary zip plus a reproduce script; a pre-registration ledger appendix.
  - Rejected: Gemini's abstract swap, which would delete the tie sentence.
  - Citation sweep (paper/manuscript/citecheck.py): 37/37 arXiv titles match; 6 non-arXiv venues were web-verified (QuanONet ICML'25, HQF-Net CVPRW'26, QuFeX QST, ResQ ICCV'25, de Keijzer npj QI'25, QIML Sci Adv'26).
- 03:45 Monitor. User: "produce results that will improve our standing; 1 qubit ≠ 1 param".
  - Launched two registered runs: the doc 38 TD replication (seed 1340, resumed at step 2,308; rule unchanged) and the doc 35 §17 FGN power extension (seeds 4–6, TRL2D and shear; registered after seeing 3 seeds).
  - RF full queue and pilot queue stopped. The in-flight full runs finish; the §18 RF-FGN pilot, killed by memguard at 03:37 when TD started, is deferred.
  - RRNCO anchor reproduces (72.160 / 23.279 / 73.005 vs 72.145 / 23.282 / 73.010); now in app:routing.
  - RB K=50 s1: register q50 0.0286, inside [0.027, 0.040]; MLP 0.0279; hybrid 0.941.
  - Paper:
    - controls matched on trainable parameters (register 244, MLP 199 = 2 hidden units per atom; 18.6M trunk; readout 3 expectations per atom); Fig. 4 legend fixed;
    - anonymized supplement built (`paper/manuscript/build_supplement.sh` → iclr2027/supplementary.zip, 5.8 MB, leak check passes, reproduce script tested);
    - reproducibility statement points to it.
  - Gemini rejected a parameter-starved/reservoir experiment as goalpost-moving; I agree.
- 03:50 Monitor.
  - RF vanilla s2 killed at 1 h: the TD replication ran at about 0.3 steps/s under three RF lanes plus the adrc jobs, and needs near-exclusive GPU (about 5 h for three arms). route10 s1 and route20-mlp s1 are finishing.
  - Shear (e) is complete (3 seeds): dressed 0.0189 (register-det) / 0.0190 (MLP-det) vs FGN 0.0170 / 0.0179, so FGN wins on every seed and core; app:fgn updated. RB K=50 s1 files committed.
  - Octave reading, 30→50 qubits (register): the added capture is all in b0 (shear 0.222→0.248 of error, b0 holds 0.43; RB 0.020→0.025 of 0.052); b1–b4 are flat. To be added to app:hscale when RB s2–3 land.
  - Gemini: invented RF pilot numbers (23.475/23.518, 23.447/23.473) and mislabelled η as q50 (0.7317). It conceded, and agreed not to run Candidate A.
- 04:00 Monitor.
  - The TD replication ran at 0.32 steps/s against a historical 1.67 alone, so route10 s1 (RF full, ~3 h in) and the fgnt/dressr runner shells were stopped. Their in-flight jobs finish. Doc 35 scope note added.
  - RB K=50 s2: register q50 0.0276 (s1 0.0286), both inside [0.027, 0.040]; MLP 0.0267 / 0.0279; hybrid 0.935 / 0.941.
  - route20-mlp full: 8.71, so K=20 is worse than K=30 for both cores (register 8.87). The K=20 shift is not register-specific.
- 04:15 Monitor.
  - §20 complete. RB register q50 0.0286 / 0.0276 / 0.0280, all inside [0.027, 0.040]; the 50-qubit registered test holds 6/6 across both rows. MLP 0.0279 / 0.0267 / 0.0275; hybrid 0.940 both. About 94% of the added capture lands in octave b0. Written into app:hscale, §5.5 and the abstract ("on both rows and cores"); OpenReview abstract text refreshed.
  - FGN power extension, TRL2D s4: register 0.1591 vs MLP 0.1619, so the register is lower on 4/4.
  - Original FGN runners paused (MHD register ×3 deferred). FGN6 restarted serial so the TD replication keeps the GPU (0.58 steps/s at step ~3,000).
  - Doc 38 pre-committed abstract wording for both outcomes; Gemini's fallback rejected.
- 04:30 Monitor.
  - TD replication at 0.85 steps/s with one serial adrc job (step ~3,430). Each 2400 s attempt resumes cleanly.
  - FGN6: shear s4 has the register lower (+5.3%), making 4/4 on both rows.
  - RB FGN register s3 gives 2/3 and p = 0.36 (RB is not in the extension).
  - Table 13 checked by eye (p. 28), fine.
  - Gemini's app:fgn sentence rejected: it called the method "fractional Gaussian noise".
- 04:45 FGN6 TRL2D s5: register 0.15923 vs MLP 0.16039, so 5/5. Shear s5 register 0.01743 (MLP running). TD at step ~4,040, ~0.9 steps/s.
  - Gemini: "homogeneous isotropic turbulence" for TRL2D corrected. Per-octave τ_b accepted only as next-paper work.
- 05:05 **Registered FGN extension verdict.**
  - **shear_flow:** the register beats the matched MLP on 6/6 seeds, −3.4%, Holm p = 0.044. It PASSES, and the register-FGN is the best probabilistic model tested on shear (0.0171 < MLP 0.0177 < dressing 0.019 < deterministic 0.021).
  - **TRL2D:** 5/6 seeds, −0.8%, p = 0.12. Not established.
  - **Paper:** app:fgn; §5.5 gets one sentence; the conclusion names two narrow regimes; the abstract says "A register-specific effect" instead of "One …". Doc 38's pre-committed wording was updated to match (still before any seed-1340 evaluation).
  - **Robustness control registered and started:** `adrc_mlpw`, a wider MLP (3 hidden units, 297 parameters, more than the register's 244), seeds 1–6, shear then TRL2D, run serially. The shear claim stands iff register < mlpw on ≥ 5/6 seeds and p < 0.05.
  - **TD replication:** step 5,242, about 1.3 steps/s when alone.
- 05:20 **Wider-MLP control FAILS the shear claim.**
  - `adrc_mlpw` (297 parameters) vs the register: 4/6 seeds, +2.2%, p = 0.24.
  - Per the registered rule the shear advantage does not survive a wider MLP. Main-text claim retracted: §5.5 now reports "beat its matched MLP (6/6, Holm p = 0.044) but not an MLP one hidden unit wider (4/6, p = 0.24)"; the conclusion is back to one regime; the abstract is back to "One".
  - app:fgn reports the full sequence; OpenReview text refreshed.
  - Correction to my own message to Gemini: mlpw beats the matched MLP on 3/6 seeds, not 4/6 (means 0.01746 vs 0.01767).
- 05:30 TRL2D mlpw done. The register is lower than mlpw on 5/6 seeds (+1.1%, p = 0.09), reported without a claim. The MLP widths differ by noise (shear 3/6, p = 0.48, which Gemini computed correctly). app:fgn updated.
  - The TD job alone runs at ~1.0 steps/s (historical 1.67). No GPU throttling is active and the job is CPU-bound single-thread, so it is pinned to the X925 cores (taskset 5-9,15-19). Register at step ~6,360.
- 05:45 TD register arm resumed (attempt 4) at step 7,430, pinned, about 1.5 steps/s; ETA 06:10 and evaluation at about 06:25.
  - Gemini proposed dropping THROTTLE_MS on the claim that the freezes were RAM thrashing. Rejected: the verified root cause is a board power latch-off, which GPU load aggravates.
- 06:00 TD register at step 8,814 (~1.5 steps/s).
  - Observation recorded in doc 38 before evaluation: the selected best-VALID checkpoint is still step 84 (0.5803) for the register; the MLP and vanilla earlier attempts also have early bests. The rule is unchanged.
  - Gemini had appended a cell to model_performance.ipynb; told it not to write repo files.
- 06:15 TD register s1340 evaluated; the selected checkpoint is step 84 (VALID 0.5803).
  - One-step TEST 0.2230. Official-start 13:30 VRMSE 1.0056 (anchor 1.0132). All-start 0.9568 (anchor 0.9707; register seeds 1337–1339 mean 0.9486).
  - Per trajectory (13:30) it matches the anchor profile: traj 5–7 at 1.31 / 1.78 / 1.32 vs anchor 1.34 / 1.70 / 1.33. The s1339 register cut traj 5 and 7 to 0.82 and 0.95.
  - MLP s1340 training started at 06:10 (pinned). No conclusion until all three arms are in.
- 06:15 Secondary analysis registered in doc 38 at 06:12, before any final-weight evaluation: step-10,000 checkpoints of all arms, seeds 1337–1340, same evals and stats.
  - It followed Gemini's argument that early best-VALID selection biases against a training-time effect. The primary is unchanged.
  - 10 finals are being evaluated now on the A725 cores (the TD run keeps the X925 cores).
- 06:30 **Doc 38 secondary (final-step weights), seeds 1337–1339 (the fourth seed is pending).**
  - Register vs vanilla, primary per-seed test: s1337 mean diff −0.176 (Holm-significant), s1338 −0.040 (p 0.027, Holm 0.055), s1339 +0.058 (worse).
  - Register vs MLP: s1337 +0.091 (worse), s1338 −0.021 (p 0.065), s1339 +0.087 (worse).
  - Official-start 13:30 VRMSE at final weights: register 1.108 / 0.970 / 1.008; MLP 0.953 / 1.055 / 1.051; vanilla 1.343 / 1.057 / 0.949.
  - Reading: the rollout edge over the MLP exists at the validation-selected checkpoints (the primary), not at the final step. Continued TD training destabilises rollouts unevenly.
  - Gemini's hypothesis (early selection biases against the register) is contradicted on these three seeds.
  - When seed 1340 completes, the paper must scope the rollout claim to validation-selected checkpoints.
- 06:45 MLP s1340 at step 2,293 (~1.67 steps/s), ETA 08:00; vanilla ~10:00.
  - Register final s1340: official 1.1009, all-start 1.0065.
  - Selected→final per arm is not uniform: MLP improves on s1337 and vanilla on s1339. Gemini's "post-peak drift" and "phase coherence" framing rejected.
  - Main-text scope wording is prepared and will be applied after the 4-seed secondary.
- 07:00 MLP s1340 attempt 2 resumed at step 3,198 (pinned). The best VALID is still 0.5750 at step 72.
  - Gemini (checked): "register the only arm below the anchor on every seed" survives at 4 seeds whatever s1340 gives (register 4/4 below 1.0132; MLP and vanilla already exceed it on some seeds).
  - td_replication_summary.py written; it reproduces Table 2's 0.9528.
- 07:10 Gemini deleted two regenerable outputs of td_replication_summary.py ("cleaned test outputs"); no tracked files were affected. Told it to stop writing to the repo; this is its second violation today.
  - MLP s1340 at step 5,017 (~1.5 steps/s), ETA 08:05.
- 07:30 MLP s1340 at step 6,535, ETA 08:05. Paper drafts for both doc 38 outcomes prepared: a §5.2 sentence, space from compressing the mean-field sentence, and an app:perseed paragraph with the 4-seed table.
- 08:20 **DOC 38 VERDICT: DOES NOT REPLICATE.** At seed 1340 the register is worse than the matched MLP on 9/9 trajectories (selected checkpoints at steps 84 and 72). Register vs MLP is Holm-significant on 2/4 seeds; at final-step weights, on 0/4.
  - Pre-committed wording applied throughout: abstract ("…but a registered fourth seed did not replicate it, so no register-specific effect is established"), intro, contribution 2, the §5.2 title and a sentence, the conclusion ("the one candidate regime … did not replicate at four seeds"), the SoTA caption and row (0.9660 over 4 seeds), and future work.
  - To stay at 9 pages: the mean-field sentence was compressed, and the arena sentence now points to Figure 4 for the numbers. app:perseed has a four-seed paragraph.
  - Vanilla s1340 is still training, for the full report and the secondary.
- 08:30 Vanilla s1340 at step 1,911. Gemini's hostile read found one ambiguity: "only arm below the anchor on every seed" is now "below the anchor on all three seeds, the only arm that is". Its other two points and its 8/10 rating were rejected.
- 10:00 **TD replication complete.**
  - Vanilla s1340 selected at step 78: 0.8664 official, 0.8939 all-start.
  - Correction: I had swapped the MLP s1340 values at 08:11. Selected is 0.9439 / 0.8996; final is 0.9841 / 0.9555. The stats were correct.
  - The register is worst at s1340, worse on 9/9 trajectories against each control.
  - Four seeds: register vs MLP 2/4, vs vanilla 1/4 (Holm). Final weights: 0/4 and 2/4.
  - app:perseed paragraph fixed; tab:tdrepl added.
- 10:05 Final pre-submission checks pass: p10 = AI statement, 31 pages, 0 BibTeX errors, 0 undefined refs, no identity strings, OpenReview abstract matches the PDF, supplement rebuilt (5.9 MB).
  - GPU idle after TD, so the paused appendix lanes (RB/MHD (d)/(e), MHD register FGN) resumed. RF seeds stay off (3 h each, nulls).
- 10:15 Accepted Gemini's point: app:perseed now states the four-seed means neutrally ("the per-seed tests support no edge").
  - RB (e) has 3 seeds: register-det 0.0368 vs FGN 0.0352, so the drawn configurations beat the dressing control on RB as on shear (the MLP-det arm has 2 seeds). app:fgn updated. MHD register FGN s1 landed.
  - Lanes continue: RB tuned-τ, MHD register FGN s2–3, remaining (e) cells.
- 10:30 The (e) lane is done. MHD dressing 0.128 vs FGN 0.139, so the dressing wins on MHD as on TRL2D. FGN wins on shear and RB.
  - Pattern added to app:fgn: the dressing wins on rows where η ≈ 0 (TRL2D, MHD) and loses where the qubits capture most of the coverage (shear, RB). It is flagged as a four-row pattern only.
- 10:45 **§21 positive control (quantum-generated targets) registered and running.**
  - Gemini's design attack was adopted before any register result: the claim is limited to harness sensitivity, and GNN competitors (gnn 57 / gnnX4 289 params) see the teacher's exact r^-6 pair interactions.
  - Mean-field register (gutz): s1 0.733, s2 0.561.
- 11:00 **User-requested debate: quantum ALGORITHMS for utilization.**
  - Gemini's verdicts:
    - Rydberg-blockade MIS for combinatorial sub-steps is the most natural fit (~150+ atoms, Ebadi et al. 2202.04372).
    - Genuine quantum sub-system dynamics needs ≳36–64 atoms in 2D with non-local correlators.
    - Amplitude estimation (fault-tolerant overheads), HHL/PDE solvers (readout, dequantization) and Born machines (continuous fields) are dead ends for this pipeline.
  - Adopted in the main text:
    - §5.5: the register beats MLPs and GNNs on quantum-generated targets but not its mean-field twin (§21);
    - conclusion: "Whether non-simulable registers running native quantum subroutines, such as analog combinatorial routing or non-local Hamiltonian dynamics, change this is open".
    - To make room, a Fig. 6 caption sentence that repeats §5.5 was cut. Still 9 pages.
- 11:00 Gemini on the next-paper design (ii): the mean-field collapses on connected correlators by construction, so the informative test is register vs edge-aware GNN/MPS at 10–20 atoms. Settled; to register after submission. MHD tuned-τ cells are landing; FGN table regenerated.
- 11:15 Tuned-τ lane done (RB register 0.0350 vs MLP 0.0358, 3/3 seeds, p = 0.06; MHD register worse). app:fgn MHD range fixed. Last MHD register FGN s3 running.
- 11:25 All registered appendix cells complete (MHD register FGN 3 seeds: 0.142 vs MLP 0.139). Final checks pass; supplement rebuilt. GPU idle; no registered run can finish before the deadline, so the RF seeds are deferred.
- 11:40 User: abstract states the thesis, not a disclaimer. Removed 'Registers are simulated exactly…none is a claim of quantum advantage' from the abstract and moved the disclaimer to the conclusion. Scope now 'world models on ten industry-relevant benchmarks (fusion, geothermal, fluid flow, vehicle routing)'. The redundant Matched-MLP clause in §3 was trimmed to fit. OpenReview text refreshed.
- 11:45 Quiet: nothing running; the GPU is kept idle until the user copies the submission files. Gemini asked for a final read of the new abstract.
- 11:55 User: the abstract rewritten as one story (question → why averaging fails → routing → 0–50-qubit hybrid scaling law with the registered 50-qubit prediction → honest limits → takeaway), without internal jargon (trunk, Lean, DC, test names). \clearpage keeps the AI statement at p10. OpenReview text refreshed.
- 12:00 Abstract reordered per the user: thesis → 0–50-qubit gains (13% / 6%) → hybrid scaling law with its registered 50-qubit prediction → placement drives it (36% vs 7%) → why averaging fails (proof) → the classical module gains as much, so the rule is general. Limits live in the conclusion and §5.2.

## 2026-09-26 13:00 — final consistency pass against the abstract
- Abstract (9579f8f): utilization over qubit count; 36% vs 7% pooled at 30 qubits; 0->50 qubits 13%/6% with the hybrid scaling law G(K)=eta*U_inf[1-(1+K/K0)^(1-a)], registered K=50 prediction held 6/6; analog bands -> qubits, digital -> classical; Pasqal mapping with detuning re-encoding.
- Body aligned (f19d33f, 93e4ef0): contributions reordered gains-first (allocation, scaling law, theory, audit); results paragraph now reports K=50 and G(K); fig_hybrid_scaling moved into the main text, fig_multistart to app:perseed; future work = long-horizon industrial digital twins and neutral-atom scaling; ten = 8 audit rows + RealPDEBench + routing (two solvers).
- Gemini final pass: numbers verified; rejected its "match across all benchmarks" (overclaim) and removal of the persistence caveat on the 36%.

## 2026-09-26 13:57 — doc 35 §22: 3-D configuration and the value of geometry + tethers (registered 13:42)
- Bug found: position head zero-initialized, all atoms at z=0, zero gradient in z; every trained layout was 2-D.
- Registered control, 3 seeds, TEST once: shear 2-D 0.02073 / frozen grid 0.02071 / 3-D bilayer 0.02110 / MLP 0.02129;
  RB 0.04261 / 0.04258 / 0.04271 / 0.04281. Neither "3-D helps" nor "geometry + tethers help" passes: the routing carries the gain.
- Abstract corrected (no longer attributes gains to per-input positions/tethers); results + app:hscale report the control.
- Figure 2 (model outputs): shear, convection, TRL2D (TEST window fixed in advance) + MHD (valid trajectory, cached).

## 2026-09-26 16:15 — doc 35 §23/§24 (3-D tethered register on every benchmark; configuration screen)
- TetheredGeometry3D (per-input x, y, z, detunings, tethers) wired into every benchmark; checkpoints + Figure 2 dumps kept.
- Heat plume: classical 0.0364 / 3-D MLP 0.0340 / 3-D register 0.0341; TORAX: 0.0321 / 0.0309 / 0.0305. Register ties its MLP.
- §24 screen (VALID only) of configuration levers on the routed co-processor running; memguard killed early jobs (too many lanes).
- §24 screen (VALID, 2 seeds): H1-H7 none promoted (best: H6 combo -1.2% shear but +0.4% RB). Default configuration stays.
- §23: TRL2D arena 3-D register 0.9694 vs 3-D MLP 1.0229 vs classical 1.0663 (3/3, p=0.010 nominal; Holm 0.05, not < 0.05);
  RealPDEBench all cores 0.00939; fusion 3-D reruns (timeouts) in progress.
- §25 registered: hybrid scaling to K=100, 200 with the law's prediction; queued behind memory.

## 2026-09-26 21:25 — disk cleanup (paper submitted: https://openreview.net/forum?id=GNJXn14y6P)
Rule: keep everything behind a reported number; remove only duplicates, unrelated/re-downloadable assets, parked caches.
Removed: ~185 GB unrelated HF LLMs (~/.cache/huggingface); data/well/planetswe 179 GB (excluded in the paper);
data/lingbot-world-v2-14b-causal-fast 80 GB (parked video); data/tokamark/zarr_partial_leftovers 38 GB (broken partial
shards moved aside by runs/tokamark/repair_nested_zarr.py; repaired copies kept); data/phase_c/{renders,latents} 11 GB
(regenerable, parked video); 56 *_latest.pt resume checkpoints (12 GB) older than a day whose final checkpoint exists.
Disk free 8.5 GB -> 514 GB. Kept: all benchmark data, checkpoints, prediction caches and results used by the paper.

## 2026-09-26 22:08 — next phase opened; scaling extended; TokaMark rerun
- Read and checked: QUOPS (arXiv 2609.12146, Proctor et al., Sandia: hardware ~1e5 below useful scale), TQI quantum
  simulation explainer, TQI/IEEE quantum-to-classical handoff (validate quantum output before handing on).
- Gemini critique: keep M1 (Rydberg MWIS move selection), M2 (projected quantum kernel), M4 (handoff gate); drop M3;
  QSVT/QLBM/VQLS excluded (exact simulation = same linear algebra as the control). Shapley credit adopted.
- §25 K=200: 11/12 runs in, every seed inside the registered capture range (shear 0.365-0.375, RB 0.0407-0.0409);
  eta ~0.90. Register vs MLP at K=200: shear 2/3, RB 1/2 so far (not a claim; §26 replication decides).
- §27 registered (TRL2D, MHD to K=50/100/200; expectation eta < 0.15), queued after S25.
- §28 registered: paper/qmethods (blockade-subspace MWIS emulator, batched; exact-additive 2-opt/relocate conflict graphs
  with RouteFinder multi-depot costs, parity with the env reward 1e-6; 23 tests). Smoke (released policy, n=16, VALID):
  rydberg reaches 99.7-99.9% of exact's local-search Shapley credit. Screen waits for the §23 EAL vanilla policy.
- §28c registered (handoff under SPAM/dephasing/shots + Shapley over gate/classical/core); VALID smoke: noise costs ~1%
  at 100 shots, handoff keeps full trust; hw10 profile added before any TEST run.
- TokaMark §23: 4 h cap ended the 3-D run at the ft_heads->ft_full boundary (VALID 0.1013); full recipe rerun, 16 h cap.

## 2026-09-26 23:06 — §29 (MHD, fusion), RouteFinder EAL, fusion leaderboard submission
- RouteFinder §23 EAL vanilla (seed 1): mean test gap 8.687% over the 16 unseen MD variants (policy saved). §28 M1
  screen crashed (VALID has 128 instances per variant, not 200); fixed, deviation recorded, relaunched.
- §29a MHD solenoidal B: TEST 0.1725 -> 0.1709 (registered 0.170-0.171). §29c shift averaging registered (VALID
  diagnostic -9.5% with the solenoidal B; blur check: error falls in every octave, small high-k shrink); TEST pending.
- §29b fusion: 3-D register and matched MLP on the SoTA trunk (MLP VALID S 0.9867-0.9868); register seeds running.
- Table 1 and the scaling figure now show a qubit budget only when all three registered seeds are in.
- Fusion leaderboard: the registered entry (7-member ensemble, held-out S 0.9910) predicted on the 874 public DIII-D
  test shots (order and frame counts checked against the raw files, the Hub file list and the stream), uploaded as a
  direct zip (Codabench submission 947029, Development phase, method name "PCA-MLP ensemble"); scoring in progress.
  Leaderboard scores are external evaluation only, never used for model selection in the paper.
- 2026-09-27 00:21 MHD §29c TEST: per-channel 0.1725 -> 0.1709 (solenoidal B) -> 0.1592 (+ 8-shift average); field-grouped 0.1853 ->
  0.1840 -> 0.1707 (published 0.1633, field-grouped). §29b fusion: 3-D register VALID 0.9877-0.9881 vs MLP 0.9867-0.9868;
  TEST singles in progress (trunk 0.9844, MLP 0.9848/0.9851). §28c seed 1: register system better overall (0.02057 vs
  0.02089) through its co-trained head+gate (0.02196 vs 0.02444); its core's Shapley 0.0009 vs the MLP core's 0.0029.

## 2026-09-27 01:49 — rebuttal revision rounds 1-2, §26 progress, §30 first result
- Revision batch (46f07b4, defb5c6, and the wording pass after it) from the GPT Codex reviewer's hostile reads: published
  column = current verified best (PDE-FM shear 0.0345 / RB 0.0415, arXiv:2511.21861 Table 3; Flowers TRL2D 0.1930 4->1,
  arXiv:2603.04430 Table 1; EddyFormer MHD 0.1160, NeurIPS 2025 Table 3); bold = best of ours; RB routed rows marked
  stratified; app:claims (only the first audit round confirmatory); Prop. 1 scoped to the insertion latent; energy score
  called observation-weighted where first defined; AI statement narrowed and discloses the Codex reviewer. 9 pages.
- §26 replication (K = 100, register vs matched MLP, TEST once): shear_flow s4-s7 register lower on 4/4 (0.02050 vs
  0.02075, 0.02014 vs 0.02098, 0.02009 vs 0.02086, 0.02047 vs 0.02088); rayleigh_benard s4-s5 2/2 (0.04093 vs 0.04160,
  0.04117 vs 0.04131); s8 / RB s6-s8 pending. Rule: >= 4/5 and one-sided paired p < 0.05.
- §30 (i) submitted-objective re-run, shear s1: variance-normalised ES 0.01795 (submitted run 0.01704), ensemble-mean
  VRMSE 0.02373 (submitted 0.02259), deterministic hybrid 0.02105 -> ES 15% below (submitted 19%); proper ES 0.00667
  (fixed TRAIN scale, not comparable in units). Gemini read test_xi0 (0.03358, the xi = 0 member) as the test VRMSE.
- §32 registered and launched (waits for 40 GB): locked-recipe retraining + full-TEST evaluation of Table 1's RB routed rows.

## 2026-09-27 02:00 — fusion 3-D ensemble selected (VALID), §28c seed 2, §33 launched
- §29d fusion: VALID-selected 6-member ensemble with one 3-D register model (ens_select_reg3d.json): full VALID S 0.9913,
  consistency 0.9677 (MLP entry 947029's ensemble: 0.9913 / 0.9674); kappa 0.9335 vs 0.9310, volume 0.9654 vs 0.9646,
  li 0.9539 vs 0.9565. TEST scoring after the _mlp3d selection (running).
- §28c seed 2 (shear K=30, register): TEST 0.02071; Shapley gate 0.01082, classical head 0.01459, core 0.00074 (2.8%).
  Seed 1: register core 0.00094 (3.6%), matched MLP core 0.00288 (11%).
- §33 (VALID-only redesign) registered and launched; screens no longer load TEST. K=100 shear: register lower VRMSE than
  the MLP on 7/7 seeds but lower captured energy on 4/7 (metric-mismatch caveat, 39-rebuttal-plan.md).

## 2026-09-27 02:12 — §26 RB s6, §27 K=200 s2, §28c MLP s2; Figure 2 row chains
- §26 RB K=100: register 0.04093 / 0.04117 / 0.04108 vs MLP 0.04160 / 0.04131 / 0.04112 (s4-s6, 3/3; s6 margin 0.1%).
  Shear s4-s7 4/4 (means over s1-s7: register 0.0200854, MLP 0.0209530, paired -0.0008676; register captures less
  energy on 4/7). s7/s8 RB and s8 shear pending.
- §27 K=200 s1-s2: TRL2D and MHD_64 tie at 4 decimals (register / MLP / classical ~0.1922 and ~0.1733).
- §28c shear K=30 s2: MLP TEST 0.02079 (core removal +7.9%; s1 +14.5%); register s2 0.02071 (core removal +7.2%).
- Figure 2: runs/s23_rows_runner.sh and runs/s23_rfrows_runner.sh now dump the Well, MHD and RouteFinder rows as their
  §23 3-D chains finish; runs/paper_refresh.sh rebuilds. §31 downloads not started (awaiting the user's direct go-ahead).

## 2026-09-27 02:21 — §31 split and sampling locked (§31c, §31d); Flowers MHD feasibility
- §31c: V1/V2 = alternating parity of sorted (file, trajectory) ids (metadata check before caches); §31d: balanced
  VALID sample, 10 starts per trajectory (gray_scott, 1,200) and 25 (active_matter, 600). Both before any cache/score.
  Download running (active_matter ~49 GB, then gray_scott ~142 GB; TEST not downloaded).
- Flowers (github t-muser/flowers, MIT, commit f1d2cad, baseline/flowers): the 4->1 recipe trains from scratch, 100
  epochs, batch 32, AdamW lr 1e-3 wd 1e-2, warmup-cosine, MSE. For MHD_64 (7 channels, 64^3) the model builds with
  dim_in 28 / dim_out 7, 23.9 M parameters. MHD_64 TRAIN: 77 trajectories x 96 windows ~ 7.4k windows/epoch, ~23k steps.
  The paper reports up to 24 h on one H200; on the GB10 (far lower memory bandwidth, 3-D convolutions) several days
  alone, not measured yet because the GPU is saturated. No weights released.

## 2026-09-27 02:28 — §34a Flowers port aligned; TORAX caption; §27 MHD K=200 s2
- §34a: the Flowers trainer now follows the authors' per-epoch schedule, shuffle-without-replacement epochs and no
  clipping; throwaway timing chunk sets 100 or 25 epochs; documented deviations: stats.yaml normalisation, VALID selection.
- Figure (all benchmarks) caption: TORAX's "clearest gain" is now labelled as the seed-1337 sample, with three-seed
  VRMSE@8 means register 0.0305 / MLP 0.0309 / classical 0.0321 (register ahead on 1 of 3; Gemini's audit, verified).
- §27 MHD_64 K=200 s2: register 0.17329, MLP 0.17327 (tie at 4 decimals, as at K=50/100).

## 2026-09-27 02:58 — §28 M1 blockade move selection: not promoted; post-classical export; bridge sender fix
- M1 screen + exact_eps control (VALID, 128 x 4 variants x 2 seeds): rydberg level with exact on average (8.381 vs 8.387, descriptive) and greedy 8.401; selector quality had little observed effect at n <= 18 in this pipeline; ceiling test FAIL (mdvrpl s2 +0.026 pp; nn start up to +0.148), combinatorial vs best1 PASS (9.425). No TEST. Appendix app:routing paragraph added.
- §33: Gemini's band reading corrected (negative bands 2-3 are the classical path; qubit path >= 0 in every band; 97% band-0 slots ~ 92% band-0 energy). Both agents accepted.
- paper/alloc/dump_post_cls.py: post-classical (cls_only) residuals for TRAIN/VALID, shear/RB s1 (VALID 0.022946 / 0.045819 reproduced), for Codex's ceiling rerun (their first run used ungated trunks 0.844 / 0.724).
- .bridge/send.sh: sender taken from the caller's tmux session (Gemini's 024539 note to Codex had been labelled "claude").
- 03:05 §27 done: TRL2D/MHD at 50/100/200 qubits stay on the classical part (seed means 0.1922 / 0.1733, register = MLP; TRL2D eta <= 2.4e-4); MHD capture inside the registered ranges on 9/9 seeds, eta < 0.15 expectation held. app:hscale paragraph added.
- 03:08 §23 RB 3-D chain done (Figure 2 row source): 3-D register neck FULL TEST 0.04282, 3-D MLP neck 0.04259 (n=34300; published best 0.0415 PDE-FM; our E-ft ensemble 0.04027). MLP ahead of the register by 0.5%. Shear 3-D chain: register neck 0.03027 full; MLP neck strat 0.0281, full TEST pending. Gate table rows regenerated (gate_table.py adds the "3-D MLP neck" label).
- 03:06 §33 export extended (idx, arm-classical, same-checkpoint no-core residuals) for Codex's registered route-execution ceiling; no-core export equals diag (0.021384 / 0.023467).
- 03:12 Codex reviewer diagnostics (isolated, uncommitted, VALID only, each protocol written before scoring; numbers verified by Claude from the exports): post-classical K30 perfect-cancellation ceiling on shear s1: static 23.0%, causal ridge 24.6%, cross-trajectory shuffled 15.1%, oracle 35.9% (allocation ceiling only). Route audit: register realizes 27% of its own selected-route headroom, MLP 51% (different co-trained baselines, no clean quantum penalty). Input-only register-vs-MLP selector: +0.20% vs all-MLP, 95% CI includes zero (oracle 2.52%, register better on 52.5% of windows); no conditional quantum value claim.
- 03:16 §33b B3 registered (fixed log_lam_xy 0.366, V x 9; Codex critique adopted: diag uses effective V, fixed scale for equal parameter count, failure rejects the recipe only). Tag s33b-b3-prereg; runner waits for §31's terminal marker.
- RB s1 stage A (VALID): register 0.043077, MLP 0.043177, cls_only 0.045819; frozen no-core register +14.7%, MLP +4.7% (shear the reverse); per-input positions/tethers inert again.
- 03:22 §26 primary cell PASS: shear K=100, seeds 4-8, register < matched MLP on 5/5 (0.02020 vs 0.02072, 2.49%, one-sided paired p = 0.0065, Holm-safe). RB K=100 4/4 so far (p = 0.070). Paper: abstract exception clause, claims row, app:hscale paragraph with the prepared caveats (graph-blind MLP, TEST-informed, graph-aware/metric-aligned controls pending).
- 03:20 §33c registered: retrained tau = 0 register control (tag s33c-tau0-prereg), after B3.
- 03:15 Codex state diagnostic (registered 03:13:07 before scoring; isolated, uncommitted; VALID s1, 64 windows per row; readouts reproduce the model within 8.4e-7): half-register (5|5) entropy mean/median/p90/max shear 0.077/0.053/0.125/0.709 bits, RB 0.018/0.002/0.023/0.834 (max 5); pair-phase p90 0.129 (matches diag_s33) but p99 0.53 shear / 5.36 RB. Reading: weak typical entanglement with a strongly interacting tail; not an effect claim. Entropies not independently recomputed by Claude (file values match the message).
- 03:31 §29d fusion ensembles (held-out TEST S): 3-D register 0.991003, 3-D MLP 0.990986, classical+MLP (submitted) 0.991017, classical 0.990973, 2-D register 0.990460. Tie; 3-D register +0.0005 over 2-D. Upload of submission_reg3d held for §29e / final step. Paper: fusion appendix sentence.
- 03:31 Paper scope fixes from Codex audit: routers trained separately per arm (routed sets overlap 40%), GNN learns its own graph, VRP/ADR-C MLP ignores coordinates, 199 vs 241 trainable, doc 35 entry times corrected to commit times.
- 03:32 K100->K200 shear (Codex paired audit, TEST-informed, descriptive): register VRMSE 0.019796 -> 0.020117 (+1.6%, p = 0.41, n = 3) while its captured energy share rises 0.3505 -> 0.3774 on 3/3 seeds; MLP 0.021065 -> 0.020309 (p = 0.14). Paper: "levels off at 200 although the captured share still rises" (was "as the reachable share saturates"). No K100/200 reruns until §33 B2 (K30, VALID) passes its gate; then a registered VALID-only K100/200 loss/target factorial.
- 03:42 Codex joint-shot covariance diagnostic (registered 03:39:59 before scoring; isolated, uncommitted; VALID, §33 A seed-1 register checkpoints, 64 windows x 3 groups per row; joint distributions reproduce expectations within 2.2e-6): exact variance of a group's 10-atom mean over the independent-marginal variance, shear Z 1.122 / X 0.926 / Y 1.053, RB 1.008 / 1.002 / 1.000. Supports the paper's "independent-marginal approximation" qualifier; no VRMSE claim, old shot-study TEST claim not retrofitted. Not cited in the paper (reviewer data, uncommitted).
- 03:42 Gemini §26 recomputation exact (p = 0.006504, Holm-safe). Accepted: abstract names "ten 10-atom groups". Rejected for now: stating that the margin disappears against the §26b controls (1/5 seeds; the registered rule decides at 5/5); its "parameter-matched" label for the 199-parameter MLP.
- 03:43 §26b GNN control decided: GNN (239) below the register on s4 and s5 (0.01960/0.02050, 0.01947/0.02014), so the >= 4/5 rule cannot pass; abstract, appendix and claims now say the K=100 shear edge holds against the smaller graph-blind MLP only and is not established against a graph-aware GNN of similar size.
- 03:49 §31 VALID stage done (retry for the OOM cell): S = 0.0735 active_matter, 0.2906 gray_scott, both route proxy -> registered prediction "proxy beats pooled on full TEST in both register and MLP arms". Prediction, results, caches and code hashed (36 files, runs/s31/lock), tag s31-prediction-lock. V1/V2 baseline gap reported as a limit (active_matter 5.2x).
- 03:49 §26 RB K=100: 5/5, 0.87%, p = 0.029 (unadjusted pass; Holm pending the K=200 cells). §26b wider MLP s5 0.020758 above the register's 0.020142 (width gate open); GNN gate already failed. app:hscale updated.
- 03:56 §33 stage A rule decided: per-input positions/tethers not causal on either row (shear 0/3, RB at most 1/3; swaps <= 0.21%); frozen core ablation +4.8 to +17.4%. §31e audit fixes (bootstrap estimand, arms-lock rechecks) committed; phase A running.
- 04:09 §33b B3 seed 1 (VALID; both rows unfavourable): shear 0.020545 vs stage-A register 0.020396 (+0.73%), RB 0.043437 vs 0.043077 (+0.84%). Regime check passes (V_eff>1 on 26.6% / 27.0% of pairs vs 1.4% / 2.2% in A); swaps still inert. On shear training shrank tau (-0.03, 0.07, 0.10, 0.09 vs ~0.34-0.39 in A), so phase q90 rose only 0.13 -> 0.31 rad. Rule needs s2/s3; no entanglement claim (state not measured on B3).
- 04:14 §31 stress analysis after the lock (Codex, read-only, V2 descriptive, prediction unchanged): active_matter proxy-vs-pooled V2 gain +6.21% register / +7.88% MLP overall, but negative on 5 of 12 V2 configurations for the register and 4 for the MLP (worst zeta9 alpha-1: -9.05% / -6.81%); gains concentrate in high-error configurations (zeta7 alpha-3 +19.6% / +21.1%). Concrete TEST failure mode: a configuration mix with more low-error regimes. One trajectory per configuration on V2.
- 04:16 §33 stage A complete (18/18): per-input configuration 0/3 on both rows; core ablation +4.8 to +19.8%. app:hscale gains one sentence (VALID mechanism check).
- 04:21 §31 metadata (Codex, file names in data/well/active_matter/.gitattributes only; TEST not opened): 45 TRAIN, 16 VALID, 21 TEST configurations; TEST in TRAIN 21/21, in VALID 5/21. TEST mix differs from VALID but is not unseen-parameter OOD relative to TRAIN (Gemini's "OOD interpolation" rejected). Prediction unchanged.
- 04:27 §26b shear decided: register lower than the GNN (239) on 2/5 (GNN mean 0.01990 vs 0.02020, p = 0.76) and than the wider MLP (297) on 2/4 (at most 3/5): the K=100 shear edge is not established against either control. Abstract/app:hscale/claims updated. B3 shear s2 0.020204 < A 0.020510 (s1 was worse). Figure 2 shear chain done (S23_WELL_shear_flow_DONE 04:25); rows dumping.
- 04:31 §26b shear complete: wider MLP register lower 3/5, p = 0.13 (fail); GNN 2/5, p = 0.76 (fail). RB controls started. app:hscale updated with complete numbers.
- 04:37 §31e phase A, gray_scott V2 (descriptive; locked prediction untouched): register proxy V2 VRMSE 0.12657 / 0.12425 vs pooled 0.15130 / 0.14486, but captured-energy share -0.0442 / -0.0482 (pooled +0.0145 / +0.0396). VRMSE and captured share diverge (per-field square-root mean vs residual-energy fraction; training on captured share, V1 stop on VRMSE). The registered endpoint is TEST VRMSE; capture is reported separately and the proxy arm is not described as capturing more energy on this row.
- 04:44 §31e arms lock PASSED Codex's independent audit (69 entries, sha 0aeecb4d); ARMS_COMMITTED created, phase B (TEST download, TEST-only cache, score once, verdict) running. Scheduling: the §28c credit and §30 proper-ES runner shells paused with SIGSTOP (PIDs 237509, 438458; their running jobs finish, no new launches) to free 40 GB windows for the §31 TEST cache and the TRL2D/MHD Figure 2 chains; resume with kill -CONT. No protocol change.
- 04:46 §31 gray_scott V2 regime audit (Codex, archived, V2 only, after the arms-lock PASS): aggregate proxy-vs-pooled V2 gain +14.45% register / +24.48% MLP, but gliders (pooled 0.856 vs 0.003-0.005 elsewhere) carry ~98% of the pooled error sum and all of the aggregate improvement; within-regime register/MLP gains: bubbles +1.5/+2.2%, gliders +14.8/+25.0%, maze +3.3/+0.8%, spirals -3.7/-2.9%, spots -1.2/-0.4%, worms -0.5/-1.1%. Reporting: the registered aggregate TEST verdict first; per-regime TEST breakdown descriptive; the aggregate is not presented as broad regime-level improvement.
- 04:54 Figure 2 now has 7 rows (shear flow and convection added from the §23 3-D chains; middle test window). Shear sample favours the register (0.0171 vs 0.0271 classical, 0.0230 MLP) but full TEST register neck 0.0303 vs MLP neck 0.0282; convection MLP ahead on sample and full split (0.0426 vs 0.0428). Caption corrected (only MHD and TokaMark omitted). Remaining rows: TRL2D (chain waiting for 40 GB), MHD, TokaMark.
- 04:56 §31 TEST acquired (both rows, pinned revision, hashed); TEST-only cache waits for 40 GB (MemAvailable ~22 GB). Paused two more runner shells with SIGSTOP (running jobs finish, no new launches): §29e fusion band models (357694; fusion upload held) and §26 K=200 cells (141818; only feed the Holm family). Paused so far: §28c, §30, §29e, §26-K200. Resume each with kill -CONT <pid> after the §31 TEST cache and the TRL2D Figure 2 chain have run. Gemini retracted the gray_scott mechanism story.
- 04:59 §33b B3 FAILED its rule (shear 1/3, RB 2/3; regime check passed). Only the fixed strong-coupling recipe is rejected. §33c tau=0 control starts.
- 05:05 §29e job exited; the TRL2D Figure 2 chain took the freed window (neck fine-tune started ~04:54). MemAvailable ~32 GB. Paused the §32 runner shell too (PID 502187; its current RB full-test cache build continues) so its next 40 GB launches do not pre-empt the §31 TEST cache. Paused: §28c 237509, §30 438458, §29e 357694, §26-K200 141818, §32 502187 (resume each with kill -CONT).
- 05:17 §31 TEST cache still gated (MemAvailable ~31 GB vs 40); also paused the §33 stage-B runner (553536) and §33c runner (640230) after their current jobs, to open the window. Paused: §28c 237509, §30 438458, §29e 357694, §26-K200 141818, §32 502187, §33 553536, §33c 640230; resume with kill -CONT after the §31 TEST cache starts.
- 05:22 Resumer bug: pgrep -f "capped.sh s31t" self-matched its launching shell, so §33 lanes were resumed at 05:17:56 although the §31 TEST cache had not started (Codex 052200). Fixed: resumer killed by PID, §33 lanes re-paused, CORRECTION line in status, resumer relaunched with a bracket-class pattern on the real cache.py process. §31 cache still waits for 40 GB.
- 05:28 Codex conditional-budget ceiling (isolated, VALID-only ideal cancellation; features causal, traced by Codex: VALID a/E/D enter only after the policy is fixed): shear adaptive K vs same-histogram cross-trajectory shift 11.1-11.4% lower ideal VRMSE (mean K ~15.6; wins all 579 shifts); RB 0.16-0.26% (seed-1 interval narrowly excludes zero; reverses on ~4% of shifts) = practically tiny. Caveats: VALID rows reused (fresh split needed), achieved effect unmeasured and likely smaller (no numerical extrapolation), not quantum-specific. Recorded as §33d design motivation only.
- 05:30 Codex scalar-difficulty control: the shear conditional-budget ceiling edge is fully explained by allocating more slots to windows with larger predicted total energy (mode-specific budgeting adds nothing on shear; tiny on RB). §33d design note in doc 35.
- 05:35 Codex committed its own files (8becf23, paper/codex + runs/codex only), breaking the single-writer rule; undone with git reset --soft (unpushed HEAD), files kept uncommitted; Codex told not to commit. §31 TEST cache started for real (active_matter, 05:30); resumer resumed §33 lanes correctly.
- 05:45 Codex read-only TRAIN->VALID coverage diagnostic (anchor/pre-gate, exploratory, no TEST): full-basis U TRAIN->VALID shear 0.687->0.706, RB 0.444->0.449, TRL2D 0.297->0.534, MHD 0.031->0.020; a TRAIN ridge on causal g predicts VALID coverage on shear/RB (Spearman ~0.82, MAE roughly halved) but not TRL2D/MHD. Supports keeping the closed form descriptive, not a cross-row forecast. No paper change.
- 05:50 Codex exploratory V1-calibration check (pre-gate coverage): V1 calibration rescues TRL2D/MHD but is worse than TRAIN on shear; LOO-trajectory choice indecisive on shear/TRL2D. Caution for any future per-row calibration; nothing registered, no paper change.
- 05:52 §26b complete: RB register vs GNN 1/5 (p 0.97), vs wider MLP 3/5 (p 0.15); both rows fail against both controls. app:hscale final numbers.

- 2026-09-27 06:12: §33c (tau = 0 retrained) VALID: RB within 0.5% of the register on 3/3 seeds (no measurable interaction
  contribution in this recipe); shear mixed (+1.26/+0.21/-0.68%). Codex's route-choice diagnostic was reproduced and
  critiqued (the denominator should be the reachable reduction; static in-sample bias); not promoted; §31 unchanged.

- 2026-09-27 06:25: RouteFinder §23 EAL-MD, 3-D register (q30-geo=3d, seed 1, 10 epochs, 10k, CRN): mean test gap 8.623% over the
  16 unseen MD variants, against 8.687% for vanilla seed 1 (0.064 pp lower, 0.7% relative; single seed, not a claim).
  The matched 3-D MLP control (q30-geo=3d-mlp, seed 1) started next in the same runner. Its result decides the reading.

- 2026-09-27 06:56: TRL2D Figure 2 chain (s23trl): the 3-D tethered register (q30-geo=3d, gated fine-tune, vrmse loss, seed 1)
  scored FULL TEST 0.19128 (n = 873), once. For reference: E-ft ensemble 0.19091, CNextU-net 0.1956 (published). It
  beats the published number by 2.2%, is 0.2% above our E-ft, and is a single seed. The matched 3-D MLP arm
  (q30-geo=3d-mlp) is now training in the same chain; the paper row waits for it.

- 2026-09-27 07:26: §31 prospective TEST verdict: prediction held on active_matter and gray_scott in both cores (proxy vs pooled
  mean seed-paired gains: register 12.3%/15.2%, MLP 14.1%/24.8%); the MLP gains more; not a quantum advantage. Lanes
  auto-resumed at the marker.

- 2026-09-27 08:32: TRL2D §23 pair complete (seed 1, gated fine-tune, vrmse loss, FULL TEST n = 873, scored once each): 3-D register
  0.19128, matched 3-D MLP 0.19098 (MLP lower by 0.00030, 0.16%). Both are below CNextU-net 0.1956 and bracket our E-ft
  0.19091. The selected necks are at steps 107/102; the gated stage ran 5,118 vs 8,394 steps (time budget). One seed,
  so descriptive: no register advantage on TRL2D.

- 2026-09-27 08:52: §33t TEST (registered, frozen checkpoints, TEST-informed): B2loss (train on mean VRMSE) holds on TEST.
  - Register: shear 0.0195/0.0199 vs 0.0205/0.0208; RB 0.0418/0.0417 vs 0.0425/0.0427.
  - The MLP improves similarly.
  - New lowest shear_flow TEST: 0.0195 (published best 0.808).

- 2026-09-27 09:03: RouteFinder EAL-MD §23 pair (seed 1, 10 epochs, CRN): 3-D register 8.623%, matched 3-D MLP 8.774%
  (1,715,198 vs 1,715,057 params), vanilla 8.687%. The register is lower than both (0.15 pp vs MLP). One seed, so
  descriptive. The MLP took 9,442 s and the register 21,127 s (same epochs). Paper: app vrp Results sentence.

- 2026-09-27 09:25: §34 Flowers MHD timing: 106 steps in 1,257 s, so 100 epochs are projected at 273,931 s (about 76 h,
  under the one-week cap). The runner set 100 epochs; the real run started at 09:23 and should end around Sep 30 13:30.
  The §23 MHD row and lanes stay paused until then.

- 2026-09-27 09:30: Codex signed residual probe (post hoc, VALID, CPU; not registered in doc 35 before it ran, recorded
  after). Field x band linear heads remove 22-25% (shear) and 13-15% (RB) of the pooled weighted residual energy, but
  shear K30 VALID VRMSE gets worse (0.0229 -> 0.0250) while RB improves slightly (0.0458 -> 0.0454). Claude traced the
  shear case to selection overshoot (weighted slope 0.42 on the selected slots) and window heterogeneity, not the square
  root. Exploratory only; outside the paper.

- 2026-09-27 09:35: signed-probe shrinkage follow-up (post hoc, VALID, CPU). Shear: TRAIN rejects the linear correction (gamma 0 on
  two seeds, neutral on the third). RB: a TRAIN-chosen gamma of .5-.75 lowers K30 VALID VRMSE by 0.5-1.0%, and the
  input-dependent route beats static and shuffled on all three seeds. At calibrated gamma the shear failure comes from
  the per-window square-root and heterogeneity effect; the selection overshoot only dominates at gamma 1. Outside the paper.

- 2026-09-27 09:45: Codex replayed the §33 A/B2loss no-core knockouts (post hoc, VALID). The core output is load-bearing in all
  16 cells, but 12/16 knocked-out models are worse than a classical model trained without a core. So the knockout
  measures co-adaptation, not contribution, and no mechanism for the metric-aligned gain is claimed.
- 2026-09-27 09:45: paper: Figure 2's new routing row got a wrapped label and a caption sentence (instance: MLP cheapest, 13.34 vs
  13.40 register, 13.65 classical; 16 variants: register lowest mean gap, 8.62% vs 8.69% and 8.77%, one seed).
  tab:sota shear keeps the registered 3-seed 0.0207 and adds the metric-aligned 0.0197 (2 seeds, matched MLP 0.0199).
  Main text ends on page 9; AI-use statement on page 10.

- 2026-09-27 09:50: paper: the registered `lin` arm (doc 33: the core's value is its increment over `lin`) is now reported. On the same
  routes a linear corrector gets 28.75% of rayleigh_benard's 36.25% and all 26.3% on shear_flow; the cores add 7.5
  points on RB and none on shear (abstract, contributions, section alloc, claims table). Gemini raised the point with
  wrong numbers (gated §22 pipeline); the allocation files confirm it. Trimmed the conclusion's future-work list and
  three other sentences so the main text still ends on page 9.

- 2026-09-27 09:59: §34c: the Flowers paper (arXiv:2603.04430, Table 5) used lr 5e-4 for Flower on MHD_64, not the repo default of 1e-3
  that §34 had fixed. Gemini found it and Claude verified it against the paper. The amendment was recorded first, then the
  real run was restarted at step 0 with 5e-4. The 178-step 1e-3 partial run is archived and unused. The run should now
  end around Sep 30 14:05.

- 2026-09-27 10:00: where the cores beat the linear corrector on rayleigh_benard (allocation study, proxy, test, 3-seed means per
  window): the top 10% of 1,200 windows carry 97% of the 7.5-point increment (top 5%: 77%); the median window changes by
  0.0001; the worst window goes 6.89 -> 4.67. On other rows the register adds at most 0.8 points (heat plume). Added to
  app:alloc.

- 2026-09-27 10:13: tab:alloc caption fix: RB's "median VRMSE -4%" belonged to the static strategy. For the listed proxy row the
  median changes by only -0.3 to -0.5%. The onset windows (4%) carry 47% of the absolute gain, but the gain without
  them is still 36.1%. Gemini's onset attribution checks out (50 windows, 55% of the cores' increment over the linear
  corrector), but its claims of an "onset artifact" and an "8-parameter MLP" do not.

- 2026-09-27 10:27: MHD_64 error profile (fine-tuned CNextU-net, 3 seeds, runs/phase_b_mhd/test_*_s133{7,8,9}.csv; 10 test runs x 96
  windows in sorted-file order): super-Alfvenic runs (Ma 2) score 0.2177/0.2174/0.2441 against 0.1364/0.1367/0.1604 at
  Ma 0.7 (ratio 1.57; 61% of the total error). Error is flat in time (start-time bins 0.183/0.181/0.185/0.189) and the
  tail is light (the worst 10% of windows hold 13.6% of the mean). Register, MLP and vanilla are identical within each
  seed. So MHD error is regime-level, which is why routed modes cannot reach it (U = 0.023). Added to app:sota.

- 2026-09-27 10:39: TokaMark 3-D register (FT-Base warm start, full recipe, seed 1337): TEST score 0.1512 against vanilla FT-Base
  0.1504 (same seed) and our best, 0.1379 (Scratch recipe). No improvement. The matched 3-D MLP arm is now training
  (about 12 h). Flowers MHD at lr 5e-4: chunk 1 ended at step 233 (1 epoch) with VALID 0.4750.

- 2026-09-27 10:41: TokaMark per-signal (seed 1337, one seed, descriptive): the 3-D register beats vanilla FT-Base on 5 of 15 signals
  (bvac_rmag -0.0033, x_point_r, magnetic_axis_z, bphi_rmag, minor_radius; each change 0.0008 or less except bvac), and
  is worse on the pressure signals (beta_normal +0.0043, beta_pol +0.0035, beta_tor +0.0022) and the triangularities
  (+0.0022/+0.0023). The Scratch recipe beats both on 14 of 15 signals (not x_point_z), so the TokaMark gap to 0.140 is
  about the recipe, not the neck.

- 2026-09-27 10:58: §36 step 0 (TRAIN only, CPU) passed. The mean-field Alfven Mach estimate from one input frame separates the
  MHD_64 regimes perfectly (0/385 snapshots misassigned; Ma 0.7 -> 0.67-0.94, Ma 2 -> 6.9-9.7). The pointwise form was
  undefined because supersonic runs contain negative-density artifacts (up to 4% of cells). The gate's VALID and TEST
  steps wait for Flowers.

- 2026-09-27 11:41: MHD VALID anchor (same metric as Flowers' validate()): the fine-tuned CNextU-net lands at best VALID 0.177 or
  0.202 by seed, in every arm (vanilla 0.1771/0.1770/0.2020). Flowers at lr 5e-4: 0.4750 (step 233), 0.2646 (543),
  0.2453 (657; ~2.8 of 100 epochs, still warming up). The TokaMark matched MLP is in ft_full (epoch 2: val 0.0952,
  against 0.0914 for the register) at ~7 min per epoch, so it should end around 13:15.

- 2026-09-27 11:59: MHD correction: the 0.177/0.202 "seed bimodality" is run length. Four fine-tunes were cut at about 7.5k of 20k
  steps (known since Sep 22). The app:sota regime sentence now uses only the full-length seeds (commit 3ef2aa6). Gemini's
  claim that Flowers "converges faster per step" is withdrawn: the step sizes differ (32 vs 8 samples), and at equal
  samples both sit near VALID 0.245. The VALID anchor for Flowers is 0.1770.

- 2026-09-27 12:23: TokaMark pair (FT-Base, full recipe, seed 1337): matched 3-D MLP 0.1500, vanilla 0.1504, 3-D register 0.1512.
  The MLP is lower on 13 of 15 signals. Descriptive, one seed. Fixed the Figure 2 TokaMark row builder (one shot's
  windows over time, NaN-aware for missing EFIT values), added the row (shot 20895: MLP 0.43, register 0.55, classical
  0.56), and put the pair in the TokaMark appendix paragraph. Main text still ends on page 9.

- 2026-09-27 13:11: Flowers MHD (lr 5e-4) chunk 4 ended at step 1193, best VALID 0.2069. With the GPU to itself it runs at 6.95 s per
  step (351 steps in 2,439 s), against 10-12 s while sharing, so 100 epochs now project to about Sep 29 morning
  (including ~3% restart overhead), not Sep 30 14:05.

- 2026-09-27 14:41: Flowers MHD chunk 6 ended at step 1857 with an end-of-chunk VALID of 0.2051, above the running best of 0.1870
  (step 1779). This is the first upward jump, at the peak learning rate (5e-4) with no gradient clipping, as in the
  authors' recipe. Best-VALID selection keeps 0.1870. The run continues unchanged; a repeat would be worth a look at
  whether training is unstable.

- 2026-09-27 15:11: Flowers MHD (lr 5e-4, from scratch) chunk 7 ended at step 2208 (epoch 9.6; 70,656 samples) with best VALID 0.1756.
  That is below our full-length fine-tuned CNextU-net's best VALID of 0.1770 (20,000 steps at batch 8 = 160,000 samples,
  from a pretrained checkpoint), with ~90 of 100 epochs left. The step-1857 jump (0.2051) was transient: 0.1943 at 1991,
  0.1850 at 2130, then 0.1756. This is VALID on our stride-2 windows; TEST against 0.1160 is scored once, after training.

- 2026-09-27 17:41: Flowers MHD: chunk 10 ended at step 3252 (epoch 14.1) with best VALID 0.1625 (0.1690 -> 0.1649 -> 0.1625), 8.2%
  below the CNextU-net full-length VALID anchor of 0.1770. There was a ~25 min memory stall at 16:30-16:55 (VS Code
  servers hold ~22 GB), and chunk 11 started without delay.

- 2026-09-27 19:41: Flowers MHD chunk 13 ended at step 4299 (epoch 18.6) with best VALID 0.1540 (0.1617 -> 0.1600 -> 0.1580 ->
  0.1563 -> 0.1540), 13% below the CNextU-net full-length VALID anchor of 0.1770. No stalls since 16:55.

- 2026-09-27 21:41: Flowers MHD chunk 16 ended at step 5344 (epoch 23.1) with best VALID 0.1506, 15% below the CNextU-net
  full-length VALID anchor of 0.1770. Noisy checks at the peak rate (0.1617 at step 4919) never reached the 10% rule.
- 2026-09-27 22:45 §37 (registered 22:43, 6cfac22): the user asked for investigations of the lagging rows and strategies tried on stratified samples. Five investigators started: MHD_64 (Flowers checkpoint vs CNextU-net per Alfven regime, error spectrum, TTA/blend), Fusion Equilibrium (S decomposition, GBDT/stacking on CPU), rayleigh_benard single model (per Ra/Pr/onset stratum, what the ensemble fixes), TokaMark (per shot/signal group, Scratch-only ensembling), routing (published SoTA columns for RouteFinder/RRNCO). One GPU job at a time via runs/strat/gpu.lock, QW_GPU_GB<=12; TEST never loaded. Flowers §34 at VALID 0.1481 (epoch 25).
- 2026-09-27 23:38 §37 (Claude alone; the user asked for no subagents): findings recorded in doc 35 (c3be4f3, 3d8a5f3).
  TokaMark: Scratch-only 3-neck ensemble S3, VALID 0.1353 vs single 0.1398 (-3.2%) and E4 0.1405, so E4 was diluted by
  FT-Base. Registered §37a: S3 on TEST once (queued). RB: x-symmetry averaging tta8 -9.7% on stratified VALID (single
  0.0426 -> 0.0384, ensemble 0.0404 -> 0.0367; x periodic, y walls confirmed from the HDF5 attrs). Registered §37b: full
  TEST once (runs/s37b_runner.sh, 25-min slices). Fusion: 75% of 1-S is shape consistency. Two input errors found:
  extract.py lags are ms/1000 (sub-ms, no temporal context), and the challenge's shared Ip time vector is wrong on
  20 kHz shots (|Ip| < 5% of peak at 57-100% of EFIT frames). §37c matched retrain registered (runs/s37c_runner.sh).
  Routing SoTA checked against the PDFs: EAL-MD 8.58% (ours 8.623%); RRNCO row best = RADAR 2.61/2.62/2.59% (ours
  3.47/3.57/3.22%). RADAR code and checkpoint are public. Stopped: three CPU neural retrains (GPU-only rule) and the MHD SWA probe.
- 2026-09-28 00:27 §37a TokaMark S3 (Scratch vanilla + MLP neck + register neck, equal weight, seed 1337) TEST 0.13343: -3.2% vs single
  Scratch 0.13787 (829/1,020 shots), -4.7% vs E4 0.1400, 4.7% under the published 0.140. Post hoc (after E4), labelled as such.
  §37 MHD stratified VALID: Flowers 0.1481 (Ma 0.7: 0.105, Ma 2: 0.191) vs CNextU-net 0.1723; 78% of the error at |k| >= 16;
  single-pass physics projections (div-free B + conserved means) -2.9%; 30-pass averaging -14.3%; the octant neck is dead
  (|k| <= 3 holds ~3% of the error). Routing: decode precision has no effect; sampling x4 -2.3% route cost.
- 2026-09-28 01:01 §37e scorer smoke (2 VALID windows, Flowers step 6276): raw 0.1059/0.1040, dfc 0.1028/0.1010 (-2.9%), shg8_dfc 0.0845/0.0837, all30_dfc 0.0825/0.0817; peak 1.48 GiB. Ready for the post-§34 TEST pass.
- 2026-09-28 01:58 §37a arm B (TRAIN-fitted six-network blend) TEST 0.13320: vs S3 -0.17% [-0.41, +0.07] (not resolved), vs the single run -3.4%. S3 stays the row's number.
- 2026-09-28 03:49 §37c Fusion matched retrain: the input corrections do not raise official S (iplag vs old -0.0006/-0.0005, CIs span 0; ip alone -0.0017/-0.0011, significant). Heads and psi R^2 improve, flux-derived consistency worsens. Rule failed; no ensemble retrain. Paper: lag unit error stated; app:sota gets the Fusion diagnosis.
- 2026-09-28 04:54 §37f Fusion: indicator and lag-floor arms all lower official S (e.g. iplagk -0.0017/-0.0025, iplagk50 -0.0045/-0.0015); rule failed; the marker and pre-breakdown readings are unsupported. Fusion input track closed; the entry stands.
- 2026-09-28 07:57 §37d routing TEST: sampling x4 decode gives register 6.031% / matched MLP 6.142% (greedy 8.623/8.774); register - MLP -0.11 pp [-0.13, -0.09], 8/16 variants, one policy per arm. Harness passed (<= 0.008 pp). Daggered rows in app:routing.
- 2026-09-28 08:00 §37g registered (post-§34): routing TW/non-TW partition test with seeds 2-3; app:routing gets one post hoc sentence.
- 2026-09-28 13:47 §37b RB TEST: single model 0.0425 (behind PDE-FM 0.0415 by 2.3%); 8-pass x-symmetry averaging 0.0383 single / 0.0366 ensemble (daggered; -9.8% / -9.1%, CIs within +-0.8 pp); check passed. Paper updated.
- 2026-10-01 04:47 §37h MHD two-backbone blend TEST 0.11686 (check exact); 0.7% behind EddyFormer 0.1160. Daggered shg8 blend 0.10426. §37i fine-tune running.
- 2026-10-01 08:11 §37k MHD: two-backbone blend with the fine-tuned Flowers TEST 0.11544 (EddyFormer 0.1160); called parity. Single model 0.1172. Paper updated.
- 2026-10-01 11:00 §37g failed (TW partition did not replicate; register = MLP over 3 seeds); §37l RouteFinder fp32 vanilla 3-seed mean 8.650 vs 8.58: parity. app:routing corrected. User: long jobs only when certain; RADAR retrain and 2nd Flowers seed not queued.
- 2026-10-01 11:29 §37m Fusion O-point loss FAILED (w 0.1: -0.010/-0.008 S; w 1: -0.24/-0.04); flattening at the axis breaks O-point extraction. Fusion stays behind.
- 2026-10-02T00:18:27+03:00 RB reporting per Gemini 001727: intro drops 0.0366† (now 'best single model 0.0425'); app:sota adds register-vs-MLP single-checkpoint gap -3.1% -> -0.8% under 8-pass TTA (seed 1337, not a replication).
- 2026-10-02T00:32:06+03:00 Fusion diagnosis (VALID sample, entry): PCA-64 is not the cap (oracle S 0.9991); consistency loss heavy-tailed (top 1% frame-scalar residuals = 38%), concentrated at shot ends (last frame = 0.46% of frames, 7.1% of total loss; last 5 = 14.5%); inputs not stale there. Direct z_axis head R2 0.9909 vs scorer Z_axis from psi 0.9781 (R 0.9763 vs 0.9710; li equal).
- 2026-10-02T00:32:06+03:00 §37n VALID: core shift toward axis heads (inference only) RZ-head +0.00048 [+0.00033, +0.00063], Z-head +0.00033, RZ-avg +0.00039; only R/Z_axis terms move. TEST once running. §37o (li interior warp) registered and scoring.
- 2026-10-02T00:37:23+03:00 §37n TEST once: axis shift RZ-head 0.991017 -> 0.991553 (+0.00054; Z_axis 0.978 -> 0.9915, R_axis 0.974 -> 0.980); replicates VALID.
- 2026-10-02T01:11:46+03:00 §37p: dedicated shape heads beat the scorer's psi-derived shape (kappa R2 0.965 vs 0.933, volume 0.987 vs 0.966, LCFS radius RMS 7.1 vs 9.9 mm). §37q VALID: radial boundary warp to head LCFS + li warp (Q-bl3) +0.00133 over §37o; entry 0.99144 -> 0.99343 on the sample. float16 storage costs -0.00038 (fix: float32 file).
- 2026-10-02T01:12:20+03:00 §37o TEST once: +li-avg 0.991587 (+0.000034 over axis shift; VALID said +0.00018): weak replication.
- 2026-10-02T01:34:35+03:00 §37r FAILED: dedicated q95/betaN heads -0.00016 (avg with trunk -0.00003); not applied.
- 2026-10-02T02:00:26+03:00 §37q TEST once: boundary warp to shape-head LCFS (+li warp) 0.991587 -> 0.992511 (+0.00092); entry 0.991017 -> 0.992511 (+0.0015). kappa 0.931 -> 0.945, volume 0.968 -> 0.982, D_LCFS 0.0146 -> 0.0130.
- 2026-10-02T02:21:32+03:00 §37s FAILED: 64-ray heads on all TRAIN frames (unstable training, 2/3 seeds) -0.00129 vs Q-bl3 on the sample.
- 2026-10-02T02:33:23+03:00 §37t FAILED (-0.00373; one diverged head seed in the mean). §37u grad clipping: 3/3 seeds stable; scoring.
- 2026-10-02T02:45:38+03:00 §37u: 64-ray clipped heads = v1 quality, no sample gain (-0.00006); head accuracy binds, not ray count.
- 2026-10-02T02:46:12+03:00 Fusion diagnostic (VALID sample look, no arm): temporal smoothing of head outputs does not help (sigma 0.5 ~ equal, sigma >= 1 worse); head error lag-1 autocorr 0.47 (bias-like).
- 2026-10-02T04:03:17+03:00 §37v TEST once: v5 heads (clipped, all TRAIN frames) 0.992511 -> 0.992912 (+0.00040); entry +0.0019.
- 2026-10-02T05:03:25+03:00 §37z closed at its gate: 60-epoch heads beat v5 on 6/7 scalars but volume -0.00006 (rule needed kappa, volume, li all up).
- 2026-10-02T05:19:33+03:00 §37w TEST once: elongation to kappa head 0.992912 -> 0.993232 (+0.00032); entry 0.991017 -> 0.993232 (+0.0022).
- 2026-10-02T05:40:19+03:00 §37aa closed at its gate (scalar-weighted heads: kappa +0.0025, li +0.0034, volume -0.00007). §37ab registered (60 ep + scalar weight, judged on S).
- 2026-10-02T07:02:11+03:00 §37y TEST once: 6-seed heads 0.993232 -> 0.993288 (+0.000056); replicates VALID (+0.00006).
- 2026-10-02T08:41:41+03:00 §37ab TEST once: v8 heads (60 ep + scalar weight) 0.993232 -> 0.993385 (+0.00015); entry +0.0024.
- 2026-10-02T09:03:59+03:00 Fusion diagnostic (sample look, no arm): axis target avg(trunk, v8-6s heads) R2 R 0.9763 -> 0.9775, Z 0.9909 -> 0.9914 (~+0.00004 S); not registered (below the cost of a TEST cycle).
- 2026-10-02T10:41:59+03:00 §37ac TEST once: 6-seed v8 heads 0.993385 -> 0.993428. Fusion pipeline frozen: local TEST 0.991017 -> 0.993428.
- 2026-10-02T10:43:13+03:00 §30 adjudicated from saved JSONs: proper-trained noise-conditioned register beats det and the VALID-tuned dressing on 3/3 seeds both rows (shear 0.0062 vs 0.0069; RB 0.0228 vs 0.0235); MLP core equal. Claim survives, not register-specific.
- 2026-10-02T10:45:17+03:00 §35 analysed: RB onset tail, register vs matched MLP delta>0 on 3/5 new seeds (mean -0.005) -> no tail edge; register beats mean-field twin 5/5 (0.55-0.85 on onset).
- 2026-10-02T10:47:34+03:00 §32 recorded: RB routed row on full TEST (classical 0.0456, MLP 0.0435, 30/50/100 qubits 0.0430/0.0425/0.0418); Table 1 switched, stratified to appendix. §30 main-text wording made unambiguous (Gemini).
- 2026-10-02T11:07:51+03:00 §23 MHD pair TEST (seed 1337): 3-D register 0.17707 = 3-D MLP 0.17708 = classical 0.17708; seeds 1338-9 not queued (tie; long).
- 2026-10-02T11:50:29+03:00 Fusion leaderboard: float32 submission (entry + §37n/q/w/li pipeline, v8 6-seed heads; 874 public DIII-D shots, 193,359 frames, local checks pass) uploading to Codabench 17739 Development (phase 29580). External evaluation only; not used for selection; not in the paper (double-blind).
- 2026-10-02T11:59:01+03:00 Fusion leaderboard 956442: public DIII-D S 0.9934 (was 0.9905); matches local TEST 0.99343 (old gap mostly float16); SoTA 0.9959 (gap 0.0025).
- 2026-10-03T03:56 Fusion GPU path: twin 6.8x faster (bit-identical), exact spline warps; GPU-built psi = official 0.99431 on sample (CPU 0.99431); weakness map (c0 60% of psi error, slow biases); §38e head screen registered
- 2026-10-03T04:16 §38f warp caps (GPU screen): best LK +0.00006 [-0.000004, +0.00016], fails CI rule; no TEST
- 2026-10-03T04:37 §38e PASS: H1 head (4096x3) 1 seed mean R2 0.98415 vs v8 s1 0.98286 (= 6-seed v8); §38g registered (corrected Ip/lag inputs under head pipeline; q/b zero-training check iplag 0.99702 vs entry 0.99649); Ip-fixed TEST+public inputs built; gpu_build.py TEST repro 0.993392
- 2026-10-03T04:51 §38d TEST PASS (GPU path): psi-specialist ensemble + post S 0.993691 vs base 0.993392 (+0.00030); new best local TEST
- 2026-10-03T05:25 paper: Fig. 1 replaced by a layer-level architecture diagram with tensor sizes (fig_arch.py: CNextU-net stages, pooled neck register, gate MLP, routed co-processor with MLP bars and gate-level Rydberg circuit); new Appendix B (fig_arch_backbones.py: FNO, CNextU-net rows, 1HP U-Net, UNetLite, TokaMind, TORAX 1-D, Fusion MLP trunk; shapes traced by arch_dump.py); main text still 9 pages
- 2026-10-03T05:37 Fusion leaderboard 957750 (psi specialists + post): public S 0.99353 (was 0.99341; local TEST said 0.99369; consistency gain did not transfer), rank 16
- 2026-10-03T05:37 §38g G1 PASS: corrected-input 4096x3 head mean R2 0.98603 (H1 cached 0.98415, v8 0.98286; 1 seed > 6-seed v8); rays 6.11 mm; 6-seed rebuild on corrected inputs queued
- 2026-10-03T05:39 §38g QF PASS: corrected-input trunk r2_qb 0.99627 (cached twin 0.99513, entry 0.99644); 50/50 with entry +0.00006 [+0.00002,+0.00010]; QF seeds 1,3 queued
- 2026-10-03T05:52 §38i PASS: psi specialist on corrected inputs r2_psi 0.999767 vs 0.999695 (-24% psi error); seeds 1,3 queued
- 2026-10-03T06:03 §38h TEST PASS: extreme-point warp X3 +0.000053 (tri +0.001/+0.002, D_LCFS +0.00035)
- 2026-10-03T06:12 paper Fig. 1 redesigned (user feedback): coloured title band + role line + drawn component for every module (CNN parts, neck insertion, gate inputs/octave split/MLP/mix, POD/router/config/register/write-back/classical head/output); caption shortened; 9 widow-line trims; main text still ends page 9
- 2026-10-03T06:27 paper Table 1: new 'Our best' column (best classical lift vs published; beaten: shear 0.0198, TRL2D 0.1909e, TokaMark 0.1334e; parity RB 0.0403e, MHD 0.1154e; fusion 0.9937 below 0.9959); app:sota fusion TEST 0.9937 + chain step 0.99369; figures regenerated
- 2026-10-03T06:51 paper Fig. 2 (user): all 12 benchmarks (incl. routing, TDC Caco-2), columns GT/classical/+MLP/+30/+50 qubits/error; TDC 50-atom variant (mr50ryd) + paper/tdc/fig2_row.py; routed 30/50 renders for the 4 Well rows in progress (adrc.py QW_DUMP_WINDOW, render_routed.py with basis check)
- 2026-10-03T07:00 §38g QF 3 seeds: r2_qb 0.99669 alone, 0.99691 50/50 with entry (+0.00007 [+0.00003,+0.00011]); combined pipeline uses qf50
- 2026-10-03T07:25 paper Table 1: all 12 benchmarks (bottom block: TRL2D rollout 13:30, RealPDE, RouteFinder, RRNCO ACVRP, Caco-2; RouteFinder best = 8.65 parity, register comparison one seed); Fig. 2: TRL2D rollout row (paper/phase_b/fig2_rollout_row.py, start-0 rollouts match multistart exactly), arena row dropped; 9 pages kept. §38x combined fusion pipeline registered (runs/s38x_runner.sh).
- 2026-10-03T07:49 §38m xp mechanism (VALID sample): lateral support and direction rules pass (R_top -0.00164, R_bot -0.00216, Z ~0; xp beats radial controls by +0.0005 S); xp gain is consistency only and costs D_LCFS (+0.00031); post-xp R error below both contour and head errors (blend effect).
- 2026-10-03T08:25 §38n: 6 xp arms (blend w, sigma_z) vs X3 all fail (deltas -0.00002..0.00000); sigma_z cuts D_LCFS cost by a third but loses li or tri; xp lever flat at X3.
- 2026-10-03T09:16 §38x X6 combined Fusion pipeline: VALID official +0.00080 [+0.00052, +0.00110]; TEST S 0.994904 (was 0.993691); public upload started. Table 1 fusion best 0.9949. §39 second machine (mephiles) set up: RouteFinder vanilla seeds 1-3 (anchor first) and Caco-2 CaliciBoost audit; CaliciBoost reproduces 0.2560 only with gpu_hist (CPU hist 0.2710). RouteFinder Table 1 row corrected to 3-seed register/MLP means.
- 2026-10-03T09:44 Fusion X6 public S 0.994542 (rank 15/84; gap to #1 0.0018, consistency 0.0011); local-vs-public gap = consistency. §39a RouteFinder anchor passes on mephiles (8.658 vs 8.687). §39b CaliciBoost reproduces (0.256) only with released per-seed subsets + gpu_hist; leakage-free 0.308, no selection 0.294; paper app:caco2. Learn2Design-2026 started on merphistophline (Strix Halo, ROCm JAX 0.9 evals/s vs H100 ~10/s; complex128 simulator); our Codabench token is invalid on the L2D server.
- 2026-10-03T10:12 §39a RouteFinder vanilla 3 seeds 8.630 (beats register 8.670 / MLP 8.678); Table 1 corrected. §40 L2D transfer probe run 1: transferred designs best 1.28 vs random 5.64.
- 2026-10-03T11:43 §40 L2D: label rule exact 40/40; warm start beats cold at 2,500 evals on run 1 (0.499 vs 1.76); transfer probe helps on 5/6 topologies; draft submission validated (static); end-to-end run 1 (8k evals) running on mephiles.
- 2026-10-03T14:48 §38p X7 (6-seed J1 heads): TEST 0.994934 (+0.00003 over X6); uploaded. L2D v2 periodic-sigmoid stalled at the pool best (debugging).
- 2026-10-03T16:09 §38t shot-bias lin arm: no gain (in-sample TRAIN residuals); L2D v1 run 8 0.433 at 3k evals.
- 2026-10-03T17:16 §38t reg arm (seed 1): 0.98712 -> 0.98712, no gain (in-sample residuals; §38u out-of-fold next).
- 2026-10-03T17:29 §39c RouteFinder routed hybrid route30 fp32 3 seeds mean 8.623 (vanilla fp32 8.65; published 8.58); seeds 4-5 queued.
- 2026-10-03T18:09 §40g hybrid TopoGen generator fails (worse than transfer on 10/10); register's L2D role moves to donor ranking.
- 2026-10-03T18:30 X8 sample 0.99518 (TEST running); RF routed hybrid 5-seed mean 8.646; L2D initial model uploaded (submission 411, predicted ~0.45).
- 2026-10-03T19:12 Fusion X8 public 0.994588 (rank 15, gap 0.0017); q/b heads transferred as predicted.
- 2026-10-03T21:19 SoTA-gap investigation (user request): doc 36 (36-sota-gaps-diagnosis.md) - per-row diagnosis and fixes; new probes: RouteFinder per-variant (parity; hybrid better on all TW variants, worse on duration-limit ones), Fusion public-shift and Thomson-geometry (negative), RRNCO local-search route found; §41 RRNCO pilot hybrid worse at lr 1e-3 (retry 1e-4); RF bf16 routed 3 seeds 8.596; mephiles back (seeds 4-5 queued).
- 2026-10-03T23:07 RRNCO: hybrid (register only, frozen RRNet) 3 seeds 3.71 vs 3.49 (worse); +test-time PyVRP local search 2.41/2.61/2.07% vs RADAR 2.61/2.62/2.59 (extra compute, marked in Table 1). Fusion: offset learnability (§38w-pre) and q/b smoothing negative - input-side levers exhausted. RouteFinder bf16 5 seeds: hybrid 8.614 vs classical 8.654; §39d TW-gated hybrid running. L2D §40i: lower lr breaks the plateau (0.526 -> 0.471 at lr 0.01).
- 2026-10-04T01:50 RouteFinder: §39d TW-gated hybrid fails (8.729); §39e sampling decodes 5 seeds hybrid 5.963 vs classical 6.022 (3/5); Table 1 RouteFinder row -> routed register 5 seeds (8.61 vs 8.65, our best 6.02# with test-time sampling). L2D: v2 (lr decay) fails 0.546; v2b froze at 1.043 (diagnostic with traces on mephiles); fastsim fails (FP32 too inaccurate). Fusion: offset learnability and q/b smoothing negative.
- 2026-10-04T06:22 L2D: donor scan (10 retired topologies) -> pool width 512 (mean start 2.09 -> 0.945); v2b passes on CUDA; v3b (adaptive 64->512 pool + step-size restarts) passes (mean 0.465 vs 0.478 over runs 1,3,4,8; run 3 2.08 -> 0.424); CUDA validator 0.433, 715/715 finite; uploaded as submission 413 (replaces 411); predicted official mean ~0.38 (0.28-0.50). ROCm results on merphistophline unreliable (shared with Steam): screens on CUDA. RRNCO safe hybrid fails gate.

## Oct 04 15:36 — Fusion consistency: realise the heads (HJl9), TEST +0.00047

- Deep dive on the consistency gap (doc 35 §38aa): the PCA basis is not the cap (k=64 ceiling 0.9977); the axis shift aimed
  at the trunk axis, not the stronger J1 heads; kappa, tri and D_LCFS lose at the X-point tips where the scorer's grid
  quantisation makes the extracted tip jump by a row or column.
- HJl9 = head-aimed axis + joint Gauss-Newton shape fit (paper/strat/fusion/joint_warp.py) + li cap 0.9: sample +0.00044
  [+0.00030, +0.00058] vs X8; TEST 0.995439 vs 0.994967.
- Notebook 20 (diagram-first, ASD-STE100): competitions + geothermal hybrid, with HTML export.
- Oct 04 16:20: HJl9 public 0.994909 (+0.00032 vs X8), rank 13; consistency gap to #1 0.0011 -> 0.0008.
- Oct 04 17:10: L2D §40t cold run 1 (organizers' NAAdamGD, random start) 0.200 at 10k evaluations vs v3b 0.46 (flat);
  ~4th of 60 in the Round 2 field at matched evaluations. Runs 3, 4, 6 running. (Correction: 0.36 is #1's mean at 15 min,
  not the field median; the field median at 15 min is 1.67 on topology 1.)
- Oct 04 20:30: Learn2Design v6 (§40u: v3b warm phase for 15k evaluations, then cold noisy-Adam draws of 10k for the
  rest of the 4 h, best over all) passes the official validator; uploaded as submission 420 (predicted ~0.34 vs 413's ~0.38).
- Oct 05 22:35: QPI ensemble finding - single hybrid heads (J1 + engaged 30-atom register +/- volume physics) do
  not beat J1, but 6 x J1 + 2 hybrid heads = VALID mean R2 0.98729 vs 0.98711 (li +0.0007, rays 5.88 mm); HJQ8 sample gate
  running. L2D faithful v6 on topology 3 matches the leader at 15 min (0.42) but stalls at 0.39; warm-long test running.
