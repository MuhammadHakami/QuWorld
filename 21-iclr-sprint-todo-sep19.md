---
title: ICLR 2027 sprint — benchmark list + big TODO (re-baselined Sep 19 2026)
date: 2026-09-19
project: quworld
tags: [todo, benchmarks, iclr, sprint]
status: LIVING — tick items here; numbers go to 15-challenge-ladder.md
---

# Priors, re-baselined (Sep 19, 12:40)
- Paper = QUOTA (ICLR 2027, due **Sep 25 AoE**, primary area *foundation or frontier models*): a few configurable neutral-atom qubits take the analog part of a physical signal inside a pretrained world model; pre-registered, matched controls, no advantage claim.
- Evidence in hand: TRL2D official campaign (headline rollout gain), sprint ladder (flat from 5q), mechanism (inference-inert; interactions matter in training; Gutzwiller ambiguous), MHD 3×3 (null). **Two of ten industrial benchmarks reported.**
- **Standing goal (user, Sep 19 13:00): results on the 10+ benchmarks above all else; the paper comes later. GPU only (no CPU training/inference). For every benchmark: clone the SoTA/baseline repo, start from its pretrained or official model and protocol, never re-implement the classical part — only the quantum register is ours. Always cross-compare with the SoTA under its own protocol.**
- **Needs user:** an `HF_TOKEN` (unauthenticated HF API is rate-limited: 500 req/5 min — Sophelio's 9,123-file download and the judge download hit it); root GPU clock check.
- **Research pointer (Sep 19 17:10):** `22-sota-research-sep19.md` = full SoTA sweep (56 new PDFs, 4 repos); competitor changes folded into the ladder (Physics-IQ 45.5, TokaMind 0.150, Sophelio S≥0.85 eligibility, Walrus/Pfeffer on RB, QCNN on Pelzer).
- Compute reality: one GB10, GPU pinned at 650 MHz / 15 W (~25 TFLOP/s; root check pending — would cut everything ~4×). Phase C teacher cache is resident (~63 GB reserved); one small GPU co-runner at a time next to it (co-running slows the teacher ~2.4×; total throughput is conserved). Memory gate: MemAvailable ≥ 25 GB before any new GPU job (earlyoom at 6 GB).
- Team reality: main thread only (no subagents). Concurrent sessions on the same worktree (Sep 19 16:20): quworld-e3 owns the Well chain, Pelzer U-Net runner, TokaMark lane, Phase-C chain and these docs; the peer that relaunched `runs/sophelio_runner.sh` (spawn scorer workers) owns the Sophelio lane and `runs/memguard.sh`. One GPU co-runner next to the teacher; message before touching another session's runner.
- **Session sprawl (Sep 19 17:40):** every context restart spawns a new Claude Code session (quworld-e3 → 10 → b3) and each leaves 1–4.5 GB of VS Code server processes resident (≈16 GB total, 7 sessions listed); with the teacher at ~110 GB this is what drove MemAvailable to 6 GB and cost two Pelzer arms. Needs user: close stale sessions (quworld-10, -bd, -e0, -80 if unused). Lane ownership now: quworld-b3 = Pelzer redo, TokaMark CNN + TokaMind lanes, Well chain (inherited from e3), Phase-C/WMB chains (detached, inherited); quworld-34 = Sophelio, WMB judging, memguard.

# The ten benchmarks (status → what closes the gap)
| # | Benchmark | Domain | Data | Harness | Result | Gap-closer (owner, compute) |
|---|---|---|---|---|---|---|
| 1 | The Well TRL2D (official) | turbulence | ✓ 7.4 GB | ✓ | ✓ headline + sliding-start | none (done) |
| 2 | The Well MHD_64 | fusion (3-D MHD) | ✓ 76 GB | ✓ | ✓ 3×3 null | none (done) |
| 3 | TORAX-Bench v1 (ours) | fusion transport | ✓ 47 MB, 800/100/100 | ✓ `train_torax.py` (GPU) | ✓ 3×3: vanilla 0.0321 / MLP 0.0295 / quantum 0.0292 (adapters beat trunk; q≈MLP) | none (done; macros filled) |
| 4 | Pelzer 1HP heat plume | geothermal | ✓ 100 prepared + **1000 prepared with 1HP_NN's pipeline (gksi)** | ✓ FNO track + ✓ **1HP_NN U-Net track** (`train_pelzer_unet.py`) | 100-dp FNO 3×3 ✓ (null) | U-Net pretrain sweep → 3 seeds → arms (GPU, queued behind TORAX); compare to Pelzer & Schulte 0.046 °C |
| 5 | Physics-IQ | video physics (i2v) | ✓ 198 frames + 6 GB GT | ✓ v2 | running | GPU chain (vanilla → quantum → LoRA) |
| 6 | WorldModelBench | video physics | repo cloned (350 instances + images) | judge = VILA ewm-qwen2-1.5b (needs VILA install) | pending | after Physics-IQ arms: 60-instance pre-registered subset × 3 arms (~5.5 h/arm), judge on GPU |
| 7 | TokaMark task_1-1 | fusion (real MAST) | 11,573 shots downloading (1.3k local) | ✓ their CNN baseline + `--variant` register (smoke end-to-end) | pending (SoTA: CNN 0.2091 TaskNRMSE, random split) | `runs/tokamark_runner.sh` queued behind the download |
| 8 | Sophelio equilibrium | fusion equilibrium | **downloading (public HF, 103.9 GB, started Sep 19 13:30)** | starter kit cloned + local scorer | pending | starter-kit baseline as trunk + register; dev deadline Oct 18 |
| 9 | rayleigh_benard | fission proxy (molten-salt convection) | ✓ all splits local (370 GB) | ✓ `well2d.py` (bit-exact vs WellDataset) + `well2d_backbone.py` (CNextU-net, proven path) | ✓ anchor reproduced: CNextU-net 0.6679 on the full test split (published 0.6699; TFNO 0.6566) | arms × 3 seeds after the Pelzer track (chain v2) |
| 10 | **shear_flow** (replaces planetswe) | 2-D shear-layer turbulence | ✓ all splits local (439 GB; 112 test traj × 200 steps, 256×512, 4 ch) | ✓ loader bit-exact vs WellDataset; CNextU-net = published SoTA 0.8080, sane on sample windows (0.044 vs persistence 0.037) | pending | anchor eval → arms × 3 seeds after RB. **planetswe dropped**: the HF data (re-uploaded 2025-03-20) matches none of the released checkpoints (CNextU-net/FNO/TFNO all give VRMSE 5–17 vs 0.09–0.37 published, old and new stats; persistence there is 0.028, below every published model anyway) |

# Big TODO (deadline-ordered; ✓ when done; numbers → ladder)

## A. GPU lane (Phase C, self-driving; check every event)
- [x] A1 teacher cache 45 traj ✓ (20:39) → A2 P1 adapters ✓ (23:19, `runs/phase_c/p1_results.json`): valid chunk-latent rollout VRMSE vanilla 0.6996 | mr30ryd 0.5838±0.0052 | lora 0.5592±0.0011 | mlp30 0.5631±0.0030 → **H2a FALSE** (quantum loses to both controls: 0/3 seed wins, seed-paired diff +0.0246 vs lora / +0.0207 vs mlp30, per-traj win rate 0.11 / 0.00, Wilcoxon two-sided p 0.0078 / 0.0039). **Confound to state:** units are time-budgeted (420 s) and the register runs 510 ms/step vs 140–160 for the controls → mr30ryd best steps 400–800 vs 1800–2000 (≈¼ of the optimisation); a fixed-step rerun of the mr30ryd units (or the A8 clock fix) is needed before H2a is called. `P1_DONE`/`TEACHER_DONE` written 23:19:53 → Physics-IQ orchestrator gate now waits only for a python-free GPU.
- [ ] A3 Physics-IQ vanilla (≈19 h) → A4 quantum → A5 LoRA (auto-skip after Sep 22 08:00)
- [ ] A6 P2 closed-loop latent rollouts + paired 198-row stats; A7 figures (rows arms × cols benchmarks)
- [ ] A8 (user, root) GPU clock check `sudo nvidia-smi -q -d POWER` / `-lgc 300,2418` — would cut A1–A6 ~4×
## B. GPU queue (benchmark lane; one co-runner next to the teacher, then free GPU)
- [~] **Reboot Sep 19 22:58 (board latch-off class: journal ends 22:54 at 53% MemAvailable, no shutdown record):** every detached runner died. Relaunched 23:11 (VS Code session) in one-co-runner order via `runs/relaunch_sep19_reboot.sh`: P1 resume (`runs/phase_c/p1_resume.sh`, 24/27 units were done) → Pelzer redo (`redo_s1338_s1339.sh`; the 18:21 PELZER_UNET_S1339_DONE was FALSE — memguard killed the s1339 pretrain at 17:47 and the script wrote DONE unconditionally; sentinel now conditional) → Well chain v2 (RB vanilla s1337 resumes from `_latest.pt` at step 5798/6000; shear_flow train split is fully local, 439 GB). TokaMark downloader relaunched (9260/11573 complete; its 8.3 GB log was asyncio traceback spew from an S3 outage → rotated to `download_shots.log.prev`, asyncio/fsspec logging silenced; `tokamark_runner.sh` now requires `done: ok N fail 0`); tokamark/tokamind/FW-A runners and the Physics-IQ + WMB orchestrators re-armed. Peer quworld-2e holds memguard (pid 33654), the Sophelio runner (8 arms, false SOPHELIO_DONE cleared) and the WMB judge chain. **Tailscale is logged out since the reboot (user action: re-login).** Sentinel lesson (both sessions): every runner that writes DONE unconditionally after a killed step produced a false sentinel today (Pelzer, Sophelio) — gate DONE on the result files.
- [x] B1 TORAX-Bench 3×3 on GPU ✓ (13:48 Sep 19; ladder + macros)
- [x] B2 Pelzer 100-dp FNO-trunk arms 3×3 on GPU (null; ladder) · [~] B2b Pelzer SoTA track RUNNING 13:48: 1HP_NN U-Net pretrain sweep → 3 seeds → arms (`runs/pelzer_unet_runner.sh`)
- [~] B3 The Well rayleigh_benard + shear_flow (CNextU-net backbone, main venv; chain v2 `runs/well2d_chain.sh`): anchor eval (full test split) → `finetune_well2d.py` arms {vanilla, mlp30, mr30ryd} × 3 seeds, fixed steps (pre-registered: 6000 steps, batch 4, lr 1e-4, valid subset 8 windows/traj) → `eval_well2d.py` single-shot test → ladder. **Finding (Sep 19 14:50): the neuraloperator FNO/TFNO checkpoints do not reproduce on this stack (TRL2D control: TFNO 5.27 / FNO 2.4–2.7 vs published 0.49/0.41, identical under neuralop 0.3.0 and 2.0.0 with converted weights, era tensorly, CPU=GPU) while CNextU-net-TRL2D reproduces (0.188 on the same windows) → TFNO track parked (`tfno_hybrid.py`, `.venv-well`, `runs/well_dl/neuralop_patch.sh`).**
- [~] B4 TokaMark: `.venv-tokamark` ✓; 11,573 shots downloading from STFC S3 (32 workers, their shuffled split heads first; ~6 h); `paper/tokamark/{train,eval}_hybrid.py` = their run_training/run_evaluation with `--variant` wrapping `encoder_mlp` (smoke on 8 shots trains end-to-end with the register); `runs/tokamark_runner.sh` (task_1-1, their config, seed 23, 3 arms) queued behind the download
- [~] B4b TokaMark SoTA-first track (owner quworld-10, Sep 19 17:25): TokaMind FT-Base = 0.150 on task 1-1 (CNN 0.209) with released weights → `baseline/tokamind` cloned + installed (`runs/tokamark/tokamind_patch.sh`: --no-deps editable, py3.13 lazy-annotation patch, runs symlink), weights local (28 MB); driver `paper/tokamark/tokamind_hybrid.py` = their run_finetune/run_eval with two patched functions (in-memory config: local zarr, seed, cache workers; every modality head wrapped as head(Bottleneck(h_cls)) after their warm-start, same wrap before load_best_weights) — verified identical to TokaMind at init, ~52.6k register params/head; `runs/tokamind_runner.sh` LAUNCHED (waits TOKAMARK_DONE → 20 GB gate → 8-shot smoke → {vanilla,mlp30,mr30ryd}×{1337,1338,1339} finetune+eval, results `runs/tokamark/results/tokamind_task_1-1_*.json`, sentinel TOKAMIND_DONE; memguard kills it first). Target: TaskNRMSE < 0.150.
- [~] B5 Sophelio (quworld-2e lane): data local; seed-1337 trio DONE 23:41 Sep 19: vanilla S 0.7838 / mlp30 0.7973 / mr30ryd 0.7824 (official scorer, 40 held-out shots); seeds 1338/1339 running (`runs/sophelio_runner.sh`, resume-safe); then round-2 shared-trunk lift toward S ≥ 0.85 (--thomson, GS-residual consistency, trunk-native scalar head; identical for all arms); dev submission by Oct 18
- [~] B6 WorldModelBench: judge OPERATIONAL (peer quworld-34: `.venv-vila`, weights local, ~8 s/question on CUDA; ≈1.5 h judging per arm on the 60-subset). **B6b (mine) WIRED: `paper/wmb/generate.py` (Physics-IQ runner pipeline + adapters, subset60, first_frame + text_instruction, mp4 = frame stem; all 60 frames pass the geometry check) + `runs/phase_c/wmb_orchestrator.sh` (waits for the Physics-IQ arm chain + free GPU → vanilla → quantum → lora; sentinels `runs/wmb/GEN_<arm>_DONE`); judging by quworld-34 per arm**
- [ ] B7 ladder + `includes/ladder_macros.tex` refresh after every finished 3-seed block
## C. Writing lane (ICLR draft, peer session owns the file; I hand text/figures)
- [ ] C1 Section 5 "frozen video world model" from A-lane numbers (Sep 22)
- [x] C2 (wired into main_iclr.tex 84885fe; cells fill via includes/ladder_macros.tex) Industrial-ladder table: rows = 10 benchmarks, cols = SoTA / vanilla / MLP / quantum, with "—" where not run
- [ ] C3 Related-work refresh: 2606.09646 (video FMs & intuitive physics), 2502.06916 (quantum-inspired adapters), 2607.07534 (LingBot-World v2 report)
- [x] C4 AI-use statement already in iclr2027/statements.tex (included after refs) · reciprocal-reviewing registration (outside page limit) + reciprocal-reviewing registration (≥1 author) — **user action**
- [ ] C5 Final audits (numbers vs logs, anonymity, 9-page main text) + fields (title/abstract/keywords/primary area) → freeze Sep 24
## D. Post-deadline (ICLR rebuttal / camera-ready)
- [~] FW-A readout families (quworld-05, Sep 19 17:55): code + tests done (`QREADOUT` env, `paper/qlayer.py`); arena runner `runs/fwa_readout_runner.sh` queued behind TOKAMIND_DONE; decision rule = round-3 advance rule (beat both `none` and the re-matched MLP on vrmse@8 across 3 seeds) before any benchmark arm adopts a non-`z` readout.
- [ ] D1 WorldModelBench full 350; D2 Sophelio dev entry (Oct 18); D3 rayleigh_benard/planetswe/BubbleML arms; D4 Cosmos-Predict2.5 transfer; D5 more seeds + Gutzwiller power; D6 hardware micro-run; **D7 register ideas from `22-sota-research-sep19.md` (FW-A…FW-H in the ladder: ANO readout, configuration rule, recurrent state, structured control, Pulser noise row, DSE, Walrus trunk)**



## September 20 continuation

- Pelzer U-Net grid closed and source-audited; quantum 0.035881 ± 0.004608 °C, MLP 0.036152 ± 0.004554, vanilla 0.036363 ± 0.004407. Quantum beats MLP on 1/3 seeds; no attribution claim. Notebook executed and manuscript cells filled.
- Rayleigh–Bénard vanilla s1337 test 0.4303 (34,300 windows); remaining arms running.
- Eleven distinct task definitions and the dynamic-configuration comparison are in `23-dynamic-benchmark-continuation.md`. BubbleML 2.0 source cloned; protocol/checkpoint gate pending. This extends the ten-task table above without counting TokaMark backbones twice.
- Manuscript working PDF rebuilt: nine-page main text, references begin page 10. Remaining benchmark cells and the bibliography rerun warning remain open.

## E. Sep 21 queue (user directives 10:20 / 15:20)
- [x] E1 video rows parked (Physics-IQ, WMB, BubbleML) — resume on a B300 or when the GPU is idle
- [~] E2 Optuna HPO campaign: round 1 ✓ (single-seed selection artefact found) → round 2 two-seed objective ✓ for adapters (Sep 22 15:44) → trunk re-selection `runs/hpo_none_ms_runner.sh`: TORAX ✓ (identical hp, 0.01582), Pelzer (running), Sophelio; TokaMark-CNN / TRL2D / RB / shear_flow not tuned (variance too high to benefit, see 30-…)
- [~] E3 logistics row → **RouteFinder** (SVRPBench dropped Sep 22: train==test byte-identical, 20% infeasible instances): RF-TE reproduced to 0.002% ✓; EAL arms harness `paper/vrp/run_rf_eal.py` (smoke running) → setting md × 3 arms × 3 seeds in `runs/replication_runner.sh` stage 4
- [~] E4 TDC ADMET (drug row): harness ✓ `paper/tdc/` (official protocol verbatim, frozen ChemBERTa-77M trunk, embeddings cached), 22 × 3 × 5 queued `runs/tdc_runner.sh` behind the HPO chain
- [ ] E5 rebuild `notebooks/18_method_and_benchmarks.ipynb` after each finished block (`uv run --no-sync python paper/build_method_notebook.py`)
- [x] E6 TokaMind 3×3 ✓ (Sep 22 02:20): vanilla 0.1509 · MLP 0.1504 · Q 0.1501 vs SoTA 0.150; n.s.
- [ ] E7 Matbench harness (materials row, replaces BubbleML): `matbench` package + ALIGNN/CGCNN pretrained trunk, register on the pooled graph embedding, nested 5-fold protocol; start with perovskites/phonons
- [x] E9 **quantitative judge** `paper/judge/judge.py` → `runs/judge/scorecard.md`; verdict `30-campaign-judgment-sep22.md` (claims 1 and 2 null at n=3; reframed thesis 'drop-in, no regret' B)
- [ ] E10 **pre-registered powered replication** `31-powered-replication-prereg.md` (Pelzer 12 · Sophelio 8 · TokaMind 4 new units; H1 Q vs MLP) — `runs/replication_runner.sh` stages 1–3, after TDC
- [ ] E11 MHD repair: re-run the 4 truncated arms (s1338 mlp30; s1339 vanilla/mlp30/mr30ryd) to 20k steps (resumable from `*_latest.pt`); then test a gate-LR multiplier / shallower insertion (gates |α|≈4e-5 → inert)
- [ ] E12 shear_flow: reproduce the published 0.808 anchor (our evaluator gives 0.9946 for the official checkpoint) or drop the published comparison
- [ ] E13 **Spectral-Routed Register (SRR)** — `32-dynamic-register-configuration.md` §5/§7: pre-register Stage B on TRL2D (K=30, zero-parameter trunk-change router; arms trunk / SRR-MLP same tiles / SRR-register 3-D pyramid / fixed-2-D geometry / Gutzwiller / random tiles; endpoint VRMSE + |k|>=17 error energy); Stage A leftovers: top-K-specific router loss (NDCG@K), calibration + abstain curve
## F. Sep 23 queue — /goal: reach and improve SoTA on every non-B300 row (memory: goal-reach-improve-sota-sep23)
- [~] F1 **Sophelio Ch1 SoTA push** (gap: ours 0.80 vs leaderboard 0.9932). Starter pipeline used 200 of 7,041 shots with a RidgeCV scalar head (R²qb 0.472).
  - **Sep 25 21:10, deferred until after the paper (user):** the live leaderboard best is 0.9959 (consistency 0.986); ours is 0.9910 on a held-out training split (consistency 0.967; κ 0.93, li 0.96). The gap is the flux decoder's boundary geometry, not the PCA ceiling (0.998). The paper table now cites 0.9959 and states the split difference.
  - Done: `paper/sophelio/extract.py` → all 7,041 shots, 1,559,340 frames, 0 failed.
  - Smoke (537 shots, 3 epochs): R²qb 0.897, R²ψ 0.975, Consistency 0.0 (blurry ψ).
  - Running `runs/sophelio_sota_runner.sh`: cache → v1 full-data trunk → **joint classical+quantum NAS** (`paper/nas/sophelio_nas.py`: Optuna 5.0 MO, VALID official S vs atoms×layers, SH pruning, LLM-seeded; couplings none / latent / **srr**).
  - Then: 3-seed TEST on the Pareto picks with matched MLP controls; Consistency diagnostics via r2_cons_each; Green's-function prior for the MAST transfer (Ch2).
- [ ] F2 **TDC ADMET → CheMeleon trunk** (the frozen ChemBERTa run was stopped before starting).
  - Reference = the reproducible leaders per the 2026 JCIM audit: MapLight, MapLight+GNN, CaliciBoost (MiniMol / XGBoost / GradientBoost leak).
  - Chemprop ≥ 2.2 is installed in `.venv-tdc`; CheMeleon weights from Zenodo 15460715 (`BondMessagePassing(**hyper_parameters)` + `load_state_dict`).
  - Next: `paper/tdc/run_chemeleon.py` — fine-tune per official seed, register / MLP bottleneck between aggregation and FFN, `evaluate_many`.
- [~] F3 **RealPDE (user, Sep 23): paper benchmark, NOT a competition entry** (user: ignore their rules; CC BY-NC licence kept).
  - Protocol = RealPDEBench (ICLR 2026) foil public splits: 78 remain / 10 in-dist / 10 out-dist test, 20→20 steps at 32×64.
  - Target (their Table 1, real-world finetuning): **Rel L2 0.0145 (U-Net)**, RMSE 0.0094; real-world training 0.0159.
  - **User (Sep 23, second message): ignore their rules incl. weights — use the competition sim_real checkpoints as OUR classical trunk; continue classical+quantum.**
  - **Overlap check DONE:** the competition train_real (82 trajectories) contains 2 of the 20 RealPDEBench test trajectories, (3750, 10°) in-dist and (3750, 0°) out-dist. Report on the **18 clean test trajectories**: in-dist Re 5312 / 6093 / 7656 / 10000 / 10781 / 11562 / 12343 / 16250 / 17031 @ 10°; out-dist (5312, 0), (5312, 20), (9218, 0), (9218, 20), (11562, 0), (12343, 0), (14687, 20), (15468, 0), (17812, 15). Also report all 20, flagged.
  - Checkpoints (`data/realpde/baseline_checkpoints/sim_real_ft/`): FNO 3-D spatio-temporal (fc0 6→64, spectral modes 4×12×16 over t,h,w, 50.4M = paper FNO); CNO 8.0M (3-D convs); Transolver 12.5M (width 256, 3 blocks). Load via `baseline/RealPDEBench/realpdebench/model/*` with hyper-parameters inferred from state_dict shapes. Metrics: `realpdebench/utils/metrics.eval_metrics` (Rel L2, RMSE, fRMSE bands, KE, FE).
  - Superseded note, kept for the record — leakage check was PENDING: the competition's example `3750_0.h5` is a RealPDEBench out-dist test trajectory, so competition checkpoints / train_real may contain test trajectories. Verify against `data/realpdebench/foil/*_test_params_real.json` before using any of them.
  - Downloads running: competition release → `data/realpde`; RealPDEBench foil real shards → `data/realpdebench`.
  - Next: reproduce U-Net 0.0145 with their code (`baseline/RealPDEBench`), then arms + NAS + SRR (their metrics include high-frequency error bands) + conformal intervals.
- [~] F4 **RouteFinder EAL** re-queued (`runs/rf_eal_runner.sh`, after HPO_MS2_NONE_DONE).
- [ ] F5 MHD anchor: the official checkpoint scores 0.5447 on our evaluator vs published 0.1633 (the log mislabels it 0.1956). Find the evaluator/normalisation mismatch; re-run the 4 truncated arms.
- [x] F6 Replication deferred before any unit ran (`31-…` addendum).
- **Sep 23, 02:26 and 02:43: two OOM events (reboot, then tmux / user-systemd killed); runs restarted (details in `history.md`).**
  - Cause, from the kernel OOM dump: my RealPDE smoke test held ~100 GB of unified GPU memory.
  - Claude-launched jobs inherit `oom_score_adj -1000`, so the kernel killed desktop processes instead of the job.
  - Fix: `runs/capped.sh` (OOM priority +800, plus a cgroup cap when systemd --user is up), `QW_GPU_GB` → `paper/gpucap.py`, and `runs/memguard.sh 14`. The guard has since correctly killed one CatBoost job that had reserved 95% of free GPU memory.
- **F1 update:**
  - Bug found: float32 feature standardisation overflowed on Thomson n_e / p_e (~1e20). **84 features were silently zeroed in v1.** Fixed with `moments64`.
  - NAS restarted with the fix. Seeds are now the v1 trunk ± coupling, and stale trials are resumed. The obsolete 200-shot `sophelio none ms2` HPO was stopped.
- **F2 update:**
  - CheMeleon vanilla (5 seeds): caco2 MAE 0.343, hia AUROC 0.964, pgp 0.933, bioavailability 0.609. All are below the reproducible leaders.
  - The classical lift is **MapLight on GPU CatBoost** (`paper/tdc/run_maplight.py`, exact features and params; RDKit-2026 fingerprint shim; `gpu_ram_part` 0.05).
  - Next: `ensemble.py` = MapLight + CheMeleon {vanilla, mlp30, mr30ryd} averaged per seed. The matched register-vs-MLP comparison is kept inside the ensemble.
- **F3 correction (supersedes the checkpoint plan above):**
  - The competition release is **not** RealPDEBench's data. Same experiment labels, different processing: on (3750, 0) u has mean 0.046 / std 0.022 against 0.280 / 0.0019, and no frame matches.
  - So the competition checkpoints are off-distribution (FNO rel-L2 0.147) and are dropped. The overlap is by flow condition only.
  - Trunk = **RealPDEBench's own released checkpoints** (`data/realpdebench_models/foil/{unet,fno,cno,transolver}/{finetune,real,numerical}.pth`). On 32 VALID windows: FNO-finetune 0.0144, CNO 0.0155, Transolver 0.0356.
  - Windows are 64×128 (their `sub_s_real = 2` of 128×256), not 32×64. The fast loader is verified equal to their `__getitem__`. The chunked metric is verified equal to `eval_metrics(batch = N)` to 3e-7.
  - Full VALID/TEST evaluation of unet / fno / cno is running (`runs/realpde/eval_trunk_finetune.log`).
  - FNO's relative high-frequency error is 0.66 against 0.17 at low frequency, so it is an SRR target.
- **E13 update (SRR):** `paper/spectral/srr.py` built and checked. A pilot is pre-declared in `32-…` (addendum) and running: `runs/srr_pilot_runner.sh`, seed 1337, 5k steps, vanilla / srr-lin / srr-mlp / srr-ryd.
- [~] F7 **Sophelio NAS, restructured into two stages (03:30):**
  - Why: the joint trials cost > 4 h each under contention.
  - Stage 1 = classical trunk (20 trials, successive halving). Stage 2 = coupling (latent / srr register configs) on the frozen best trunk, coupling-only training on cached latents (`runs/sophelio_nas2_runner.sh`).
- [~] F8 **Labelled seed ensembles (classical lift; each arm = mean of its own 3 seeds, so the arms stay matched):**
  - TokaMark 1-1: record/replay through TokaMind's own metric code (`runs/tokamind_ensemble.sh`). Replay verified exact.
  - TRL2D one-step TEST: `eval_official.py a.pt+b.pt+c.pt` (`runs/well_ensembles.sh`).
  - TDC: MapLight + CheMeleon-arm per seed (`paper/tdc/ensemble.py`).
  - Next: RealPDE (U-Net / FNO / CNO, weights chosen on VALID), rayleigh_benard / shear_flow, Pelzer.
- [ ] F10 **RouteFinder EAL is parked.**
  - The vanilla EAL-full fine-tune (md, seed 1) diverges to NaN action probabilities after ~25 min, under both fp16-mixed (their default) and bf16-mixed. The earlier 1-epoch smoke passed.
  - Next: trace the first non-finite tensor (`torch.autograd.detect_anomaly` on a short run; check the env's multi-depot feature normalisation and the fresh EAL embedding init; try lr 1e-4). Rerun vanilla only once it is stable.
  - The anchor (zero-shot RF-TE-100) already reproduces: mean gap 3.126% vs 3.124%.
- [ ] F9 Matbench deferred: a crystal-GNN trunk on 5-fold CV is too heavy while the GPU is shared by 5 jobs. Start once the Sophelio NAS and SRR pilot free it.
- Memory rule: every launch goes through `runs/capped.sh` with `QW_GPU_GB`. Jobs that would push MemAvailable under ~20 GB wait for the RealPDE eval to finish (RF EAL and the SRR pilot are requeued behind it).
- [ ] E8 (optional 14th row) DrivAerNet++ drag prediction (manufacturing) / Ship-D (ships): geometry-subset download, Transolver/PointNet trunk

## G. Sep 23 evening — user thesis: the register takes the analog part, the classical trunk the rest (doc 33)

- [x] Registered the allocation-strategy study (`33-allocation-strategy-study.md`). Code in `paper/alloc/`; runners in `runs/alloc_*`.
- [x] Lean: `gain_le_ceiling`, `ceiling_mono`, `pooled_gain_le_dc` (`formal/QuotaScaling/Utilization.lean`).
- [x] Amendments:
  - 1: global context for tile co-processors.
  - 2: planetswe excluded.
  - 3: uniform VALID/TEST windows on rayleigh_benard and shear_flow.
- [x] Rows complete (3 seeds): TRL2D (captures 0), RealPDE U-Net (pooled +0.12 %), Pelzer (static +9.2 % vs pooled +1.45 %; oracle +15 %).
- [x] K sweep on Pelzer: pooled flat, routed rising.
- [x] Probe B (analog vs digital): the analog-edge rule is NOT met (register wins only at ω = 2). B2 exploratory arms (rydlin, mlpsin) added.
- [x] Sophelio single TEST: 0.9910. `\SophTEST` filled.
- [x] Remaining rows: RealPDE FNO (3 seeds), rayleigh_benard, shear_flow, MHD_64. Division-of-labor pairs (hyb vs mlp2).
- [x] Paper §5.5 (`sec:alloc`), Table `tab:alloc`, Fig. `fig_alloc`, appendix `app:alloc`: fill the `\Alloc*` macros via `paper/alloc/fill_paper.py` and `probe_table.py`. Update the abstract, contributions and conclusion. Keep the main text ≤ 9 pages.

## H. Sep 24 — /goal: raise SoTA gains from evaluation errors (docs 35 §8, 36)

- [x] ADR-C v1 on pelzer, rayleigh_benard and shear_flow, 8 arms × 3 seeds. Routing gains are real; the register ties MLP, gutz and GNN.
- [x] Fixed the `capped.sh` unit-name collision and reran the RB seed-3 cells.
- [x] Full-TEST persistence audit (`paper/alloc/persistence.py`, `runs/persistence/`):

  | Row | Persistence | Published SoTA |
  |---|---|---|
  | rayleigh_benard | 0.2024 | 0.6566 |
  | shear_flow | 0.0646 | 0.8080 |
  | MHD_64 | 0.418 (trunks win 100 % of windows) | — |
  | TRL2D | 0.576 | — |

- [x] Registered v2 (band-wise POD in dyadic FFT shells, per-band provable chain) and doc 36 (band-gated anchor plus exact per-window gate). Smoke passed.
- [ ] v2 runs (`runs/alloc_adrc_v2_runner.sh`): 5 rows, including TRL2D (v1 + v2) and MHD.
- [ ] Gate runs (`runs/alloc_bgp_runner.sh`), full TEST: shear mr30ryd, RB mr30ryd, TRL2D released, then the vanilla trunks.
- [ ] RB fine-tuned-trunk v1 stacking (`runs/alloc_rb_ft_runner.sh`), full TEST.
- [ ] TokaMind-Scratch reproduction (`runs/tokamind_scratch_runner.sh`; paper 0.140 = SoTA): 8-shot smoke, then vanilla s1337, then the register/MLP arms.
- [ ] Paper: Well rows must show persistence (doc 36 §2). Add the gate/ADR-C results table and the per-band decomposition to the appendix. Keep the main text ≤ 9 pages.

## I. Sep 26 — final day (deadline ~15:00 local): raise acceptance odds, close registered cells
- [x] §19 vanilla-trunk scaling in the paper; §20 K = 50 on shear: 3/3 seeds inside the range, and the MLP captures more. Stated as "captured fraction did not fall, as registered" (abstract, §5.5, app:hscale).
- [~] §20 RB K = 50: 6 runs, runner restarted with a 4-job cap at 03:25.
- [x] app:fgn table: (d) tuned-τ and (e) window-scaled dressing columns. (e) beats FGN on TRL2D (0.147 vs 0.152/0.158); FGN beats (e) on shear s1–2.
- [~] app:fgn remaining cells: RB/MHD (d) and (e), the FGN RB ryd s2–3 and the MHD ryd runs.
- [~] RouteFinder EAL: route20-mlp s1, route10 s1, vanilla s2 running; then route30/route30-mlp s2, route10-mlp, s3.
- [~] RRNCO anchor test (pilot lane).
- [x] Figure 1 overlaps; duplicate QPA bib entry; abstract/contributions framing (simulation enables matched controls; QPA precedent).
- [x] `openreview_abstract.txt`, the plain-text abstract matching the PDF. **User action: paste it into the OpenReview abstract field before the deadline.**
- [ ] Final pass before upload: anonymity grep (clean at 03:20), p10 starts with the AI statement, 0 BibTeX errors, no "??".
