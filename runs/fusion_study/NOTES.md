# Fusion study (user Oct 10: "proper study incl. competition blogs; we are stuck in a local minimum, need a breakthrough")
Sources read 2026-10-10:
- Codabench 17739 pages (runs/fusion_study/page_*.md): rules, evaluation, data, submission format.
- Forum https://www.codabench.org/forums/17429/ is behind a login wall (session cookie; API token not accepted) -> needs the user's browser.
- Website announcements (fusion-equilibrium-challenge.sophelio.io/announcements/):
  * Oct 6 forum tab bug; Oct 1 clarifications: dev feedback may select pipelines (disclose); final submission counts for both
    awards unless nominated; G_ratio divides by the TEAM's best DIII-D S; only GPU hours count toward 24 GPU-h.
  * Sep 23: no MAST equilibrium data; reproduction package (repo at pinned SHA: source, env, weights, train + inference
    entrypoints) for every Final submission; top 5 per challenge rebuilt; ties by R2psi then D_LCFS; per-shot detail withheld.
  * Aug 18: Ch2 gate raised to S_model >= 0.85 (no sandbagging).
  * Aug: data erratum - magnetics_plasma_current_times wrong on ~69% of DIII-D shots (data_fixes.py in starter kit; v1.1.1).
- Organizers' paper: arXiv 2609.01750 "The Fusion Equilibrium Challenge: Inferring Magnetic Geometry Without Magnetic Diagnostics".
- Leaderboard fact sheets: top entries carry no method names.
Our state: public 0.994909 (rank 22/121, Oct 4); gap to #1 (0.9967) = psi 0.00005, q/b 0.00037, D_LCFS 0.00037, consistency 0.00092.
Exhausted at our architecture: history, fixed-coord Thomson, 6-seed ensemble, width/depth, epochs, Green's split, pixel MSE.
- Organizers' paper (runs/fusion_study/paper.txt): baselines only (PCA+Ridge, PCA+GBT, MLP on 20 PCA, conv decoder); public board
  scores 50% of the held-out shots with bootstrap CIs, the blind half decides; planned extra notebooks: a transformer baseline and a
  "kinetic-equilibrium pretext task" (Thomson + GS). No winning-method hints. Forum = only remaining source (login needed).
## Gap anatomy (public, us vs #1): composite gap 0.0018 = consistency 0.00092 + D_LCFS 0.00037 + q95/betaN 0.00037 + psi 0.00005.
   The psi pixel term is NOT the issue in composite terms; boundary/shape-derived terms are (li, kappa, volume, Z_axis; D_LCFS).
## Breakthrough candidates (to rank and test):
 1. Physics-consistent decoding: predict GS source profiles (p', FF' params) + boundary flux and SOLVE free-boundary GS with the
    known coil currents -> shapes/li consistent by construction (also the only credible route to Ch2/MAST, a separate award).
 2. Boundary-first decoding: predict LCFS (X-point, gaps, contour) and axis directly, then build psi conditioned on them
    (our HJl9 post-processing does a weak version: shape heads + Gauss-Newton fit).
 3. Loss aligned with the scorer: differentiable consistency/D_LCFS terms (li via |grad psi|^2, LCFS via soft level set) in training.
 4. Temporal sequence model over the whole shot (li = current-profile memory) - history features only gave +0.00005 in psi R2,
    but li/consistency were never measured for it.
## Plan B1 (started 2026-10-10 ~17:20): LCFS-aligned training loss
- train parquet ships efit_lcfs_r / efit_lcfs_z / efit_lcfs_n (true LCFS contour per frame, "bonus context").
- B1a lcfs_extract.py: per frame resample the contour to 32 arc-length points -> cache_k256/lcfs32.npy (rows aligned with X.npy).
- B1b sota.py cfg "lcfs_w": loss = mean_k (psi_pred(contour_k) - psi_true(contour_k))^2 (bilinear basis evaluation, as the O-point
  loss), + optional X-point |grad psi_pred|^2; targets D_LCFS / kappa / tri / volume directly.
- B1c test: full-data PSI-F config seed 1 with lcfs_w vs without, official VALID-sample composite (score_val 40) on mephiles.
- (2026-10-10T17:10) B1a done: cache_k256/lcfs32.npy (1,559,340 frames x 32 points, all frames have a contour). B1b: sota.py cfg
  "lcfs_w" (psi error at the true LCFS points, normalised like the map loss). B1c running on mephiles (runs/meph_b1.sh): PSI-F seed 1,
  90 epochs, lcfs_w 0 / 1 / 5, official VALID-sample composite on 40 shots (starter kit + scorer deps installed in .venv-vrp).
- (2026-10-10T17:31) **B1c result - FAIL**: official VALID sample (40 shots) control vs lcfs_w 1: consistency 0.9676 -> 0.9428, D_LCFS 0.0131 ->
  0.0153, kappa 0.931 -> 0.891, li 0.956 -> 0.902, R_axis 0.958 -> 0.915; r2_psi unchanged (0.999836 / 0.999835). Pinning psi to the
  true value on the contour distorts the interior and does not even move the boundary closer. lc5 stopped. (Control per-scalar:
  kappa 0.931, li 0.956, R_axis 0.958 are the weakest consistency scalars on this model.)
- (2026-10-10T18:11) **Ip time-axis audit**: our mag_fix re-anchoring vs the organizers' data_fixes.fix_d3d_ip_times (erratum #6, in the starter
  kit since Aug; our starter copy predated it): rebuilding the 14 Ip columns changes 4,932 / 7,041 cache shots (median 3.2% of Ip, max
  100%) and 586 / 874 public shots (max 55%). X_ipfix_all.npy / X_ipfix_public.npy written; IPF_s1 (PSI-F seed 1 on X_ipfix) running on
  mephiles vs B1_ctrl (quick 0.999794; official sample consistency 0.9676, D_LCFS 0.0131).
## TODO (ipfix rebuild of HJl9): HJl9 = runs/s38x_runner.sh pipeline (gpu_build --psi psiF --qb qf50 --heads <s38e6.log S38E6_DONE>
   --kw post-processing) + later post-processing steps. Retrain on QW_MAGFIX=X_ipfix: PSI-F seeds 1/2/3 (IPF_s1 done, s2/s3 running on
   mephiles; checkpoints written by sota_train under the tag), the QF q/b models behind "qf50", and the 6 head seeds of s38e6; add a
   "psiFI" entry to gpu_build PSI dict; run gpu_build --split sample with QW_MAGFIX=X_ipfix vs HJl9 (0.99562); TEST once; public.
   mephiles now has scorer deps (.venv-vrp) and the Fusion cache; mephiles trains a psi model in ~25-35 min.
- (2026-10-10T18:42) IPF_s2 (PSI-F seed 2 on X_ipfix) VALID quick 0.999838 vs PSI-F s2 0.999767 / other seed-2 variants 0.999821-0.999829 (+0.00001 .. +0.00007).
- (2026-10-10T18:53) IPF_s3 quick 0.999847 vs PSI-F s3 0.999834. IPF seeds 1/2/3 mean 0.999837 vs PSI-F 0.999820 (+0.000017).
- HJl9 components to retrain on X_ipfix (gpu_build.py): PSI dict "psiF" = s38i_PSI-F{,-s1,-s3}.pt -> add "psiFI" = IPF_s{1,2,3} (trained on
  mephiles; copy the .pt files to the Spark); QF = s38g_QF{,-s1,-s3}.pt (config in the s38g runner) used by --qb qf50 (blend with the
  "entry" scalars); heads = head_boundary.py seeds 1-6 (runs/s38e6_runner.sh, head_boundary_v8_s1..s6); the "entry" base model too.
  Then: QW_MAGFIX=X_ipfix gpu_build.py --split sample --tag HJl9I (same args as HJl9) vs HJl9 0.99562 -> TEST -> public.
## (2026-10-10T20:36) Study round 2: where the gap lives (doc 35 §41q)
- PCA-64 is not the limit (truncation R2 0.9999998). psi error = 43% uniform offset (R2psi only), 25% last 10% of shots, top 10%
  shots 61%.
- Ip-sign symmetry: 8.6% of cache / 8.7% of public shots are reversed-Ip (corrected Ip; the raw cached Ip column is broken and
  undercounts them). HJl9 reversed shots S 0.99461 vs normal 0.99571 -> canonicalisation worth ~ +0.0001. canon.py + J1CN test.
- Consistency == the J1 heads' R2 per scalar (post-processing transfers them fully). The lever is better scalar heads
  (kappa 0.976, li 0.977, R_axis 0.982 are the weak ones) and q95 / betaN heads (0.9956 vs ~0.998 for #1).
- §41c history failed on psi R2, but it was built from the RAW cache (broken Ip time base on ~70% of shots, EFIT frames only,
  no limited ramp-up) and never tested on li / kappa. New: hist_full.py (24 full-rate, sign-invariant features from breakdown:
  int Ip dt, Ip lags to 2 s, t90, dIp/dt EWMAs 0.1-3 s, OH flux swing, Te-based resistive clock) -> resid_probe.py screens
  whether they predict the heads' residuals (GroupKFold by shot) before any head retrain.
- QFI q/b regression = machine / no-clip config on mephiles (QFC control equally bad), not the Ip fix.
- L2D: 467 / 471 stuck in "Submitting" for 15-24 h; 420 (v6) is the last "Submitted".
## (2026-10-11T01:55) Forum (pasted by the user; login wall)
- Threads: HF pointer route (Aug 3), Ip erratum (Aug 18), Ch2 gate (Aug 18), rules update (Sep 23), clarifications (Oct 2 + Oct 5
  follow-up), "Why we decided to leave the Fusion competition" (gsc-x, Oct 7; content not seen yet), stuck submissions (943711 / 940120).
- No shared method / notebook in the organizer posts; Sep 23: "Challenge 1 is saturated at the top of the development board"
  (G. Pourcel), #1 vs #2 separated by 0.0001 then.
- Submission: HF pointer route (private HF dataset repo + fine-grained read token in manifest.json, full 40-char revision SHA,
  file diii_d_public_test.npz at repo root; Final: diii_d_private_test.npz). Direct ~2-3 GB uploads spend ~1 h in transfer at
  0.5 MB/s and count against the 15 GB storage quota; pointer = < 1 min. We have no HF token yet (user's account needed).
- Final phase (Oct 19-26, blind private fold decides; dev board is practice): 3 submissions, last scored one counts (or nominate
  per award by email); reproduction package REQUIRED with each Final submission (public repo at pinned SHA or archive + sha256 in
  the Code URL field: source, pinned env, trained weights, training entrypoint from scratch, inference entrypoint, data
  provenance per machine, declared compute); top 5 per challenge rebuilt; <= 24 GPU-h on one 80 GB A100; rebuild tolerance to be
  posted before Oct 19. Tie-break: R2psi then D_LCFS. AI assistants allowed with disclosure in the methods report.
- Award #2 (MAST zero-shot G_ratio, needs DIII-D S >= 0.85) and the 4 honorable mentions (best R2psi / qb / consistency / D_LCFS,
  judged on MAST) need MAST predictions - we have only ever submitted DIII-D. MAST path may use DIII-D labels, MAST inputs,
  geometry, own solver; never MAST equilibria or probe-derived constants.
- (2026-10-11T02:02) gsc-x "Why we decided to leave" (Oct 7): a general essay on nonlinear plasma control / actuator linearity and
  metastability (it reads as if about a control benchmark, not equilibrium reconstruction); no data, scoring or method information. Closed.
## (2026-10-11T10:16) F2 submitted; §41r thp screen
- F2 (976610) uploaded 10:12 (psi agreement vs F1b 0.99999); expected ~0.99523 (0.99515-0.9953), stated before the build.
- Starter repo (Oct 11 00:29 UTC): no v1.1.1 - v1.1.0 + data_fixes.py is final incl. the Final-phase inputs; scorer 3.2.0 = Ch2 gate only.
- GitHub search (API): no participant repos for the challenge (only the starter + a fork). The top-5 "shared profile" stays unexplained.
- New lever under test: thp = Thomson channels placed on psi_N of the predicted psi (psiF trunks; LCFS via the GPU scorer twin),
  Te / ne on psi_N + volume averages / peaking. Screen J1THP vs J1THS (Spark, TRAIN only) and QHP vs QH_s1, read out on ALL
  VALID + TEST frames (vt_eval.py, 743 shots, shot bootstrap). Rule +0.0005 with CI > 0.
- L2D D2 run (descent_d2_r3run1) died Oct 10 03:43 after step 1; its waits were cleared.
## (2026-10-11T11:01) Final-phase readiness: test-fold switch + rehearsal
- QW_FOLD=<fold dir> (paper/strat/fusion/fold.py) drives every fold-dependent path; private-fold files get _private suffixes
  (X_magfix / X_ipfix / thz / H_full / F_fut / thp / ctx), gpu_build writes diii_d_private_test.npz. Defaults unchanged.
- runs/final_fold.sh = all inputs-only feature blocks for a fold (extract, magfix, ipfix --fold-only, thz, hist_full --fold-only,
  fut --fold-only, thp raw + feat). Then the same gpu_build command with QW_FOLD set.
- Rehearsal on a 5-shot fold made of public files (diii_d_fold_smoke): all feature blocks bit-identical to the public rows; the
  F2 build reproduces F2's q95 / betaN to 1e-6 and psi to <= 1.3e-3 abs (GPU batch composition in the iterative post-processing).
- mag_fix check (c) (pre-breakdown Ip on < 5% of 20 kHz shots) is asserted only with >= 50 such shots (smoke folds report it).
- (2026-10-11T11:05) L2D: submissions 467 and 471 now 'Submitted' (no longer stuck); 471 (v9) is the last one -> the evaluated entry (deadline 16 Oct 11:59 UTC).
## (2026-10-11T11:46) Public competitor material on the HF hub (search: fusion-equilibrium / diii-d)
- fallings/fusion-fresh-diii-d-20261010 (public, MIT, Oct 10): a participant's reproduction archive + METHODS.md. Local dev S 0.994843
  (713 own-dev shots: psi 0.999839, q/b 0.996961 (q95 0.99675 / betaN 0.99718), D_LCFS 0.00989, consistency 0.98189) - below our F2
  public 0.99539; not the top-5 profile. Method: 64-PCA ridge + 3x512 SiLU residual net; psi-gradient loss (|grad psi|^2/R weighted);
  a frozen width-512 surrogate of the official psi->scalar functionals as an auxiliary loss (end-to-end consistency); q95 / betaN = NN
  (ratio + Thomson-moment inputs) blended with GPU CatBoost (5,000 it, depth 8) at tree weights 0.25 / 0.20 chosen on dev; Gaussian
  temporal psi averaging, sigma 10 ms on true timestamps. Their repro: 1 x RTX 3080 + Threadripper; OpenAI Codex agent disclosed.
- Emulated-Inc/sci-diiid-equilibrium-without-magnetics (CC BY 4.0, Oct 10): a re-host of 5,472 of the challenge's diii_d_train shots
  (0.5 ms grid, Ip on the corrected base) - no new labelled data.
- Transferable: a GBDT family for q95 / betaN (diverse member; our q/b members are 0.73-0.82 error-correlated). The surrogate-loss
  idea is what our head-aimed post-processing already does at inference.
## (2026-10-11T12:28) Lane split (user): Claude = Fusion (DGX Spark), Codex in tmux codexqu = Learn2Design (merphistophline)
- Measured: J1AH head 32 min on the Spark vs 102 min on merphistophline (seed 8: 10:31 -> 12:13). J1AH s8 copied (8 seeds).
- merphistophline's register-head job (§41u) crashed at import (no optuna in its env) -> moved to the Spark queue after §41w
  (runs/s41u_qpiah.sh). merphistophline is left idle for Learn2Design.
## (2026-10-11T15:40) Per-term gap vs the top five (public detailed reports; ours = F2 976610) and the levers it points to
- psi pooled R2: ours 0.999841 vs 0.999884-0.999931 (#1 0.999927: 2.2x smaller error). D_LCFS 0.0102 vs 0.0068-0.0084. q95 / betaN 0.99639 /
  0.99557 vs 0.99815-0.99867 / 0.99775-0.99814 (2.3-2.7x smaller error). Consistency vs #1: li -0.0053, Z_axis -0.0040, kappa -0.0035,
  tri_bot -0.0028, volume -0.0025, tri_top -0.0022, R_axis +0.0015 (ours better). All teams share the inputs (18 F coils, ECOILA, bcoil,
  core / tangential Thomson, Ip), so the gap is modelling.
- §42a (registered): q/b heads + inputs-only physics scalings (log Bt/Ip, pressure moments, betaN-like, beta_p-like, Greenwald-like).
- §42b (registered): psi trunks + offset-free level-set LCFS loss (the B1c absolute contour pin failed by fighting the uniform offset).
- §41z: the 64-ray boundary polygon has a D_LCFS floor of 0.024 (true rays) - rays cannot steer D_LCFS.
- Pseudo-labelling deprioritised: no input shift (AUC 0.507) and a 4096x4 student memorises the teacher at the test inputs.
## (2026-10-12T01:10) Session state: F3 public 0.995565 = rank 16 / 127 (#1 argonclouds 0.9967, #10 0.9963); gap 0.0011
- Live board (Oct 12 00:50): top 10 within 0.9963-0.9967; ours psi 0.9998 / q/b 0.9971 / D 0.0102 / cons 0.9855 vs #1 0.9999 / 0.9984 /
  0.0068 / 0.9881. F3 public per scalar: R 0.9846, Z 0.9909, kappa 0.9790, tri 0.9938 / 0.9925, volume 0.9907, li 0.9673.
- §42h re-read: PQ2's chain "loss" was one VT shot (2996, a wrong first-frame O-point); without it PQ2_VB +0.00014 [+0.00008, +0.00021],
  D -0.00027 -> §42m axis guard (gpu_build --guard) + PQ2 / PQ2+LS mix arms; F4 (PQ2A x3 [+ psiFA] + guard) auto-builds if one passes.
- New queued levers: §42n level-set LCFS loss on PQ2 (D_LCFS), §42o PQ2 + history inputs, §42p li current-profile proxies (Te^1.5
  moments) for the shape heads. Ruled out tonight: head / raw-psi scalar blending (+2e-5 S), public axis-jump smoothing (jumps are
  real), public head-feature shift (thz / H_full KS <= 0.035).
- merphistophline is Learn2Design-only (user); Fusion runs on the Spark alone.
## (2026-10-12T17:47) Oct 12 state
- F4 public 0.995575 (Codabench 980200), rank 16; #1 0.9967. Psi-trunk VT gains carried at ~0.1x; q/b heads at 1.5x, shape heads ~1x.
- Closed today: §42n level-set loss, §42o psi + history, §42p li proxies for shape heads (+0.000385 < +0.0004 bar; li +0.0015 real),
  §42j q/b 120 epochs, §42r q/b + li proxies, §42q X-point (D is 82% at the X-point but an oracle saddle warp does not reduce it).
- Running: §42s q/b + out-of-fold plasma-shape features (physics stacking); runs/s42s_promote.sh auto-trains QHAR2S x3, builds and
  uploads F5 if it passes (expected 0.995575 + 0.225 x VT delta).
- Ops: Oct 12 collapses were the bridge agent's local JAX imports on the Spark (prealloc); memguard v5.1 (28 GB, kills L2D/JAX first),
  memlog flight recorder; <= 2 Fusion GPU jobs. Repro package follows F4 (psi-trunk magfix config bug fixed).
## (2026-10-12T19:01) Oct 12 evening
- Gap to #1 (F4 vs 969695-type top): psi 0.00004, q/b 0.00020, D_LCFS 0.00033, consistency 0.00053 of S. Per scalar we trail on Z / kappa /
  tri / volume / li and lead on R_axis.
- §42s seed 2 +0.000159 (mean 0.000172 < +0.0002): QHAR2S not adopted under its rule. F5c (post hoc, labelled): J1RLA (liphys + qphys2)
  shape heads + QHAR2S q/b heads, expected ~0.99569; training on the Spark (~10 h with sharing), auto build + upload.
- §42t NEW LEVER: D_LCFS is driven by mm-level X-point placement. Continuous (Newton) saddle oracle: D 0.01118 -> 0.00767 (-31%) on 60 VT
  shots (§42q's oracle failed only because its target was grid-quantised). X-point labels for all 1.56M frames (xp_labels.npz), X-point head
  XP1 (TRAIN only) training -> VT arms VBc / VBxh10 / VBxh0; F6 (= F5c + xh with all-data XPA heads) auto-builds if one passes.
## (2026-10-13T14:18) Oct 13 state
- F7 public 0.995744 (Codabench 982788) = new best: F5c heads (J1RLA + QHAR2S) + continuous X-point step (xhe, XPA heads) + second li
  pass. F6 0.995629, F6L 0.995696 (li2 is a public lever VT cannot see), F5c 0.995606. #1 (xiaoxiao5503) 0.99697; gap 0.0012 (li / kappa /
  Z / volume consistency 0.0008 of S, D 0.0002, q/b 0.0002).
- §42w: our worst public shots are the most out-of-distribution ones (unusual shaping-coil combinations); OOD shots carry ~30% of head
  SSE on VT. Closed: physics flux split (external flux only 0.82 linear in the measured currents), mirror augmentation, recipe /
  ensemble diversity, gating, time smoothing, continuous tri, aux X-point targets (§42v +0.000075).
- §42z: EFIT fails on a few last frames (betaN -14.5 / q95 1.7); the q/b heads learned that signature and fire on look-alikes (VT +143,
  public shot 476 -12.3). Clamp to the label range: VT betaN 0.807 -> 0.995; F7C uploaded (pending). In every later build (--qbclamp).
- §42y: second kappa pass + li x2 on public (kappa 14 -> 2, li 93 -> 9 frames off head): F7KC to upload on the next UTC day.
- Queued (serial; a public build always runs alone): step-4 VT (xh variants) -> F8 = final candidate (F7 with QHAR2 q/b heads, clamp;
  repro package ~19.5 GPU-h) -> §42x mixup head screen -> if it passes, J1RLMA x8 + F9.
- Ops: F7 was killed twice by memguard (diagnostics / head training beside it; VS Code servers hold ~20 GB). Public builds run alone.
- 2026-10-14T05:53 §43h: layered register fast path (QW_QFAST=1, qlayer.apply_layer_1q): register was 88% of a training step; 3.6x per step; exact up to
  rounding (complex128 grads 8e-14; 7 tests). Screen queued after F12K: J1QPIpiF_s1 (fast-path check) + J1QRL_s1 (register + J1RL inputs)
  -> VT ensembles at share 1/7 -> if pass, J1QRLA s1-2 -> F13 / F13K (runs/s43h.sh, log s43h.log).
- 2026-10-14T11:27 F12K public 0.995850 (986792) vs F8K 0.995858: all-data register heads not adopted; final candidate stays F8K. §43h
  (fast-path register heads: J1QPIpiF check + J1QRL with J1RL inputs) running on the Spark.
## (2026-10-15T04:10) Backbone / head / transfer decomposition of the consistency term (user: analyse each part)
- runs/strat/fusion/stage_decomp.py on the stored VT arms (743 shots, 163,910 frames): R^2 of f_j(raw trunk psi) / the head ensemble /
  f_j(edited psi), the edit's transfer slope (edit move / head-trunk gap) by shot phase, per-shot offset share.
- With the heads the arm used (J1 s1-6; arm PQmix~g_VBxhe10e = steps shift, bw, xhe, kw, xp, jgn, li): mean over 7 scalars trunk 0.98494,
  heads 0.98712, final 0.98551. Transfer is complete for R_axis / Z_axis (slope 1.00 / 0.98), volume 0.88, kappa 0.89; tri 0.60-0.63
  (known since §38aa: the scorer quantises tri to the grid; tip warps trade tri for D_LCFS, net ~0); **li: slope 0.66-0.69 in the first
  90% of each shot but -0.21 in the last 10%** (li RMSE there head 0.078 -> final 0.090; final li R^2 0.9728 is below the raw trunk's
  0.9739). The newest TRAIN-only heads (J1RL / J1QH / J1QRL) reach 0.98819 (li 0.9801, kappa 0.9784).
- So the backbone is not the consistency bottleneck: the heads carry more than the final psi; the losses are tri (grid quantisation) and li
  at the end of the shot. The arm predates li2 / kw2 (F7 / F8K), so the current pipeline is measured next: runs/strat/fusion/
  s44_transfer_vt.sh (150 seeded VT shots, newest TRAIN-only heads in F8's proportions, arms VBxhe10 / F8K / F8K + 2 extra li passes).
  Upper bound if the edited psi matched the newest heads: consistency +0.0027 -> S +0.0005 (W_CONS 0.20) vs the 0.0011 gap to #1.
- (2026-10-15T04:25) **CORRECTION of the 04:10 entry.** Its "trunk" column used PQmix~g_ns, which is the no-shift CHAIN (steps bw, kw, xp, jgn,
  li - runs/s42l.sh), not the raw trunk; the "17% transfer" and "final li below the trunk" statements are withdrawn. With a true raw arm on
  the same psi set (PQ2_raw -> PQ2_VB, J1 heads): mean over 7 scalars raw trunk 0.97592, heads 0.98712, final 0.98579 - the edits deliver
  ~88% of the head advantage (transfer slope R_axis / Z_axis 1.00, kappa 0.99, volume 0.97, li 0.95, tri 0.79-0.83). **The consistency
  limit is head accuracy** (li 0.977 / kappa 0.976 heads), plus tri's grid quantisation. li error is concentrated at the current quench:
  57% of the VT li SSE sits in the last 10% of frames, 2.6% of those frames (|err| > 0.2) hold 84% of it; the reference li there rises
  sharply (1.5 -> 2.6-4.0 while the volume drops 17 -> 13 m^3: physical, not EFIT noise) and the heads follow it in part (shot 849: head
  2.73 / 3.14 / 2.64 vs truth 2.58 / 2.96 / 3.24). Edit bug: the F7-era chain PQmix~g_VBxhe10e reaches li 0.9728 vs 0.9760 for PQ2_VB;
  on quench frames its li overshoots (shot 849 frame 218: final 5.99, head 2.64, truth 3.24) -> a per-frame accept-if-better guard on the
  li step; s44_transfer_vt.sh measures whether F8K's li2 / kw2 already fix it.
- 2026-10-15T04:50 **s44_transfer_vt (150 seeded VT shots, newest TRAIN-only heads in F8's proportions):** S 0.99565 for VBxhe10, F8K and F8K + 2 li
  passes alike (consistency 0.98644 / 0.98643 / 0.98643): li2 / kw2 change nothing on VT. Heads mean 0.98777 vs final 0.98615. li transfer is
  complete here (head 0.98302, final 0.98301, slope 1.00 - no overshoot frames in this sample). **The remaining transfer loss is tri_bot:
  head 0.99389 -> final 0.98598 (slope 0.62)**, 3.6x tri_top's (0.99561 -> 0.99340); the pre-xhe chain (PQ2_VB, J1 heads) lost less on
  tri_bot (slope 0.79) -> the X-point steps (xhe / xp) likely degrade the bottom tip; kappa 0.97955 -> 0.97823. Next: the li guard on the 16
  worst-li VT shots (35.5% of the li SSE; s44f_liguard_vt.sh), then a tri_bot-aware X-point step (gate the xhe / xp move by its tri_bot
  effect vs the head) - weighed against D_LCFS, which xhe was adopted for.
- 2026-10-15T05:05 **li at the current quench is a backbone failure, not an edit failure.** 16 worst-li VT shots (35.5% of the li SSE): F8K li final
  0.736 vs head 0.859; the li accept-if-better guard changes nothing (identical S 0.97886: the warp is not what overshoots); 4 extra li
  passes change nothing; a wider beta cap (2.0) is harmful (li R^2 -0.60, cons 0.924 -> 0.897). Step trace on quench shots 849 / 3136
  (cumulative arms): raw 0.8345 -> shift 0.8376 -> bw 0.8418 -> xhe 0.8696 -> **kw 0.8433 (-0.026)** -> xp / jgn ~0 -> li ~0 -> li2 0.8922
  (+0.050) -> kw2 0.9271 (+0.035): the kappa warp damages li there and the later passes recover part. The raw trunk li on those frames is
  far off (849 frame 218: trunk 5.31, head 2.64, truth 3.24). Label-free per-frame trunk selection among 7 trunks by agreement with the
  heads (trunk_select.py, 743 VT shots): S 0.99530 vs best single 0.99531 (cons +0.00025, li +0.0006, but psi / D worse) - every trunk
  fails on the same frames. Not adopted. The quench li fix needs a trunk that has learned the current quench (end-of-shot weighting).
- 2026-10-15T06:10 **§44g step ablation of the F8K chain (150 VT shots, newest TRAIN-only heads, paired): the post-processing is saturated.** S: F8K
  0.99565; without xhe 0.99563 (D 0.01058 vs 0.01020, cons +0.0001); without xp 0.99564 (D 0.00991 better, cons -0.0002); **without jgn
  0.99551 (jgn is worth +0.00014)**; without the first kappa warp 0.99564. Every step but jgn is within +-0.00002 (noise); xp / xhe trade
  consistency against D_LCFS. Together with today's quench analysis (li guard / extra li passes / wider cap / trunk selection all ~0) and
  §38y (shot-phase inputs, end weighting) and §41t (future context, 5-fold proxy CV -0.00034): the terminal-frame li jump is an input-
  information gap (inputs see at most +20 ms ahead). **Fusion state:** the remaining 0.0011 gap to #1 is head / trunk accuracy (top teams'
  psi 1-R^2 is 0.38x ours); no cheap lever is left in post-processing. F8K / F13K stay the final candidates (final phase Oct 19-26).
- 2026-10-16T09:50 **§45h q/b Huber: reject** (QHR2Hu_s1 vs QHR2c_s1 on all VT shots +0.000081 [-0.000053, +0.000201]; betaN +0.00016, q95 0). With the
  shape arm (+0.0001) Huber is a real but small effect (~+0.00004 S at public transfer); not worth the all-data retrain. F14K stays best.
- 2026-10-16T10:22 **§45i + §46.** Seed ensembles saturated (8 -> 16 J1RLA seeds +0.00005 mean-7; q/b 3 -> 8 seeds +0.00004). Challenge 2 (MAST zero-shot,
  Award #2: we have G = 0) tried as a physics free-boundary entry (paper/strat/fusion/fb/); on labelled DIII-D frames even EFIT's own external
  field + Ip + R_out give psi R2 0.955 and negative consistency with a fixed current profile -> stopped, no MAST upload. F14K stays final.
- 2026-10-16T14:23 **Public competitor study re-read (Anulum, docs/STUDY.md §4.16 / §6, ideas only).** Their largest public step (+0.000198) was 215
  sub-frame window summaries (Ip offsets, coil changes over +-5 / +-10 ms); our magnetics block has carried 13 point lags since §37c. **§45m
  subf block (151 window means / slopes / ripple / Thomson pulse timing): proxy CV -0.00002 (t -0.1), Z -0.0014 -> FAIL.**
- 2026-10-16T14:23 **§45n public Z_axis deficit diagnosed** (label-free on public): error ratio public / VT is 1.72 for Z vs 0.76-1.19 for the others; 5.5 sd
  below 874-shot VT bootstrap draws; the official O-point code = our GPU twin on all 193k F14K public frames (0 differ); no shot class
  (sampling rate, Ip sign) explains it. On the CV folds the top-10% OOD TRAIN shots have Z 0.941 vs 0.993 (half of all Z SSE); none of 16
  tested levers moves OOD Z measurably; a vacuum B_R-null physics feature explains <= 1% of the OOD residual. Worth +0.00012 S at most.
- 2026-10-16T14:23 **§45o Ip-sign router registered** (post hoc origin: canonical heads lose on normal shots but win on reversed ones; router on saved VALID
  preds q/b +0.00010 [+0.00004, +0.00017], shape +0.00011 [-0.00003, +0.00028]). Fresh TRAIN-only twins J1RLCS_s1 / QHR2CS_s1 training;
  gpu_build gained "route:<normal>|<reversed>" (smoke-tested). final_f15k.sh + a smoke-fold rehearsal waiter are ready for F15K.
- 2026-10-16T16:42 §45o Ip-sign router PASSES for both families (VT 743 shots): q/b +0.00026 [+0.00013, +0.00042], shape +0.000215 [+0.000094, +0.000367].
  F16 = F15 (or F14 if F15K < F14K) + route:<base>|<canonical twins> on reversed-Ip shots; expected public +~0.00006. Runner runs/f16_router.sh.
- 2026-10-17T00:41 F15K public 0.995949 (994544; expected 0.99593: in range) = +0.000056 vs F14K, rank 14/145; adopted for the final by the hidden-set rule
  (VT + public shot-weighted delta +0.000044). Smoke rehearsal OK. F16K (router) building, expected ~0.99601.
- 2026-10-17T03:22 F16K public 0.996167 (994647; expected ~0.99601) = +0.000218 vs F15K, rank 13; Z_axis 0.9908 -> 0.9934 (the public Z deficit was mostly
  reversed-Ip shots). Adopted for the Final (hidden-set rule). Next: route the register / tree / XP / entry parts too (§45q).
