# 38 — TRL2D rollout-aware (TD) fine-tuning: 5-seed replication (registered Sep 25 14:30, before any new run)

## Why
The register-specific effect in the paper rests on 3 seeds (1337–1339):
- Official start: the register is the only arm below the anchor at 13:30.
- Sliding start: the primary test is significant on 2/3 seeds against each fine-tuned control.

Two more seeds per arm test whether the effect replicates. The result is reported whichever way it goes.

## Runs (identical recipe to round 2, `paper/phase_b/run_round2_td.sh`)
- **Arms and seeds:** {vanilla, mlp30, mr30ryd} × seeds {1340, 1341}.
- **Command:** `finetune_td.py <v> <seed> 2400 1e-4 10000 4`, with the same resume-until-`target_reached` loop and THROTTLE_MS=15.
- **Evaluations,** unchanged scripts:
  - `eval_official.py`: one-step test CSV;
  - `eval_rollout_pertraj.py`: official-start rollout, pertraj npz;
  - `eval_rollout_multistart.py`: sliding start, 9 starts × 9 trajectories.

## Analysis (fixed now)
All 5 seeds are reported (1337–1341).
- **(A) Official start:** 13:30 VRMSE per seed and arm. Count the seeds on which the register is below both fine-tuned controls, and the seeds on which it is below the anchor (1.0132).
- **(B) Primary test,** per the existing pre-specification. For each seed, a per-trajectory paired Wilcoxon (one-sided) on start-averaged 13:30 VRMSE:
  - `paired_stats_multistart.py mr30ryd vanilla 1337 1338 1339 1340 1341`
  - the same with mlp30.
  - Holm-correct across the **5** seeds; this replaces the 3-seed correction for the 5-seed statement.
- **(C) Seed level:** across the 5 seeds, a paired t-test and a sign count of the start-averaged 13:30 VRMSE, register vs each control.

## Decision rule
- **Replicates:** (B) is significant after Holm-5 on ≥ 3 of 5 seeds against **each** fine-tuned control. The paper keeps "one register-specific effect survives", with the 5-seed numbers.
- **Otherwise:** the paper says the effect does not replicate at 5 seeds (it was 2/3 at 3 seeds), and the abstract and conclusion are weakened accordingly.
- There is no further seed extension either way.

## Amendment, Sep 25 14:30, before any result (no replication run had passed step ~160)
Three parallel lanes gave lower total throughput than one run: 0.9 vs ~1.6 steps/s, because the GPU is the limit. Two seeds per arm would therefore take ~10 h and miss the submission.
- **Scope:** reduced to **one new seed (1340)** per arm, run serially (mr30ryd, mlp30, vanilla; ~1h45m each).
- **Analysis:** (A)–(C) over 4 seeds, 1337–1340, with Holm across 4.
- **Decision:**
  - "Replicates" iff (B) is significant on ≥ 3 of 4 seeds against each fine-tuned control. Given 2/3 before, seed 1340 must be significant against both.
  - Otherwise, the paper says it did not replicate at 4 seeds.

---
# Addendum (Sep 25 14:40): z-domain band filter (ZBF), registered before any fit
**Diagnosis:** `paper/alloc/spectral_diag.py`, run on the TEST sample as a diagnostic only, nothing fitted. It gives in-sample upper bounds on the MSE a per-band two-tap filter could cut: shear ≤ 8.3%, TRL2D ≤ 3.6%, RB ≤ 0.3%.
**Filter:** on top of the registered E-ft selection, per field c and Fourier octave b:
ŷ' = x_t + Σ_b P_b(α_cb·(ŷ − x_t) + β_cb·(x_t − x_{t−1})).
**Fit:** (α, β) by least squares with weights 1/var(y) (approximating VRMSE), on stratified TRAIN windows (1 per trajectory × 8 time bins, rng 7).
**Rows:** shear_flow and TRL2D. RB is skipped, since its bound is ≤ 0.3%.
**Rule:**
- Keep ZBF iff stratified VALID (valk=2, rng 2, the same sample as the E-ft selection) improves by ≥ 0.5% relative.
- A kept filter gets one full TEST.
- The result is reported either way; the filter is classical.

## Stopped, Sep 25 15:15, by the user's time constraint (no long runs before submission)
- **Where it stopped:** mr30ryd s1340 at ~2,100 of 10,000 steps; mlp30 and vanilla had not started.
- **No evaluation was run:** no official-start, sliding-start or test score exists for seed 1340, so no replication result was seen.
- **Paper:** keeps its 3-seed statements as already qualified. Replication is future work.
- **ZBF amendment (15:16, before any TEST score):** because of the time limit, a kept filter is scored once on the stratified TEST sample (testk=2, rng 3; the same sample as the shear E-ft stratified estimate), not on the full split.

## Resumed, Sep 26 03:35 (user: "we still have some time… produce results"; long registered runs allowed). No result had been seen.
- **State at resumption:** no evaluation of seed 1340 was ever run. On the training side, mr30ryd reached step 2,308 (its last validation checks use VALID only), and mlp30 and vanilla stopped at ≤ 148 steps.
- **Protocol:** unchanged (`runs/td_repl_serial.sh`): serial mr30ryd, mlp30, vanilla, each resumed from its `_latest.pt` to 10,000 steps, followed by the three evaluations.
- **Analysis and decision rule:** unchanged (4 seeds, Holm across 4). "Replicates" iff seed 1340 is significant against both fine-tuned controls under the primary sliding-start test.
- **Reporting:**
  - If every arm finishes before submission, the paper reports the outcome either way, weakening the abstract and conclusion if it does not replicate.
  - If not, the paper keeps its 3-seed statements, and the 4-seed result goes to the rebuttal.
- **GPU priority:** the RouteFinder full queue stops launching new runs at 03:35; the in-flight runs finish. That queue's remaining runs (seeds 2–3 of the null, the §18 FGN arms) are deferred.

## Pre-committed abstract wording (Sep 26 04:20, before any seed-1340 evaluation)
- **Replicates** (seed 1340 significant against both fine-tuned controls under the primary test, Holm across 4). The abstract keeps "One register-specific effect survives", and "(Holm-significant on two of three seeds; …)" becomes "(Holm-significant on three of four seeds; …)". The per-seed numbers are updated.
- **Does not replicate.** Replace the clause from "One register-specific effect survives:" through "so deployment needs no quantum hardware." with:
  > "Under temporal-difference fine-tuning, long-horizon rollout error on the turbulent radiative layer (TRL2D) fell 7.5–7.7% against both continued classical controls on three seeds (Holm-significant on two), but a registered fourth seed did not replicate it, so no register-specific effect is established."
  - The conclusion's "the one regime where the register helps … is real at stated power, narrow, and arises in training rather than inference" becomes "the one candidate regime, long-horizon rollouts under rollout-aware training, did not replicate at four seeds".
  - Contribution 2's "One scoped exception" sentence gets the same qualifier.
- **Incomplete by submission.** No change to the paper; the 4-seed result goes to the rebuttal.
- **Rejected** (Gemini's draft): "two exploratory seeds" (there were three); "a matched classical core matches or exceeds the register" across all benchmarks (overstates the null).
- **Update, Sep 26 05:10, before any seed-1340 evaluation (text moved after the §17 shear result).**
  - The abstract now reads "A register-specific effect survives:"; the pre-committed "does not replicate" replacement for that clause is unchanged.
  - The conclusion now names two regimes. If the rollout effect does not replicate, it becomes: "…the regime where the register helps, noise-drawn configurations on shear flow, is real at stated power and narrow; the rollout effect did not replicate at four seeds."
- **Update, Sep 26 05:20, before any seed-1340 evaluation.** The §17 shear result failed its wider-MLP control, so the abstract is back to "One register-specific effect survives:" and the conclusion to the one-regime sentence. The original pre-committed wording (04:20) applies again.
- **Sep 26 05:32 (speed only; recipe unchanged):** the TD process and its runner shell are pinned to the Cortex-X925 performance cores with `taskset 5-9,15-19`. The job was single-thread CPU-bound at about 1.0 steps/s; pinned, it runs at about 1.5. Model, data, steps, throttle and evaluation are all unchanged.
- **Observation, Sep 26 05:58, before any seed-1340 evaluation; the protocol is not changed.**
  - Selection is by rollout VALID (the registered round-2 rule; the best checkpoint is kept).
  - At step ~8,800, mr30ryd s1340's best VALID vrmse8 is still 0.5803, from step 84 of the first attempt; no later check has gone lower.
  - The earlier mlp30 and vanilla s1340 attempts also saved early bests (0.5750 at step 72; 0.5746 at step 78).
  - Seeds 1337–1339 selected late checkpoints (steps 5,726–9,309).
  - Whatever checkpoint the rule selects is evaluated; the outcome is reported under the pre-committed wording.
  - Validation runs every 180 s of wall clock, so this run has more checks per step than round 2 did. That gives it more selection opportunities, never fewer.

## Secondary analysis: final-step checkpoints (registered Sep 26 06:12, before any final-checkpoint evaluation)
- **Why:** seed 1340's selected checkpoints come from the first ~80 steps, so the primary test there compares nearly un-fine-tuned models. Gemini: early selection biases against a training-time effect. The primary rule and its pre-committed wording are UNCHANGED; this analysis cannot replace them.
- **What:** every arm's step-10,000 weights (`<v>_s<seed>_latest.pt`, symlinked as `runs/phase_b2/final/<v>_s<seed>.pt`), for seeds 1337–1340 and arms {mr30ryd, mlp30, vanilla}.
  - The same `eval_rollout_pertraj.py` and `eval_rollout_multistart.py` produce names `<v>final_s<seed>`.
  - The same `paired_stats_multistart.py` then runs register vs each control over the four seeds, with Holm across 4.
- **Report:** in the appendix as "secondary, registered after the primary selection was seen for the register at seed 1340", whichever way it goes. It cannot turn a failed primary into "replicates". If the primary fails and this passes, the text says exactly that.
- **Schedule:** the ten finals that exist now are evaluated immediately on the efficiency cores (0-4, 10-14) so the TD run keeps the X925 cores. mlp30 and vanilla s1340 follow when their training ends.

## VERDICT (Sep 26 08:15): DOES NOT REPLICATE
- **Primary (VALID-selected checkpoints),** register vs matched MLP. Per-trajectory one-sided Wilcoxon on start-averaged 13:30, Holm across the 4 seeds:
  - s1337: mean diff −0.037, Holm 0.20 (ns)
  - s1338: −0.060, Holm 0.008 (sig)
  - s1339: −0.069, Holm 0.008 (sig)
  - s1340: **+0.057** (the register is worse on 9/9 trajectories), p = 1.0
  - Significant on 2/4 seeds (< 3/4). Seed 1340 is not significant against the MLP, so the rule fails whatever the vanilla comparison gives.
- **Seed 1340 values** (selected checkpoints: register step 84, MLP step 72):
  - register: official 1.0056, all-start 0.9568, one-step 0.2230
  - MLP: official 0.9841, all-start 0.9555, one-step 0.2221
- **Secondary (final-step weights), register vs MLP:** Holm-significant on 0/4 seeds; mean diffs +0.091, −0.021, +0.087, +0.051.
- **Vanilla s1340** is still training (for the full report and the secondary). Its comparison cannot change the verdict.
- **Applied now,** per the 04:20 pre-commitment: the abstract clause, the conclusion, contribution 2 and a §5.2 sentence. The vanilla counts get filled in when that arm lands.
- **Correction and completion (Sep 26 10:00).**
  - In the 08:15 entry I swapped the MLP seed-1340 values. They are: selected (step 72) official 0.9439, all-start 0.8996; final official 0.9841, all-start 0.9555. The paired statistics read the npz files directly and were correct.
  - Vanilla s1340 (selected step 78): official 0.8664, all-start 0.8939; final official 1.1456, all-start 1.1031.
  - At s1340 the register (0.9568 all-start) is worse than both controls on 9/9 trajectories.
  - **Four-seed primary, Holm across 4:** register vs MLP significant on 2/4 seeds, vs vanilla on 1/4. Means (official / all-start): register 0.9660 / 0.9506, MLP 1.0101 / 0.9780, vanilla 0.9892 / 0.9635.
  - **Secondary (final weights):** register vs MLP 0/4, vs vanilla 2/4. Means: register 1.0467 / 1.0242, MLP 1.0109 / 0.9723, vanilla 1.1236 / 1.0877.
  - **Verdict unchanged:** DOES NOT REPLICATE. app:perseed now has the paragraph and Table tab:tdrepl (td_replication_summary.py).
- Sep 26 11:55: at the user's request the abstract was rewritten as a single story. The non-replication is now stated as 'a register-specific gain in long rollouts did not survive a registered fourth seed' (same substance as the 04:20 pre-commitment; details in §5.2 and app:perseed).
- Sep 26 12:00: at the user's request the abstract leads with gains and the scaling law. The rollout non-replication moved out of the abstract; it is stated in §5.2, the conclusion and app:perseed.
