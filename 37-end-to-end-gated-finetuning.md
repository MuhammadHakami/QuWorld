# End-to-end fine-tuning with a persistence-anchored band gate and a metric-aligned loss (registered Sep 25 2026, 00:10)

## Why (user, Sep 24: "only insisting on one strategy over fine tuning")

Every Well-row gain so far came from correctors stacked on **frozen** trunks. The persistence audit (doc 36) explains why the trunks themselves are weak on rayleigh_benard and shear_flow:
- They were trained with plain MSE on normalized fields, which gives low-variance windows almost no weight.
- The benchmark's VRMSE divides each window's error by that window's variance, so those same windows dominate the mean.

The fix belongs in training, not only in post-hoc stacking.

## Model (`paper/alloc/finetune_gated.py`)

y_hat = x_t + Σ_b α_b(z) P_b(trunk(x) − x_t), over 5 dyadic l∞ FFT bands.

- **Trunk:** the fine-tuned CNextU-net, all weights trainable, in three matched variants:
  - vanilla;
  - mlp30 (matched MLP in the neck);
  - mr30ryd (30-atom register in the neck).
- **Gate (classical, digital router):** α per field and band, from **causal** features only:
  - log band energies of p − x_t and Δx, normalized by var(x_t);
  - mean and log-variance of x_t;
  - BatchNorm with running statistics.
  - It is initialised at the TRAIN-fitted blend of the matching ADR-C cache.
- **Loss:**
  - `vrmse`: the benchmark metric itself, the mean over windows and fields of sqrt(mse/var(y)). The target variance is used only inside the loss.
  - `mse`: the old objective.

## Protocol

- **Selection:** a stratified VALID sample. For RB, k = 2 windows per trajectory × 8 time bins, validated as an estimator in doc 36 §7.
- **Iteration:** 40-minute budgets.
- **Reporting:** the stratified TEST estimate is printed for iteration and labelled as such. The VALID-selected configuration is scored **once** on the full TEST split (`--full-test`).

**Factorial on rayleigh_benard, seed 1:**
- {mr30ryd, vanilla, mlp30} × vrmse + gate;
- mr30ryd × mse + gate (isolates the loss);
- mr30ryd × vrmse + plain (isolates the gate).

Then shear_flow with the winner's settings. Then seeds 2–3 for the selected cell and its matched classical trunks.

## Decision rules

- **Classical lift:** kept if it lowers stratified VALID.
- **Quantum-specific claim:** register-neck < both matched classical trunks (vanilla, mlp30) on seed-paired VALID and TEST, 3 seeds. Otherwise the gain is credited to the loss, the gate and fine-tuning.

## Stacking

The ADR-C co-processor (doc 35 §8–9, causal gate features after the doc 36 §8 fix) can then be stacked on the fine-tuned model. Per-band ceilings are reported as before.

## Implementation notes (before the first full run, Sep 25 00:25)

- **Gate initialisation.** α₀ is fitted inside the script by the same closed-form TRAIN rule on 400 random TRAIN windows, for every trunk. This keeps initialisation identical across the matched trunks, instead of depending on which ADR-C cache exists.
- **Stable gate inputs.** The gate's BatchNorm uses cumulative running statistics (momentum None).
- **Learning rates.** The gate learns at 3e-4 and the trunk at 5e-5.
- **Smoke test** (90 s, 33 steps): the pipeline runs end to end. After 33 steps VALID was worse than at the start, with the gate's running statistics not yet settled. That is the reason for the two changes above.
- **Budget.** 3,600 s of wall time per cell, VALID every 600 s on the stratified k = 1 sample (1,400 windows), 2 concurrent lanes (`runs/ft_gated_runner.sh`).

## Run log

- **00:33.** memguard (MemAvailable 13 GB) killed the first register-neck RB cell at step ~400, when its VALID was 0.1427 → 0.1229. The vanilla cell had reached 0.1441 → 0.1191 by step 424.
  - Low-value lanes were paused: TRL2D and RB-vanilla frozen gate arms, the TRL2D ensemble caches, and the MHD caches.
  - The register-neck cell was restarted from its s1337 checkpoint (`runs/ft_gated_lane1.sh`), with the same budget and settings as every other cell.
  - Nothing from the killed partial run is used.
- **00:50, amendment for the shear_flow and TRL2D lane (`runs/ft_gated_runner2.sh`), before any of its cells completed.**
  - First VALID points showed that joint training from step 0 disturbs the trunk before the gate has learned. RB vanilla went 0.119 → 0.173 by step 1,066; shear register-neck went 0.047 → 0.060 by step 347. RB register-neck kept improving (0.143 → 0.101 by step 599).
  - The new schedule is two-stage: gate-only for 900 s with the trunk frozen (gate learning rate 1e-3), then joint training with the trunk at 2e-5.
  - The RB factorial keeps the original schedule (5e-5, no warm-up), so its cells stay comparable with each other. The RB winner will be re-run under the two-stage schedule if that schedule wins on shear_flow.
  - The partial shear cell was discarded.
- **01:10, amendment (replaces the 900 s warm-up before any of its cells completed): exact gate pre-training.**
  - **Why.** The end-to-end gate was far behind the frozen-trunk ADR-C gate (VALID ≈ 0.09 vs 0.043 on RB), because it had seen too little data.
  - **How.** `--gate-pretrain N` runs the frozen trunk once over N random TRAIN windows and caches the causal gate features plus the exact band statistics. The gate is then trained on the exact per-window VRMSE (Parseval; Lean `gate_energy_exact`) for 60 epochs, selected on the stratified VALID. Joint fine-tuning follows, trunk at 2e-5 and gate at 1e-3.
  - **Check (TRL2D smoke, 200 windows).** The exact VALID after pre-training (0.19276) equals the VALID of the model's actual predictions (0.19276).
  - **Scope.** The shear_flow and TRL2D lane now uses N = 3,000 for all three trunks. RB gets the same schedule for its three trunks after its v1 factorial.

## Results so far (Sep 25 04:15; seed 1; stratified test estimates)

**RB, register-neck trunk:**

| Cell | Schedule | Stratified test |
|---|---|---|
| VRMSE + gate | joint | 0.0742 |
| VRMSE, no gate | joint | 0.433 (no gain) |
| MSE + gate | joint | 0.1415 (no gain over the blend) |
| VRMSE + gate | two-stage | **0.0425** (= the exact pre-trained gate; the joint stage adds nothing) |

**RB, joint schedule, other trunks:** vanilla 0.0854; MLP neck 0.0659.

**shear, two-stage:**
- register neck 0.0271 (the joint stage adds nothing);
- vanilla 0.0298 (joint 0.0301 → 0.0294 VALID);
- MLP neck: its pre-training gives 0.0284 VALID; the joint stage is running.

**Reading.**
- The gain needs both the metric-aligned objective and the band gate.
- Exact gate pre-training on the frozen trunk captures essentially all of it.
- End-to-end trunk updates add little or nothing at this budget.
- The trunk ranking (register neck ≤ MLP neck < vanilla at the pre-training stage) is descriptive: one seed, no claim.

## Registered Sep 25 07:55, before any ensemble score: equal-weight trunk ensemble (E-ft)

- **Members:** per row, the two-stage (`_pre`) gated fine-tuned models of the three matched trunks: register neck, vanilla, MLP neck. On MHD_64 there are two, because no MLP-neck cell exists.
- **E-ft:** the equal-weight mean of the members' gated predictions ŷ. No weights are fitted.
- **Selection:** on the same stratified VALID sample used in training (valk=2, rng seed 2), choose between E-ft and the single member with the best VALID score. Only the selected model is scored on the full TEST split, and only once.
- **Attribution:** E-ft is a classical ensemble. The register is one member among matched members. No quantum credit is taken.
- **Script:** `paper/alloc/ft_ensemble.py`.
