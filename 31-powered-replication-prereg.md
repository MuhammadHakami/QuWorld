# Pre-registration — powered replication of the adapter comparison (Sep 22 2026, before any new data)

Committed to git **before** any run below is launched. Motivation and power analysis: `30-campaign-judgment-sep22.md`
§2, which shows that n = 3 seeds cannot resolve the ≲2% effects the pooled data allow.

## Hypotheses

- **H1 (primary): Q vs matched MLP.** Across the three powered rows, the Rydberg register at its selected
  configuration outperforms the parameter-matched MLP adapter (reference budget, `q30-mlpref`).
- **H2 (secondary): Q vs trunk.** Same rows, register vs the trunk alone.
- **H3 (mechanism, Pelzer only; stage 2, launched only if H1 or H2 passes):** the effect vanishes when
  entanglement is removed (mean-field / Gutzwiller propagation of the same register) or when the register shrinks
  to 5 atoms.

## Rows, arms, configurations

| Row | Arms (configuration = the round-2 `ms2` selection, read from `runs/hpo/<bench>/<arm>_ms2_final.json`) | Replication unit | New units |
|---|---|---|---|
| Pelzer 1HP (tuned) | trunk · MLP (ref budget) · Q | (trunk seed ∈ {1337, 1338, 1339}) × (adapter seed ∈ {2001…2004}) | 12 |
| Sophelio S (tuned) | trunk · MLP · Q | full retrain per seed ∈ {1340…1347} | 8 |
| TokaMark 1-1, TokaMind trunk | vanilla · mlp30 · mr30ryd (their warm-start protocol) | fine-tune seed ∈ {1340…1343} | 4 |

The configurations are frozen by the finished Optuna studies. The trunk arm's round-2 selection on Pelzer and
Sophelio is still running at the time of writing, and these rows use whatever it selects; nothing is re-tuned
after this document.

## Endpoints and analysis (fixed now)

- Per row: seed-paired relative difference, sign-corrected so > 0 = first arm better, computed by
  `paper/judge/judge.py`.
- **Primary analysis uses the NEW units only.** Seeds 1337/1338 took part in the validation-based selection and
  are excluded from the primary test. All-seed results are reported as sensitivity.
- Pooling: DerSimonian–Laird random effects over the three rows, 95% CI, two-sided.
- **H1 passes** iff the pooled Q-vs-MLP CI excludes 0 in Q's favour **and** at least 2 of 3 rows have a positive
  point estimate. Same rule for H2. Anything else is reported as a null, with the pooled CI as the upper bound
  on the effect.
- No outlier removal and no early stopping of the replication on interim results. A crashed unit is re-run with
  the same seed; a unit that fails twice is reported as missing, not replaced.

## Expected power (from the Sep 22 spreads)

Paired SD (% of metric): Pelzer 2.1, Sophelio 3.0, TokaMind 0.7. With 12 / 8 / 4 new units, the per-row MDEs are
roughly ±1.3%, ±2.5% and ±1.1%. Pooled, a true +1.5% effect should be detected with power above 0.8. A true
effect of 0.5% or less will most likely come out null; that is acceptable, because it would be too small to matter.

## Order on the GPU

HPO trunk selection (running) → TDC ADMET (queued) → **this replication: Pelzer → Sophelio → TokaMind** →
RouteFinder EAL. TokaMind is the most expensive (≈ 7 GPU-h per seed for the three arms), so it goes last and can
be cut to 3 new seeds, a decision recorded here in advance: if it is cut, H1/H2 are evaluated on the units that
completed and the cut is reported.

## Deferral (Sep 23 2026 ~01:20, before any unit ran)

The user set a new goal: reach and improve SoTA on every non-B300 benchmark
(`memory: goal-reach-improve-sota-sep23`). The Sep 23 triage showed that the Sophelio row's starter pipeline sits far
below the Challenge-1 leaderboard (S 0.79 vs 0.9932), so replicating Q vs MLP *on that pipeline* is no longer the
most informative use of the GPU. `runs/replication_runner.sh` was stopped while it was still waiting for TDC.
`runs/replication/status` is empty, so no replication data exist, and the hypotheses, endpoints and analysis
above stand unchanged for whenever the replication is run: on the current configurations, or on the SoTA-class
trunks once they exist, in which case this document gets a dated addendum before that run.
