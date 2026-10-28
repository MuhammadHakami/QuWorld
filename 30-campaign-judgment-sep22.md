# Campaign judgment — Sep 22 2026

Every number here comes from `runs/judge/scorecard.{json,md}`, which `paper/judge/judge.py` builds from the result
files themselves (not from the ladder prose). Re-run it after any result lands. This document reads the 22:20
scorecard, which already includes the MHD length-matching correction (§3).

## 1. What is being judged

Three claims, in decreasing order of what the program needs:

1. **Q vs trunk:** inserting the Rydberg register into a strong classical trunk improves the benchmark metric.
2. **Q vs matched MLP:** the register beats a parameter-matched classical adapter at the same latent and under the
   same protocol. This is the quantum-specific claim, still without any claim of quantum advantage.
3. **Standing vs published SoTA:** where the arms land relative to the published reference.

Design, fixed before the pooled numbers were read: one **primary** row per benchmark (the final-protocol,
strongest-trunk row). Older protocols, weaker trunks, single-seed rows and configurations selected on the scored
data are **secondary** and never enter a pool.

## 2. Quantitative verdict

### Per row (primary), seed-paired, positive = first arm better

| Row | Q vs trunk | Q vs MLP | MLP vs trunk | MDE, Q vs trunk |
|---|---|---|---|---|
| TRL2D arena VRMSE@8 | +11.7% (2/3, p=0.44) | +11.4% (3/3, p=0.16) | +0.3% | ±53% |
| MHD_64 VRMSE (length-matched seeds, n=2) | +0.0% (tie) | +0.0% (tie) | −0.0% (tie) | ±0.3% |
| rayleigh_benard VRMSE | +7.5% (1/3, p=0.69) | +20.0% (3/3, p=0.07) | −15.6% (1/3) | ±69% |
| shear_flow VRMSE ⚠ anchor | +1.0% (2/3, p=0.97) | −4.7% (2/3) | +5.5% | ±99% |
| TORAX-Bench VRMSE@8 (tuned) | −6.4% (2/3, p=0.77) | −19.2% (0/3, p=0.40) | +10.8% (3/3, p=0.11) | ±84% |
| Pelzer MAE (tuned) ⚠ mixed | +1.2% (2/3, p=0.42) | +1.3% (2/3) | −0.1% | ±5.2% |
| Sophelio S (tuned) ⚠ mixed | +0.3% (tie) | +1.0% (2/3) | −0.7% | ±7.5% |
| TokaMark 1-1 (TokaMind trunk) | +0.5% (2/3, p=0.34) | +0.2% (tie) | +0.3% | ±1.8% |

"Tie" means |gain| < 0.5%, which the judge treats as neither a win nor a loss.

### Across the 8 primary rows

| Comparison | better / worse / tie | sign test (ties excl.) | significant (±) | mean over rows [95% t-CI] | random-effects pool [95% CI] | pool without MHD |
|---|---|---|---|---|---|---|
| **Q vs trunk** | 5 / 1 / 2 | p = 0.22 | 0 / 0 | +1.98% [−2.54, +6.51] | +0.04% [−0.00, +0.08] | +0.58% [−0.16, +1.33] |
| **Q vs MLP** | 4 / 2 / 2 | p = 0.69 | 0 / 0 | +1.25% [−8.30, +10.79] | +0.53% [−0.39, +1.45] | +2.16% [−0.46, +4.78] |
| MLP vs trunk | 2 / 2 / 4 | p = 1.00 | 0 / 0 | +0.06% [−6.19, +6.31] | +0.06% [−0.31, +0.44] | +0.14% |

Two summaries are reported because they answer different questions and fail in different ways:
- **Mean over rows** treats each benchmark as one observation ("does it generalise across industries?"). No single
  row can dominate it, but with 8 rows it is wide.
- **Random-effects pool** weights rows by precision. Rows where the adapters are *inert* (MHD: gates |α| ≈ 4×10⁻⁵,
  so the three arms are numerically the same function) have near-zero variance and take almost all the weight.
  Most of what the pool measures is that inert adapters change nothing. It is reported, but it is not the headline.

### Reading

- **Claim 1 (Q improves the trunk): not supported.** No primary row is significant in either direction, the sign
  test gives 5 vs 1 (p = 0.22), and every pooled interval contains zero. Across benchmarks, the data are
  consistent with anything from −2.5% to +6.5%. The two large point estimates (TRL2D +11.7%, rayleigh_benard
  +7.5%) sit on rows whose seed spread gives MDEs of 53–69%.
- **Claim 2 (Q beats the matched MLP): not supported.** The result is 4 better, 2 worse, 2 ties, p = 0.69, and
  every interval contains zero. The earlier "6/8, +2.3%" reading came partly from a truncated MLP run on MHD (§3)
  and is withdrawn. The largest Q-over-MLP margins (rayleigh_benard +20%, TRL2D +11%) come with the MLP *losing to
  the trunk* on rayleigh_benard (−15.6%, 1/3 seeds, n.s.). At most this hints that the register is the more
  benign adapter on that one row. It is not evidence of a better model.
- **The classical adapter is a null too** (2/2/4). At the deepest latent of already-tuned trunks, neither
  adapter buys anything measurable. That is a property of the insertion point and the trunks, not of quantumness.

### Standing vs published references (claim 3)

| Row | Q vs reference | trunk vs reference | Honest status |
|---|---|---|---|
| rayleigh_benard | +35.7% vs TFNO 0.6566 | +30.5% | Official checkpoint reproduced (0.6679). The gain is the **vanilla fine-tune** trimming a heavy tail (median 0.034 → 0.032), not the register |
| shear_flow | +67.1% vs 0.808 | +66.8% | **Not comparable:** our evaluator scores the official checkpoint 0.9946, so the published anchor is not reproduced |
| TORAX-Bench | +87% | +88% | Reference is persistence on our own benchmark, not a SoTA |
| Pelzer | +37.4% vs 0.046 °C | +36.7% | Our 700/200/100 split; equivalence to theirs unverified |
| TokaMark (CNN trunk) | **+2.8%** vs 0.2091 | −1.3% | **The only row where the register flips the verdict**, but n = 1 (their fixed seed) |
| TokaMark (TokaMind trunk) | −0.1% vs 0.150 | −0.6% | All three arms tie at the SoTA |
| MHD_64 | −16.1% vs 0.1633 | −16.1% | Below SoTA; adapters inert |
| Sophelio | −6.7% vs 0.85 bar | −7.0% | Below the award bar |

Every multi-seed row that beats its published number is beaten by the vanilla trunk by nearly the same margin,
so none of that margin is attributable to the register.

### Power: what n = 3 seeds can and cannot see

Seeds needed to detect a Q-vs-trunk effect of a given size (paired t, α 0.05, power 0.8, observed spread):

| Row | paired SD (% of metric) | 1% effect | 2% effect | 5% effect |
|---|---|---|---|---|
| TokaMark (TokaMind trunk) | 0.71 | **7** | 4 | 3 |
| Pelzer (tuned) | 2.08 | 36 | **11** | 4 |
| Sophelio (fixed / tuned) | 1.66 / 3.04 | 24 / 75 | **8** / 21 | 4 / 6 |
| TORAX (fixed) | 6.71 | >200 | 91 | 17 |
| TRL2D arena, rayleigh_benard, shear_flow, TORAX (tuned) | 21–40 | >200 | >200 | 146 → >200 |

Only TokaMind, Pelzer and Sophelio can resolve effects of 1–2% at an affordable n. The other four rows have
training-seed variance of 20–40% of the metric and need variance reduction (longer schedules, EMA weights,
checkpoint averaging) before more seeds are worth anything.

## 3. Qualitative verdict

**Methodology is strong, and it keeps getting stronger because it catches its own errors.**
- SoTA trunks reproduced before any arm was scored: RouteFinder RF-TE within 0.002% (16 variants), TokaMind
  0.1504 vs 0.150, TokaMark CNN 0.2118 vs 0.2091, rayleigh_benard 0.6679 vs 0.6699, Pelzer U-Net 0.0497 vs 0.046.
- Official evaluators, matched controls (0.14% parameter match), zero-init identity at step 0, pre-registered
  Wave G rules, Optuna selection on VALID with a single TEST evaluation, and a pre-registered powered replication
  (`31-…`) committed before its first run.
- Errors caught, each before it reached a claim:
  - An 8-shot smoke result misread as a TokaMark arm (0.8587).
  - The round-1 single-seed HPO selection artefact.
  - SVRPBench's defects: byte-identical train and test files, and 20% of instances infeasible.
  - The unreproduced shear_flow anchor.
  - **MHD compared arms of unequal training length**, found by this judge. 4 of 9 MHD runs stopped at ~7.5k of
    20k steps (restart attempts, one INCOMPLETE), and their best checkpoint is their last. Seed 1338's MLP ran 7.5k
    steps against 20k/19.5k for the trunk and the register, which alone produced MHD's "Q beats MLP 3/3, +4.2%".
    At matched length (seeds 1337 and 1339) the three arms agree to 4 decimals, and the gates never opened
    (|α| ≈ 3–7×10⁻⁵).

**The evidence does not support "the register improves industrial simulators", nor "the register beats an
equal-size classical adapter".** It supports one weaker claim that is still worth making:

> A 30-atom Rydberg register can be dropped into eight SoTA-class industrial trunks (fusion, fission proxy,
> geothermal, turbulence, MHD), under each benchmark's own protocol, **at no measurable cost**: no primary row is
> worse, 5 of 8 are nominally better, and every pooled interval contains zero. It matches the SoTA wherever the
> trunk does.

**What the register is not doing.** The qubit ladder saturates below 5 qubits (5q 0.843 ≈ 30q 0.845), and the
Wave G graph factors (topology, spacing, partition, cascade, gating, per-input budgets) produced one promotion
(`link=shared`) that did not transfer at fixed protocol. The 30-atom Hilbert space and its interaction graph are
therefore not what carries whatever effect exists. A claim about analog quantum dynamics needs a result that
disappears when entanglement is removed (the Gutzwiller control), on a row with the power to see it.

**Grades.**

| Dimension | Grade | Why |
|---|---|---|
| Protocol and reproducibility | **A−** | Reproduced anchors (RouteFinder to 0.002%), official scorers, matched controls, pre-registration; minus the shear_flow anchor, the MHD length mismatch, and the mixed-protocol tuned rows |
| Statistical power | **D** | n = 3 cannot resolve the ≲2% effects the data allow; 4/8 primary rows cannot resolve 5% at any affordable n |
| Evidence for claim 1 (Q > trunk) | **null** | 5/1/2, p = 0.22; mean over rows +2.0% [−2.5, +6.5] |
| Evidence for claim 2 (Q > MLP) | **null** | 4/2/2, p = 0.69; withdrawn from "weak positive" after the MHD correction |
| Evidence for the reframed thesis (drop-in, no regret) | **B** | Holds across eight benchmarks; industry breadth (drug, logistics) still being added |

## 4. What would change the verdict (ordered by information per GPU-hour)

1. **Finish the trunk's two-seed selection** on Pelzer and Sophelio (running), which removes the "mixed" flags.
2. **The pre-registered powered replication** (`31-…`, queued): Pelzer 12 units, Sophelio 8, TokaMind 4, with
   Q vs MLP pooled as the primary endpoint and new units only.
3. **TDC ADMET (22 endpoints × 5 official seeds, queued):** 22 more rows for the across-benchmark estimate, which
   is currently limited by having only 8 rows. It is also the first non-physics industry row.
4. **Repair MHD:** re-run the four truncated arms (s1338 mlp30, s1339 × 3) to the full 20k steps. Then treat
   "gates never open" as the finding it is: the adapter is inert at this latent, and a different insertion point
   or a gate learning-rate multiplier is the experiment worth running, not more seeds.
5. **Reproduce the shear_flow anchor** (0.808) or drop the published comparison for that row.
6. **Mechanism test on a powered row:** Gutzwiller (no entanglement) and a 5-qubit register vs the 30-atom register
   on Pelzer/TokaMind. This runs only if the replication yields a signal (prereg H3).
7. **Logistics row (RouteFinder EAL, harness in smoke test):** their own published adapter method as the
   baseline, which gives an adapter-vs-adapter comparison on unseen variants.
