# 36 — Where we fall short of SoTA: diagnosis and proposed fixes (2026-10-03, 21:19)

Scope: every Table 1 row plus the two live competitions. "Behind" = our best is worse than the published/leader number;
"ensemble-only" = ahead only with an ensemble, behind as a single model. Evidence is from doc 35 unless a new probe is
named. Proposals are ranked by expected gain per GPU-hour; none is run yet unless marked "running".

## Status at a glance

| Benchmark | Published / leader | Ours | Status | Root cause (short) |
|---|---|---|---|---|
| Learn2Design (mean loss, lower better) | -0.211 (Round 2 #1) | ~0.45 predicted (submission 411) | far behind | optimizer plateaus at ~3k of ~150k evaluations; poor donors on some topologies |
| Fusion equilibrium (S) | 0.9963 (#1 public) | 0.99459 public, rank 15 | behind 0.0017 | consistency scalars: per-shot offsets not predictable from inputs |
| RRNCO ACVRP (gap %) | 2.61 (RADAR) | 3.49 (released RRNet) | behind 0.88 pp | RADAR's asymmetry modelling; our model is RRNCO as released; hybrid pilot hurt |
| RouteFinder EAL-MD (gap %) | 8.58 | 8.63 classical; 8.596 routed hybrid (bf16, 3 seeds) | parity | seed spread (0.3-0.9 pp per variant) larger than the gap |
| Caco-2 (MAE) | 0.256 (not re-derivable) | 0.276 | behind the printed number | published selection used test data; honest reproduction 0.308 (we beat that) |
| Convection (VRMSE) | 0.0415 | 0.0403 ensemble; 0.0418 single | ensemble-only | best single model 0.7% above |
| MHD_64 (VRMSE) | 0.1160 | 0.1154 ensemble; 0.1172 single | ensemble-only | Flowers backbone single model 1.0% above |
| TRL2D, shear, rollout, RealPDE, TokaMark | - | ahead | ahead | - (register adds ~0 on TRL2D/MHD) |

## 1. Learn2Design (largest gap, deadline 16 Oct 11:59 UTC)

Diagnosis (competition_data/round2, §40):
- Early we are competitive: v1 reaches 0.37-0.58 on good-transfer topologies by 3k evaluations; the leader's 15-min mean
  (~10k evaluations) is 0.36. The leader then improves for the full 4 h (0.08 at 60 min, -0.211 at 240 min, time-to-best
  239 min on every run). v1 is flat from ~2-4k evaluations; pool restarts never beat the first basin.
- Every team runs ~10-11.5 evaluations/s on the H100 (no batching advantage): the race is the use of ~150k evaluations.
- Donor transfer: nearest strings differ in 6-9 of 21 positions; the best of 32 donors sits at a random Hamming rank;
  topology 3 stays at 2.08 (field 15-min median).
- Good public designs sit on bounds (reflectivity 0/1, tunings at multiples of 90 deg, masses 200, lengths 0.1/4000):
  the landscape is partly combinatorial; Adam in sigmoid space cannot make those jumps.
- Development is evaluation-starved: ~0.9 evaluations/s locally (FP64 on consumer GPUs), so a 150k horizon is never
  tested.

Fixes:
1. Long-horizon search (§40i probe queued; code in submission.py hop_mode): after the first plateau, short Adam segments
   from perturbed copies of the incumbent - random hops and *structured* hops (jump 1-4 parameters to bound / 90-deg
   values) - instead of pool restarts. Next step after the probe: an elite population (4-8 incumbents, PBT-style
   replace-worst) for the 150k horizon.
2. Wider donor pool (§40j scan queued): ~1,000 donors cost ~100 s of the 4 h on the H100; then the hybrid register
   ranker (code ready) for the competition's classical+qubit component.
3. Dev speed (§40k running): complex64 LU + complex128 refinement for the solves (dev only), to test 150k-evaluation
   horizons in hours. If accurate, every optimizer decision moves from 3k-evaluation proxies to full-length tests.
Expected: matching the field's 4-h p10 (~0.16) is realistic if (1) works; the leader's -0.21 needs (1)+(2) both.

## 2. Fusion equilibrium (rank 15; gap 0.0017 = consistency 0.0011, q/b 0.00031, D_LCFS 0.00027, psi 0.00005)

Diagnosis:
- Consistency scalars (kappa, li, R_axis weakest) carry ~75% of 1-S. Their error is mostly a per-shot offset (oracle
  shot-mean removal +0.0041 mean R2), but the offset is not learnable from inputs: ridge with context (§38r-pre), pooled
  shot-bias co-processor in-sample (§38t, all arms 0.0000) and out-of-fold (§38u, -0.00016).
- New today: public test inputs show no covariate shift vs local splits (mean |SMD| 0.028 vs 0.030-0.038); core Thomson
  channel geometry varies between shots but does not explain the offsets (eta^2 <= 0.005). Local-to-public differences
  are shot-sampling noise.
- q/b: the leader's q95/betaN error is half ours (R2 0.9980 vs 0.9959); our q/b heads gained the most recently (QH).
Fixes (in order):
1. Whole-shot sequence heads: a bidirectional temporal encoder over each shot's frames trained end-to-end on the scalar
   targets (not on residuals). Untested hypothesis: shot dynamics (response of the shape to coil-current changes)
   identify the plasma state that the per-frame heads miss. Gate: VALID sample mean R2 > +0.001 before any TEST.
2. q/b with the same sequence encoder (the leader's q/b error is half ours; q/b heads are where width and corrected
   inputs paid off).
3. Final phase (19-26 Oct): retrain heads/trunks on TRAIN+VALID for the private fold (+~5% data).
Honest outlook: the cheap levers are exhausted; (1) is the only untested idea with a plausible mechanism for the 0.0011.

## 3. RRNCO ACVRP (0.88 pp behind RADAR)

Diagnosis: our row is the released RRNet (reproduces 72.160 vs 72.145 published). RADAR adds SVD-initialised asymmetric
distance embeddings and Sinkhorn-normalised attention, trained on RRNCO's real-city data; its release has only a
synthetic-ACVRP checkpoint. §41 pilot: the routed register on the frozen RRNet made it worse (3.70 vs 3.49%) after one
epoch at lr 1e-3; lr 1e-4 retry running.
Fixes:
1. Test-time local search (cheap, labelled as extra compute): RRNCO's own environment ships a PyVRP local search
   (rrnco/envs/rcvrp/local_search.py). Apply it to the decoded best-of-808 solutions. Likely the largest single gain; it is
   not like-for-like with RADAR's greedy number, so it gets a dagger, as RouteFinder's sampling row did.
2. Safe hybrid: register write-back gated (per-input gate initialised closed) with a KL penalty to the frozen policy,
   so REINFORCE noise cannot pull the policy away; lr 1e-4.
3. Like-for-like: retrain RADAR on RRNCO's data (days of GPU; only if 1-2 leave the row behind and the paper needs it).

## 4. RouteFinder EAL-MD (parity)

New probe (per variant, 15 of our runs vs Table 11): the published value lies inside our seed range for all 16 variants.
Our deficit concentrates in closed-route duration-limit variants (MDCVRP, MDVRPL, MDVRPBL: +0.17 pp), while the routed
hybrid is better than published on every time-window variant (MDVRPTW 9.31 vs 9.43, MDOVRPTW 6.05 vs 6.24, MDVRPLTW 9.37
vs 9.47, MDVRPBTW 8.35 vs 8.42) and worse on duration-limit ones (MDVRPL 9.68 vs 9.31, MDVRPBL 13.08 vs 12.70).
Fixes:
1. Variant-routed register: the router already sees the variant attributes; gate the register by the time-window /
   duration-limit flags (on for TW, off for L). Expected about -0.1 pp mean if the pattern holds over seeds 4-5
   (running on mephiles).
2. More seeds (4-5 running) to state parity or a lead with a CI instead of 3-seed means.
3. Test-time sampling (-2.3% route cost measured, dagger).

## 5. Ensemble-only rows (convection, MHD) and Caco-2

- Convection: best single 0.0418 vs 0.0415 (0.7%); MHD: single 0.1172 vs 0.1160 (1.0%). Fix: one more fine-tune
  stage or a second backbone seed; low priority, since both rows are ahead with a labelled ensemble.
- Caco-2: the printed 0.256 needs CaliciBoost's per-seed test-selected subsets; honest selection gives 0.308 and ours
  is 0.276. Nothing to fix on the benchmark side. The register is 20.7% worse than classical there (906 training
  molecules: the register core is the noisiest part in a small-data regime). Fix: drop the register from that row or
  gate it on GBDT residuals; low priority.

## 6. Cross-cutting: why the register attachments failed

| Attachment | Outcome | Mechanism |
|---|---|---|
| Fusion shot-bias (§38t/u) | 0.0000 / -0.00016 | target residual not learnable out of sample |
| L2D TopoGen generator (§40g) | worse than random | MSE generation averages distinct good designs |
| RRNCO write-back (§41 pilot) | -0.2 to -0.3 pp | write-back drifts under noisy REINFORCE on a frozen policy |
| TRL2D / MHD necks | eta ~ 0 | classical part already captures the predictable part |
| RouteFinder routed | TW better, L worse | effect depends on the variant; mean washes out |

Proposed rules for every future competition attachment:
1. Learnability gate first: an out-of-fold probe (ridge or GBDT, minutes) must find learnable signal in the target before
   a register is trained on it (would have saved §38t/u).
2. Safe by construction: per-input gate initialised closed, plus a penalty on the write-back norm (or a KL term to the
   frozen policy for RL), so the hybrid cannot start below the classical model.
3. Select, don't generate: use the register to rank or route (L2D donor ranker, RouteFinder variant routing), where
   the classical part still produces the solution.
4. Route by input: where the effect has a sign that depends on the input (RouteFinder TW vs L), make the gate depend
   on it explicitly.
