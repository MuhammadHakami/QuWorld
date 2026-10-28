# Sophelio Entry Plan — Architecture, TTA, Submission Mechanics, Timeline

Scaffold date: 2026-08-17. Companion docs: [INTEL.md](INTEL.md) (recon),
[download_plan.md](download_plan.md) (dataset fetch). Working code:
`baseline/sophelio-starter/` (starter kit clone + our `my_experiments/`).

## 0. What the scaffold established today (measured, not assumed)

All run CPU-only on the 3 local DIII-D demo shots (`parquet_data/`, pulled via the
LFS media endpoint) with the **vendored official scorer** (`fusion_scoring/`, metric
v3.1) — identical code to Codabench.

| Check | Result |
|---|---|
| Scorer harness self-test (`local_score.py --mode perfect`, 2 HF-streamed shots) | **S = 1.0000 PASS**; HF streaming works unauthenticated |
| Raw vacuum physics ψ (F-coils, analytic Green's functions) | R² outside plasma envelope **+0.04** — vacuum field alone explains almost nothing, because the plasma's 1.4 MA dominates ψ even outside the envelope |
| Free per-frame fit of GT on the 19-coil Green's basis (outside envelope) | R² **0.966** (coils) / **0.999** (+ Ip blob regressor) — reproduces the README's "R²≈0.94" claim and validates our Green's-basis code |
| **Calibrated Green's-basis linear model** (21 global params: per-coil effective turns + Ip blob + offset; trained on 1 demo shot, officially scored on 2 held-out) | **S = 0.3895** — R²ψ 0.7007, R²{q95,βN} < 0 (zeros), 1−D_LCFS 0.041, Consistency **0.0000**, LCFS extraction fails on 87% of frames |

Three design-driving conclusions:

1. **The README's vacuum-prior R²≈0.94 is a *fit-capacity* number, not a zero-knowledge
   prediction.** The Green's basis is the right scaffold, but the coefficients must be
   *learned* (calibration + plasma term), not taken from physics constants. Our
   "residual over the vacuum prior" architecture should treat the per-coil Green's maps
   as conditioning features with learned calibration, plus a learned plasma-contribution
   field — not as a fixed additive prior.
2. **Consistency (20%) + D_LCFS (10%) is where naive models die**: an R²ψ = 0.70 model
   earns exactly 0.00 on both. Blurred or slightly-warped flux maps bleed 30% of S long
   before R²ψ moves. Sharp, contour-faithful ψ is the actual objective;
   `local_score.py`'s per-scalar breakdown is the debugging tool from day one.
3. Units as *actually shipped* (measured on demo shots, contradicting the README's
   units table which claims "A"): DIII-D F-coils **kA·turn** (max ≈ 183–390),
   Ip/ECOILA/bcoil **kA** (Ip max ≈ 1378). Multiply by 10³ for SI. INTEL.md was right.

## 1. Architecture mapping (our stack → this task)

```
inputs (per shot, irregular)                         frozen/trained
──────────────────────────────                       ──────────────
coil tokens   (R, Z, w, h, θ1, θ2, I(t))  ┐
thomson tokens (R, Z, Te(t), ne(t))       ├─► set-tokenized encoder ──► per-efit_times
global tokens (Ip(t), TF, grid extent)    ┘    (frozen after pretrain)     latent z_t
                                                                            │
                       30q Rydberg adapter (adaptive register geometry, ◄───┤
                       conditioned on machine-geometry tokens)              │
                                                                            ▼
ψ̂(t,R,Z) = Σ_j ĉ_j(z_t)·I_j(t)·G_j(R,Z)  +  Ip(t)·Ĝ_plasma(z_t; R,Z)  +  f_θ(z_t; R,Z)
            └── learned coil calibration ──┘   └── learned plasma field ──┘  └ residual ┘
q̂95(t), β̂N(t) = scalar head(z_t)          (genuinely non-derivable from ψ — by design)
```

- **Set tokenization is mandatory for Ch2**: input dimensionality differs (21 vs 14
  current channels; Thomson channel counts vary per shot *within* DIII-D — 22 chord
  layouts in train; MAST ships 812 conductor elements). Anything with a fixed input
  vector cannot transfer. MAST elements aggregate per `coil_input_column` (P2=20,
  P3=8, P4=23, P5=23, P6=4 turns) or stay element-level with shared embeddings.
- **Vacuum Green's basis as conditioning features**: `my_experiments/vacuum_prior.py`
  now computes G_j(R,Z) analytically from shipped `coil_R/Z/width/height/angle1/angle2`
  (incl. the six sheared-parallelogram F-coils) — machine-agnostic physics, the concrete
  bridge for zero-shot MAST. Learned ĉ_j replaces the ambiguous turn conventions
  (ECOILA's is genuinely unresolved; the linear fit assigns it −150×).
- **Decode in physical coordinates** using `efit_grid_R/Z` (both 65×65 but different
  extents: DIII-D R∈[0.84,2.54], MAST R∈[0.06,2.0], Z∈[−2,2]) — coordinate-conditioned
  decoder, never pixel space, never a shared PCA basis (kidney-bean vs D-shape).
- **Temporal context is legal and the leader uses it** ("seq + mlp"): causal sequence
  encoder over the discharge emitting per-`efit_times` states. Our TD/pushforward
  objective (wave-1.5) recast as a *self-supervised auxiliary loss* on latent temporal
  coherence — an auxiliary regularizer here, not the task. Never interpolate targets;
  shot-level splits.
- **Aux supervision to sharpen ψ**: train-only labels (`efit_r_axis/z_axis`, `efit_li`,
  LCFS contours `efit_lcfs_r/z`) supervise the decoder so extracted contours are crisp —
  directly targets the 30% Consistency+D_LCFS block that killed the linear baseline.
  Consider a differentiable axis/contour penalty, and NO per-sample ψ normalization
  (amplitude is scored; predict scale from Ip/ampere-turns).
- **Framing discipline for the paper**: this is the *perception/posterior half of a
  control world model* — equilibrium state estimation from actuators + kinetic sensors.
  Not rollout evidence. The quantum adapter's capacity is spent on the genuinely
  nonlinear plasma residual; the coil-driven part is analytic.

## 2. Ch2 strategy — TTA on MAST (target: honest G_ratio ≥ 0.654)

Assets available *now*, no submissions spent:
- 1,206 unlabeled MAST test shots (coils+Thomson) streamable during the whole dev phase.
- **3 labeled MAST demo shots** in the starter repo (28348/28350/28351, with
  `efit_psirz`) + the shipped `fusion_scoring/masks/mast_envelope.npz` → we can run a
  *local MAST S-proxy* (adapt `my_experiments/vacuum_prior.py`'s scoring driver with
  `MACHINE="MAST"`). Tiny n, but catches sign/amplitude/topology blowups for free.

TTA recipe (per our bake-off recipes, in escalating order of ambition):
1. **Input-statistics alignment**: refit per-machine normalization on MAST test inputs
   (legal: inputs, not GT). Absorbs the kA·turn↔kA mess and the ~5× flux-scale gap.
2. **Self-supervised consistency at test time**: masked-signal reconstruction over
   coil/Thomson tokens; temporal-coherence (TD-pushforward) loss on latents across
   `efit_times`. Adapt only norms + register-geometry conditioning; encoder/decoder
   stay frozen.
3. **Adaptive-register geometry conditioned on machine tokens** (spherical vs
   conventional) — the concrete instantiation of the adaptive-register thesis; Ch2's
   G_ratio column *is* our retention-under-domain-shift claim, third-party scored.
4. Entropy-style adaptation of register conditioning if (2) is stable.

Failure modes designed out up front (all documented by organizers): sign (scorer fits
one global bit — free), **amplitude** (MAST swing ≈ ⅓ of DIII-D, NOT normalized —
predict from Ip/ampere-turns), grid meaning (physical-coordinate decoder), 114
early-campaign MAST shots with two-grid union time base (mask per column, `np.isfinite`).

Open rulings before we rely on anything MAST-labeled: FAIR-MAST leakage boundary
(INTEL §7.1 — forum question) and whether G_ratio sandbagging gets patched (§7.2 —
decides ratio-optimization vs absolute MAST S; we build for absolute MAST S either way).

## 3. Submission mechanics checklist

One-time setup:
- [ ] Decide team identity (one registration, one contact email; solo aliases get
      deactivated — INTEL §7.4) **before** the first probe.
- [ ] Register at codabench.org/competitions/**17739** (the starter README's §6 link
      to 17456 is stale).
- [ ] HF **write** token → `uv run huggingface-cli login` (stays local).
- [ ] Create private predictions repo:
      `uv run python submission_skeleton.py --max-shots 5 --repo <us>/fusion-eq-predictions`
- [ ] Create **fine-grained read-only** token scoped to that repo (script refuses
      write/classic tokens by design).

Every submission:
- [ ] `uv run python submission_skeleton.py --max-shots 0 --repo <us>/fusion-eq-predictions --read-token hf_…`
      (**default is 5 shots** — a format check that scores ~everything missing; the
      real thing needs `--max-shots 0`). Streams the test fold twice (build+validate):
      budget **hours**, not minutes — or run inference from the local dataset copy and
      score-file via `--pred`.
- [ ] ψ as **float16** (skeleton default), scalars float32. **NEVER `np.round(psi, 3)`**
      — keeps R²ψ at 0.9999 while destroying ~35% of MAST Consistency.
- [ ] DIII-D file mandatory + all-finite; add `mast_public_test.npz` to enter Ch2.
      Keys `shot_XXXX_{psirz,q95,betaN}` in HF stream order, 0-indexed.
- [ ] Offline gate first: `validate_submission.py <file> --config <config> --max-shots 0`,
      then `local_score.py --pred <file> --n-shots 20 --skip 7000` — the 5/day, 100-total
      budget is for the leaderboard, not for debugging.
- [ ] Upload `submission_pointer.zip` (manifest: repo_id + pinned 40-char SHA + read
      token; ~0 quota) on the Submit tab. Fact sheet: `method_name`, `code_url`,
      `compute_light` checkbox.
- [ ] **`Force_Last`**: the latest submission replaces our leaderboard entry — never
      let a regression be the last dev-phase submission; re-submit the best-known-good
      before any pause.

Final phase (Oct 19–26): `auto_migrate = false` → we must **actively** rerun the model
on the newly released private-fold inputs and submit `*_private_test.npz` (3 slots
total, 5400 s scoring limit each). Plan a full dry run of this pipeline in early Oct.

Prize path (if leading): OSI-licensed code + 1–2 page methods report + pinned env;
organizers re-score from source. Keep the training pipeline reproducible from day one.

## 4. Compute & timeline

| Window | Work | Compute |
|---|---|---|
| **Now → Aug 29** | Scaffold (done): starter kit cloned, scorer exercised, Green's-basis code written, floors measured. Forum questions posted (leakage boundary, G_ratio gate, final-phase mechanics). Registration decision. | CPU only |
| **~Aug 29 (GPU frees up)** | Fetch test folds (10.4 GB) then `diii_d_train` (93.4 GB) per [download_plan.md](download_plan.md). Build streaming-free data pipeline + per-shot preprocessing cache. **First probe submission** (Green's-fit-class floor) to verify mechanics end-to-end — costs 1 of 100. | network/disk |
| **Sep (dev iterations)** | Frozen encoder pretrain → adapter training → ψ decoder; ablations vs matched classical control (paper needs the pair, the leaderboard doesn't). MAST TTA experiments gated by the local 3-shot MAST proxy. `local_score.py` on ≥100 held-out train shots as the selection metric; leaderboard ≤2 submissions/week, target ≤40 total used by Oct 1. Ch1 sanity bar: composite ≥0.95 for credibility (top-9 are ≥0.9877; we are not winning Ch1 and don't try). | GB10, shared with paper runs — adapter jobs are small; schedule around phase-B-style overnights |
| **Oct 1–18** | Freeze best DIII-D + best MAST-TTA checkpoint; confirm the *last* dev submission is our best (Force_Last); fact sheet + code_url; compute-light claim decided by measured adapter wall-clock (<2 h end-to-end on GB10 or don't claim it). | |
| **Oct 19–26 (final)** | Download private-fold inputs immediately on release; inference + build + pointer-push; 3 submissions: (1) main entry early, (2) confirmation, (3) reserve. Budget a full day of wall-clock for the first. | inference only |

Full-data commands (once the dataset lands, from `baseline/sophelio-starter/`):

```bash
# baselines on real data volume
.venv/bin/python experiments.py --source local --local-data-dir ../sophelio-data --n-shots 200
# our model → predictions file → offline gate → official local score
.venv/bin/python validate_submission.py submission/diii_d_public_test.npz --config diii_d_public_test --max-shots 0
.venv/bin/python local_score.py --pred my_preds_holdout.npz --n-shots 41 --skip 7000
```

## 5. Open questions (delta vs INTEL §7)

Carried over: FAIR-MAST leakage boundary; G_ratio sandbagging patch; final-phase input
release timing; team identity; compute-light scope; detailed-results leakage for TTA.
New from today:
1. **ECOILA turn convention** — physics-constant inclusion *hurts* (R² 0.04→−0.09);
   learned calibration handles it, but ask on the forum anyway (affects any
   physics-fixed prior on MAST's solenoid too).
2. README units table says DIII-D currents ship in "A"; the data says kA·turn / kA.
   Doc bug worth confirming so our MAST unit handling (×10³, ÷turns) is anchored.
3. The 3 demo DIII-D shots are consecutive same-campaign shots (203702–4) — our
   S = 0.39 floor is optimistic for its class; re-measure on ≥20 random held-out train
   shots once the dataset lands before quoting it anywhere.
