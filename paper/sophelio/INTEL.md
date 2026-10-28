# Sophelio Fusion Equilibrium Challenge (NeurIPS 2026) — Recon Intel

Recon date: **2026-08-17**. All facts sourced; see Sources at the bottom. No dataset was downloaded.

- Codabench competition: https://www.codabench.org/competitions/17739/ (API: `/api/competitions/17739/`)
- Starter kit: https://github.com/Sophelio/fusion-equilibrium-challenge-starter (MIT)
- Dataset: HF `Sophelio/fusion-equilibrium-challenge` (CC BY 4.0, ~133 GB total per Sophelio PR)
- Organizers: Sophelio (Craig Michoski, CEO; Matthew Waller `mwaller`, Tapan Ganatma Nakkina
  `ganatma` are the Codabench admins) + General Atomics (DIII-D) + UKAEA FAIR-MAST + UT Austin IFS.
  DOE-funded (DE-FC02-04ER54698 + four OFES awards).
- Status at recon: **84 participants, 134 submissions, 24 leaderboard entries**. Registration
  auto-approves.

---

## 1. Task specification

Reconstruct the tokamak magnetic equilibrium from **non-magnetic diagnostics only** — no magnetic
probe/flux-loop array. Train on DIII-D (conventional tokamak), predict on DIII-D held-out shots
**and zero-shot on MAST** (spherical tokamak, no training data by design).

### Inputs (all splits, per shot = one HF row; nested arrays)

| Group | DIII-D | MAST |
|---|---|---|
| Coil currents `magnetics_*` | 18 F-coils (`F1A`–`F9B`, **kA·turn**), `ECOILA` (kA), `bcoil` (TF, kA), `plasma_current` Ip (kA) — 21 signals | P2–P6 L/U (10, **kA·turn**), `Solenoid`, `TF`, `EFPS`, Ip (all kA) — 14 signals |
| Thomson scattering | `thomson_core_*` (vertical chord, R≈1.94 m const; 40–54 ch, varies by shot), `thomson_edge_*` (horizontal, 10 or 14 ch): Te (eV), ne (m⁻³) | core (~130 ch, per-channel R 0.25–1.5 m), edge (~16 ch); horizontal midplane laser, chord Z=0 |
| Machine geometry (every row) | `coil_R/Z/width/height/angle1/angle2` — 19 lumped rectangles (matches EFIT `mhdin.dat`; 6 F-coils are skewed parallelograms), `thomson_chord_R/Z` (**per-shot variable** — 22 layouts in train, do not cache) | 812 individual conductor elements (solenoid = 656; turn count = rows sharing `coil_input_column`: P2=20, P3=8, P4=23, P5=23, P6=4), chords at Z=0 |
| Scaffolding | `efit_times` (ms; when to predict), `efit_grid_R/Z` (65-pt grid coords, m) | same |

Key input gotchas (from Data page + starter README):
- **Time bases differ wildly**: DIII-D magnetics at 20 kHz (70% of shots) or 2 kHz (M varies, 6
  lengths — never hard-code), Ip on its own ADC (30,719 samples), Thomson ~1300–1900 core /
  200–500 edge samples; EFIT ~300 frames/shot (DIII-D T: 1–445, median 241; MAST T: 5–98, median
  58). Aligning inputs onto `efit_times` **is part of the task**. Never interpolate targets.
- 114 early-campaign MAST shots use a two-grid union time base with per-column NaN holes — mask
  per column.
- Currents are already in ampere-turns for shaping coils (Green's-function-ready);
  `ECOILA` and MAST Ip/TF/sol/EFPS are plain kA. Per-machine normalization absorbs everything.
- `magnetics_dsep` is **EFIT-derived, not an input, not scored** (DIII-D = separatrix–limiter
  clearance; MAST = δR_sep divertor balance — different quantities). Ip **is** an allowed input.
- Corpus is **diverted plasmas only** (limited frames stripped from `efit_times`).

### Outputs (what you predict)

Per shot, one prediction per `efit_times` timestamp:

| Key | Shape | Meaning |
|---|---|---|
| `shot_XXXX_psirz` | `(T, 65, 65)` | poloidal flux ψ(R,Z), V·s/rad, dense grid both machines |
| `shot_XXXX_q95` | `(T,)` | edge safety factor |
| `shot_XXXX_betaN` | `(T,)` | normalized beta |

**That is the entire contract.** LCFS boundary, axis position, κ, δ, volume, li are all **derived
from your submitted ψ by the scorer** using the same functionals applied to ground-truth ψ. A
scalar-regression head cannot earn geometry points your flux map doesn't support. `q95`/`βN` are
submitted because they need `F(ψ)`/`p(ψ)` which ψ(R,Z) cannot contain.

### Dataset splits (HF configs)

| Config | Shots | Scored frames | Contents |
|---|---|---|---|
| `diii_d_train` | 7,041 | — | inputs + all targets |
| `diii_d_public_test` | 874 | 193,359 | inputs only |
| `mast_public_test` | 1,206 | 69,075 | inputs only (**no `mast_train` exists**) |

262,434 total frames ≈ 1.9 GB in float16. Final phase swaps in **two private folds of comparable
size**. MAST EFIT labels appear in **no released config** — only in 3 demo parquet shots in the
starter repo (`mast_shot_28348/28350/28351`, shipped "purely for dFL visualization").

---

## 2. Scoring — exact formulas (Evaluation page, verbatim math)

Per machine, composite `S_model ∈ [0,1]`:

```
S_model = 0.55·R²ψ + 0.15·R²{q95,βN} + 0.10·(1 − D_LCFS) + 0.20·Consistency
```
R² terms clipped at 0, `D_LCFS` clipped at 1 before blending (trivial constant scores 0).

- **R²ψ** (55%): single **pooled** R² over all grid points × all timesteps × all test shots
  (per-shot mean reported only as diagnostic). **Sign-invariant**: scorer fits one global sign bit
  per machine (never per shot/frame), reports it back as `psi_sign`. Needed because DIII-D EFIT
  stores axis at ψ-minimum, MAST EFIT++ at ψ-maximum. **Amplitude is NOT normalized** — MAST's
  flux swing is ~⅓ of DIII-D's and getting magnitude right is part of the task.
- **R²{q95,βN}** (15%): mean of the two pooled R²s of the submitted scalars.
- **D_LCFS** (10%): length-normalized symmetric Hausdorff distance between LCFS contours the
  scorer extracts from your ψ vs. true ψ, divided by shot's time-averaged true LCFS major radius.
- **Consistency** (20%): for 7 ψ-derived scalars `{R_axis, Z_axis, κ, δ_top, δ_bot, V, li}`,
  pooled R² of `f_j(ψ_pred)` vs `f_j(ψ_gt)` (same code both sides; li in ITER li(2) norm),
  `Consistency = mean_j max(0, R²cons,j)`. If derivation fails on your map where GT has a value,
  the frame is **mean-substituted (≈0 earned)**, not skipped. Six scalars well-posed on 100% of
  frames; li 99.99% (D3D) / 99.5% (MAST).

**Challenge 2 (cross-machine):**
```
G_ratio = S_MAST / S_DIII-D     admissibility gate: DIII-D R²ψ > 0.6
```
DIII-D-only entries score G_ratio = 0. One submission enters both challenges.

Awards: one per challenge (two cash awards, amounts not published). Honorable mentions: best R²ψ,
best R²{q95,βN}, best Consistency, best D_LCFS. **Compute-light badge**: self-declared fact-sheet
checkbox, "trains end-to-end <2 h on a single GPU or CPU" — a recognition, not a track.

Critical scoring trap (Submission format page): **do not round/truncate ψ** — `np.round(psi,3)`
keeps R²ψ at 0.999965 but destroys ~35% of the MAST Consistency term (MAST triangularity varies
over a narrow range). float16 is safe (what the starter kit writes).

---

## 3. Submission mechanics

**Predictions-file submission, not code submission.** Scoring runs CPU-only in a pinned Docker
image (`ghcr.io/sophelio/fusion-eq-scorer@sha256:f56a3d29…`), 5400 s execution limit, one worker
(shared queue).

- Files: `diii_d_public_test.npz` + `mast_public_test.npz` (Final phase:
  `*_private_test.npz`) — exact names, keys `shot_XXXX_{psirz,q95,betaN}` in **HF stream order,
  0-indexed**. Extra keys ignored; DIII-D predictions must be finite.
- **Recommended route**: push `.npz` to a **private HF dataset repo**; upload a few-hundred-byte
  `submission_pointer.zip` containing `manifest.json`:
  `{"repo_id": …, "revision": "<40-char commit SHA>", "token": "hf_<fine-grained read-only>"}`.
  Scorer pulls from HF CDN (~50 MB/s) instead of Codabench storage (~0.5 MB/s ≈ 1 h upload).
  Pinned SHA makes submissions immutable for the reproduction check.
- Fallback: zip the `.npz` at archive root and upload directly (counts against 15 GB Codabench
  quota; leaderboard pins only the last submission, `Force_Last`).
- Starter kit does all of it: `submission_skeleton.py --max-shots 0 --repo you/repo --read-token
  hf_…` (**default is 5 shots** — a format check that scores nearly everything missing).
  `validate_submission.py` shape-checks offline; `local_score.py --n-shots 20 --skip 7000` runs
  the **identical scorer modules** (`fusion_scoring/`, copied unmodified) on held-out train shots,
  incl. `--mode perfect|zeros` harness self-tests.

### Phases & limits

| Phase | Window | Limits | Notes |
|---|---|---|---|
| Development (phase id 29580) | Jul 27 → **Oct 18, 2026** | **5/day, 100 total** | public test fold, public leaderboard, iterate freely |
| Final (phase id 29581) | **Oct 19 → 26, 2026** | **3 total** | blind private folds, leaderboard hidden until close; "single highest-scoring Development entry per team is the basis for final ranking; a few confirmation submissions allowed" |

`auto_migrate_to_this_phase: false` on Final → **we must actively submit private-fold predictions
in the Oct 19–26 window** (rerun model on newly released private inputs, new filenames).

### Minimal valid submission / update policy

- Minimum: **one `.npz` for DIII-D only** — all 874 shots, every `efit_times` frame (T per shot),
  three keys per shot, correct filename, finite values. That places on Ch1; G_ratio shows 0.
  Adding `mast_public_test.npz` (1,206 shots) enters Ch2.
- **Yes, entries update freely through the dev phase** (5/day, 100 total). Leaderboard rule is
  `Force_Last` — **your latest submission replaces your leaderboard entry**, so a regression in
  your last submission clobbers a better earlier score on the public board. Old submissions can be
  deleted from the Resources tab (15 GB quota; pointer route ≈ 0 quota).
- Fact sheet per submission: `method_name` (shown on leaderboard), `code_url`, `compute_light`
  checkbox.

### Rules that bind us

1. One team registration, single contact email; duplicate solo aliases deactivated.
2. **External public datasets (explicitly: "other tokamak archives, OMFIT-produced equilibrium
   tables") and public pre-trained models are permitted with disclosure** in the methods report.
3. Shot-level splits mandatory; leakage/memorization of hidden GT disqualifying.
4. Prize path: top-3 per award release code (OSI license) + 1–2 page methods report + pinned
   environment; organizers **re-score leading entries from source** before awarding.
5. Submitting predictions you didn't produce = team-wide disqualification.
6. NeurIPS Code of Conduct.

---

## 4. Leaderboard snapshot (Development, 2026-08-17, 24 entries)

Source: `GET /api/phases/29580/get_leaderboard/` (public, no auth).

| # | owner | method | d3d_S | G_ratio | mast_S | d3d R²ψ | d3d qb | d3d D_LCFS | d3d Cons | mast R²ψ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pshishkin | seq + mlp (compute-light✓) | **0.9932** | 0 | 0 | 0.9998 | 0.9957 | 0.0119 | 0.9756 | — |
| 2 | dmitryzhevn | v11 | 0.9925 | 0.6525 | 0.6477 | 0.9998 | 0.9954 | 0.0122 | 0.9728 | 0.9732 |
| 3 | mnn31 | | 0.9921 | 0.5642 | 0.5597 | 0.9998 | 0.9920 | 0.0134 | 0.9737 | 0.8483 |
| 4 | ruiqi_feng | exp40 free-boundary physics | 0.9918 | **0.6536** | **0.6482** | 0.9994 | 0.9908 | 0.0115 | 0.9736 | **0.9648** |
| 5 | dgaytanv | | 0.9918 | 0 | 0 | 0.9998 | 0.9899 | 0.0133 | 0.9735 | — |
| 6 | mwaller (organizer) | | 0.9908 | 0.5564 | 0.5513 | 0.9997 | 0.9939 | 0.0139 | 0.9662 | 0.9274 |
| … | | | | | | | | | | |
| 16 | ekaterinauu | | 0.9568 | 0.6134 | 0.5869 | 0.9985 | 0.9925 | 0.0319 | 0.8097 | 0.9368 |
| 23 | **hornedyak** | | 0.6998 | **0.9427** | 0.6597 | 0.9149 | 0.6390 | 0.1371 | 0.0720 | 0.9622 |

Competitive read:
- **Ch1 is nearly saturated**: top-9 all ≥0.9877; leader's D3D R²ψ = 0.9998, Consistency 0.9756.
  Differentiation is now in the 3rd/4th decimal, mostly via Consistency + D_LCFS. Winning Ch1
  outright looks very hard by October; matching ~0.99 is table stakes for credibility.
- **Ch2 is wide open and gameable.** Best "honest" G_ratio ≈ 0.653–0.654 (dmitryzhevn,
  ruiqi_feng; both with d3d_S ≈ 0.992). But **hornedyak sits at G_ratio 0.9427 by sandbagging
  DIII-D** (d3d_S 0.6998, R²ψ 0.9149 — just above the 0.6 gate) while keeping MAST R²ψ at 0.9622.
  The gate is on R²ψ only, not on d3d_S, so the ratio rewards a deliberately mediocre-but-uniform
  model. Whether organizers will patch this is an open question — if not, the Ch2 meta-game is
  "maximize MAST absolutely, keep DIII-D no better than MAST".
- Best absolute MAST numbers seen: R²ψ 0.973 (dmitryzhevn), mast_S 0.648 — the MAST composite is
  dragged down by Consistency/scalars, not by R²ψ. That's where headroom lives.
- Several strong D3D entries have **negative** MAST R²ψ (catskills −1.09, maximml −1.40, Slash
  −200) — classic transfer failure/sign-or-amplitude blowups.

---

## 5. Starter kit & baselines (github.com/Sophelio/fusion-equilibrium-challenge-starter)

Authors: Waller, Michoski, Ganatma Nakkina, Sammuli, Boyes, Clark, Smith, Nazikian
(Sophelio / General Atomics / UT Austin IFS).

- `experiments.py` / `experiments_torch.py`: PCA-on-ψ baselines (PCA 2–3 components ≈ 99% of ψ
  variance; component 1 alone ≈ 92%). Ridge/RF/MLP on 20 PCA coefficients reach SSIM ≈ 0.87–0.90
  on ~100 shots; raw-pixel 5M-param MLP overfits badly at small n (MSE 0.239 vs 0.005 for PCA-MLP).
- Example `local_score.py` output shown in README: S = 0.8143 (R²ψ 0.9312, qb 0.7458, D_LCFS
  0.1179, Cons 0.5103) on 20 held-out shots — i.e., the naive baseline's weak spot is Consistency.
- **Cross-machine demo: coil-mapping model gets SSIM 0.83 on DIII-D → 0.10 on MAST** (transfer
  fails: flux ranges differ ~5×+sign, geometry differs, one machine's PCA basis can't represent
  the other).
- **Physics nugget** (README "worked use"): building each coil's **vacuum Green's function** from
  shipped `coil_R/Z/…` geometry + ampere-turn currents reproduces ψ outside the plasma at
  **R² ≈ 0.94 analytically** — the coil-driven part of ψ need not be learned at all.
- MODELING_GUIDE.md pushes "synthetic diagnostics" for transfer: machine-agnostic physics features
  (pe = ne·Te profiles, stored-energy proxy, q ~ Bt/Bp from TF+Ip, aggregate shaping-field
  up/down asymmetry) instead of raw per-machine coil vectors; per-sample ψ normalization ideas
  (caution: amplitude IS scored); separate PCA bases per machine + shared physics feature space.
- Warning in kit: training rows contain `efit_*` ground truth — `local_score.py` is meaningless if
  the model reads them.

---

## 6. Mapping OUR system onto the task

Our stack: frozen encoder over (coil currents + Thomson profiles) → **30-qubit Rydberg adapter**
(adaptive-register geometry, cf. wave-1/1.5 adaptive-geometry results) → decoder to ψ(R,Z).

### Where it maps cleanly

- **Encoder/adapter/decoder is exactly the required shape**: per-frame inputs → 65×65 field. The
  65×65 output and (T,) scalars fit our TORAX-Bench-style decoder heads. `q95`/`βN` need a small
  separate head (they're genuinely not derivable from ψ — the organizers designed it that way).
- **Physics-prior residual framing**: compute the vacuum/coil-driven ψ analytically from shipped
  coil geometry + ampere-turns (R²≈0.94 for free outside the plasma) and let the network predict
  only the **plasma contribution** (Grad-Shafranov source term). This is a strong, honest fit for
  the "frozen classical backbone + small quantum adapter learns the hard residual" narrative — the
  adapter's capacity is spent on the genuinely nonlinear part, and it shrinks the domain gap for
  MAST (Green's functions are machine-agnostic physics).
- **Machine-agnostic tokenization is mandatory for Ch2**: input dimensionality differs (21 vs 14
  current channels; Thomson channel counts vary per shot even within DIII-D). Encode coils as
  tokens (R, Z, w, h, skew, ampere-turns) and Thomson chords as tokens (R, Z, Te, ne) → a set
  encoder is the natural frozen front end. Anything with a fixed input vector cannot transfer.

### Framing care vs. our world-model narrative

- **This is static inverse-problem state estimation, not rollout.** Scoring is frame-wise; no
  dynamics are scored, no autoregression required. We must NOT present this as world-model rollout
  evidence. Honest framing: it is the **perception/posterior half of a control world model** —
  inferring latent plasma state from actuators + kinetic sensors, exactly what a tokamak-control
  world model consumes. "Equilibrium state estimation module."
- Temporal context is still legal and useful (leader's method is literally "seq + mlp"): inputs
  are full time series, so a causal sequence encoder over the discharge, emitting per-`efit_times`
  states, is fine. Our **TD/pushforward auxiliary objective** (wave-1.5) can be recast as a
  *self-supervised auxiliary loss* on latent temporal coherence — defensible, but it's an
  auxiliary regularizer here, not the task. Never interpolate targets; splits shot-level.
- Rounding/consistency trap matters for us: a quantum-adapter decoder that produces slightly
  blurred ψ will bleed the 20% Consistency term (κ/δ from LCFS bounding geometry) long before R²ψ
  moves. `local_score.py`'s per-scalar breakdown is the debugging tool; run it from day one.

### Zero-shot MAST ↔ adaptive-register / TTA story

- **MAST unlabeled inputs are available during the dev phase** (1,206 shots of coils+Thomson).
  Label-free test-time adaptation — input-statistics alignment, self-supervised objectives
  (masked-signal reconstruction, temporal consistency), entropy-style adaptation of the register
  conditioning — appears legal: rules ban leaking *ground truth*, not using test *inputs*.
  This is a near-perfect stage for the **adaptive-register** claim: register geometry conditioned
  on machine geometry tokens (spherical vs conventional) is a concrete, physical instantiation of
  "the quantum register adapts its geometry to the domain."
- Known transfer failure modes to design around (all documented by organizers): sign flip
  (scorer normalizes — one global bit, free), **amplitude** (MAST swing ≈ ⅓ of DIII-D — NOT
  normalized; predict scale from Ip/ampere-turns, not per-sample normalization), grid meaning
  (both 65×65 but different physical extents — decode in physical coordinates using
  `efit_grid_R/Z`, not pixel space), kidney-bean vs D-shaped topology (a DIII-D PCA basis cannot
  represent MAST — avoid machine-specific output bases).
- **Strategic fit**: Ch1 is saturated; **Ch2/G_ratio and absolute MAST S are where a
  physics-structured adapter can actually place** — current honest ceiling is mast_S 0.648, and
  the award metric literally quantifies *retention under domain shift*, which is our
  adaptive-register thesis stated as a leaderboard column. Even a mid-pack Ch1 entry with top-3
  honest G_ratio is a publishable, third-party-scored evidence point for the paper.

### Practical notes for our hardware (GB10, 121 GiB unified)

- Do not bulk-download 133 GB blindly; HF `streaming=True` works and the starter is built around
  it. If we later cache, DIII-D train dominates; per-shot parquet rows are self-contained.
- Scoring is CPU-only on their side; **no training-compute constraint at all** — the only compute
  incentive is the self-declared <2 h compute-light badge (our simulated 30-qubit adapter's
  training cost decides whether we can claim it).
- Full submission build streams the test fold twice (build + validate); budget hours, not minutes.
  Keep 5/day / 100 total in mind — `local_score.py` replaces leaderboard probing.

---

## 7. Open questions

1. **FAIR-MAST leakage boundary (the big one).** MAST test shots are real, identifiable shots
   (`MAST_25607`-style IDs) whose EFIT++ reconstructions exist in UKAEA's *public* FAIR-MAST
   archive. Rule 2 explicitly allows "other tokamak archives" with disclosure, while the fair-play
   rule makes leaking hidden GT disqualifying. Where exactly is the line — is training on public
   FAIR-MAST equilibria for *non-test* MAST shots allowed (breaking "zero-shot" in spirit but
   apparently not in letter)? Needs a forum question before we rely on any MAST-labeled signal;
   also determines how seriously to take rivals' G_ratio numbers. (The 3 labeled MAST demo shots
   in the starter repo are a miniature version of the same question.)
2. **G_ratio sandbagging** — the gate is DIII-D R²ψ > 0.6 only; hornedyak's 0.9427 via a hobbled
   DIII-D model is currently winning Ch2. Will organizers patch (e.g., gate on d3d_S or take
   max-with-floor)? Affects whether we optimize the ratio or absolute MAST S.
3. **Final-phase mechanics**: when are private-fold *inputs* released, and does the Oct 19–26
   window give enough wall-clock for our full inference + upload (3 submissions only, 5400 s
   scoring limit each)? Phase description says dev-best "is the basis for final ranking" with
   "confirmation submissions" — confirm on the forum exactly how dev scores and final-fold scores
   combine.
4. **Team identity**: one contact email per team; solo aliases get deactivated — decide our
   registration (personal vs. team account) before the first probe submission.
5. **Compute-light badge**: does adapter training on the emulated 30-qubit register fit <2 h
   end-to-end on GB10, and is the frozen encoder's pretraining counted as "training end-to-end"?
   (Self-declared, but the methods report must survive scrutiny.)
6. **Detailed results content**: submissions expose per-scalar Consistency R² and
   derivation-failure rate — verify what else leaks (e.g., psi_sign per machine) that could guide
   TTA without spending many submissions.
7. **Prize amounts** are unannounced ("two cash awards"); co-authorship on the lessons-learned
   paper + NeurIPS workshop invitations are confirmed for top teams — relevant to whether we cite
   this as third-party validation in the ICLR 2027 paper before winners are announced (final
   results presented at NeurIPS, December 2026).

---

## Sources

- Codabench API (fetched 2026-08-17): `https://www.codabench.org/api/competitions/17739/`
  (pages: Overview 84632, Data 84633, Evaluation 84634, Rules 84635, Submission format 84636;
  phases 29580/29581; leaderboard 19740; fact_sheet; terms).
- Leaderboard: `https://www.codabench.org/api/phases/29580/get_leaderboard/` (public endpoint).
- Starter kit README + MODELING_GUIDE.md (raw from
  `github.com/Sophelio/fusion-equilibrium-challenge-starter`, branch `main`, fetched 2026-08-17).
- Sophelio announcement: `https://sophelio.io/fusion-equilibrium-challenge-neurips-2026-competition/`
  (prizes, organizers, 133 GB / ~9,100 discharges figure); Yahoo Finance PR
  (`finance.yahoo.com/technology/ai/articles/sophelio-launches-first-fusion-competition-130000527.html`);
  NeurIPS 2026 competitions announcement (`blog.neurips.cc/2026/07/28/neurips-2026-competitions-announced/`).
- Note: the starter README's §6 links registration to competition id **17456** — an older/stale id;
  the live competition is **17739**.
