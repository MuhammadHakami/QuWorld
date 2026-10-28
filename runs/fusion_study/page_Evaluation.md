# Evaluation

Scoring runs automatically on submission (CPU-only). Each machine gets a composite score
`S_model ∈ [0, 1]`. The competition runs **two challenges**, ranked by two columns on the single
leaderboard — an intra-machine DIII-D challenge and a cross-machine transfer challenge (see
[Two challenges](#two-challenges) below).

## Composite score (per machine)

```
S_model = 0.55 · R²ψ  +  0.15 · R²{q95,βN}  +  0.10 · (1 − D_LCFS)  +  0.20 · Consistency
```

with the R² terms clipped at 0 and `D_LCFS` clipped at 1 before the blend (so a trivial constant
prediction scores 0, not a large negative number).

**The design principle: you submit a flux map and only the two scalars that a flux map cannot
contain.** Everything else — the axis position, boundary shape, volume, internal inductance, the
x-point gap — is *derived from your submitted `ψ`* by the scorer, with the **same published
functional applied to the ground-truth `ψ`**. A separate regression head can no longer earn scalar
points that your flux map does not support: if your `ψ` does not imply the right geometry, the
Consistency term falls, and a perfect `ψ` scores every term perfectly by construction.

### R²ψ — flux-map accuracy
Pooled coefficient of determination of the predicted flux map, over **all grid points × all EFIT
timesteps × all test shots** of the machine (a single pooled R², not a per-shot average). A
per-shot mean R²ψ is also reported as a diagnostic. Both machines use a dense, finite 65×65 grid.

> **You do not have to guess the flux sign — scoring is sign-invariant.**
> The two machines' reconstruction codes store `ψ` with **opposite sign conventions**: DIII-D's
> EFIT puts the magnetic axis at the flux **minimum**, while MAST's EFIT++/FAIR-MAST puts it at
> the **maximum**. This is a provenance difference between two implementations, not physics —
> the machines agree on current direction.
>
> A model trained on DIII-D and transferred to MAST therefore predicts a *correct* MAST
> equilibrium with the *opposite* overall sign. To stop that costing you the challenge, the
> scorer determines the global sign of your submitted flux map for each machine and scores you
> under it. It is **one bit per machine**, fixed once over the whole test fold — never per shot
> or per frame — and it is reported back to you as `psi_sign` in your results (`+1` = as
> submitted, `−1` = normalized), together with what `R²ψ` would have been unnormalized.
>
> **Only the sign is normalized — never the amplitude.** Getting the magnitude of `ψ` right is
> still part of the task, and the two machines differ there too (MAST's flux swing is roughly a
> third of DIII-D's).

### R²{q95,βN} — the two predicted scalars
Mean of the pooled R² of **`q95`** (edge safety factor) and **`βN`** (normalized beta) against the
stored EFIT values. These are the only scalars you submit: both require information that is not in
`ψ(R,Z)` at all — `q95` needs the toroidal field function `F(ψ)` and `βN` the pressure profile
`p(ψ)` — so they remain genuine prediction targets.

### D_LCFS — boundary alignment
Length-normalized symmetric Hausdorff distance between the predicted and reference
last-closed-flux-surface contours, divided by the shot's time-averaged true LCFS major radius;
smaller is better. **You do not submit a boundary contour** — the scorer extracts the LCFS from
your predicted flux map with the same published procedure it applies to the ground-truth flux map,
so a perfect flux map yields `D_LCFS = 0`.

### Consistency — seven ψ-derived scalars
For each of

```
R_axis,  Z_axis,  κ (elongation),  δ_top,  δ_bot (triangularity),  V (volume),
li (internal inductance)
```

the scorer computes `f_j(ψ_pred)` from your flux map and scores it against `f_j(ψ_gt)` computed
from the ground-truth flux map **with the same function**, as a pooled R² over the valid frames:

```
R²cons,j = 1 − Σ (f_j(ψ_pred) − f_j(ψ_gt))² / Σ (f_j(ψ_gt) − mean)²
Consistency = mean_j max(0, R²cons,j)
```

The derivations are the standard EFIT-literature forms (axis = interior O-point of `ψ`; κ/δ from
the LCFS bounding geometry; `V = 2π∫∫R dR dZ` inside the LCFS; `li` in the ITER li(2)
normalization via `Bp = |∇ψ|/R` and Ampère's law). Because the same code runs on both sides,
unit and normalization conventions cancel — what is scored is whether *your* flux map implies the
same geometry as the true one.

**Valid frames — effectively all of them.** The dataset is now **diverted timeslices only**, so the
limited/diverted mask earlier versions applied to the shape scalars is gone. Measured on the
reference folds, six of the seven scalars are well-posed on **100%** of frames; only `li` falls
short, at 99.99% (DIII-D) and 99.5% (MAST). A frame leaves a scalar's average **only** where the
*ground-truth* derivation itself fails there — never because of boundary type.

If a derivation fails on **your** map where the ground truth has a value, that frame is not
skipped: it is penalized by mean-substitution and earns ~0. `D_LCFS` likewise drops a frame only
when the *reference* contour extraction fails. The per-scalar breakdown and your
derivation-failure rate appear in each submission's detailed results.

> **Why `dsep` is gone.** It was an eighth scalar. The DIII-D and MAST `dsep` columns are not the
> same physical quantity — DIII-D's is EFIT's a-file separatrix↔limiter *clearance*, MAST's is
> divertor *balance* (δR_sep) — so a single functional could not score both machines coherently.
> With the corpus restricted to diverted frames, the classification role it also played is gone.

## Cross-machine generalization (Award #2)

```
G_ratio = S_MAST / S_DIII-D            (admissibility gate: DIII-D S ≥ 0.85)
```

The fraction of DIII-D performance retained under zero-shot transfer to MAST. Values near 1
indicate near-complete transfer.

**For the award, the denominator is your best DIII-D.** On the leaderboard, `G_ratio` is computed
per submission. For Award #2, `S_DIII-D` is your team's best DIII-D composite across all its
successfully scored Final submissions, and the gate applies to that best score; `S_MAST` comes from
the submission that counts for Award #2 (see Rules → *Which Final submission counts*). With a single
Final submission the two figures are identical. The effect is that a weaker DIII-D model, in any
of your submissions, cannot raise your Award #2 score.

The admissibility gate is on the **full DIII-D composite**, and the scorer enforces it: an entry
with `S_DIII-D < 0.85` is reported as `G_ratio = 0` (a note in Detailed Results says why). This
replaced the original `R²ψ > 0.6` gate on 2026-08-18: that gate constrained only the `0.55·R²ψ`
term of the denominator, so a deliberately weakened DIII-D entry could inflate the ratio roughly
3× without breaking any rule ([starter-kit issue #7](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/7)).
Gating the composite bounds the whole denominator — and `S ≥ 0.85` implies `R²ψ ≥ 0.727`, so it is
strictly stronger than the old gate.

`G_ratio` is not capped, and values above 1 are possible in principle — they mean an entry scored
higher on the unseen machine than on the machine it trained on. The gate bounds it in practice:
with `S_DIII-D ≥ 0.85` and `S_MAST ≤ 1`, no admissible entry can exceed `1/0.85 ≈ 1.18`.

**Ties.** Ties on the composite `S_model` are broken first by `R²ψ`, then by `D_LCFS` (lower is
better).

## Two challenges

There is **one leaderboard with a column for each challenge** — click a column to rank by it. You
enter both with the same submission; include predictions for whichever machines you attempt. In
the Final phase you may nominate a different submission for each award (see Rules).

**Challenge 1 · Intra-machine (DIII-D)** — the **`Ch1: DIII-D S`** column (`S_model` on DIII-D), the
default sort. **Open to everyone**; you only need DIII-D predictions to place here.

**Challenge 2 · Cross-machine (`G_ratio`)** — the **`Ch2: G_ratio`** column
(`G_ratio = S_MAST / S_DIII-D`; admissibility gate: DIII-D `S ≥ 0.85`, enforced by the scorer). Because `G_ratio` needs
**both** machines, entering it requires a DIII-D (Challenge-1) submission **plus** MAST predictions —
a DIII-D-only entry scores `G_ratio = 0`. So you can attempt Challenge 1 alone, but Challenge 2
always includes Challenge 1.

The public (Development) leaderboards score the public test fold continuously so you can iterate;
final winners are determined on the **blind private fold** in the Final phase.

## Score yourself locally — don't spend submissions finding out

`R²ψ` is easy to compute on your own. `D_LCFS` and `Consistency` are **30% of the composite** and
need contour extraction plus the seven ψ-derived functionals — which is exactly where a
plausible-looking flux map and a good one come apart.

The starter kit ships the metric so you don't have to burn submission slots to see it:

```bash
python local_score.py --n-shots 20 --skip 7000     # held-out DIII-D training shots
```

It prints the same four-term breakdown as the leaderboard, plus a per-scalar R² for the Consistency
term so you can see *which* derived quantity your ψ is getting wrong. `fusion_scoring/` holds this
scoring program's own modules, copied unmodified — the same functionals run here and on the
platform.

What differs is the **data**: locally you score training shots you hold out, the leaderboard scores
withheld test shots. So treat it as a faithful proxy for model selection, not a leaderboard
prediction. You can confirm the harness is honest at any time — a perfect prediction must score
exactly `1.0` and zeros exactly `0.0`:

```bash
python local_score.py --mode perfect --n-shots 2
python local_score.py --mode zeros   --n-shots 2
```