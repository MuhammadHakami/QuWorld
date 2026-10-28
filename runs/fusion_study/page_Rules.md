# Rules

1. Each team registers on Codabench with a single valid contact email; aliases that already exist
   as solo participants are deactivated.
2. External public datasets (other tokamak archives, OMFIT-produced equilibrium tables) and
   publicly available pre-trained vision or scientific foundation models are permitted, with
   explicit disclosure in the methods report. **See rule 3 for the one exception.**
3. **No MAST equilibrium data.** Equilibrium reconstructions of MAST or MAST-U from any source —
   the FAIR-MAST archive, mirrors of it, datasets derived from it, or model weights trained on it —
   may not be used to train, fine-tune, calibrate, validate or select any model or component that
   produces MAST predictions, including hyperparameter search and checkpoint selection. This covers
   flux maps, LCFS contours, EFIT-derived scalars, and aggregates of them such as bases or priors.
   The withheld MAST targets are drawn from that archive, so using it is training on the test set.
   **Permitted and unaffected:** the MAST inputs shipped with the challenge and public MAST *input*
   data, used without targets — including self-supervised pretraining, target-domain normalization,
   distribution alignment and label-free test-time adaptation; MAST machine geometry, grids, wall
   and coil descriptions; equilibria from other machines; and synthetic equilibria from your own
   free-boundary solver, which may be calibrated against MAST inputs but not against MAST EFIT
   reconstructions. The three labelled MAST demo shots in the starter kit are for format inspection
   and visualization only. **The competition's own feedback is not covered:** the scores and
   Detailed Results the Development phase returns for your submissions may be used to compare, tune
   and select MAST pipelines, and that use must be stated in the methods report. The line is
   whether you can score against MAST equilibria yourself, offline — that is barred; the scores the
   platform returns for a submission are not. **Probes are not feedback:** a submission built to
   measure the labels rather than to be judged as a candidate (for example a constant, shifted or
   rescaled prediction) reveals label statistics, and those statistics — and any constant fitted to
   them or chosen by computing scores offline from them — count as MAST equilibrium data. The
   starter kit's `fusion_scoring/masks/mast_envelope.npz` is a union of MAST LCFS contours: use it
   only inside the official scoring derivation, never as a model input, feature, loss mask or prior.
   Effective 2026-09-23 (feedback clause 2026-10-01; probe and mask clauses 2026-10-04);
   Development-phase entries are not judged retroactively.
4. No restriction is placed on programming language or framework. AI coding assistants may be used
   for any part of an entry, provided the workflow is disclosed in the methods report and the entrant
   takes responsibility for the submission.
5. All train/test splits must respect shot-level boundaries.
6. **Every Final-phase submission must be accompanied, at submission time, by a reproduction package
   reference:** a Git repository at a pinned commit SHA on any public forge, or a versioned archive
   on any public host identified by its SHA256. The package must contain model source, a pinned
   environment specification, the trained weights used for the submission, a training entrypoint
   that rebuilds the model from scratch, an inference entrypoint that regenerates the submitted
   predictions, and a data-provenance statement per machine. Declare your training compute budget;
   training must be reproducible within 24 GPU-hours on a single 80 GB A100-class GPU or equivalent.
   Only GPU work counts toward that ceiling: CPU-only data generation (for example synthetic
   equilibria from your own solver) must be declared with its hardware and wall-clock but is not
   capped, and the package may ship the generated data with SHA256 checksums for the rebuild to
   start from, provided the code that generates it is included.
   Organizers rebuild the **top five entries per challenge**.
7. Top three teams in each award category must release source code under an OSI-approved licence
   and submit a 1–2 page methods report before prizes are paid.
8. Organizing-team members with access to the hidden ground truth are excluded from prize
   eligibility.
9. All participants follow the NeurIPS Code of Conduct.

## Submissions

- **Limited to 5 per day and 100 in total per team** to discourage leaderboard probing.
- Splits are strictly **shot-level** — never by timestep.
- Private-fold EFIT reconstructions are never released; leakage or memorization is disqualifying.
- **Ties** on the composite `S_model` are broken first by `R²ψ`, then by `D_LCFS` (lower is better).
- A Final-phase submission lost to a **platform fault** (stuck in the queue, or failed for a reason
  on our side) does not cost you a slot: email the organizers and we remove it so the slot returns.
- **Every submission must be your own team's work.** Submitting predictions you did not produce —
  by pointing your `manifest.json` at another team's repo, or by uploading their files as your
  own — is plagiarism and is **disqualifying for every member of the team**.
- To make that enforceable, **organizers rebuild and re-score the leading entries from source**
  before any prize is awarded (see *Prize eligibility*). A submission whose score cannot be
  reproduced from the code and environment the team hands over is removed from award consideration.
  A pinned commit SHA is required precisely so that what was scored cannot be changed afterwards.
- The recommended submission route keeps your predictions in a **private** repo read through a
  token scoped to it alone, so nothing you submit is exposed to other teams (see **Submission
  format**). Organizers can read submitted repos in order to re-score them, and nothing else.

## Awards

One award per challenge — the mapping is one-to-one.

- **Challenge 1 (intra-machine) → Award #1:** highest composite `S_model` on the hidden DIII-D test
  set.
- **Challenge 2 (cross-machine) → Award #2:** highest `G_ratio = S_MAST / S_DIII-D`, among entries
  with a DIII-D composite `S ≥ 0.85` (raised 2026-08-18 from `R²ψ > 0.6`; the scorer enforces the
  gate, so an entry below it shows `G_ratio = 0` — see the Evaluation page). Needs predictions for
  **both** machines — a DIII-D-only entry scores `G_ratio = 0`. `G_ratio` is not capped; the gate
  bounds any admissible entry at `1/0.85 ≈ 1.18`. **For the award, `S_DIII-D` is your team's best
  DIII-D composite across all its successfully scored Final submissions**, and the gate applies to
  that best score; `S_MAST` comes from the submission that counts for Award #2. With a single
  Final submission this is exactly the leaderboard's `G_ratio`; the organizers compute it otherwise.
- **Honorable mentions:** best `R²ψ`, best `R²{q95,βN}`, best `Consistency` (the ψ-derived
  scalar agreement), best `D_LCFS` — each judged on the **MAST** blind private fold, using the
  submission that counts for Award #2. Compared at full precision; exact ties are shared.
- **Which Final submission counts.** By default, your most recent Final submission that scored
  successfully counts for both awards and the honorable mentions; a submission that fails to score
  never displaces an earlier one. You may instead nominate one scored Final submission for Award #1
  and a different one for Award #2, by emailing `fusionec26@sophelio.io` with the submission IDs
  before the Final phase closes (2026-10-26 00:00 UTC). Because Award #2 divides by your best DIII-D score (above),
  nominating a weaker DIII-D model for Award #2 cannot raise your `G_ratio`.
- **Compute-light badge:** models that train end-to-end in under 2 h on a single GPU *or CPU* (self-declared
  via the submission fact sheet). It applies across **both** challenges — it is a recognition, not a
  separate track or leaderboard.

## Prize eligibility

Award candidates are validated by **rebuilding them, not only by re-running their predictions**.
From the reproduction package required with every Final-phase submission (rule 6), the organizers
retrain the model from its training entrypoint, re-score the result, and separately check that the
shipped weights reproduce the submitted predictions. **We do this for the top five entries per
challenge**; the binding prize condition remains the top three of rule 7, and asking five leaves
room when an entry above cannot be reproduced.

Final prize candidates additionally submit a 1–2 page methods report. An entry that cannot be
rebuilt from what it hands over stays on the leaderboard but is ineligible for an award.