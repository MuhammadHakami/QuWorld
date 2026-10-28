# Pass 2: simulated ICLR 2027 review of the revised draft (Sep 23, after the Pass 1 fixes)

Rating moved from 4 to 6 (marginally above). The rest of this review lists what still costs points.

## What Pass 1 fixed

- The claims now match the evidence: an audit null, one scoped effect, and a mechanism.
- The pending-cell table is gone.
- A machine-checked proposition explains the null, which answers the "theory lacking" and "necessity of quantum" patterns.
- Compute cost is reported.
- Hardware mapping is covered in the appendix.
- The template is compliant: official ICLR 2027 files, no layout overrides, statements before the references.
- The register pooling is now described correctly (the Sep 20 draft wrongly said "attention pool").

## Remaining weaknesses

**R1. The proposition-to-evidence link is looser than the prose implies.**
- The proposition is about the *neck latent*: the register reads and writes only its spatial DC mode.
- The band measurement is on the *output field*.
- A DC change at the neck can in principle reach other output bands through the nonlinear decoder.
- The text says "the register acts where the proposition says it can". The correct statement: the proposition bounds what the register can see and write at the neck, and the output measurement shows the effect stayed at output k = 0, so the decoder did not spread it.

**R2. The abstract hides the post-hoc status of the 81-rollout protocol.** "Holm-significant on two of three seeds in an 81-rollout protocol" should say that the protocol was registered after the official-start results.

**R3. "Released state-of-the-art physics world models" overstates the trunks.**
- The TRL2D arena FNO is internal.
- TORAX-Bench is our own benchmark and trunk.
- The Sophelio row uses the starter-kit U-Net.
- Five of eight trunks are released or official-pipeline models.

**R4. Two different Sophelio pipelines are used without saying so.** Table 1 uses the starter pipeline (S ≈ 0.79); §5.4 uses the full-data pipeline (S ≈ 0.986). Readers will be confused.

**R5. Atoms vs qubits.** The title says qubits, the text says atoms. Define once: one atom is one qubit.

**R6. The red placeholder in Table 6** (Sophelio TEST) must be resolved before submission. The single TEST scoring is running; the rule was committed before any TEST score.

## Minor

- The scaling figure's axis text is small (existing figure). Acceptable in a figure, but regenerate at a larger font if time permits.
- There is space for about two more pages. Consider a schematic of the three insertions (pooled, band-limited, routed) in a later revision.

## Scores (post-Pass-1)

| Criterion | Score |
|---|---|
| Soundness | 3 / 4 |
| Presentation | 3 / 4 |
| Contribution | 3 / 4 |
| Rating | 6 |
| Confidence | 4 |

Fixes R1–R5 are applied in Pass 2. R6 is blocked on the running TEST scoring.
