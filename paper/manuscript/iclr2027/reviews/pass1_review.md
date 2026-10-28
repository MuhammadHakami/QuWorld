# Pass 1: simulated ICLR 2027 review of `main_iclr.tex` (Sep 20 draft), written Sep 23 2026

The reviewer persona reads the submission against the evidence now in the repository and against the rejection patterns mined from OpenReview (next section).

## Rejection patterns from close papers (ICLR 2024–2025 reviews and meta-reviews)

Sources:
- Public review mirrors: `smallari/openreview-iclr-peer-reviews` (19,076 ICLR 2024–2025 papers, with reviews and decisions) and `ulab-ai/ResearchArcade-openreview-reviews` (meta-reviews for the 48 ICLR 2025 quantum forums).
- OpenReview itself now serves a browser challenge to scripts, and no ICLR 2026 mirror was found.

Closest papers:

| Paper | Venue | Decision |
|---|---|---|
| Quantum-PEFT | ICLR 2025 | Accept |
| QPA, a quantum circuit-based compression perspective for PEFT | ICLR 2025 | Accept |
| Quantum Neural Fields | ICLR 2025 | Reject |
| HiQ-Lip | ICLR 2025 | Reject |
| GFlowNet ansatz design | ICLR 2025 | Reject |
| Noise-resilient PQC training | ICLR 2025 | Reject |
| Quantum entanglement for attention | ICLR 2025 | Withdrawn |
| Fully Quanvolutional Networks | ICLR 2025 | Reject |
| EQGNN for MILP | ICLR 2024 | Reject |
| EQGNN for MILP | ICLR 2025 | Accept |
| eQMARL | ICLR 2025 | Accept |
| Trainability vs dequantization | ICLR 2025 | Accept |

Recurring causes, with the verdicts they drove:

1. **The quantum component is never shown to be necessary.**
   - QNF meta-review: "lacked concrete evidence to substantiate the indispensability of quantum computers".
   - QPA discussion: "questioned the necessity of QNNs over classical alternatives".
   - eQMARL reviewer: "'None' performs similarly well … how can one understand the role played by entanglement".
2. **Simulation only, no noise or hardware.** HiQ-Lip reject, QNF reject, entanglement-attention withdrawn, and QPA reviewer 4.
3. **Scale.** Few qubits and toy tasks: the GFlowNet meta-review ("only a few qubits"), and QNF (6 qubits; "unclear why more qubits leads to worse performance").
4. **Compute cost is not reported, or is inconsistent.** QPA reviewers 1 and 5; HiQ-Lip runtime inconsistencies.
5. **Weak or selected baselines, dataset selection bias.**
   - FQN: "12 of 30 UEA datasets … selection bias".
   - EQGNN: comparison with SOTA classical GNNs, and a parameter-count mismatch letting the classical model overfit.
   - QNF: MLP baseline "lacks contextualisation".
6. **Overclaiming and misleading hardware claims.** QPA reviewer 2 on FTQC claims; the non-commutative DTI paper's "groundbreaking" language.
7. **No theory or mechanism.** In the ICLR QML set, "theory lacking" appears in 29 % of rejected vs 8 % of accepted papers. Accepted QML papers either proved something (dequantization) or answered necessity with added experiments in rebuttal.
8. **Presentation.**
   - Too much background, or claims and caveats tangled into long sentences (QNF reviewer 3, entanglement-attention).
   - Unfinished tables or placeholder cells, which weighed against PDE-Diffusion's rejection.
9. **PDE and neural-operator papers specifically:**
   - few or simple PDEs;
   - results that do not beat FNO or official baselines;
   - benchmark-only contributions sent to the Datasets & Benchmarks track (PINNacle);
   - long-horizon and generalisation evidence expected.

## Summary of the submission

QUOTA inserts a small input-conditioned Rydberg register (≤ 30 simulated atoms) at the pooled neck of released physics world models. It tests the register against parameter-matched classical controls under pre-registered kill rules. The headline is a long-horizon rollout gain on The Well's TRL2D under temporal-difference fine-tuning: Holm-significant on 2 of 3 seeds in a post-hoc, higher-power sliding-start protocol. Additional parts are a qubit-count ladder (flat), a dequantization suite (inference-inert), a mean-field control (ambiguous), an MHD null, and an 11-row benchmark table with mostly pending cells.

## Strengths

- **Unusually careful controls.** Parameter-matched MLP (to 0.001 %), a no-interaction ablation, a mean-field (Gutzwiller) control, pre-registered kill rules, Holm correction, and a trajectory cluster bootstrap. This directly answers pattern 5 and is rare in QML.
- **Anchored to released SoTA checkpoints and published numbers.** The work fine-tunes official checkpoints rather than toy baselines, which answers patterns 5 and 9.
- **Honest scope statements:** no advantage claim, and classical simulability stated.
- **Dequantization as a first-class result.** The trained register is inference-inert, so the deployed model is classical.

## Major weaknesses

**W1. The title and abstract overclaim relative to the evidence (pattern 6).**
- "A Few Configurable Qubits Improve Long-Horizon Rollouts" rests on one dataset: 9 test trajectories and 3 seeds, with the higher-power protocol registered after the official-start results were known.
- The repository's own matched-control audit over 8 SoTA-class trunks is a null: register vs trunk 5/1/2 rows (sign p = 0.22), register vs MLP 4/2/2 (p = 0.69), random-effects +0.04 % [−0.00, +0.08].
- Three later rows are exact ties between register and MLP: RealPDE U-Net, RealPDE FNO, and Sophelio SRR coupling.
- A reader who finds the ladder will read the title as cherry-picking. Only the MHD and Pelzer nulls appear in the paper.

**W2. Necessity of the quantum component (pattern 1).**
- The register is inference-inert, the mean-field control is ambiguous, the 5-qubit register equals 30 qubits, and on most benchmarks a matched MLP does as well.
- The paper never states plainly what the register adds over a classical core with the same interface. That is exactly the question that sank QNF and nearly sank QPA.

**W3. No account of *why* the register rarely matters (pattern 7).**
- The authors have a machine-checked result that explains it: the pooled bottleneck is blind to every zero-mean perturbation (Lean `pool_blind`, `pool_blind_mode`), so it sees only the DC mode.
- They also have spectral evidence on TRL2D: the DC band holds 2.9 % of the error energy, and the register's only measurable effect is at k = 0 (−0.49 % ± 0.21, 3/3 seeds).
- Neither is in the paper. The appendix instead spends a page on a speculative scaling framework ("we propose, but do not establish").

**W4. Unfinished benchmark table (patterns 8 and 9).**
- Table `tab:ladder-industrial` is half pending cells ("—", red markers).
- The prose says "eleven-benchmark program, with remaining evaluations pending", while completed rows (TokaMark, rayleigh_benard, shear_flow, TORAX, Sophelio, RealPDE) are absent.

**W5. Compute cost is not reported (pattern 4).** The limitations section says so explicitly. Reviewers will ask for it. Measured: +18 % wall-clock per step for the exact-statevector register vs +1.4 % for the matched MLP.

**W6. Hardware realism (pattern 2).**
- Simulation only.
- Signed per-layer local detunings and clamped distances are not mapped to device constraints.
- No noise beyond 128-shot readout.
- The inference-inertness result turns this into a partial strength (the deployed model needs no quantum hardware), but the paper does not frame it that way.

**W7. Statistical power and garden of forking paths.**
- Two rounds, sequential registration, and a post-hoc power extension.
- Minimum detectable effects are not given.
- Reviewers will count degrees of freedom: 5 comparators, 3 seeds, 2 protocols.

**W8. Missing related work.** QPA (ICLR 2025) is the closest "quantum at training, classical at inference" precedent. The trainability-vs-dequantization theory (ICLR 2025) bears on inertness. TokaMind, RealPDEBench, spectral bias of neural operators, and the SoTA anchors actually used should also be cited.

**W9. Presentation (pattern 8).**
- The abstract is long and caveat-dense, with parenthetical p-values.
- Contribution bullets carry their own hedges.
- The results section is a wall of numbers.
- The main text barely fits 9 pages only because of float-spacing overrides in the preamble; the ICLR 2027 template forbids format changes.
- Required statements are placed after the references, but the ICLR 2027 template puts them at the end of the main text, before the references.

**W10. Reference correctness.**
- TokaMark's leaderboard reference must be the current TokaMind Table 2: task 1-1 best is TokaMind-Scratch 0.140 (FT-Base 0.150).
- Some internal notes use 0.150 as "SoTA".
- The paper does not yet cite TokaMark numbers; when it does, it must use 0.140.

## Minor

- "Eleven-benchmark program" includes BubbleML, which was dropped for Matbench. The program itself changed; the table must match reality.
- MHD appears both as a "cross-domain grid" subsection and in the table. The row must use the length-matched seeds (the unequal-length grid is flagged in the audit).
- The Pelzer published-split caveat belongs in the table caption, not the body.
- Several claims cite 2026 preprints. Verify every one resolves.

## Scores (pre-revision)

| Criterion | Score |
|---|---|
| Soundness | 3 / 4 |
| Presentation | 2 / 4 |
| Contribution | 2 / 4 |
| Rating | 4 (marginally below) |
| Confidence | 4 |

## What would change the rating (the Pass 1 fix list)

**F1. Reframe honestly.**
- Make the title and abstract the audit: across N SoTA-class trunks, the register ties a matched classical core.
- There is one register-specific effect, TRL2D long-horizon under TD fine-tuning.
- A proven insertion-site limit explains the ties: moving the insertion to where the error lives yields gains, but a classical core captures them equally.

**F2. Add a short theory-plus-evidence section:**
- Proposition (pooling is DC-blind), machine-checked.
- TRL2D band decomposition: DC holds 2.9 % of the error energy, and the register's effect sits only at k = 0.
- Corollary: the ceiling of a pooled insertion is the DC share of the error.

**F3. Replace the pending-heavy table with a completed matched-control audit table.** Use the scorecard's primary rows plus the RealPDE and Sophelio coupling rows, with the across-row sign test and pooled effect. Move the video rows to a one-line "not attempted" note.

**F4. Add "moving the insertion" results:**
- RealPDE low-mode correctors, on two trunks: register = MLP, both below the released U-Net.
- Sophelio spectral routing: +0.0012–0.0015 with either core.

**F5. Report compute:** +18 % vs +1.4 % per step, measured on the TRL2D neck at batch 4 on a shared GPU.

**F6. A hardware-mapping and deployment paragraph.** Inference-inert means no quantum hardware is needed at deployment. Also list what would have to change for device execution.

**F7. Compress the TRL2D sections.** Per-seed tables and the full sliding-start statistics move to the appendix. Keep one table, and add minimum detectable effects.

**F8. Related work additions:** QPA, trainability vs dequantization, TokaMind, RealPDEBench, and spectral bias.

**F9. Template compliance:**
- Remove the spacing overrides.
- Place the AI-use statement (required), Ethics and Reproducibility statements before the references.
- Make the AI-use statement follow the template's disclosure structure.

**F10.** Add SoTA context as by-products, credited to classical components, in the appendix. Do not use it as a headline.
