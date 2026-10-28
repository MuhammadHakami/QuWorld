# Allocation strategies for the classical + quantum co-processor: registered study (Sep 23 2026, before any result)

User request (Sep 23): the paper's thesis is that a quantum register should not be a stand-alone solver. Its advantage lies in continuous (analog) dynamics, so it should handle the analog part of the problem that the classical world model gets wrong, while the classical trunk keeps the rest.

Input-dependent allocation has many strategies. The question is which one is optimal for the classical + quantum split, measured across 10+ benchmarks. The goal is to show in practice what is proved on paper.

## What is proved and what is not (must stay explicit)

| Statement | Status |
|---|---|
| A pooled insertion reads and writes only the DC mode, so its gain is capped at the DC share of the error | Lean: `pool_blind`, `pool_blind_mode`, `broadcast_orthogonal` |
| A corrector restricted to a component set S can remove at most the error energy inside S | Lean: `residual_after_band_removal` (orthonormal basis) |
| Per-component energies add up exactly | Lean: `band_energy_split` |
| With a budget of K components and separable gains, top-K **by gain** is optimal | Lean: `pointwise_min_optimal`, `swap_improves`, `swap_card` |
| A quantum core extracts more of the allocated error than a matched classical core | **Not proved.** Empirical hypothesis H_Q, tested below |

The theory proves **where** the co-processor must work (the utilization half). It does not prove that qubits do the job better than a classical core of the same size. At ≤ 30 simulated atoms per register, any register is classically simulable, so no advantage claim is possible or made.

## Design: frozen trunk + routed co-processor

- **Digital part:** the released or SoTA trunk, frozen. Its predictions are cached once per split.
- **Analog part:** the trunk's residual `r = y − p`, written in an orthonormal multi-level Haar basis. One tile is (level, orientation, position) with all channels. The coarsest approximation coefficients are included as tiles.
- **Co-processor:**
  - It reads features of K allocated tiles.
  - Features are the Haar coefficients of `p − x_t` (the trunk's own predicted change), `p`, `x_t` and `x_t − x_{t−1}` at the tile, plus log energy, level and orientation.
  - It writes corrections to exactly those tiles.
  - Because the basis is orthonormal, the corrected MSE equals the stored residual energy minus the gain on the allocated tiles, exactly as in the Lean theorem. Every L2-type metric (VRMSE, rel-L2, NRMSE) is therefore computed exactly from coefficient-space energies, with no full-field reconstruction.
- **Register:**
  - The atoms are the allocated tiles, split into registers of 10 blockade-separated atoms.
  - Atom position comes from the tile (x, y from position, z from scale), so geometry is set per input by the allocation. Detunings come from tile features.
  - Readout is ⟨Z_i⟩, ⟨X_i⟩, ⟨Y_i⟩ per atom.
- **Write-back mode "res":** `corr_i = W f_i + A o_i`.
  - The linear path W is shared by every arm.
  - A is zero-init, so each core starts as the linear corrector, and the core's added value is measured directly as the increment over the `lin` arm.

## Strategies (where the atoms go, per input)

| Id | Strategy | Input-dependent? | Parameters |
|---|---|---|---|
| pooled | Global means → register → one broadcast constant per channel (the audit's insertion) | yes (DC only) | 0 |
| static | Fixed K tiles with the largest mean TRAIN residual energy (the best fixed allocation on average) | no | 0 |
| proxy | Per-input top-K tiles by the energy of the trunk's predicted change `p − x_t` | yes | 0 |
| learned | Per-input top-K by a scorer trained on TRAIN to predict residual energy (re-ranks the proxy shortlist of M = 256) | yes | ~2k |
| gain | Per-input top-K by a scorer trained to predict the co-processor's own realized per-tile gain (the Lean-optimal rule, "top-K by gain") | yes | ~2k |
| random | K random tiles per input (control) | yes, uninformative | 0 |
| oracle | Top-K by true residual energy (diagnostic ceiling only; uses the target) | — | — |

## Cores

- `lin`: no core.
- `mlp`: DeepSets MLP over each group of 10, parameter-matched to the register.
- `ryd`: the Rydberg register.
- Controls at the best strategy only:
  - `gutz`: mean-field; interactions kept, entanglement forbidden.
  - `ryd2d`: z = 0, so no cross-scale or signed geometry.
  - `mlpX4`: 4× the parameters, a capacity reference.

## Budget

- K = 30 atoms (3 × 10) is the primary setting.
- K sweep {10, 30, 100, 300} for `pooled` and `proxy`, `ryd` and `mlp`, 1 seed. K = 300 means 30 independent 10-atom patches.

## Benchmarks

Priority order, all frozen SoTA/official trunks:

1. The Well TRL2D (CNextU-net).
2. shear_flow (CNextU-net).
3. rayleigh_benard (CNextU-net).
4. planetswe (CNextU-net).
5. RealPDE foil with the U-Net trunk.
6. RealPDE foil with the FNO trunk.
7. MHD_64 (CNextU-net, 3-D).
8. Pelzer 1HP (U-Net).
9. TORAX-Bench (1-D profiles).
10. TokaMark 1-1 (TokaMind, time series).
11. TDC ADMET (discrete molecular inputs; the "digital" negative control, from the CheMeleon register arms already running).

Rows 8–11 are added only as time allows before Sep 25 AoE. Any row not run is reported as not run.

## Protocol

- **Splits:** co-processors train on TRAIN windows, early-stop on VALID, and are scored on TEST once per final configuration. Scorer and threshold choices use VALID only.
- **Seeds:** 3 per (benchmark, strategy, core) at K = 30.
- **Metrics per test sample:**
  - the benchmark's L2 metric;
  - the utilization ceiling `U = Σ_{i∈S} r_i² / ‖r‖²`;
  - the captured fraction `(‖r‖² − ‖r'‖²) / ‖r‖²`.

## Predictions and decision rules (fixed now)

**H_U, utilization governs the gain (theory → practice):**
- **P1 (accounting check):** captured ≤ U on every sample. This must hold; a violation means a bug.
- **P2:** At K = 30, every benchmark has causal routing (proxy, learned or gain) ≥ pooled on TEST. Across (benchmark × strategy) cells, realized gain increases with causal U (Spearman ρ > 0 with a bootstrap CI over benchmarks excluding 0).
- **P3:** Pooled gain is flat in K, and proxy-routed gain increases with K. This is the "utilization, not qubit count" prediction.
- **P4:** Gain routing ≥ energy routing (proxy or learned) in a majority of benchmarks. This is the Lean top-K-by-gain rule in practice.

**H_Q, a quantum core exploits the allocated analog part better:**
- `ryd` vs `mlp` (and vs `lin`) at each benchmark's VALID-best strategy, paired over seeds.
- Across benchmarks, report a sign test plus a random-effects pooled relative gain.
- **Positive only if** register > MLP in a majority of benchmarks **and** the pooled CI excludes 0.
- If positive, `gutz` and `ryd2d` decide whether entanglement or geometry carries it.
- **Otherwise** report the tie. The paper then states that allocation, not core type, carries the gain at this scale.

## What this can and cannot show

- **It can show** that the utilization theory predicts practical gains across many benchmarks, and which input-dependent strategy is best for a fixed atom budget.
- **It cannot show** a quantum advantage.
- **Frozen-trunk correction is not fine-tuning:** the audit's training-mediated TRL2D effect is a different mechanism and stays a separate result.

## Amendment 1 (Sep 23, 21:35): global context for tile co-processors

**What was seen:** after the first seed-1 cells on TRL2D and RealPDE (U-Net), every tile strategy captured ≈ 0 of its ceiling, while the pooled corrector captured about a third of its (DC) ceiling on RealPDE.

**Why that is a confound:** tile co-processors saw only their own tile's coefficients, whereas the pooled co-processor sees the window's global statistics. The comparison therefore conflated *where* the atoms write with *what information* they get.

**Change:** every tile co-processor now also receives the pooled global-context vector (TRAIN-standardized), so tile strategies have at least the pooled information.
- It is the same for every strategy and every core.
- The register reads it through its detuning map, and the matched MLP's width is re-matched automatically.

**Results handling:**
- The v0 cells are archived in `runs/alloc/results_v0/` and are not pooled with v1.
- Nothing else changes: strategies, cores, K, seeds, splits and decision rules stay as registered.

## Amendment 2 (Sep 23, 22:10): planetswe excluded

**Reason:** the released planetswe checkpoints do not reproduce on the downloaded data. VRMSE is 5–17 against a published 0.09–0.37, as recorded on Sep 19 in `21-iclr-sprint-todo-sep19.md`, where planetswe was already dropped from the benchmark list. A co-processor that "corrects" a broken trunk would inflate every strategy.

**Handling:**
- Its cache is built for the record only and moved to `runs/alloc/cache_excluded_planetswe/`.
- No grid cell runs on it.

**Anchor flags for the paper** (released checkpoints, our evaluator): shear_flow gives 0.9946 vs a published 0.8080, and MHD_64 gives 0.5447 vs a published 0.1633. Both stay in the study as "released trunk, our evaluator" rows, as in the audit, and are flagged in the table.

**Study rows:** TRL2D, rayleigh_benard, shear_flow, MHD_64, RealPDE (U-Net), RealPDE (FNO), Pelzer. That is 7 rows over 6 datasets.

## Sub-study B (registered Sep 23, 22:15, before any run): analog vs digital inductive-bias probe

**Question:** is there a class of maps on which the register beats a parameter-matched classical core? The user's thesis predicts an edge on analog (continuous, oscillatory) maps and none on digital (threshold) maps. Theory supports only a hypothesis here: register outputs are trigonometric polynomials of the detunings (Schuld–Sweke–Meyer; Lean `reupload_degree`), with frequencies set by the Hamiltonian spectrum.

**Setup:** `paper/alloc/bias_probe.py`, using the same cores and interface as the study.
- **Groups:** 10 atoms, each with feature f_i ~ U[−1, 1]^4 and a random site on a 4×4×2 grid.
- **Readout:** a linear map on the core's (Z, X, Y) output.

**Targets per atom** (u, v are fixed random unit vectors):
- `analog(ω)`: y_i = sin(ω f_i·u) · cos(ω mean_j f_j·v), for ω ∈ {0.5, 1, 2, 4, 8};
- `digital`: y_i = sign(f_i·u) · sign(mean_j f_j·v);
- `parity`: y_i = sign(f_i·u) · sign(f_{i+1}·u) (next atom in the group, cyclic).

**Cores:**
- ryd and gutz.
- mlp: nearest width to the register's parameter count. It receives the site coordinates as features.
- mlp+: the next width up, never fewer parameters than the register. Added after the smoke test showed mlp at 75 vs 84 parameters, before any sweep result.
- mlpX4.

**Training and scoring:**
- 20k training, 5k validation and 10k test groups, 3 seeds.
- Adam at 3e-3 for 3,000 steps, batch 256, best-VALID checkpoint.
- Metric: TEST R².

**Decision rule:**
- **Analog edge supported only if** ryd > mlp on TEST R² for 3/3 seeds at ≥ 3 of the 5 ω values, and ryd ≥ mlp on average over analog targets.
- **Mechanism:** gutz ≈ ryd means the edge is not entanglement.
- **Scope:** whatever the outcome, this is a statement about inductive bias at ≤ 10 atoms, not an advantage claim.

### Sub-study B2 (exploratory, post hoc: added Sep 23 22:55 after the ω ≤ 4 results)

**What was seen:** the registered register arm uses a tanh-bounded detuning encoding and loses to the MLP at ω = 4 (R² 0.87 vs 0.94–0.97). All MLPs fail at ω = 8 (R² ≈ 0).

**Question:** is that limit a property of the register family, or of our encoding? The textbook re-uploading form encodes inputs linearly into the angle, which gives a trigonometric-series function class.

**Arms:**
- `rydlin`: the same register with detuning `base + W f`, no tanh.
- `mlpsin`: a parameter-matched DeepSets MLP with sin activation, the classical control with the same trigonometric bias.
- Both are compared with the registered arms at ω ∈ {4, 8, 16}, 3 seeds, same protocol.

**Status:** exploratory. Neither arm enters the registered decision rule for sub-study B, and any result is reported as post hoc.

## Amendment 3 (Sep 23, 23:20): uniform VALID/TEST windows for rayleigh_benard and shear_flow

**What was seen:** before any rayleigh_benard cell was analysed beyond seed-1 pooled, the trunk's mean on the cached TEST windows was 1.24 (median 0.043), against 0.668 on the full split.

**Cause:** evenly spaced starts per trajectory (7 per trajectory on rayleigh_benard) put the initial transient (t0 = 3, near-rest fields, VRMSE ≈ 8) at 1/7 of the windows, against 1/196 in the full split. shear_flow uses the same design (11 per trajectory).

**Change:**
- VALID/TEST windows for these two rows are now a uniform random sample of all (trajectory, start) windows, so the mean estimates the full-split metric.
- TRAIN (already uniform) and the static set are reused.
- Old caches are archived in `runs/alloc/cache_v0_evenly_spaced/`, and the seed-1 cells computed on them in `runs/alloc/results_v0/`.

**Other rows:**
- TRL2D and MHD_64 already cover (nearly) every test window.
- RealPDE uses all TEST windows and a random VALID subset.
- Pelzer uses whole splits.
