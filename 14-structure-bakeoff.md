---
title: Structure-Search Bake-off — full design (AG-2c)
date: 2026-08-16
project: quworld
tags: [bakeoff, evolution, eprop, spsa, design]
status: adopted-2026-08-16
---

# ON-THE-GO STRUCTURE OPTIMIZATION BAKE-OFF (AG-2c design)

Substrate for all strategies: the `kmix30` K=4 bank (`paper/qlayer.py::KMixtureRegister` — 4 × MultiRegister(3×10 qubits) + Gumbel-softmax router), frozen world-model backbone. Structure params exposed to adaptation: 4×60-dim atom coordinates (or log-couplings w_ij), per-site detunings, router logits — ~200–1,000 params total. Constraints hard-enforced in a sealed evaluator (5 µm floor, 38 µm FOV, gauge canonicalization), never as penalties (AI-CUDA-Engineer / Factorio lesson).

---

## (1) STRATEGY ROSTER

### (a) Gradient v2.1 — adjoint reference (known recipe)
- **Update**: Adam(lr 0.05–5.0 cosine) on ∂L/∂θ, full backprop through statevector sim. L = L_push (stop-grad pushforward, backprop last step only, 2202.03376) + c·L_TD (EMA-target latent consistency, c=0.1–0.5, ρ=0.5, H=3–5). Params = bounded log-couplings w_ij ~ log V_ij → differentiable stress-layer realization → positions; exponent homotopy R⁻ᵖ→R⁻⁶; positions on slower timescale than drives.
- **Compute/step**: 1 fwd + 1 bwd through 3×2¹⁰ statevector ≈ 3 eval-units (bwd ≈ 2× fwd).
- **HIL executability: NONE.** Requires adjoint through analog evolution; stochastic parameter-shift (Banchi–Crooks 2005.10299) needs mid-evolution gate insertion — unavailable on Fresnel/Aquila. Simulation-only oracle. This is the bake-off's ceiling, not a deployable arm.

### (b) Evolutionary population-of-registers (QD)
- **Representation**: genome = one bank slot = (60 bounded coords [3 regs × 10 atoms × 2D] with min-dist repair operator, router-row logits). Filesystem genomes `runs/bakeoff/population/*.json` + append-only JSONL lineage (parent id, operator, fitness, accept/reject) — Prime-Agent discipline, gives free operator-ablation data.
- **Operator portfolio** (≥5 distinct operators per lineage, General-Agent diversity constraint): per-atom Gaussian jitter σ=0.05·FOV (~1.9 µm); pairwise atom swap; sub-register block crossover (exchange a whole 10-atom register between parents — CycleQD merge analog); Gram-spectrum mutation (perturb eigenvalues of pair-distance Gram, structure-preserving SVD analog); lattice re-seed; register→register atom transfer; evolved split-point crossover (M2N2 2508.16204).
- **Fitness**: two-tier, QuantumNAS trick (2107.10845). **Tier-0**: inherit adapter+router weights from the trained bank, score = val-slice VRMSE forward pass (~2 s GPU, no retraining). **Tier-1** (elites only): one 5-min sprint (router-only or 2,500-step) → sprint-val rollout VRMSE (or TD error on the stream for the continual arm). Per-sample fitness sharing (M2N2) to protect niches around samples the 0.1934 config fails.
- **Generation vs 5-min-sprint budget**: tier-0 supports CMA-MAE batch 36 freely (~1,800 evals/hr); tier-1 at 12 sprints/hr → λ=8–12 elites/generation, ~8 generations per overnight (96 sprint slots). Matches EVQE's real-hardware regime (pop 8, gen cap 2 per window).
- **QD archive**: pyribs CMA-MAE, GridArchive 20×20, learning_rate 0.01, 3–5 EvolutionStrategyEmitters (ranker='imp', σ₀=0.05·FOV). Behavior descriptors: (1) blockade-graph edge density (frac pairs r_ij<R_b); (2) interaction-spectrum entropy of J_ij=C₆/r_ij⁶ eigenvalues; optional dim-3: cross-register coupling ratio (parameterizes the 3×10 factorization). Novelty gate: reject mutants within ε of archived descriptor vectors before paying for a sprint (AI-Scientist/ShinkaEvolve pattern, sim threshold ~0.99). Parent sampling DGM-style: p ∝ sigmoid((0.1956−VRMSE)/τ)·1/(1+children); never discard.
- **Library (one aarch64 GB10 box)**: **pyribs v0.11.0** — pure NumPy/CPU, pip-clean on aarch64, zero GPU footprint, evolves alongside campaign jobs. cmaes/pycma as scalar fallback; CBXpy polarized-CBO as the K-niche variant. Avoid QDax/evosax (JAX would contend for the GPU).
- **Compute/adaptation step**: λ tier-0 fwd passes (λ eval-units) + occasional tier-1 sprint.
- **HIL: HIGH** — evaluation = submit jobs, read scalars; precedents: EVQE on IBMQ Vigo, (1+λ)-ES on NISQ (2012.13453), Ebadi 289-atom closed loop.

### (c) e-prop / three-factor online rule
- **Trace (factor 1, measured-data-only)**: eligibility of detuning δ_j w.r.t. site i: `e_ij^t = λ_tr·e_ij^{t-1} + C_ij^t`, with C_ij^t = ⟨Z_iZ_j⟩_t − ⟨Z_i⟩_t⟨Z_j⟩_t the measured connected correlator — Kubo linear-response surrogate for ∂⟨Z_i⟩/∂δ_j; λ_tr = exp(−Δt/τ) plays e-prop's α.
- **Learning signal (factor 2)**: L_i = Σ_k B_ik·δ_k, fixed random B (random e-prop, Bellec 2020) or B = current readout rows (symmetric); DFA random projection covers the classical readout hop (Nokland 1609.01596, Nakajima Nat Comms 13:7847).
- **Third factor**: broadcast scalar TD error δ_t from the frozen world model (the E5 TD signal). **Update**: Δδ_j = −η Σ_t δ_t Σ_i B_i e_ij^t (reward-based e-prop form, F_γ low-pass on the product).
- **Geometry for free**: chain through V_ij: ∂V_ij/∂r_ij = −6C₆/r_ij⁷, so Δr = (ZZ-trace) × (analytic V-derivative), aggregated per atom, then 5 µm repair + gauge canonicalization.
- **Caveats (state in paper)**: equal-time correlator ≠ true retarded susceptibility → biased trace (analog of e-prop dropping cross-unit paths); ignores router-mediated credit (DFA covers it approximately); possible sign errors deep in blockade — periodic sign-check against an SPSA/MGD probe. Validation: 10-qubit statevector, cosine(trace update, stochastic-PSR oracle gradient); accept if biased-but-descent-aligned. Fallback factor-1: MGD orthogonal perturbation codes per parameter across jobs, demodulate scalar TD error (McCaughan 2303.03986).
- **Compute/step**: 1 fwd pass + O(30²) trace arithmetic = 1 eval-unit. Cheapest arm.
- **HIL: HIGH** — consumes only ⟨Z_i⟩, ⟨Z_iZ_j⟩ (already measured) + one scalar; no adjoint, no mid-evolution gates, zero extra hardware cost per step.

### (d) SPSA / QN-SPSA — hardware-standard control
- **SPSA**: ĝ_k = [L(θ_k+c_kΔ_k) − L(θ_k−c_kΔ_k)]/(2c_k) · Δ_k⁻¹, Δ_k ~ Rademacher ±1; θ_{k+1} = θ_k − a_k ĝ_k; a_k = a/(A+k+1)^0.602, c_k = c/(k+1)^0.101, A ≈ 10% of max iters; Qiskit-style calibration (c = shot-noise scale, first step ≈ 2π/10 magnitude, 25–50 cal evals — budget this per new geometry, SPSA is hyperparameter-sensitive per 2310.09468).
- **QN-SPSA** (Gacon, Quantum 5:567): θ_{k+1} = θ_k − a_k(F̄_k+βI)⁻¹ĝ_k, F̂ from 4 overlap + 2 loss evals = 6 evals/step; resamplings 4–8 under shot noise; hessian_delay N (plain SPSA first).
- **Compute/step**: 2 eval-units (SPSA) / 6 (QN-SPSA), independent of d — the point of the control.
- **HIL: HIGHEST** (published standard on this hardware class).

---

## (2) BAKE-OFF PROTOCOL

- **Common substrate**: single shared checkpoint — E2's winning kmix30 bank trained at tcool=0.03 (2,500 fixed steps). All strategies adapt ONLY structure params (geometry/couplings, detunings, router); backbone+readout frozen.
- **Continual stream** (same for all): segment A = TRL2D tcool=0.03 (base) → segment B = tcool=3.16 (E5 regime shift) → segment C = TORAX-Bench pilot discharges (E6 transfer). Adaptation data: identical ordered stream of N=512 trajectories per segment.
- **Budget parity in eval-units** (not steps): fwd=1, bwd=2. Fixed 7,500 units/segment ≡ 2,500 gradient steps ≡ 3,750 SPSA steps ≡ 1,250 QN-SPSA ≡ 7,500 e-prop steps ≡ 7,500 evo tier-0 evals. Wall-clock cap: one 5-min sprint per (strategy, segment, seed); 3 seeds (1337/1338/1339) → 36 sprints ≈ 3 GPU-hours at campaign cadence.
- **Controls**: RLS-only readout adaptation always-on (FORCE, P=I/α, α=10) — every strategy must beat it; frozen no-adapt arm = forgetting floor.
- **Metrics**: (1) post-adaptation rollout VRMSE@8 on held-out val of the new regime; (2) adaptation sample count to reach within 5% of that arm's final (sample-efficiency curve); (3) forgetting = VRMSE regression on segment-A val after B (and A∪B after C); (4) eval-units actually consumed. Selection on val only; TEST single-shot, champion only (existing discipline).
- **Kill rules (pre-registered, commit = timestamp)**: (a) gradient — 2× NaN/divergence per seed, or loses to RLS-only on segment B across all seeds; (b) evo — elite fails to beat weight-inheritance parent by >0.1% after one 96-slot night, or archive coverage <10% after 2 nights (degenerate operators); (c) e-prop — sim-validation cosine <0.2 vs stochastic-PSR oracle, or hardware-arm update direction anti-correlates with SPSA probe on >50% of checks; (d) SPSA — control arm, kill only on calibration divergence. Global: monitor flags >2× wall-clock, kill at 3×.
- **Extends E5/E2**: E5's {geometry-only, full-adapter, frozen} scope axis becomes scope × strategy; E2's verdict picks the substrate bank. Results → `results.tsv` + `16_structure_bakeoff.ipynb`; winner promotes per AG-3 (Phase-B Wilcoxon rematch vs 0.1934).

---

## (3) NOVELTY LEDGER

| Arm | Claim | Closest prior art (gap) |
|---|---|---|
| (b) evo+QD | First population/QD archive over **analog hardware register geometries**, co-evolved with an input-conditioned router, weight-inheritance fitness | CBO 2504.06702 (population but single static geometry, VQE, no router — and re-verify the 90/10 stat before citing, not found in fetched HTML); ReCon 2408.13389 (positions on Aquila but Nelder-Mead, single geometry); QD-VQC 2504.08459 (only QD-in-quantum paper — gate-based, no geometry, no hardware); QuantumNAS 2107.10845 (co-search, but circuits+mapping, digital) |
| (c) e-prop | **First eligibility-trace/three-factor learning rule over quantum Hamiltonian parameters** (detunings/geometry) from measured observables × broadcast TD error — standalone claim | Kimura 2021 (traces only in classical value update of a VQC agent); quantum-reservoir "gradient-free" = offline readout fitting only; no e-prop/Hebbian construction through quantum dynamics exists as of Aug 2026 |
| (d) SPSA | None claimed — honest hardware-standard control | Spall; Gacon QN-SPSA; Ebadi 2022 closed-loop |
| (a) grad v2.1 | Modest: R⁻⁷-pathology-safe log-coupling+stress-layer+homotopy recipe inside a frozen world model | Synthesis of 2504.06702 landscape analysis + PulserDiff/DAVQE |
| Bake-off itself | First controlled comparison of hardware-executable vs adjoint-requiring rules for **online quantum structure adaptation on a continual stream** | Bio-plausible-rule comparisons (2406.16062) are classical-only; QAS surveys (2406.06210, 2604.26289) cover no analog geometry |

---

## (4) PAPER ANGLE — "learning rules executable on real hardware"

The x-axis of the bake-off is executability, and it is physically forced: analog Rydberg admits no parameter-shift (generators multi-eigenvalue; stochastic PSR needs mid-evolution gate insertion Fresnel/Aquila lack), so adjoint gradients (a) are simulation-only. Frame (a) as the oracle ceiling and report the **hardware-executability tax**: ΔVRMSE(oracle → rule that runs on the machine) at matched eval-unit budget. Three outcomes, all publishable: (i) e-prop/evo close the gap → "continual structure adaptation on analog quantum hardware needs no gradients," with the ZZ-correlator trace as a new hardware-native learning primitive (needs only observables the block already measures — zero marginal hardware cost, unlike SPSA's 2 extra jobs/step); (ii) they beat the oracle (CBO precedent: non-gradient methods win on divergent R⁻⁶ landscapes) → stronger; (iii) large tax → the first quantified cost of hardware-honesty for quantum structure learning, with SPSA as the honest deployable floor. Carry per-platform caveats: signed per-site detunings are Pasqal-DMM-roadmap/Aquila-experimental; positions are the high-gain, per-job-free lever (MappableRegister trap selection).

---

## (5) IMPLEMENTATION ORDER (max information per GPU-hour)

1. **Harness (≈1 GPU-hr)**: sealed evaluator + continual stream (A/B/C segments) + eval-unit accounting + RLS always-on baseline + frozen arm. Blocks everything; also upgrades E5.
2. **SPSA arm (same day, no new deps)**: ~50 lines + calibration phase. Establishes the deployable floor and the probe used by (c)'s kill rule. Runs in existing sprint slots.
3. **e-prop sim validation (GPU-light, decision gate)**: 10-qubit statevector, ZZ-trace vs stochastic-PSR oracle, report cosine vs perturbation strength; MGD fallback wired. Cheapest test of the paper's biggest standalone claim — do before investing anywhere else.
4. **Gradient v2.1**: already scheduled AG-1 work; shares code with the campaign, lands as the oracle arm.
5. **Evo tier-0 (CPU-free lunch)**: `pip install pyribs`, weight-inheritance fitness fn, CMA-MAE archive + lineage JSONL; evolves overnight between GPU jobs at zero GPU cost; tier-1 elite sprints only after the archive shows structure.
6. **Full 4×3×3 crossing (~3 GPU-hr)** once every arm passes smoke tests; verdicts to notebook 16.
7. **Champion**: single-shot TEST, then Phase-B paired-Wilcoxon rematch vs 0.1934 (AG-3).

Files: plan lives at `/home/mohhakami/Documents/paper/quworld/13-adaptive-geometry.md` (AG-2c); code substrate `/home/mohhakami/Documents/paper/quworld/paper/qlayer.py` (KMixtureRegister/MultiRegister); results to `/home/mohhakami/Documents/paper/quworld/results.tsv` and `notebooks/16_structure_bakeoff.ipynb`; genomes/lineage under `/home/mohhakami/Documents/paper/quworld/runs/bakeoff/`.

*Source: workflow wf_12f3045d-a61.*
