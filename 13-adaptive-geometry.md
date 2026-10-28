---
title: Qubits-on-the-Go — adaptive register geometry program (big plan + todos)
date: 2026-08-16
project: quworld
tags: [adaptive-geometry, v2, todo, gpu-throughput, pasqal]
status: live-checklist
---

# Qubits-on-the-Go: real utilization of adaptive registers

**User thesis (2026-08-16):** flat results come from using qubits *inflexibly*. Neutral atoms can restructure per simulation — so the register geometry itself becomes (a) **learnable**, (b) **input-conditioned**, (c) **continually adapted** via temporal-difference signals across simulations, inside the exact Pasqal 2D/3D envelope. Companion fix: converged-checkpoint + one-step MSE + zero-init gate = **gradient starvation** — the objective moves to rollout/TD so the quantum path has signal to earn.

**Throughput policy (user):** hypothesis sprints run **on the GPU concurrently** with the campaign (step-targeted arms stay fair; ~26% base util leaves headroom); monitor flags >2× wall-clock degradation. Everything detached + resumable + absolute-path launched.

## AG-0 — Research integration ✅ (design brief `wf_ada0b17d-3fb`, 2026-08-16)

**Hardware envelope (pulser-core 1.9.0 device objects, verified):** Pasqal Fresnel: **2D only** (Register3D is simulation-only; 3D = Barredo lab demo, never productized), ≤80 atoms (calibrated layout: 61-trap triangular, 5.0 µm → **max 30 atoms = exactly our 3×10**), min spacing 5 µm, FOV 38 µm radius, Ω_max 12.566 rad/µs, sequence ≤6 µs (τ_eff ~3.2 µs), **no local detuning today** (DMM roadmap: negative-only, per-site weights ∈[0,1]); QuEra Aquila: 256 sites, 4 µm, free per-shot geometry, local detuning experimental. **Architecture-deciding fact:** geometry can change per-JOB for free within one calibrated layout (MappableRegister trap selection); NEW layouts cost hours of calibration; per-input geometry = one job per input = throughput death.

**Verdicts adopted:**
- **Conditioning architecture: HYBRID (c)** — K-register router (per-job trap-subset selection = free hardware path, Gumbel-softmax training) + slow local refinement of the K anchor geometries (calibration-timescale outer loop). Continuous per-input geometry (`cg30`) stays as the *simulation expressivity probe*, presented as such.
- **Geometry learning recipe (R⁻⁷-pathology-safe, cites 2504.06702 + PulserDiff):** learn bounded **log-couplings** w_ij ~ log V_ij (O(1) gradients), realize positions via a differentiable stress layer (DAVQE objective), lattice-anchored softplus-bounded displacements with hard 5 µm floor, **homotopy on the exponent** (train R⁻ᵖ, anneal p→6), gauge canonicalization (anchor atom + axis + lexicographic order), positions on a slower update timescale than drives, Adam+cosine lr 0.05–5.0.
- **TD/continual recipe (exact):** L_push = pushforward with stop-grad inner steps, backprop last step only (Brandstetter 2202.03376) + L_TD = EMA-target latent consistency, ρ=0.5, H=3–5 (TD-MPC2 2310.16828), λ-return mixing 0.95 (DreamerV3); c·L_TD with c=0.1–0.5. **TTA:** adapt only a 2–8-dim context vector ξ (CODA 2202.01889 linear hypernet) 5–10 steps from a Reptile meta-init, TENT-style. **Forgetting:** per-simulation geometry bank + shared drive/encoder (SLE-FNO pattern, beats EWC/replay at 2603.20410) + online **QFI-EWC with the FULL Fisher** (affordable at 2k params — paper-worthy detail) + shrink-and-perturb between rounds.
- **Novelty CONFIRMED:** learned input-conditioned atom geometry = zero prior art; MoE-over-registers unclaimed; theory cover: Cesa–Pichler geometry-universality (PRL 131:170601), Hamiltonian-encoded input evades the static-injection nonlinearity ceiling (2607.12035 Thm 3), and **capacity is non-monotonic in the interaction graph (PRA 112:012622) → no static register is task-universal** — the theoretical statement of the user's thesis.
- **Deployability caveat to carry honestly:** our per-qubit signed detunings are Pasqal-DMM-roadmap (negative-only) / Aquila-experimental — state per-platform constraints in the paper; positions are the high-gain lever (δV/V = −6 δr/r).

## AG-1 — v2 implementation (paper/qlayer.py + trainers)
- [x] `ConditionedGeometryBlock`: per-input atom positions = base (learnable) + bounded hypernet shift; per-sample blockade energies, batched; 2D and 3D modes; spacing clamp (hardware envelope)
- [x] K-register mixture (`kmix`): bank of K geometries + learned router (soft in training, hard at deploy — K precompiled registers = trivially hardware-real)
- [x] Sprint trainer: K-step pushforward/TD rollout loss (`rollout_k` arg) — the gradient-signal fix, applies to quantum AND control arms
- [ ] **v2.1 (post-wave-1, per design brief):** log-coupling parameterization + differentiable stress-layer realization + exponent homotopy + gauge canonicalization; sg-pushforward L_push (current td_loss backprops all steps — replace) + EMA-target L_TD; CODA context vector ξ for TTA; QFI-EWC full-Fisher forgetting guard
- [ ] Test-time adaptation mode: ξ-only adaptation, 5–10 steps, Reptile meta-init — after E1/E4 verdicts
- [ ] Hardware-envelope module: 5 µm floor / 38 µm FOV / Ω≤12.566 / seq≤6 µs; Pulser MappableRegister export of router trap-subsets; 3D flagged simulation-only

## AG-2 — GPU experiment ladder (sprint loop, 3 seeds each, kill rules pre-stated)
- [ ] **E1 conditioned vs static geometry**: `cg30` vs `mr30ryd` vs `mlp30`, rollout VRMSE@8 — kill: conditioned ≤ static across seeds
  - wave-1 seed-1337 (wall-clock): cg30 0.438/1.108 BEAT static 0.485/1.116 at matched steps (~3.1k both ✓); mlp30 1.008 not comparable (5.0k steps — MLP cores are cheaper per step; cross-type wall-clock comparisons are inherently throughput-confounded, as in round 1)
  - [ ] **wave 1.5: fixed-step E1 trio** (2,500 steps each, à la round-2s) + seeds 1338/1339
- [ ] **E2 mixture vs continuous**: `kmix30` (K=4) vs `cg30` — decides the deployable architecture
- [ ] **E3 2D vs 3D registers**: `cg30-3d` vs `cg30` — does the extra interaction-graph freedom matter?
- [ ] **E4 objective fix isolated**: rollout_k ∈ {1,4} × {mr30ryd, mlp30, none} — does TD/pushforward alone un-starve adapters (incl. classical)? *Most important control in the program*
- [ ] **E5 continual/on-the-go**: train regime tcool=0.03 → few-shot adapt to tcool=3.16 (TD steps, geometry-only vs full-adapter vs frozen) — the "different simulation" claim
- [ ] **E6 cross-domain generality**: winning variant, unchanged, on TORAX-Bench pilot discharges (CPU-generated data, GPU training)
- Cadence: ~5-min sprints, ≥12 experiments/hour interleaved with campaign; results → `results.tsv` + notebook 03 verdict cells

## AG-2c — Structure-search bake-off (user 2026-08-16: evolution + bio-plausible)
- [ ] Research in flight (`wf_12f3045d-a61`): Prime Agent evolution machinery, Sakana toolkit (CycleQD/QD archives, evolutionary merge, island models), evolutionary-quantum prior art (novelty check), e-prop/three-factor rules on measured ⟨Z⟩ × TD error
- [ ] **Bake-off**: four strategies adapt the SAME K-register bank on the SAME continual stream — (a) gradient v2.1 · (b) evolutionary population-of-registers w/ QD archive (hardware-honest: evolution between jobs; CBO precedent beat gradients 90/10) · (c) e-prop-style eligibility traces × broadcast TD signal (**no backprop through quantum evolution → hardware-in-the-loop executable — potential standalone claim**) · (d) SPSA (hardware-standard control)
- [ ] Metrics: post-adaptation rollout VRMSE, adaptation sample count, forgetting; kill rules per strategy; extends E2/E5
- [ ] Notebook `16_structure_bakeoff.ipynb`

## AG-3 — Promotion path
- [ ] Winner → Phase-B round 2 on CNextU-net (with rollout objective) → the paired-Wilcoxon rematch vs 0.1934
- [ ] Winner → LingBot chunk-boundary adapter (H2) and fusion tracks (TokaMark/Sophelio/TORAX-Bench)
- [ ] Paper claim upgrade (C1→C1′): *input-conditioned, continually-adapted register geometry* — pending AG-0 novelty confirmation
- [ ] Dequantization suite re-run on adaptive winner (χ-sweep must cover per-input geometries)

## Kill rules & honesty
E1–E5 each pre-registered before their first run (commit = timestamp). If adaptive geometry never beats static under the fixed objective, that negative goes in the paper next to the positive TD-objective result (if E4 delivers one). No claim without its notebook (E-series → notebook `15_adaptive_geometry.ipynb`).
