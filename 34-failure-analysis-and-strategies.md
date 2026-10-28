# Where the classical + quantum model fails, and strategies to use it better (Sep 24 2026)

Written at the user's request after the allocation study (doc 33). The numbers are TEST numbers from `runs/alloc/failure.{md,json}` (`paper/alloc/failure.py`), `runs/judge/scorecard.json`, `runs/sophelio_sota/status` and `runs/tdc_ensemble/stack_table.txt`. The interpretive view is in `notebooks/19_models_and_interpretation.ipynb`.

## 0. Design principle (user, Sep 24): underfitting 30 qubits is a routing failure

The structure gives each available qubit a specific analog signal, with an ideal configuration: x, y, z placement and tethering. If 30 qubits cannot reach the error, or underfit because they are handed too much, the failure belongs to **routing, configuration and navigation**, not to the structure or the qubit count. Every failure mode below is therefore read as a routing, configuration or navigation shortfall.

What the current implementation leaves undone:
- **Routing unit:** each qubit is handed a raw Haar tile, not a compressed analog signal.
- **Configuration:**
  - x, y come from tile position with two learned global scales, and z from scale;
  - there is no per-input learned placement;
  - there is no tethering between the 10-atom groups;
  - groups are formed by spatial (Morton) order, not by signal correlation.
- **Navigation:** every input and every field gets the same budget.

**Both directions are routing failures.**
- **Underfitting:** the qubits are handed more signal than they can carry (raw tiles covering diffuse error).
- **Overfitting:** the qubits are handed too little usable signal, so they fit noise.
  - Examples: random tiles; oracle tiles whose residual is unpredictable from what is routed with them (the TRL2D 64× MLP got worse on test); pure threshold ("digital") structure, where the register scores R² 0.07 in the probe.

**The router's objective is therefore maximal qubit utilization:**
- Each qubit gets an appropriate analog signal: smooth, oscillatory, predictable from what is routed to it.
- The signal comes with a configuration tuned to it (x, y, z, tethering, grouping).
- Each qubit is loaded to its capacity, neither starved nor overloaded.
- Digital structure (thresholds, discrete or categorical features, shocks) goes to the classical world model.

In Lean terms, the router maximises the captured share under `gain_le_ceiling`. It does this by choosing the signal each qubit receives, not by adding qubits.

**Measurable per qubit:**
- the captured error per qubit;
- the train–test gap of the qubit-carried correction (the overfitting indicator);
- the register-vs-classical gain on that component (the analog indicator from the probe).

A router that uses these three signals is the natural next step: R1–R4 below plus the S6 gate.

The TRL2D predictability failure is a different kind: even oracle placement captures nothing, so a different signal has to be routed there (for example S5), not just a better route for the same one.

## 1. Failure decomposition (frozen trunks, register co-processor at each row's VALID-best strategy, 30 atoms)

The trunk's weighted TEST error energy splits exactly into four parts (Lean `band_energy_split`, `gain_le_ceiling`):

- **captured:** removed by the co-processor;
- **predictability failure:** the error was on allocated tiles but was not removed;
- **allocation-rule failure:** an oracle with the same 30 atoms would have reached it;
- **routing-coverage failure:** the error lies beyond the 30 most-erroneous tiles, because the router hands each qubit a raw tile instead of a compressed analog signal (see the principle above).

| Row | Captured | Predictability failure | Allocation-rule failure | Routing-coverage failure | Where the metric error sits |
|---|---|---|---|---|---|
| TRL2D | 0.000 | 0.231 | 0.229 | **0.540** | pressure 71 %; spread over windows |
| rayleigh_benard | **0.645** | 0.010 | 0.000 | 0.345 | velocity 100 %; worst 10 % of windows = **90 %** of the metric (convection onset) |
| shear_flow† | 0.495 | 0.000 | 0.001 | **0.503** | pressure **97 %** |
| MHD_64† | 0.025 | 0.001 | 0.002 | **0.972** | density 53 %, magnetic field 37 % |
| RealPDE (U-Net) | 0.000 | 0.021 | 0.043 | **0.935** | v component 77 % |
| RealPDE (FNO) | 0.065 | 0.008 | 0.025 | **0.901** | the u bias is corrected (−19 %), the v component is not (−0.4 %) |
| Pelzer heat plume | 0.188 | **0.239** | 0.170 | 0.404 | single field; gain +11.5 % on the worst windows |

† The released checkpoint does not reproduce its published number under our evaluator.

**Reading:** the four failure modes, in order of size.

1. **Routing-coverage failure dominates on 5 of 7 rows (40–97 %).** The router's unit is wrong: one Haar tile per qubit spreads diffuse error over too many components. This is a routing failure, not a qubit-count one. The K sweep shows the same thing from the other side: a raw-tile router needs 10× the qubits to cover the error (shear 11 % → 68 %, MHD 0.9 % → 7.9 % from 10 to 300). The fix is a compressed routing unit (R1), not more qubits.
2. **Predictability failure** is the whole story on TRL2D: 23 % of the error is on the allocated tiles, but nothing predicts it, even under the oracle and with a 64× MLP. It is also 24 % on Pelzer.
3. **Allocation-rule failure** reaches 17–23 % on TRL2D and Pelzer, where the causal rules miss tiles the oracle would take.
4. **Field and time imbalance.** VRMSE normalises per field, so a low-variance field dominates the metric: pressure is 71 % on TRL2D and 97 % on shear_flow. On rayleigh_benard, 90 % of the metric sits in the onset transient. Our tiles bundle all fields together and every window gets the same budget.

**Rows outside the study:**

| Benchmark | Where it fails |
|---|---|
| Sophelio Ch1 | TEST 0.9910 vs leader 0.9932. Scalars and ψ are at the leader (ψ R² 0.99962, qb 0.9947). The gap is the **boundary shape**: consistency 0.967 vs 0.976 and D_LCFS 0.0145 vs 0.0119. |
| TokaMark 1-1 | 0.1454 vs 0.140. The best published model is TokaMind-*Scratch*, but our trunk is FT-Base; the register ties the MLP. |
| TORAX-Bench | The tuned MLP adapter is best (0.0141) and the register worst (0.0168). |
| TDC | Worst ranks on small, heterogeneous endpoints: bioavailability #21, caco2 #8–9, CYP3A4-substrate #9, DILI #3–6. Inputs are discrete graphs and the register trails the MLP on 13 of 22 endpoints. |
| TRL2D rollouts | The register's one effect (0.953 vs 1.03) arises under TD fine-tuning (training-mediated, inference-inert), not in frozen correction. |

## 2. Relevant research (Sep 2026)

- **Digital-analog learning on Rydberg arrays:** alternating single-qubit digital gates with native analog evolution gives shorter, more noise-robust circuits than purely digital ones ([2401.02940](https://arxiv.org/abs/2401.02940); [QST 2024](https://iopscience.iop.org/article/10.1088/2058-9565/ad9177)).
- **Quantum reservoir computing on Rydberg arrays** for chaotic and turbulent time series:
  - [practical QRC on Rydberg arrays, 2602.00610](https://arxiv.org/pdf/2602.00610);
  - [collective amplification, 2511.15047](https://arxiv.org/pdf/2511.15047);
  - [robust QRC for chaos, Proc. R. Soc. A 2025](https://royalsocietypublishing.org/rspa/article/481/2324/20250550/234346/Robust-quantum-reservoir-computers-for-forecasting);
  - [hybrid QRC for low-dimensional turbulence, Chaos 2025](https://doi.org/10.1063/5.0334540);
  - [QRC on chaotic maps, PRR](https://link.aps.org/doi/10.1103/qwvv-xqfq).
- **Data-adapted bases for residual correction:**
  - [PODNO: POD neural operators, 2504.18513](https://arxiv.org/pdf/2504.18513);
  - [residual-based error correctors for neural operators, 2210.03008](https://arxiv.org/pdf/2210.03008);
  - [DeepONet error = sum of mode errors, 2602.21910](https://arxiv.org/pdf/2602.21910).
- **Stochastic (generative) closures for the unpredictable small scales:**
  - [generative AI for subgrid turbulence, PRF](https://journals.aps.org/prfluids/abstract/10.1103/vb6j-f876);
  - [transport-based stochastic closures, 2602.17089](https://arxiv.org/html/2602.17089);
  - quantum generative models such as [spectral Born machines, 2607.06675](https://arxiv.org/html/2607.06675v1).
- **Uncertainty decomposition and value-of-information routing:**
  - [calibration-aware PDE-surrogate uncertainty, 2602.11090](https://arxiv.org/html/2602.11090);
  - [structure-aware epistemic UQ for neural operators, 2603.11052](https://arxiv.org/html/2603.11052);
  - [value-of-information routing for expert mixtures, 2608.02528](https://arxiv.org/html/2608.02528).

## 3. Strategies

All eight stay within our novelty (utilization-first allocation of a small Rydberg budget inside a pretrained classical world model, with Lean bounds and matched classical controls). Each is aimed at a measured failure mode.

| # | Strategy | Failure it targets | Why it is faithful | Test (prediction) |
|---|---|---|---|---|
| S1 | **Allocate modes, not pixels.** Build a data-adapted orthonormal basis of TRAIN residuals (POD/PCA per field, or wavelet packets chosen by residual energy) and route atoms to the modes with the largest predicted reducible error. | Routing-coverage failure (40–97 %) | Every Lean bound holds in any orthonormal basis (`band_energy_split`, `gain_le_ceiling`); only U changes. | U@30 and captured@30 of POD vs Haar on all 7 rows. Predicted: large U gains on MHD, RealPDE and TRL2D. |
| S2 | **Per-field allocation.** Tiles become (field, scale, position) with metric weights, so atoms go to the field that dominates the normalised metric (pressure). | Field imbalance (TRL2D 71 %, shear 97 %) | The same allocation principle at a finer unit. | Captured on pressure; VRMSE per field. |
| S3 | **Navigation over time under the same 30-qubit budget.** The router moves qubits to the transient or high-reducible-error windows; the total budget is unchanged. | Concentration: rayleigh_benard's worst 10 % of windows hold 90 % of the metric | Utilization across inputs, not only within one. | Mean-budget-matched gain vs fixed K. |
| S4 | **Predictability-aware (value-of-information) routing.** Route by predicted *reducible* (epistemic) error, estimated by cross-fitted ensembles of the co-processor, instead of total error. | Predictability and allocation-rule failure (TRL2D, Pelzer); P4 failed with noisy single-fit gains | This is the Lean top-K-by-gain rule with a calibrated gain estimate. | Captured@30 vs proxy/learned. Honest prior: TRL2D may have no reducible error at all. |
| S5 | **Register as a stochastic closure for the unpredictable part.** The fine-scale residual a deterministic co-processor cannot touch is sampled from the register's measurement shots (a Born-machine head conditioned on the trunk state), for rollouts. | Predictability failure; long-horizon drift | This is the quantum-native role: an analog register samples correlated distributions natively. It links to our only register-specific effect (TD rollouts). | Rollout VRMSE, spectra and CRPS on TRL2D/MHD vs a matched classical sampler (Gaussian or small flow). No advantage claim at ≤ 10 qubits per register. |
| S6 | **Digital/analog division of labour with a learned per-component gate.** Each allocated component goes to the register (smooth, oscillatory) or to the classical core (thresholds, shocks, discrete) by predicted gain. A digital-analog ansatz (single-qubit digital layers between analog blocks) makes the register expressive enough for its share. | The probe (register R² 0.07 on thresholds, no edge on smooth maps); `hyb` concatenated readouts without routing | This is the user's thesis made operational and testable. | Gate statistics (do smooth components go to the register?), gated hybrid vs gated two-MLP control. |
| S7 | **Register as temporal memory (quantum reservoir)** for the sequence benchmarks: TokaMark, TORAX, Sophelio shot sequences, and rollout steps. | Time-series rows where the register currently ties or loses | Analog dynamics as memory of the time series is where the Rydberg QRC literature reports promise. | vs an echo-state network of matched size and a GRU; forecasting horizons. |
| S8 | **Allocation under fine-tuning.** Put the routed register inside TD fine-tuning (rollout-aware training), the regime of our one register-specific effect, instead of on a frozen trunk. | Frozen correction cannot reach training-mediated effects | Combines both of our positive mechanisms. | TRL2D rollout VRMSE@13:30 vs routed MLP and pooled register, 3 seeds, registered first. |

### Routing, configuration and navigation fixes (priority under the design principle)

| # | Fix | Shortfall addressed |
|---|---|---|
| R1 (= S1) | **Route compressed analog signals, not raw tiles.** Each qubit receives one coefficient of a learned residual basis (POD/PCA per field, or a learned encoder of a front), so 30 qubits cover the error. | Routing unit; coverage failure (40–97 %) |
| R2 | **Learned configuration.** Per-input x, y, z from a bounded hypernetwork (hardware-valid by `lattice_sep`); **tethering** as learned inter-register edges or auxiliary relay atoms between groups (doc 32 §1; the Wave-G cascade/shared links are precedents); groups formed by signal correlation, not Morton order. | Configuration |
| R3 (= S2, S3) | **Navigation:** route per field (pressure gets its share) and across inputs and time under the same 30-qubit budget (the onset windows on rayleigh_benard). | Navigation |
| R4 (= S4) | **Route by fixable (reducible) error.** | Allocation-rule failure |

**Classical-side fixes for the SoTA gaps** (not quantum claims):
- Sophelio: allocate a coupling to the boundary/LCFS modes (S1 on ψ).
- TokaMark: switch the trunk to TokaMind-Scratch.
- TDC: MiniMol/CaliciBoost-style features for the weak small-data endpoints.

## 4. Suggested order (deadline Sep 25 AoE, then camera-ready)

1. **S1 + S2 on existing trunks, before the deadline if possible.** This needs a re-cache that stores POD/per-field coefficients (trunk inference again, about 10–30 min per row) and reuses `coproc.py` unchanged. It directly attacks the dominant failure mode.
2. **S3 and S4:** cheap on top of the caches.
3. **S6** (the thesis test) and **S8** (the regime where the register mattered). Register protocols first.
4. **S5 and S7:** new heads, for the camera-ready or the next paper.
