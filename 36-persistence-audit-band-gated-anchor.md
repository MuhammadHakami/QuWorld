# Persistence audit and band-gated anchor (registered Sep 24 2026, before any blend run)

## 1. What the evaluation errors showed

ADR-C v1 found very large "co-processor" gains on two Well rows:
- shear_flow: 88–90% from every arm, including the all-classical linear head;
- rayleigh_benard: 56% from the static linear corrector.

The component features include projections of x_t and of p − x_t, so a linear head can move the trunk toward persistence.

**Persistence** (ŷ(t0+1) = x(t0)), on the same uniform TEST windows (`paper/alloc/persistence.py` scratch check, 1,200 windows):

| Row | Persistence VRMSE mean (median) | Published SoTA | Our fine-tuned trunks (full TEST) |
|---|---|---|---|
| shear_flow | **0.0636** (0.0345) | CNextU-net 0.8080 | 0.25–0.27 |
| rayleigh_benard | **0.2053** (0.1768) | TFNO 0.6566 | 0.4221 (register) / 0.4303 (vanilla) |
| turbulent_radiative_layer_2D | 0.5761 (0.5723) | CNextU-net 0.1956 | 0.1929–0.1930 |

On shear_flow and rayleigh_benard the published leaders, and our trunks, are worse than persistence under the official mean VRMSE. The means are heavy-tailed: the RB trunk has median 0.0345 against mean 0.668.

Hypothesis H0: the trunk's tail errors fall on windows where persistence is nearly exact (low-variance targets, slow evolution). On TRL2D the models beat persistence by 3×, so the audit changes nothing there.

The full-TEST persistence numbers, joined per window with every existing trunk CSV, are in `runs/persistence/<row>_summary.json`. They report:
- the share of windows where the trunk beats persistence;
- the per-window oracle min(trunk, persistence).

## 2. Consequence for claims (binding)

- Every Well-row table shows **persistence next to the published SoTA**.
- On a row where persistence beats the published SoTA, "beats SoTA" is never written alone. The baseline to beat is the stronger of the two.
- The co-processor's gain is credited to the stack it sits on. Any quantum-specific statement still needs the §6 rule of doc 35: beat the classical GNN on the same learned graph.

## 3. Strategy: band-gated persistence blend (classical, frequency-aware "digital" anchor)

`paper/alloc/adrc_cache.py`, env `ADRC_BLEND=5` (= `ADRC_BANDS=5`, amended in §5 before any full run):
- p' = x_t + Σ_b α_{c,b} P_b(p − x_t), over 5 dyadic l∞ FFT shells (the same projector family as doc 35 §8).
- α is fitted per field and band, in closed form, by metric-weighted least squares on the 1,000 TRAIN snapshot windows (Parseval). α = 1 recovers the trunk and α = 0 gives persistence.
- The blend is where the analog/digital split starts on these rows. The classical side keeps what is carried over (persistence) plus the trunk's reliable bands. The residual that remains is what the router may send to qubits.

The ADR-C v2 components (band-wise POD on the same 5 bands, doc 35 §8) are then built on the blended residual r' = y − p'.

## 4. Registered runs

**Rows and trunks:**
- rayleigh_benard with the fine-tuned register-in-the-neck trunk mr30ryd s1337;
- shear_flow with the fine-tuned mr30ryd s1337;
- turbulent_radiative_layer_2D with the released anchor trunk.

**Splits:** α and the co-processor are fitted on TRAIN, every choice is made on VALID, and TEST is used once. For the SoTA-comparable number, RB and shear use the **full** official TEST split (`ADRC_TEST_ALL=1`).

**Arms:** the doc 35 §8 arms (cls_only, static_lin, adrc_ryd, adrc_gnn, adrc_mlp, frozen_ryd, adrc_gutz, rand_ryd) × 3 seeds.

**Reported per row:**
- published SoTA;
- persistence;
- unblended trunk (`trunk0_test`);
- blended trunk (`trunk_test`);
- each arm's corrected TEST VRMSE, with the per-band decomposition and the qubit slot share per band.

**Decision rules:**
- The blend is a classical improvement. It is kept only if it lowers VALID VRMSE for the blended trunk.
- A quantum-specific claim needs adrc_ryd < adrc_gnn on a majority of rows, with the pooled CI excluding 0. Otherwise the gain is credited to routing, configuration and the classical anchor.

## 5. Amendment (Sep 24, before any full run): exact per-window gate

**Full-TEST persistence audit (`runs/persistence/<row>_summary.json`).**

| Row | Persistence mean (median) | Trunk mean range, 9 fine-tuned arms (published-checkpoint anchor) | Share of windows where trunk beats persistence | Per-window oracle min(trunk, persistence) |
|---|---|---|---|---|
| rayleigh_benard (34,300 windows) | **0.2024** (0.1749) | 0.356–0.571 (0.6679) | 84–85% | **0.044–0.048** |
| shear_flow (21,952 windows) | **0.0646** (0.0347) | 0.224–0.345 (0.9946) | 54–63% | **0.036–0.039** |

The trunks beat persistence on most windows but lose the mean in a heavy tail, so the headroom is a *routing* decision per window.

**Design.**
- The gate and the POD components use the **same 5 bands**, so every mode lies inside one gate band.
- The cache stores, per window, field and band:
  - gY = ‖P_b(y − x_t)‖²;
  - gYD = ⟨P_b(y − x_t), P_b d0⟩;
  - gDD = ‖P_b d0‖²;
  - the causal gX = ‖P_b Δx‖²;
  - ay = ⟨y − x_t, φ⟩ and fd0 = ⟨d0, φ⟩, where d0 = p − x_t.
- For **any** per-window α(C, B), the gated residual has:
  - energy Σ_b (gY − 2α gYD + α² gDD);
  - coefficients ay − α_{band(m)} fd0.
- **Both are exact**, so the gate trains jointly with every arm on the exact metric.
- `adrc.py --gate` computes α = α_fit + 1.5 tanh(MLP(z)). z is built only from causal inputs: log band energies of p − x_t and Δx, plus the global statistics. The last layer is zero-initialised, so training starts at the TRAIN-fitted blend.
- A new arm, `none` (no co-processor), isolates the gate.
- The gate is classical. It is the digital router between the carried-over state and the trunk. Every arm, the register and all its classical controls, gets the identical gate.

**Checks and disclosure.**
- **Smoke (scratch cache, 96/48/48 windows, RB mr30ryd trunk):** anchor consistency at α = α_fit is 3.6e-4 relative in the coefficients and 5e-5 in the energy (fp16 basis). Per-band chain violations are ≤ 0.
- **Disclosure:** the gate strategy was motivated by the TEST-split persistence audit above, not by VALID. The smoke checks also read 48 TEST windows. No hyperparameter was chosen from either. Every arm and setting is selected on VALID, and the full runs use TEST once.

**Runs (`runs/alloc_bgp_runner.sh`), all arms with `--gate`, 3 seeds:**
- none, cls_only, static_lin, adrc_ryd, adrc_gnn, adrc_mlp, frozen_ryd, adrc_gutz, rand_ryd;
- on shear_flow mr30ryd, rayleigh_benard mr30ryd and TRL2D released, all on the full TEST split;
- then the vanilla trunks.

**Lean (`formal/QuotaScaling/BandGate.lean`; standard axioms only, checked in `formal/Audit.lean`):**
- `band_span_le_energy` (Bessel inside a band) gives span_b ≤ full_b.
- `gate_energy_exact` and `gate_coeff_exact` show the gate's exactness from the cached statistics.
- `gate_opt_le` shows the best scalar gate dominates every gate. `gate_opt_le_persistence` and `gate_opt_le_trunk` specialise it: the per-band oracle gate is no worse than persistence or the trunk.

## 6. Amendment (Sep 24, 23:05, before these runs): ensemble trunks on TRL2D, fine-tuned MHD trunks

**First results.** Shear_flow (register-neck trunk, full TEST):

| Step | VRMSE |
|---|---|
| Trunk | 0.2297 |
| TRAIN-fitted blend | 0.0451 |
| Gate alone | 0.0273 |
| Every co-processor arm | 0.0223–0.0226 (register ≈ GNN ≈ mean-field ≈ static linear) |

TRL2D (released trunk): 0.1934 → blend 0.1926 → gate 0.1923.

**Added runs.** TRL2D's best trunks are 3-seed ensembles, so the gate and the arms are also run on ensemble-mean trunks. `adrc_cache.py` now accepts a comma-separated `ADRC_CKPT`, and the trunk prediction is the members' mean:
- `turbulent_radiative_layer_2D:gate_ens3_mr30ryd` (register-neck seeds 1337–1339);
- `turbulent_radiative_layer_2D:gate_ens3_vanilla` (the matched classical ensemble).

MHD_64 gets the gate on its fine-tuned trunks, vanilla and mr30ryd s1337 (`runs/alloc_bgp_mhd_runner.sh`). There, persistence is beaten on every window (0.4183), so the gate can only shrink the trunk's high-k increment.

The rules are the same as before: TRAIN fits, VALID selects, TEST is used once. On TRL2D and MHD, the comparison is against published 0.1956 and claimed 0.1633.

## 7. Amendment (Sep 24, 23:20, user direction): stratified samples for iteration, full TEST only for the selected stack

Building a full-TEST cache through the register-neck trunk on rayleigh_benard (34,300 windows) takes hours, so iteration moves to stratified samples.

**Sampling.** `ADRC_VALID_STRAT=k` / `ADRC_TEST_STRAT=k` draw k windows per (trajectory × one of 8 equal time bins). The early-time heavy tail is therefore represented pro rata. The estimator was validated against the full-TEST per-window CSVs (20 draws):

| Row, trunk | k = 4 estimate | Full split | Uniform 1,200-window sample |
|---|---|---|---|
| RB, vanilla | 0.4226 ± 0.0047 | 0.4303 | 0.4986 |
| RB, persistence | 0.2024 ± 0.0004 | 0.2024 | — |
| shear, register-neck | 0.2314 ± 0.0007 | 0.2297 | 0.2407 |

**Runs.**
- RB iterates on `rayleigh_benard:gatest_{mr30ryd,vanilla}_s1337`: VALID k = 2 (2,800 windows), TEST k = 4 (5,600).
- The stack selected on VALID is then scored once on the full TEST split.
- shear_flow keeps full TEST, because its register-neck cache already exists and the vanilla trunk is fast.

## 8. Correction (Sep 24, 23:55): the first gate runs leaked the target variance

**The leak.** `adrc.py` built the gate's input z from log(gDD / D) and log(gX / D). But D = npix · var(y) is the **target** frame's variance, which is not causal. It hints exactly at the low-variance heavy-tail windows.

**Fix.** z is now normalized by var(x_t), the last input frame's variance, taken from the cached global statistics. Every other feature was re-audited: all are projections of p − x_t, p, x_t and Δx, or their global statistics. The target enters only the loss and the reported metric.

**Consequences:**
- All 116 gate-run JSONs are archived in `runs/alloc/adrc_results_leaky_gate/`, and no number from them is used.
- The earlier gate figures in this doc and in history (shear 0.0273 / 0.0223; RB 0.0422 / 0.0413; TRL2D 0.1923 / 0.1921) are **void**. The rerun supersedes them (`runs/alloc_gate2_runner.sh`).
- The TRAIN-fitted band blend (the anchor: shear 0.0451, RB 0.142 stratified, TRL2D 0.1926) uses no per-window input and stands.
- The non-gate ADR-C runs never used z and stand.
