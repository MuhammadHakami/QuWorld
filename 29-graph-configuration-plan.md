---
title: Wave G — register graph configuration bake-off (pre-registered Sep 21 2026)
date: 2026-09-21
project: quworld
tags: [pre-registration, graph, configuration, qubit-scarcity, arena]
status: PRE-REGISTERED 01:20 Sep 21 — decision rules fixed before any arm ran
---

# Wave G: which register *graph* should a few qubits form?

**User directive (Sep 21 00:50):** improve the quantum arm's benchmark numbers by testing qubit-scarcity strategies through configurable register graphs (connection mechanisms, per-input "on the go" configuration) aimed at the hardest analog part of each problem, since neutral-atom arrays can realise any geometry.

**Observation that motivates the wave.** The register's interaction graph V_ij ∝ |r_i − r_j|^-6 is set by the atom positions. Every model so far started from the same jittered square grid at unit spacing, and the Sep 20 routing diagnosis measured geometry gradients of order 1e-11: the positions do not move in training. So every ladder row used one arbitrary graph. This wave makes the graph a factor.

## Factors (all at 30 simulated qubits, `paper/model.py` grammar `q30-...`)

| Axis | Levels | Mechanism |
|---|---|---|
| Topology family | grid (ref) · chain · ring · tri | `qlayer.topology_positions`: nearest-neighbour spacing fixed, degree 4 / 2 / 2+loop / ≤6 |
| Spacing (coupling strength, graph sparsity) | 1.0 (ref) · 0.7 (8.5× coupling) · 1.5 (0.09×, near product state) | scales all positions |
| Partition (scarcity strategy) | 3×10 (ref) · 6×5 · 2×15 | R patches of n atoms; qubit ladder said 5q ≈ 30q, so 6×5 tests many-small |
| Inter-patch link (connection mechanism) | independent (ref) · cascade · shared | cascade: patch k's controller also reads patch k−1's readout (classical feed-forward; zero-init → identical to ref at init); shared: one controller for all patches (graph diversity only) |
| On-the-go participation | off (ref) · gate | per-input, per-atom drive gate g_i(h) ∈ (0,1); an undriven atom stays in \|0⟩ and leaves the blockade graph; init ≈ 1 everywhere |

One-factor-at-a-time around the reference (12 configurations), each with its own parameter-matched MLP control (`-mlp` suffix; budgets within 1.3%).

## Protocol (the arena protocol that produced the paper's quantum result)

- `paper/train.py <variant> <seed> 100000 2000 4`: TRL2D 3-regime subset, **fixed 2,000 steps**, TD/pushforward k = 4, batch 16, lr 1e-3; metric = validation VRMSE@8 (8-step rollout) with VRMSE@1 recorded.
- Stage 1 (screen): all 12 configurations + reference q30 + mlp30 + none at seed 1337.
- Stage 2 (confirm): the **top 3 configurations by Stage-1 VRMSE@8** (ties → lower VRMSE@1) and their matched controls at seeds 1338 and 1339, plus the reference trio at those seeds if not already on file (`runs/ag15_*_k4_s133*.log` provide q30 ≡ mr30ryd, mlp30, none).
- Decision rule: a configuration is *promoted* to the benchmark rows only if its 3-seed mean VRMSE@8 is below the reference register on 3/3 seed-paired comparisons AND below its own matched control on 3/3. Otherwise it is reported as a null in the ladder.
- Gate report: for `-gate` arms, mean participation (qubits in use per input) on the validation set is recorded; a gate that stays at ≈1 is a null for the scarcity claim even if the metric improves.
- Confounds stated up front: one-factor-at-a-time cannot find interactions; a fixed step budget favours mechanisms that learn fast; the arena is a 3-regime subset, so promotion is a hypothesis for the benchmark rows, not a result.

## Compute and scheduling

≈13 min per register arm, ≈8 min per control (Aug 16 timings, GPU alone). Stage 1 = 12 + 12 + 3 arms ≈ 4.5 GPU-h; Stage 2 ≈ 2.5 GPU-h. Runs as a GPU co-runner (`runs/waveG_runner.sh`): starts only when MemAvailable ≥ 40 GB and no Physics-IQ generate process holds the GPU; the arena trainer is first in memguard's shed order. Results: `runs/waveG/<variant>_s<seed>.log`, status `runs/waveG/status`, summary `runs/waveG/summary.md`, sentinel `WAVEG_STAGE1_DONE` / `WAVEG_DONE`.

## What comes after (not part of this pre-registration)

Promoted configurations enter TORAX, Pelzer U-Net and Sophelio as "register configuration only" adaptations (the foundation rule), each with the same 3-arm × 3-seed protocol as the existing rows. Interactions between axes and the per-benchmark configuration search follow once Stage 2 is in.

## Interim note (03:25 Sep 21, Stage 1 in progress — rules unchanged)

The `-mlp` controls share one architecture and budget (1,338,552 parameters) but not one initialisation: each configuration's register is built before its control and consumes a different slice of the seed's random stream. Their Stage-1 spread (0.920 for `q30-part=6x5-mlp`; 1.018–1.021 for the other five) is therefore a direct estimate of single-seed initialisation noise on this arena: ≈ ±0.05 with 0.1 outliers. Stage-1 differences among register configurations of that size (chain 0.873, sparse 0.885 vs reference 0.902; dense 0.930) are within it. Stage 1 ranks candidates only; Stage 2's seed-paired 3/3 rule against both the reference and the matched control is the decision, as pre-registered.

## Stage 1 result (05:15 Sep 21; seed 1337; `runs/waveG/summary_stage1.md`)

Reference q30 = 0.9020 (regression anchor vs Aug 16 mr30ryd 0.9026); mlp30 1.0978; none 1.2473. Configurations (VRMSE@8 / matched control): chain **0.873** / 1.020 · ring 0.903 / 1.021 · tri 0.903 / 1.020 · sp=0.7 0.930 / 1.019 · sp=1.5 **0.885** / 1.018 · 6×5 1.012 / 0.920 · 2×15 **0.881** / 1.060 · cascade 1.325 / 0.850 · shared **0.865** / 1.121 · gate 0.935 / 1.195 (participation 0.985) · gate+cascade 0.996 / 0.988 (0.986) · tri+sp=0.7 0.925 / 1.019.

Reading (screen only): outside the ±0.05 init band are two losses, the cascade feed-forward link (worse than the bare trunk) and the 6×5 partition (control level). Inside the band, every gain points to sparser/weaker/larger-patch graphs with less controller diversity (chain, spacing 1.5, 2×15, shared). Both gates stayed open (≈0.985) → the per-input scarcity mechanism is a null as implemented (no sparsity pressure in the objective). Stage 2 selected shared, chain, 2×15 (runner, 05:15).

## Addendum (pre-registered 05:25 Sep 21, before any run): scarcity pressure on the gate

Stage 1's gate never closed because nothing in the objective charges for a qubit. `GATE_PENALTY=λ` in `paper/train.py` adds λ·(mean drive gate) to the TD loss. Arms: `q30-gate` × λ ∈ {0.03, 0.1, 0.3}, seed 1337, 2,000 steps, k = 4 (`runs/waveG_gate_runner.sh`, chained behind WAVEG_DONE). Read-out: participation (qubits in use per input) and VRMSE@8 against the Stage-1 reference 0.902 and the ungated gate 0.935. The scarcity claim needs participation to drop well below 1 with VRMSE@8 not worse than the reference; a gate that closes at the cost of accuracy is reported as the price of scarcity, not a win. Single seed: screen only.

## Stage 2 result and verdict (07:10 Sep 21; `runs/waveG/summary.md`)

| Configuration | seeds 1337 / 1338 / 1339 | mean ± sd | vs ref (paired) | vs control (paired) | Verdict |
|---|---|---|---|---|---|
| reference q30 (3×10 grid) | 0.9026 / 0.8258 / 0.8058 | 0.8445 ± 0.051 | — | 3/3 | baseline |
| **q30-link=shared** | 0.8651 / 0.8166 / 0.8039 | **0.8285 ± 0.032** | **3/3** | **3/3** (controls 1.121 / 0.910 / 0.837) | **PROMOTE** |
| q30-geo=chain | 0.8730 / 0.8214 / 0.8028 | 0.8324 ± 0.036 | 3/3 | 2/3 (lost to its control 0.795 on 1338) | null (runner-up) |
| q30-part=2x15 | 0.8805 / 0.8118 / 0.8620 | 0.8515 ± 0.036 | 2/3 | 2/3 | null |

**Reading.** The promoted mechanism removes controller diversity: one detuning/drive controller for all three patches, so the patches differ only by their (trainable, effectively frozen) interaction graphs, with 4,200 fewer parameters than the reference. Adding directed inter-patch coupling (cascade) was the worst configuration; the participation gate without a scarcity price never closed. The promoted gain is 1.9% on the mean and consistent in sign on every seed, inside the arena's per-seed noise on any single seed. Next: `q30-link=shared` enters TORAX, Pelzer U-Net and Sophelio as a configuration-only adaptation (3 arms × 3 seeds each, same protocols as their existing rows); chain is the runner-up candidate for the per-benchmark search; the gate-penalty addendum runs now.

## Addendum result (07:46 Sep 21): a price closes every gate

λ = 0.03 / 0.1 / 0.3 → participation 0.0008 / 0.0003 / 0.0001 with VRMSE@8 0.989 / 0.930 / 0.940 (reference 0.902, ungated gate 0.935). Even the smallest price made the optimiser drop all 30 qubits on every input: the register's marginal value per driven qubit is below 0.03 loss units, so pricing yields all-or-nothing, not per-input allocation, and the closed-gate models are register-free models whose spread (0.93–0.99) is the usual init noise. **Verdict: scarcity-by-price is a null.** The follow-up design is a fixed per-input budget (each input must choose *which* k of 30 atoms to drive, k fixed; a top-k or Gumbel-k selector over the gate logits) so the mechanism is allocation rather than pricing, tested on the arena at k ∈ {10, 20} against the 3×10 reference and a random-k control. Queued after the transfer runs.

## Transfer result (09:55 Sep 21): the arena winner does not transfer

`q30-link=shared` + matched control, 3 seeds, on the three rows' own protocols: TORAX 0.0292±0.0022 vs control 0.0287±0.0015 (2/3; per-seed within 0.2% of the reference register) · Pelzer U-Net 0.03651±0.0045 vs 0.03613±0.0044 °C (0/3; at vanilla level) · Sophelio S 0.7842±0.0095 vs 0.7816±0.0163 (1/3; 2/3 vs reference). **Null on all three.** A configuration that passed the 3/3 rule on the arena (TRL2D subset, TD-k4 rollout, FNO trunk) is indistinguishable from the reference register and from a matched MLP on other trunks and objectives. Reading: the graph effect measured on the arena is specific to that rollout regime; per-benchmark configuration must be searched on the benchmark itself (the FW-B / config-search route), not transferred. This is reported in the ladder as evidence against transfer, alongside the promotion.

## Next (queued after this): per-input budget selector

Scarcity-by-price failed (gates close globally). The allocation version: each input drives exactly k of 30 atoms, chosen by a top-k over the gate logits (straight-through), k ∈ {10, 20}; controls = the 3×10 reference (k = 30), a random fixed-k mask, and the matched MLP. Arena protocol, seed 1337 screen then 3 seeds for any k that beats the reference. Question answered: does choosing *which* qubits per input beat using all of them?

## Wave G-k (pre-registered 10:20 Sep 21): per-input budget selector

Mechanism (`paper/qlayer.py`, grammar `q30-k=<k>[-randk]`): each register drives exactly k of its 10 atoms per input, chosen by a top-k over a linear selector on the pooled latent (straight-through softmax gradient); idle atoms stay in \|0⟩ and leave the blockade graph. Arms at seed 1337, arena protocol (2,000 steps, TD k=4): `q30-k=3`, `q30-k=5`, `q30-k=7` (9 / 15 / 21 atoms per input), each with (a) its random-fixed-subset control `-randk` (same budget, no choice) and (b) its matched MLP `-mlp`; reference = q30 (all 30). Decision: a budget k is *promising* if the chosen-k arm beats both its random-k control and the reference at seed 1337, in which case it goes to seeds 1338/1339 with the same 3/3 rule as Stage 2 (vs reference and vs MLP control) plus 3/3 vs random-k. Read-out: VRMSE@8, participation (must equal k/10), and whether the selection varies across inputs (a selector that always picks the same atoms is a static sub-register, not on-the-go allocation). Runner `runs/waveG_k_runner.sh`, chained behind TRANSFER_DONE, same GPU gate.

## Wave G-k result (11:44 Sep 21): per-input budgets do not rescue scarcity

| atoms driven per input | chosen (top-k) | random fixed subset | matched MLP |
|---|---|---|---|
| 9 of 30 (k=3) | 1.080 | 1.082 | 1.105 |
| 15 of 30 (k=5) | 1.090 | 1.061 | 1.100 |
| 21 of 30 (k=7) | 1.052 | 1.063 | 1.102 |
| 30 of 30 (reference) | **0.902** | — | 1.098 |

Every budget below the full register collapses to control level, and choosing *which* atoms never beats a random fixed subset beyond noise (±1%). **Verdict: null.** Together with the price gate (all-or-nothing) this closes the on-the-go allocation question on the arena: the effect needs complete 10-atom patches, and qubit scarcity is a question of register *size* (the qubit ladder: 5q ≈ 30q) rather than per-input participation. The remaining open axis is per-benchmark configuration search (the Optuna campaign), where topology / spacing / depth / link are tuned on each benchmark's own validation split.
