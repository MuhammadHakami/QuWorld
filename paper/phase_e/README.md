# Phase E — Dequantization suite (B4/B5 + finite-shot)

Question (08-scoreboard B4/B5, 11-master-plan Phase E): does the quantum
block's contribution to the paper's headline TD result survive classical
surrogacy — MPS-style chi-truncation, product-state (no-entanglement)
ablation, and finite-shot readout? Either outcome is reported.

## Targets

The three TD-fine-tuned quantum checkpoints behind the headline result:
`runs/phase_b2/mr30ryd_s{1337,1338,1339}.pt` — CNextU-net hybrids whose neck
carries a `MultiRegister` of 3 x 10-qubit Rydberg-analog blocks
(`paper/qlayer.py`), pooled-latent conditioned, `<Z_i>` readout through the
zero-init gated bottleneck (`paper/model.py::Bottleneck`).

## Protocol

Metric everywhere: **valid rollout VRMSE@8** — `finetune_td.validate_rollout`
(k=8, stride_t=8), the TD campaign's selection metric, on the 9-trajectory
TRL2D valid split (`data_ram.py` cache). Checkpoint weights are never
modified; surrogates are eval-time wrappers around the trained registers
(`paper/phase_e/dequant.py`), verified to reproduce each register's own
forward exactly (max readout diff 0.0e+00) before use. Intact re-evals
reproduce the recorded checkpoint selection values to <=3e-4 (the recorded
values for s1338/s1339 came from the campaign's final stride_t=4 eval; this
suite holds stride_t=8 for every row, so all deltas are like-for-like).

1. **Product-state ablation (B5)** — entanglement surgically disabled, drive
   and input-conditioned detunings kept, two ways: (a) `prod_zero`: blockade
   interaction energies set to 0 exactly; (b) `prod_far`: trained geometry
   scaled 100x (spacing >> blockade radius; residual energies ~1e-11 vs
   trained maxima ~15). Both give exactly factorized single-qubit evolution
   (verified: final-state entanglement entropy = 0).
2. **chi-sweep Schmidt-truncation surrogate (B4)** — the final 1024-dim state
   of each register is Schmidt-decomposed across the 5|5 qubit bipartition
   (torch SVD of the 32x32 reshape), truncated to rank chi in
   {1,2,4,8,16,32}, renormalized, then read out and propagated through the
   trained W_out. chi=32 is exact (pipeline sanity: matches intact to 4e-6).
   Entanglement entropy and cumulative Schmidt weights are measured on 256
   real valid latents captured at the neck (the exact input the quantum core
   sees).
3. **Finite-shot readout** — exact `<Z_i>` replaced by s-shot binomial
   estimates of the marginal P(z_i=+1), s in {128, 1024, 8192}
   (`torch.binomial`, seeded per config: seed*10007 + sum(ord(name))).
4. **Load-bearing controls** (added after 1-3 came back null, to separate
   "readout content is surrogate-able" from "readout is unused"):
   `const0` — register readouts replaced by zeros (no information flows
   through the quantum core at eval; trained W_out bias kept); `gate_off` —
   the bottleneck's alpha gate zeroed (adapter contributes nothing to the
   field at eval; the model is the fine-tuned trunk alone).

Run/reproduce (GPU-light; allocator capped at 25%, batch = 9 valid trajs):

```
uv run paper/phase_e/run_suite.py                  # all seeds, all configs
uv run paper/phase_e/summarize.py                  # table below
```

Raw numbers: `runs/phase_e/results.json` (incremental, interrupt-safe).
Log: `runs/phase_e/suite.log`.

## Results

Valid rollout VRMSE@8 (lower is better), measured 2026-08-23:

| config | s1337 | s1338 | s1339 | mean +/- std | delta vs intact |
|---|---|---|---|---|---|
| intact (exact statevector) | 0.578391 | 0.576815 | 0.563994 | 0.573066 +/- 0.007896 | — |
| product-state: blockade -> 0 | 0.578406 | 0.576821 | 0.564001 | 0.573076 +/- 0.007899 | +0.000009 |
| product-state: far geometry (100x) | 0.578406 | 0.576821 | 0.564001 | 0.573076 +/- 0.007899 | +0.000009 |
| chi=1 Schmidt truncation | 0.578399 | 0.576817 | 0.564003 | 0.573073 +/- 0.007894 | +0.000007 |
| chi=2 | 0.578400 | 0.576821 | 0.563981 | 0.573067 +/- 0.007908 | +0.000001 |
| chi=4 | 0.578400 | 0.576822 | 0.563995 | 0.573072 +/- 0.007901 | +0.000006 |
| chi=8 | 0.578393 | 0.576810 | 0.564005 | 0.573069 +/- 0.007889 | +0.000003 |
| chi=16 | 0.578388 | 0.576802 | 0.563987 | 0.573059 +/- 0.007896 | -0.000007 |
| chi=32 (exact rank) | 0.578387 | 0.576804 | 0.563990 | 0.573060 +/- 0.007895 | -0.000006 |
| 128-shot readout | 0.578406 | 0.576829 | 0.564002 | 0.573079 +/- 0.007900 | +0.000013 |
| 1024-shot readout | 0.578399 | 0.576821 | 0.563997 | 0.573072 +/- 0.007899 | +0.000006 |
| 8192-shot readout | 0.578398 | 0.576815 | 0.563994 | 0.573069 +/- 0.007899 | +0.000003 |
| CONTROL: readout -> zeros | 0.578412 | 0.576833 | 0.563982 | 0.573076 +/- 0.007915 | +0.000009 |
| CONTROL: adapter gate off | 0.578405 | 0.576821 | 0.563972 | 0.573066 +/- 0.007915 | -0.000001 |

Every configuration is within |delta| <= 1.3e-5 of intact — ~600x smaller
than the seed-to-seed std (0.0079) and ~0.002% of the metric.

Mechanism measurements (256 real valid latents, per register, 3 seeds):

- **Entanglement entropy (5|5 cut)**: 0.032–0.089 bits across all 9 trained
  registers (max possible: 5 bits). chi=1 Schmidt weight 0.9907–0.9973
  (mean 0.9926); chi=8 weight > 0.999996 everywhere. Product-state ablation
  entropy = 0.000000 (ablation verified).
- **Readout regime**: mean |<Z_i>| = 0.991 — the registers sit in a
  near-saturated, near-computational-basis regime. Input-driven readout
  variation (std across latents, RMS over the 30 dims): 0.0133 (s1337),
  0.0006 (s1338), 0.0003 (s1339).
- **Readout perturbation RMS** (surrogate vs exact <Z>, 30 dims):
  product-state ~0.009, chi=1 ~0.004, 128-shot ~0.012 — all leave the
  rollout metric unchanged, consistent with the controls.
- **Trained geometry**: min inter-atom distance 0.87–0.92 blockade radii,
  blockade energy maxima 14.6–16.9 (far ablation residual ~1.5e-11);
  tau ~= 0.30, omega ~= 0.50 (near init).

Runtime: ~21 s per rollout eval on the GB10, run concurrently with the
Phase-B MHD fine-tune campaign (GPU ~96% busy throughout); allocator capped
at 25%, no OOM, no interference observed in the MHD logs.

## Verdict

**The quantum block's contribution does NOT survive dequantization as an
inference-time effect — because there is no inference-time effect to remove.**

1. Every surrogate — product-state (no entanglement), chi=1 truncation,
   128-shot readout — reproduces the intact valid rollout VRMSE@8 to
   <= 1.3e-5 (600x below seed noise): the trained registers are weakly
   entangled (<= 0.09 of 5 bits, chi=1 weight >= 0.991) and read out in a
   saturated <Z> ~= +/-1 regime, so the eval-time computation is trivially
   classically simulable.
2. The controls sharpen this into the real finding: zeroing the entire
   readout, or the adapter's alpha gate, ALSO leaves the metric unchanged
   (<= 2.4e-5) — the fine-tuned adapter is inference-time inert, so the
   quantum arm's headline advantage on the official test rollouts
   (0.9528 vs controls ~1.03) is a TRAINING-DYNAMICS effect: the quantum
   bottleneck steered the trunk's TD fine-tuning into a better basin, and
   that benefit is permanently baked into the trunk weights.
3. Consequences for the paper: (a) B5/B4 rows report "dequantizes at
   inference; benefit is optimizer-path-dependent" — the honest claim is
   quantum-influenced optimization, not quantum inference; (b) hardware
   deployment is trivially shot-robust (128 shots suffice, and in fact the
   register could be removed at deployment); (c) the decisive follow-up
   experiment is a TRAINING-side ablation — fine-tune with the product-state
   (blockade-zeroed) register from step 0 and test whether the trained-in
   advantage still appears (this suite only proves the eval-time claim).

The suite reports the outcome as measured, per the pre-registered protocol
("either outcome is reported").
