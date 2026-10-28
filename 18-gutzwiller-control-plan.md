# Gutzwiller (mean-field) control — pre-registered 2026-09-06 14:05 local

**Question.** The no-entanglement ablation zeroed the blockade term, which removes
entanglement AND the interaction nonlinearity at once. Does the training-side
effect need entanglement, or only interactions?

**Control.** Same `mr30ryd` architecture, same weights/initialisation, same TD
protocol (10k steps, k=4, lr 1e-4, seeds 1337/1338/1339, rollout-valid VRMSE@8
selection, single-shot official test rollout). The only change: the diagonal
blockade phase exp(-i tau E), E = sum_{i<j} V_ij n_i n_j, is replaced by its
mean-field (Gutzwiller/Hartree) decoupling, a per-qubit local phase
theta_i = tau * sum_{j!=i} V_ij <n_j>. The register is a product state by
construction (zero entanglement) but keeps the V_ij(r) nonlinearity, the
trainable geometry, and the input-conditioned detunings. Because the phase is
diagonal it leaves occupations unchanged, so the explicit step is the exact
mean-field dynamics for this Trotter structure (no self-consistency loop).

**Decision rule (fixed before any run).** Compare the three `gutz` arms'
official-test 13:30 VRMSE with the entangled arms (0.9528 +/- 0.0542, 3/3 seeds
below the 1.0132 anchor), the no-entanglement arms (1.0156, 1/3 below) and the
controls (~1.03).
- gutz tracks the entangled arms (>=2/3 seeds below anchor, mean within one SD
  of 0.9528): the effect needs interactions, not entanglement -> the paper's
  mechanism claim becomes "interaction-induced nonlinearity during training".
- gutz tracks noent/controls (<=1/3 seeds below anchor, mean >= 1.00):
  entanglement is necessary during optimisation -> the stronger claim is
  restored with the confound removed.
- anything else is reported as ambiguous; no post-hoc rule.

**Pilot.** The sprint arena (2k steps, k=4, 3 seeds) runs first as a smoke test
and early signal only; its seed variance (SD 0.05-0.25) makes it non-decisive.

**Artifacts.** `paper/qlayer.py` ansatz `rydberg_mf`; `paper/tests/`;
`paper/phase_b/gutz_shim.py`, `finetune_td_gutz.py`, `eval_rollout_test_gutz.py`;
checkpoints `runs/phase_b2/gutz_s{seed}*.pt`; logs `runs/phase_b2/td_gutz_s*.log`.

**Amendment (2026-09-06 14:10, before any run started).** The control is wired
as the native variant `gutz30` in both harnesses instead of shim scripts:
training via stock `paper/phase_b/finetune_td.py gutz30 ...`, evaluation via
stock `eval_rollout_test.py` (it derives the variant from the checkpoint name),
checkpoints `runs/phase_b2/gutz30_s{seed}*.pt`, logs `runs/phase_b2/td_gutz30_s*.log`,
arena pilot logs `runs/ladder_gutz30_k4_s*.log`. Protocol and decision rule unchanged.

## Result (2026-09-06 19:50, single-shot official test, eval_rollout_test.py on runs/phase_b2/gutz30_s{seed}.pt)

| seed | 6:12 | 13:30 | below anchor 1.0132? |
|---|---|---|---|
| 1337 | 0.5029 | 1.1882 | no |
| 1338 | 0.5221 | 0.9592 | yes |
| 1339 | 0.5397 | 1.0011 | yes |

Mean 13:30 = 1.0495 ± 0.1219 (sample SD). Selection bests 0.5847/0.5870/0.5838.

**Verdict (fixed rule applied, no post-hoc changes):** AMBIGUOUS under the pre-registered rule: mean-field 13:30 = 1.0495 ± 0.1219 (seeds 1.1882/0.9592/1.0011), 2/3 below anchor 1.0132; not within one SD of the entangled 0.9528 (rule A fails on the mean), not ≤1/3 below anchor (rule B fails on seeds). Point estimate sits at control level (vanilla 1.0301, MLP 1.0322, noent 1.0156) and ~2 entangled-SDs above the entangled arms; 6:12 matched (0.5216). Direction favors entanglement-matters but is not decisive at n=3 — more seeds (Future Work i) required.

Logs: runs/phase_b2/rollout_gutz30_s{1337,1338,1339}.log. Reported in the ICLR version's mechanism section; the workshop PDF (v3.1) already labels the arena pilot non-decisive.

### Addendum: sliding-start view of the control (2026-09-06 20:00; protocol of 19-multistart-power-plan.md, npz `runs/phase_b2/multistart_gutz30_s*.npz`, canonical single-shot official logs `runs/phase_b2/eval_test_gutz30_s*.log`)

| arm | all-start 13:30, seed mean ± SD | per seed |
|---|---|---|
| quantum (entangled) | 0.9486 ± 0.0231 | 0.9620 / 0.9618 / 0.9219 |
| no-entanglement | 0.9664 ± 0.0264 | 0.9437 / 0.9602 / 0.9954 |
| **Gutzwiller (mean-field)** | **0.9782 ± 0.0418** | 1.0222 / 0.9732 / 0.9390 |
| vanilla fine-tune | 0.9867 ± 0.0196 | 1.0091 / 0.9784 / 0.9727 |
| matched-MLP fine-tune | 1.0041 ± 0.0160 | 0.9993 / 1.0221 / 0.9911 |

Quantum vs Gutzwiller, pre-specified primary test (per-trajectory Wilcoxon on start-averaged 13:30, Holm across seeds): 0/3 significant (one-sided p = 0.41 / 0.29 / 0.15; win fractions 0.44 / 0.44 / 0.56); secondary 81-pair test 0/3; cluster-bootstrap CIs of the mean difference all include zero ([-0.239, +0.088], [-0.035, +0.009], [-0.045, +0.010]).

**Reading.** Across starts the mean-field control sits between the entangled arm and the classical fine-tunes and is not distinguishable from the entangled arm at this power; the official-start verdict is AMBIGUOUS. Taken together with the arena pilot (mean-field = entangled on all seeds) and the no-entanglement comparison (1/3), the entanglement question is unresolved at three seeds, while the register-versus-classical-bottleneck gap remains the robust, powered finding. ICLR wording: "interactions appear necessary at the official start; entanglement is not separable from interaction nonlinearity at three seeds"; more seeds are the only way to resolve it.

Full pre-specified statistics for quantum vs Gutzwiller (from `paired_stats_multistart.py mr30ryd gutz30 1337 1338 1339`): mean diff −0.0602 / −0.0115 / −0.0172; primary p 0.4102 / 0.2852 / 0.1504, Holm 0.5703 / 0.5703 / 0.4512; secondary win 0.40 / 0.53 / 0.51, secondary p 5.4e-1 / 2.0e-1 / 1.2e-1.
