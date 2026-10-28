# Sliding-start rollout protocol (statistical power) — pre-registered 2026-09-06 14:20 local

**Problem.** The Well's official rollout scores one 30-step rollout per test
trajectory from t=0, so paired long-horizon tests have n=9 and were reported as
underpowered in the WM-PAI paper (1/3 seeds Holm-significant vs MLP, 0/3 vs
vanilla).

**Protocol (ICLR version).** Identical models, normalisation, history (4 frames),
horizon (30 steps) and windows (6:12, 13:30) as the official protocol, but
rollouts start at t in {0, 8, 16, ..., 64} (stride 8; the last start still fits
4+30 frames inside the 101-frame trajectories). Start 0 reproduces the official
numbers exactly and is checked against `runs/phase_b2/pertraj_<name>.npz`.
This yields 9 starts x 9 trajectories = 81 paired rollouts per model.

**Pre-specified statistics.**
- Primary: per-trajectory paired Wilcoxon (n=9) on the 13:30 VRMSE averaged
  over the 9 starts (start-averaging reduces within-trajectory noise without
  inflating n). Comparisons: quantum vs vanilla, vs matched-MLP, vs anchor, and
  vs the noent and Gutzwiller arms, per seed, Holm-corrected across seeds.
- Secondary: per-(trajectory, start) paired Wilcoxon (n=81), reported with the
  caveat that starts within a trajectory are correlated, plus a cluster
  bootstrap over trajectories (10,000 resamples) for the mean paired difference.
- Also reported: 6:12 window, win fractions, and how the advantage depends on
  start time (later starts probe later mixing-layer dynamics).
No claim is decided by this protocol; it is a measurement-power extension.

**Artifacts.** `paper/phase_b/rollout_windows.py` (tested helpers),
`paper/phase_b/eval_rollout_multistart.py` (writes
`runs/phase_b2/multistart_<name>.npz`), `paper/phase_b/paired_stats_multistart.py`.

## Results (2026-09-06 14:35 local; 13 models, 9 starts x 9 trajectories, all start-0 rows identical to the official per-trajectory arrays)

**All-start 13:30 VRMSE (lower is better), seed mean ± SD; start-0 column = official protocol.**

| arm | all-start 13:30 | per seed | start-0 (official) per seed |
|---|---|---|---|
| mr30ryd | 0.9486 ± 0.0231 | 0.9620 / 0.9618 / 0.9219 | 0.9927 / 0.9745 / 0.8911 |
| noent | 0.9664 ± 0.0264 | 0.9437 / 0.9602 / 0.9954 | 1.0202 / 0.9701 / 1.0564 |
| vanilla | 0.9867 ± 0.0196 | 1.0091 / 0.9784 / 0.9727 | 1.0787 / 1.0216 / 0.9901 |
| mlp30 | 1.0041 ± 0.0160 | 0.9993 / 1.0221 / 0.9911 | 1.0328 / 1.0060 / 1.0578 |
| anchor (no fine-tune) | 0.9707 | — | 1.0132 |

**Pre-specified tests, quantum (mr30ryd) vs each comparator, per seed.** Primary = per-trajectory paired Wilcoxon on start-averaged 13:30 VRMSE (n=9), one-sided quantum-better, Holm across seeds (* = significant at 0.05). Secondary = per-(trajectory, start) pairs (n=81, correlated within trajectory). CI = trajectory cluster bootstrap of the mean paired difference.

| comparator | seed | mean diff | primary win | primary p | Holm | secondary win | secondary p | bootstrap CI95 |
|---|---|---|---|---|---|---|---|---|
| vanilla | 1337 | -0.0471 | 0.78 | 0.0195 | 0.0391* | 0.59 | 5.0e-03 | [-0.0833, -0.0129] |
| vanilla | 1338 | -0.0167 | 0.78 | 0.1016 | 0.1016 | 0.57 | 2.3e-02 | [-0.0393, +0.0101] |
| vanilla | 1339 | -0.0508 | 0.89 | 0.0039 | 0.0117* | 0.77 | 7.5e-08 | [-0.0702, -0.0260] |
| mlp30 | 1337 | -0.0372 | 0.78 | 0.1016 | 0.1016 | 0.53 | 1.9e-02 | [-0.0795, +0.0079] |
| mlp30 | 1338 | -0.0603 | 1.00 | 0.0020 | 0.0059* | 0.75 | 5.2e-08 | [-0.0889, -0.0339] |
| mlp30 | 1339 | -0.0692 | 1.00 | 0.0020 | 0.0059* | 0.77 | 1.0e-08 | [-0.0929, -0.0462] |
| anchor | 1337 | -0.0086 | 0.56 | 0.5449 | 1.0000 | 0.44 | 4.9e-01 | [-0.0981, +0.0725] |
| anchor | 1338 | -0.0089 | 0.44 | 0.6328 | 1.0000 | 0.48 | 5.0e-01 | [-0.0900, +0.0571] |
| anchor | 1339 | -0.0488 | 0.56 | 0.1504 | 0.4512 | 0.56 | 3.5e-02 | [-0.1206, +0.0249] |
| noent | 1337 | +0.0183 | 0.11 | 0.9512 | 1.0000 | 0.30 | 1.0e+00 | [-0.0706, +0.0737] |
| noent | 1338 | +0.0016 | 0.44 | 0.5000 | 1.0000 | 0.54 | 4.6e-01 | [-0.0333, +0.0391] |
| noent | 1339 | -0.0735 | 0.89 | 0.0039 | 0.0117* | 0.67 | 9.5e-05 | [-0.1152, -0.0350] |

**Reading (pre-specified statistics only).**
- vs fine-tuned vanilla: primary Holm-significant on 2/3 seeds, secondary on 3/3; the quantum arm's long-horizon edge over the matched classical fine-tune is now a powered result, not a directional one.
- vs matched-MLP: primary 2/3, secondary 3/3 — same conclusion against the parameter-matched bottleneck.
- vs anchor: primary 0/3. Across all starts the un-fine-tuned anchor is strong on late starts; the quantum arm is below it on average but not significantly. The paper's 'only arm below the anchor on 3/3 seeds' statement holds at the official start; it does not generalise to all starts as a powered claim. Note the fine-tuned controls are *worse* than the anchor across starts, so TD fine-tuning degrades late-start behaviour for the controls but not for the quantum arm.
- vs no-entanglement arm: primary 1/3 (seed 1339 only). Under the broader protocol the product-state register is close to the entangled one on two seeds, so 'interactions necessary during optimisation' is supported at the official start but is NOT a general powered claim; together with the arena pilot (mean-field = entangled on all seeds) this points to the register's nonlinear analog map rather than entanglement. The official-protocol Gutzwiller arms decide the ICLR wording.
- Ordering across starts: quantum < noent < anchor < vanilla < MLP: any Rydberg-structured register beats the classical bottleneck; entanglement adds a small further edge on one seed.

Artifacts: `runs/phase_b2/multistart_<name>.npz`, logs `runs/phase_b2/multistart_<name>.log`, status `runs/phase_b2/multistart_status`. Reproduce the tables with `uv run paper/phase_b/paired_stats_multistart.py mr30ryd <vanilla|mlp30|anchor|noent> 1337 1338 1339`.

Figure: `paper/manuscript/fig_multistart.pdf` (script `paper/figs_multistart.py`; palette validated with the dataviz validator, all checks pass; anchor drawn as a dashed neutral reference). Left: seed-mean 13:30 VRMSE per start; right: paired quantum-minus-comparator difference per start with the seed range as a band.
