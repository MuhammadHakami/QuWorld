# TORAX-Bench v1 — Dataset Card

Actuator-conditioned 1D tokamak core-transport rollouts for training and
evaluating surrogate / world models. 1,000 simulated discharges generated with
[TORAX](https://github.com/google-deepmind/torax) v1.4.3 (JAX, CPU, float64).

## Generation spec

- **Base config:** `torax.examples.basic_config` (circular geometry,
  R_major=6.2 m, a_minor=2.0 m, B_0=5.3 T, elongation 1.72, n_rho=25;
  constant-chi transport model; linear theta-method solver with Pereverzev
  terms; no pedestal model).
- **Evolved equations:** ion heat, electron heat, current (psi), and electron
  density — all four (the v0 pilot evolved only the two temperatures; its n_e
  and q were prescribed and static in time).
- **Time grid:** fixed dt = 0.05 s, t_final = 6.0 s, `exact_t_final` — every
  discharge saves the same uniform grid of **121 frames** (t = 0.00 … 6.00 s).
- **Generator:** `paper/torax_bench/generate_v1.py`, seed **20260816**,
  scenarios drawn i.i.d. uniform from the 12-dim space below
  (`data/torax_bench_v1/manifest.json` freezes the exact draws).

### Scenario space (12 dims, all uniform)

| Dim | Range | Meaning |
|---|---|---|
| `ip_flat` | 10–18 MA | flattop plasma current |
| `ip_frac0` | 0.4–0.9 | Ip(0) as fraction of flattop |
| `t_ip_ramp` | 0.5–2.5 s | Ip ramp end time (piecewise-linear ramp, then hold) |
| `P0`, `P1`, `P2` | 20–120 MW | heating power at waveform knots |
| `t_P1` | 1.0–2.5 s | first heating knot time |
| `t_P2` | 3.0–4.5 s | second heating knot time |
| `nbar0` | 0.6–1.1 x 10^20 m^-3 | initial line-averaged density (edge BC = nbar0/2) |
| `puff0`, `puff1` | 0.5–5 x 10^22 s^-1 | gas-puff particle source before/after step |
| `t_puff` | 1.5–4.5 s | gas-puff step time |

Time-varying actuators: `Ip(t)` (piecewise-linear ramp + flattop),
`P_heat(t)` (piecewise-linear, 3 knots at 0/t_P1/t_P2, held after t_P2),
`S_gas_puff(t)` (step at t_puff). Heating deposition is a Gaussian fixed at
rho_norm = 0.3 (width 0.25, default electron fraction 2/3).

### Per-discharge file format (`discharge_#####.npz`)

| Key | Shape | Units | Notes |
|---|---|---|---|
| `T_e`, `T_i` | (121, 27) | keV | electron / ion temperature on `rho_norm` |
| `n_e` | (121, 27) | m^-3 | electron density on `rho_norm` |
| `q` | (121, 26) | – | safety factor on `rho_face_norm` |
| `actuators` | (121, 3) | A, W, s^-1 | commanded `[Ip, P_heat, S_gas_puff]` waveforms on the saved time grid |
| `scenario` | (12,) | – | raw sampled dims, order = `manifest.json:scenario_names` |
| `t` | (121,) | s | saved times |
| `rho_norm` / `rho_face_norm` | (27,) / (26,) | – | radial grids (cells + boundaries / faces) |

All arrays float32. Dataset size: 44 MB (0.044 GB, 1,000 files, ~45 KB each,
compressed npz).

## Splits (frozen)

`paper/torax_bench/splits.json` — train/valid/test = **800/100/100** by
discharge id, random permutation with seed **20260816**. Do not resplit.

## Intended tasks

1. **One-step prediction:** given frame t (4 profile channels) and the
   actuator values, predict frame t+1 (dt = 0.05 s). Metric: VRMSE@1.
2. **Rollout:** autoregressive prediction over 8 steps (0.4 s) given the
   actuator waveforms; VRMSE@8 (and full-horizon rollouts as a stretch task).

**Metric — VRMSE** (variance-normalized RMSE, per The Well convention): for
each (discharge, target frame, channel), `sqrt(mean_x (u - u_hat)^2 /
(var_x(u) + 1e-12))` over the radial grid, averaged over frames then
discharges, reported per channel and as a 4-channel mean.

## Baselines (CPU, `paper/torax_bench/baselines.py`, frozen splits)

VRMSE, lower is better (computed 2026-08-17 on the frozen test split):

| baseline | T_e | T_i | n_e | q | mean |
|---|---|---|---|---|---|
| persistence@1 | 0.0158 | 0.0159 | 0.0098 | 0.0123 | **0.0134** |
| persistence@8 | 0.1224 | 0.1167 | 0.0355 | 0.0947 | **0.0923** |
| climatology | 0.6375 | 0.5400 | 0.2053 | 0.8357 | **0.5546** |

A learned model must beat persistence at the matched horizon to demonstrate
skill; climatology bounds the "ignore the input entirely" regime.

## Physics caveats

- **TORAX is a 1D flux-surface-averaged core-transport model** — it evolves
  radial profiles only; no 2D/3D MHD, no turbulence dynamics, no sawteeth or
  ELM crashes, no scrape-off layer. Its transport equations have been verified
  against RAPTOR with **<5% NRMSD** (Citrin et al., TORAX paper).
- v1 uses the **constant-chi transport model** and **circular geometry** — a
  deliberately simple, fully-reproducible setting. Profile dynamics are
  therefore smoother and more diffusive than QLKNN/experimental discharges;
  treat absolute numbers as benchmark-internal, not machine predictions.
- No pedestal model; edge boundary conditions are simple prescribed values.
  Fusion, ohmic, ei-exchange, bootstrap, and generic current sources are
  active with base-config defaults.
- Actuator "commands" are exact boundary conditions / source magnitudes
  (no actuator dynamics or control latency).

## License

**Apache-2.0.** Derived from TORAX (Copyright DeepMind Technologies Limited,
Apache-2.0); generation configs based on `torax.examples.basic_config`.

## Provenance / regeneration

```
<torax-venv-python> paper/torax_bench/generate_v1.py 1000 data/torax_bench_v1
python paper/torax_bench/baselines.py
```
Resumable; failures logged to `data/torax_bench_v1/failures.log`.

Generation run (2026-08-17, single CPU process, DGX Spark GB10 aarch64):
**1000/1000 ok, 0 failed**, 1223 s wall (~1.2 s/discharge), peak RSS ~0.9 GB.
