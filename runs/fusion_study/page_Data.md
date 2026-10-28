# Data

All data is on Hugging Face: **[`Sophelio/fusion-equilibrium-challenge`](https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge)** (CC BY 4.0). Pull it with the `datasets` library — no machine credentials required.

## Configs

| Config | Split | Shots | Scored frames | Contents |
|---|---|---|---|---|
| `diii_d_train` | `train` | 7,041 | — | DIII-D shots, **full** (inputs + all targets) — train here |
| `diii_d_public_test` | `public_test` | 874 | 193,359 | DIII-D shots, **inputs only** (targets withheld) — predict here |
| `mast_public_test` | `public_test` | 1,206 | 69,075 | MAST shots, **inputs only** — predict here (zero-shot; MAST has no train split) |

Each shot is one row; every time series and 2-D map is a nested array in that row. "Scored frames"
is the total number of `efit_times` timestamps across the fold — that is how many flux maps a
complete submission contains (262,434 in all, ≈1.9 GB in `float16`).

The **Final** phase swaps in two private folds of comparable size, blind until it closes.

## What you predict on

The public-test configs ship **inputs only** — all EFIT-derived targets (`efit_psirz`, the
equilibrium scalars, the LCFS, and `magnetics_dsep`) are withheld. Two scaffolding columns are
kept so you know *when* to predict:

- **`efit_times`** — the EFIT timestamps; predict one value per timestamp, in this order.
- **`efit_grid_R` / `efit_grid_Z`** — the 65-point flux-grid coordinates (metres).

## Inputs (non-magnetic only)

- **Coil currents** (`magnetics_*`): the shaping/field coils — commanded actuators, *not*
  measurements of the plasma field. DIII-D: F-coils + `ECOILA`, `bcoil`, plasma current. MAST:
  P-coils + `Solenoid`, `TF`, `EFPS`, plasma current.
- **Thomson scattering** (`thomson_*`): electron temperature and density profiles.

> `magnetics_dsep` carries a `magnetics_` prefix but is **EFIT-derived, not an input.** It is
> withheld on the test configs. It is not a scored target either — see **Evaluation**.
> `magnetics_plasma_current` (Ip) *is* an allowed input.

> **Diverted plasmas only.** Every timeslice in the dataset is a diverted plasma; limited frames
> have been removed from `efit_times` and every column on that axis. On a limited plasma the
> boundary is set by contact with the wall, and wall geometry is not contained in ψ(R,Z) — so it
> was never inferable from the inputs you are given. The full-rate `magnetics_*` and `thomson_*`
> inputs still cover the whole discharge; only `efit_times` is non-contiguous.

> **Machine geometry ships as inputs.** Every row of every split carries the PF-coil geometry
> (`coil_R`, `coil_Z`, `coil_width`, `coil_height` plus the parallelogram skew
> `coil_angle1`/`coil_angle2`, joined to the current columns by `coil_input_column`) and the
> Thomson chord positions (`thomson_chord_R/Z`, all subsystems). DIII-D describes 19 lumped
> rectangles — every F-coil row verified exactly against EFIT's own `mhdin.dat`, six of them
> carrying a non-zero skew angle; MAST ships 812 individual conductor elements — different
> granularity, both correct for their machine. DIII-D chord positions vary shot to shot; do not
> cache them.

> **There is no `coil_turns` column, and you do not need one.** Shaping-coil currents are already
> in **ampere-turns**: each DIII-D F-coil's turn count is folded into `magnetics_F*`, and MAST's
> `p{2-6}{l,u}_current` are natively `kA·turn`. That is the quantity a Green's-function
> calculation wants, so `F4A` (~127 kA·turn) and MAST `p4u` (~114 kA·turn) are directly
> comparable. Plasma current is in **kA** on both machines, as are DIII-D `ECOILA`/`bcoil` and
> MAST `sol`/`tf`/`efps`, for which no trustworthy turn count exists.

> **Open caveat on the MAST solenoid.** `magnetics_sol_current` reads exactly **2×** the FAIR-MAST
> level-2 `pf_active` solenoid current, and we have not resolved which convention is correct. Its
> geometry has 656 single-turn elements. If your forward model uses the solenoid quantitatively,
> treat that factor as an open question. `tf_current` and DIII-D `bcoil` are toroidal field coils
> with no poloidal rectangle, and `efps_current` is not a coil — it is a feed current at the P2
> link board.

Inputs live on different time bases (magnetics ≫ EFIT ≫ Thomson); aligning them onto `efit_times`
is part of the task. **Never interpolate the targets.** Split by shot, never by timestep. Every
time array is in **milliseconds** on both machines. Full schema: the dataset card on Hugging Face,
plus the starter kit's `README.md` and `MODELING_GUIDE.md`.