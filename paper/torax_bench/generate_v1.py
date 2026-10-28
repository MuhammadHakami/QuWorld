"""TORAX-Bench v1 generator — actuator-conditioned tokamak profile rollouts.

Upgrades over the v0 pilot (generate.py):
  * All four transport equations evolved (ion heat, electron heat, current,
    density) — in v0, n_e and q were prescribed and static in time.
  * 12-dim scenario space with TIME-VARYING actuators: Ip ramp (rate +
    flattop), piecewise-linear heating waveform with 3 knots (sampled knot
    times and powers), and a gas-puff step (density actuator).
  * Fixed-dt time stepping (uniform 0.05 s grid, 121 frames over 6 s) —
    v0 used the adaptive chi stepper (~24 non-uniform frames).
  * Full commanded actuator waveforms saved per discharge as (T, 3) arrays,
    not just scalar multipliers.

Runs on CPU (jax) — the GPU stays with the training campaign.

Usage: <torax-venv-python> paper/torax_bench/generate_v1.py [n] [out_dir]
Resumable: skips discharge ids already on disk; failures logged to
<out_dir>/failures.log and re-attempted on the next invocation.
"""

import copy
import importlib
import json
import logging
import pathlib
import sys
import time

import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "data/torax_bench_v1")
SEED = 20260816

T_FINAL = 6.0          # discharge length (s) — fixed so every trajectory
FIXED_DT = 0.05        # shares the same uniform 121-frame time grid
HEAT_GAUSSIAN_LOCATION = 0.3   # off-axis deposition (base config default: 0.0)

# 12-dim scenario space. Knot COUNTS are fixed across discharges so every
# ToraxConfig shares one jit-compiled step function (values stay dynamic).
SPACE = {
    # Ip ramp: Ip(t) rises piecewise-linearly from ip_frac0*ip_flat at t=0
    # to ip_flat at t_ip_ramp, then holds flattop.
    "ip_flat": (10e6, 18e6),       # flattop plasma current (A)
    "ip_frac0": (0.4, 0.9),        # Ip(0) as fraction of flattop
    "t_ip_ramp": (0.5, 2.5),       # ramp end time (s)
    # Heating waveform: piecewise-linear generic_heat P_total through knots
    # (0, P0), (t_P1, P1), (t_P2, P2), held at P2 afterwards.
    "P0": (20e6, 120e6),           # heating power at t=0 (W)
    "P1": (20e6, 120e6),           # heating power at first knot (W)
    "P2": (20e6, 120e6),           # heating power at second knot (W)
    "t_P1": (1.0, 2.5),            # first heating knot time (s)
    "t_P2": (3.0, 4.5),            # second heating knot time (s)
    # Density: initial line-averaged density + gas-puff particle-source step.
    "nbar0": (6e19, 1.1e20),       # initial line-avg density (m^-3)
    "puff0": (0.5e22, 5e22),       # gas puff rate before step (particles/s)
    "puff1": (0.5e22, 5e22),       # gas puff rate after step (particles/s)
    "t_puff": (1.5, 4.5),          # puff step time (s)
}

ACTUATOR_NAMES = ["Ip_A", "P_heat_W", "S_gas_puff_per_s"]


def sample(rng: np.random.Generator) -> dict:
    return {k: float(rng.uniform(*b)) for k, b in SPACE.items()}


def build_config(base: dict, s: dict):
    """Return a new ToraxConfig for scenario `s` (base dict is not mutated)."""
    import torax

    cfg = copy.deepcopy(base)
    cfg["numerics"] = {
        "t_final": T_FINAL,
        "fixed_dt": FIXED_DT,
        "evolve_ion_heat": True,
        "evolve_electron_heat": True,
        "evolve_current": True,
        "evolve_density": True,
    }
    cfg["time_step_calculator"] = {"calculator_type": "fixed"}
    cfg["profile_conditions"] = {
        "Ip": {0.0: s["ip_frac0"] * s["ip_flat"], s["t_ip_ramp"]: s["ip_flat"]},
        "nbar": s["nbar0"],
        "n_e_right_bc": 0.5 * s["nbar0"],
    }
    cfg["sources"]["generic_heat"] = {
        "P_total": {0.0: s["P0"], s["t_P1"]: s["P1"], s["t_P2"]: s["P2"]},
        "gaussian_location": HEAT_GAUSSIAN_LOCATION,
    }
    cfg["sources"]["gas_puff"] = {
        "S_total": ({0.0: s["puff0"], s["t_puff"]: s["puff1"]}, "step"),
    }
    return torax.ToraxConfig.from_dict(cfg)


def actuator_waveforms(s: dict, t: np.ndarray) -> np.ndarray:
    """Commanded actuator values on the saved time grid, shape (T, 3)."""
    ip = np.interp(t, [0.0, s["t_ip_ramp"]],
                   [s["ip_frac0"] * s["ip_flat"], s["ip_flat"]])
    p_heat = np.interp(t, [0.0, s["t_P1"], s["t_P2"]],
                       [s["P0"], s["P1"], s["P2"]])
    puff = np.where(t < s["t_puff"], s["puff0"], s["puff1"])  # step mode
    return np.stack([ip, p_heat, puff], axis=1).astype(np.float32)


def extract(tree) -> dict:
    """Pull [t, T_e, T_i, n_e, q] + radial coords from the output DataTree."""
    prof = tree["profiles"]
    out = {"t": np.asarray(prof["T_e"]["time"].values, dtype=np.float32)}
    for name in ["T_e", "T_i", "n_e", "q"]:
        out[name] = np.asarray(prof[name].values, dtype=np.float32)
    out["rho_norm"] = np.asarray(prof["T_e"]["rho_norm"].values,
                                 dtype=np.float32)
    out["rho_face_norm"] = np.asarray(prof["q"]["rho_face_norm"].values,
                                      dtype=np.float32)
    return out


def validate(arrs: dict) -> None:
    for k, v in arrs.items():
        if not np.all(np.isfinite(v)):
            raise RuntimeError(f"non-finite values in {k}")
    n_t = len(arrs["t"])
    expected = int(round(T_FINAL / FIXED_DT)) + 1
    if n_t != expected:
        raise RuntimeError(f"unexpected time length {n_t} (want {expected})")


def write_manifest(scenarios: list) -> None:
    import torax

    path = OUT / "manifest.json"
    if path.exists():
        return  # keep the frozen manifest from the first invocation
    path.write_text(json.dumps({
        "version": "v1",
        "seed": SEED,
        "n": N,
        "space": SPACE,
        "t_final": T_FINAL,
        "fixed_dt": FIXED_DT,
        "heat_gaussian_location": HEAT_GAUSSIAN_LOCATION,
        "actuator_names": ACTUATOR_NAMES,
        "scenario_names": list(SPACE.keys()),
        "base_config": "torax.examples.basic_config",
        "torax_version": torax.__version__,
        "scenarios": scenarios,
    }, indent=1))


def run_one(i: int, s: dict, base: dict, path: pathlib.Path) -> None:
    import torax

    cfg = build_config(base, s)
    tree, _hist = torax.run_simulation(cfg, progress_bar=False)
    arrs = extract(tree)
    acts = actuator_waveforms(s, arrs["t"].astype(np.float64))
    arrs["actuators"] = acts
    validate(arrs)
    np.savez_compressed(
        path,
        scenario=np.array(list(s.values()), dtype=np.float32),
        **arrs,
    )


def main() -> None:
    logging.getLogger().setLevel(logging.WARNING)  # silence per-run chatter
    try:  # torax logs "Starting simulation" etc. through absl at INFO
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.WARNING)
    except ImportError:
        pass
    mod = importlib.import_module("torax.examples.basic_config")

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    scenarios = [sample(rng) for _ in range(N)]  # deterministic across resumes
    write_manifest(scenarios)

    fail_log = OUT / "failures.log"
    t_all = time.perf_counter()
    n_ok, n_fail, n_skip = 0, 0, 0
    for i, s in enumerate(scenarios):
        f = OUT / f"discharge_{i:05d}.npz"
        if f.exists():
            n_skip += 1
            continue
        t0 = time.perf_counter()
        try:
            run_one(i, s, mod.CONFIG, f)
            n_ok += 1
            if n_ok <= 2 or n_ok % 50 == 0:
                print(f"[{i + 1}/{N}] ok in {time.perf_counter() - t0:.2f}s",
                      flush=True)
        except Exception as exc:  # noqa: BLE001 — log and continue generation
            n_fail += 1
            msg = f"discharge_{i:05d} {type(exc).__name__}: {exc}"
            print(f"[{i + 1}/{N}] FAILED: {msg}", flush=True)
            with fail_log.open("a") as fh:
                fh.write(msg + "\n")
            f.unlink(missing_ok=True)  # never leave partial files behind
    wall = time.perf_counter() - t_all
    print(f"GENERATION_DONE ok={n_ok} skipped={n_skip} failed={n_fail} "
          f"of {N} in {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
