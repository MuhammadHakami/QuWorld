"""TORAX-Bench v0 generator — actuator-conditioned tokamak profile rollouts.

Runs on CPU (jax) — the GPU stays with the training campaign. Samples a
scenario space around torax.examples.basic_config via Sobol-like uniform
draws, simulates each discharge, and writes per-discharge profile histories
[Te, Ti, ne, q] x radial cells x saved steps + the actuator vector.

Usage: <torax-venv-python> paper/torax_bench/generate.py [n_discharges] [out_dir]
Resumable: skips discharge ids that already exist on disk.
"""

import copy
import importlib
import json
import pathlib
import sys
import time

import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                   "data/torax_bench_v0")
SEED = 20260816

# Scenario space (fractional multipliers on the base config + absolutes).
# Kept deliberately simple for v0; the full bench widens this per the spec.
SPACE = {
    "Ip_mult": (0.7, 1.3),          # plasma current scale
    "nbar_mult": (0.7, 1.3),        # line-averaged density scale
    "heat_mult": (0.5, 2.0),        # external heating power scale
    "t_final": (4.0, 6.0),          # discharge length (s)
}


def sample(rng: np.random.Generator) -> dict:
    return {k: float(rng.uniform(*b)) for k, b in SPACE.items()}


def build_config(base: dict, s: dict):
    import torax

    cfg = copy.deepcopy(base)
    pc = cfg["profile_conditions"]
    pc["Ip"] = {0: pc["Ip"][0] * s["Ip_mult"]} if isinstance(pc.get("Ip"), dict) \
        else pc.get("Ip", 15e6) * s["Ip_mult"]
    if "nbar" in pc:
        pc["nbar"] = pc["nbar"] * s["nbar_mult"]
    if "generic_heat" in cfg.get("sources", {}):
        src = cfg["sources"]["generic_heat"]
        if "P_total" in src:
            src["P_total"] = src["P_total"] * s["heat_mult"]
    cfg["numerics"]["t_final"] = s["t_final"]
    return torax.ToraxConfig.from_dict(cfg)


def profiles_from(result) -> dict:
    """Extract [Te, Ti, ne, q] histories from a torax result (tuple in 1.4.3)."""
    hist = result[0] if isinstance(result, tuple) else result
    ds = hist.simulation_output_to_xr() if hasattr(hist, "simulation_output_to_xr") \
        else hist
    dt = ds.dataset_by_name if hasattr(ds, "dataset_by_name") else None
    prof = dt["profiles"] if dt is not None else ds["profiles"]
    out = {}
    for name in ["T_e", "T_i", "n_e", "q"]:
        key = name if name in prof else name.replace("_", "")
        if key in prof:
            out[name] = np.asarray(prof[key].values, dtype=np.float32)
    return out


def main() -> None:
    mod = importlib.import_module("torax.examples.basic_config")
    import torax

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    scenarios = [sample(rng) for _ in range(N)]
    (OUT / "manifest.json").write_text(json.dumps(
        {"seed": SEED, "space": SPACE, "n": N, "scenarios": scenarios}, indent=1))

    t_all = time.perf_counter()
    done = 0
    for i, s in enumerate(scenarios):
        f = OUT / f"discharge_{i:05d}.npz"
        if f.exists():
            done += 1
            continue
        t0 = time.perf_counter()
        try:
            cfg = build_config(mod.CONFIG, s)
            result = torax.run_simulation(cfg)
            prof = profiles_from(result)
            if not prof:
                raise RuntimeError("no profiles extracted")
            np.savez_compressed(f, actuators=np.array(list(s.values()),
                                                      dtype=np.float32), **prof)
            done += 1
            print(f"[{i+1}/{N}] ok in {time.perf_counter()-t0:.1f}s "
                  f"shapes={ {k: v.shape for k, v in prof.items()} }"
                  if i < 3 else f"[{i+1}/{N}] ok {time.perf_counter()-t0:.1f}s",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 — log and continue generation
            print(f"[{i+1}/{N}] FAILED: {type(exc).__name__}: {exc}", flush=True)
    print(f"GENERATION_DONE {done}/{N} in {time.perf_counter()-t_all:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
