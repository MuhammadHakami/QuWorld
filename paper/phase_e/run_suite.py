"""Phase-E dequantization suite over the TD-fine-tuned quantum checkpoints.

Targets runs/phase_b2/mr30ryd_s{1337,1338,1339}.pt (CNextU-net hybrids with a
3x10-qubit Rydberg MultiRegister neck). Per seed:
  1. sanity: statevector replication vs the checkpoint's own forward
  2. geometry facts (trained atom spacing, blockade energy scales)
  3. 256 real valid-split latents -> Schmidt spectrum / entanglement entropy
     per register (+ product-state control) and cumulative Schmidt weights
  4. valid rollout VRMSE@8 (finetune_td.validate_rollout, THE selection
     metric) for: intact | product-state (zero-blockade, far-geometry) |
     chi-truncation chi in {1,2,4,8,16,32} | finite shots s in {128,1024,8192}

GPU-light by design: batch is the 9-trajectory valid split; allocator capped
at 25% so the concurrent MHD campaign keeps priority. Results are written
incrementally (atomic) to runs/phase_e/results.json — safe to interrupt and
re-run; finished measurements are skipped unless --force.

Usage: uv run paper/phase_e/run_suite.py [--seeds 1337,1338,1339]
           [--configs all|comma-list] [--force]
"""

import argparse
import gc
import json
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase_b"))

import dequant  # noqa: E402

_ARGV = sys.argv[:]
sys.argv = _ARGV[:1]  # finetune_td parses sys.argv at import time
import finetune_td  # noqa: E402
sys.argv = _ARGV
import data_ram  # noqa: E402
from cnext_hybrid import ROOT, load_hybrid_checkpoint  # noqa: E402

OUT = ROOT / "runs" / "phase_e"
RESULTS = OUT / "results.json"
DEV = "cuda"
CHIS = [1, 2, 4, 8, 16, 32]
SHOTS = [128, 1024, 8192]
ROLLOUT_CONFIGS = (["intact", "prod_zero", "prod_far"]
                   + [f"chi{c}" for c in CHIS] + [f"shots{s}" for s in SHOTS]
                   + ["const0", "gate_off"])
N_LATENTS = 256


def wrapper_factory(name: str):
    """Config name -> register wrapper (None for the intact model)."""
    if name == "intact":
        return None
    if name == "prod_zero":
        return lambda qb: dequant.ProductStateReadout(qb, "zero")
    if name == "prod_far":
        return lambda qb: dequant.ProductStateReadout(qb, "far")
    if name.startswith("chi"):
        chi = int(name[3:])
        return lambda qb: dequant.ChiTruncatedReadout(qb, chi)
    if name.startswith("shots"):
        shots = int(name[5:])
        return lambda qb: dequant.ShotReadout(qb, shots)
    if name == "const0":
        return dequant.ConstantReadout
    if name == "gate_off":
        return "gate_off"  # handled specially in eval_rollout
    raise ValueError(f"unknown config {name}")


def load_results() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {"protocol": {
        "checkpoints": "runs/phase_b2/mr30ryd_s{seed}.pt (TD-fine-tuned)",
        "metric": "valid rollout VRMSE@8, finetune_td.validate_rollout "
                  "(k=8, stride_t=8, 9 valid trajectories)",
        "bipartition": "qubits [0..4] | [5..9] of each 10-qubit register",
        "chi_sweep": CHIS, "shots_sweep": SHOTS,
        "far_factor": dequant.FAR_FACTOR, "n_latents": N_LATENTS,
    }, "seeds": {}}


def save_results(res: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=2))
    tmp.replace(RESULTS)


def stable_seed(seed: int, name: str) -> int:
    return seed * 10007 + sum(ord(c) for c in name)


@torch.no_grad()
def geometry_facts(core) -> dict:
    facts = []
    for qb in core.regs:
        eng = qb._blockade_energy()
        eng_far = dequant.blockade_energy(qb, qb.pos * dequant.FAR_FACTOR)
        facts.append({
            "min_pair_dist": round(dequant.min_pair_dist(qb), 4),
            "eng_max": round(eng.max().item(), 4),
            "eng_far_max": float(f"{eng_far.max().item():.3e}"),
            "tau": [round(t, 4) for t in qb.tau.tolist()],
            "omega": [round(o, 4) for o in qb.omega.tolist()],
        })
    return {"registers": facts}


@torch.no_grad()
def entanglement_stats(core, h: torch.Tensor) -> dict:
    stats = {}
    for r, qb in enumerate(core.regs):
        sv = dequant.schmidt_singular_values(qb, h)
        ent = dequant.entropy_bits(sv)
        p = sv.square()
        p = p / p.sum(-1, keepdim=True)
        cum = {f"chi{c}": round(p[:, :c].sum(-1).mean().item(), 6)
               for c in CHIS}
        zero_eng = torch.zeros(2**qb.n, device=qb.pos.device)
        ent0 = dequant.entropy_bits(
            dequant.schmidt_singular_values(qb, h[:64], eng=zero_eng))
        stats[f"reg{r}"] = {
            "entropy_bits": {
                "mean": round(ent.mean().item(), 4),
                "std": round(ent.std().item(), 4),
                "min": round(ent.min().item(), 4),
                "max": round(ent.max().item(), 4),
            },
            "entropy_bits_prod_zero_mean": round(ent0.mean().item(), 6),
            "cum_schmidt_weight": cum,
        }
    return stats


@torch.no_grad()
def readout_deltas(core, h: torch.Tensor, seed: int) -> dict:
    deltas = {}
    for name in ["prod_zero", "prod_far"] + [f"chi{c}" for c in CHIS] + \
                [f"shots{s}" for s in SHOTS]:
        torch.manual_seed(stable_seed(seed, name))
        deltas[name] = round(
            dequant.readout_rms_delta(core, wrapper_factory(name), h), 6)
    return deltas


@torch.no_grad()
def readout_variation(core, h: torch.Tensor, chunk: int = 64) -> dict:
    """How much the exact 30-dim readout actually varies with the input."""
    dev = next(core.parameters()).device
    outs = []
    for i in range(0, h.shape[0], chunk):
        outs.append(core(h[i:i + chunk].to(dev)).cpu())
    z = torch.cat(outs)
    return {
        "mean_abs_z": round(z.abs().mean().item(), 4),
        "input_std_rms": round(z.std(0).square().mean().sqrt().item(), 6),
        "per_qubit_std_max": round(z.std(0).max().item(), 6),
    }


def eval_rollout(model, valid, name: str, seed: int) -> float:
    torch.manual_seed(stable_seed(seed, name))  # fixes finite-shot draws
    t0 = time.perf_counter()
    wrap = wrapper_factory(name)
    if wrap is None:
        v = finetune_td.validate_rollout(model, valid)
    elif wrap == "gate_off":
        with dequant.gate_disabled(model):
            v = finetune_td.validate_rollout(model, valid)
    else:
        with dequant.surrogate_registers(model, wrap):
            v = finetune_td.validate_rollout(model, valid)
    print(f"    rollout {name:10s} vrmse8={v:.6f} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return v


def run_seed(seed: int, valid, configs, res: dict, force: bool) -> None:
    key = str(seed)
    rec = res["seeds"].setdefault(key, {})
    ckpt = ROOT / "runs" / "phase_b2" / f"mr30ryd_s{seed}.pt"
    print(f"[seed {seed}] loading {ckpt}", flush=True)
    model = load_hybrid_checkpoint("mr30ryd", str(ckpt)).to(DEV).eval()
    core = model.neck.bottleneck.core

    worst = dequant.verify_state_replication(core, DEV)
    rec["state_replication_max_diff"] = float(f"{worst:.3e}")
    print(f"    state replication ok (max diff {worst:.2e})", flush=True)

    if force or "geometry" not in rec:
        rec["geometry"] = geometry_facts(core)
        save_results(res)

    need_latents = force or any(k not in rec for k in (
        "entanglement", "readout_rms_delta", "readout_stats"))
    if need_latents:
        h = dequant.capture_latents(model, valid, N_LATENTS, dev=DEV)
        rec["entanglement"] = entanglement_stats(core, h)
        rec["readout_stats"] = readout_variation(core, h)
        save_results(res)
        rec["readout_rms_delta"] = readout_deltas(core, h, seed)
        save_results(res)
        del h
        print("    entanglement + readout deltas done", flush=True)

    roll = rec.setdefault("rollout_vrmse8", {})
    for name in configs:
        if not force and name in roll:
            print(f"    rollout {name:10s} cached "
                  f"({roll[name]:.6f})", flush=True)
            continue
        roll[name] = eval_rollout(model, valid, name, seed)
        save_results(res)
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1337,1338,1339")
    ap.add_argument("--configs", default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    configs = (ROLLOUT_CONFIGS if args.configs == "all"
               else args.configs.split(","))
    for c in configs:
        wrapper_factory(c)  # validate names before any GPU work

    torch.cuda.set_per_process_memory_fraction(0.25)
    _, valid = data_ram.load()
    print(f"valid split {tuple(valid.shape)}; configs={configs}; "
          f"seeds={seeds}", flush=True)

    res = load_results()
    for seed in seeds:
        run_seed(seed, valid, configs, res, args.force)
    save_results(res)
    print(f"done -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
