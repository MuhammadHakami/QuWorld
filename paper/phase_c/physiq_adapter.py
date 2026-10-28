"""P1 adapter loading for the Physics-IQ quantum / lora arms (P2 hook).

Native layout = what paper/phase_c/p1_train.py writes (train_unit :301-304,
copied to runs/phase_c/p1_<arm>_s<seed>.pt by main :437-439):
    {"model_state_dict", "arm": "mr30ryd"|"lora"|"mlp30", "seed", "lr",
     "step", "valid": {"rollout", "onestep", ...}, "params", "k_steps",
     "core_out"}
Module = p1_train.ChunkAdapter(arm): forward(x_pred, prev) -> x_pred +
alpha * r, where prev is the adapter's OWN previous adapted chunk (zeros at
chunk 0; p1_train.rollout chains detached outputs). StatefulAdapter replays
that chain at P2 time: reset(B) once per video (batch), then one call per
(chunk, sample) in the order the sampling loop makes them. Input x0 is the
(16, 3, 60, 104) fp32 raw chunk latent — the tensor teacher_cache.py cached
as x0_pred — hence the 60x104 geometry requirement.

Best seed: p1_results.json["arms"][<arm>]["per_seed"][<seed>]["valid_rollout"]
(lowest wins; p1_train.aggregate). Legacy/ad-hoc shapes ({<arm>: {"best_seed"}}
or per-seed dicts with a metric key containing rollout/vrmse/rmse/loss) are
accepted; anything else is reported, not guessed (--adapter-ckpt overrides).

Fallback checkpoint layouts for ad-hoc tests: a whole nn.Module, or
{"builder": "<module>:<callable>", "config", "state_dict"}; those are called
statelessly as module(x0[None])[0].
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PHASE_C_RUNS = REPO / "runs" / "phase_c"
P1_RESULTS = PHASE_C_RUNS / "p1_results.json"
ARM_PREFIX = {"quantum": "mr30ryd", "lora": "lora"}  # p1_<prefix>_s<seed>.pt
METRIC_PRIORITY = ("valid_rollout", "rollout_vrmse", "rollout", "vrmse", "rmse", "loss")
LATENT_SHAPE = (16, 3, 60, 104)
P1_KEYS = ("model_state_dict", "arm")


class AdapterError(RuntimeError):
    """Checkpoint / results file does not satisfy the documented contract."""


# --------------------------------------------------------------------------
# checkpoint selection
# --------------------------------------------------------------------------

def ckpt_path(arm: str, seed: int) -> Path:
    return PHASE_C_RUNS / f"p1_{ARM_PREFIX[arm]}_s{seed}.pt"


def available_ckpts(arm: str) -> dict[int, Path]:
    pat = re.compile(rf"^p1_{re.escape(ARM_PREFIX[arm])}_s(\d+)\.pt$")
    found = {}
    for p in PHASE_C_RUNS.glob(f"p1_{ARM_PREFIX[arm]}_s*.pt"):
        m = pat.match(p.name)
        if m:
            found[int(m.group(1))] = p
    return found


def _metric_value(entry) -> float | None:
    """Lowest-is-best metric from a per-seed entry (exact key first, then substring)."""
    if isinstance(entry, (int, float)):
        return float(entry)
    if not isinstance(entry, dict):
        return None
    for want in METRIC_PRIORITY:
        if isinstance(entry.get(want), (int, float)):
            return float(entry[want])
    for want in METRIC_PRIORITY:
        for k, v in entry.items():
            if want in k.lower() and isinstance(v, (int, float)):
                return float(v)
    return None


def best_seed_from_results(results: dict, arm: str) -> tuple[int, str]:
    """(seed, reason) per the contract; raises AdapterError otherwise."""
    prefix = ARM_PREFIX[arm]
    arms = results.get("arms") if isinstance(results.get("arms"), dict) else {}
    section = arms.get(prefix) or results.get(prefix) or results.get(arm)
    if section is None:
        raise AdapterError(f"p1_results.json has no section for '{prefix}' "
                           f"(arms: {sorted(arms)}, top-level: {sorted(results)[:12]})")
    if isinstance(section, dict) and "best_seed" in section:
        return int(section["best_seed"]), "best_seed field"
    per_seed = None
    if isinstance(section, dict):
        per_seed = section.get("per_seed") or section.get("seeds") or section
    if not isinstance(per_seed, dict):
        raise AdapterError(f"'{prefix}' section is not a per-seed mapping: {type(section).__name__}")
    scored = {int(str(s).lstrip("s")): _metric_value(v) for s, v in per_seed.items()
              if str(s).lstrip("s").isdigit()}
    scored = {s: v for s, v in scored.items() if v is not None}
    if not scored:
        raise AdapterError(f"no seed metrics ({METRIC_PRIORITY}) under '{prefix}': {list(per_seed)[:6]}")
    seed = min(scored, key=scored.get)
    return seed, f"lowest valid rollout VRMSE {scored[seed]:.5g} among seeds {sorted(scored)}"


def select_ckpt(arm: str, override: Path | None = None) -> tuple[Path, str]:
    """Checkpoint path + human-readable reason."""
    if override is not None:
        if not Path(override).exists():
            raise AdapterError(f"--adapter-ckpt not found: {override}")
        return Path(override), "explicit --adapter-ckpt"
    found = available_ckpts(arm)
    if P1_RESULTS.exists():
        seed, why = best_seed_from_results(json.loads(P1_RESULTS.read_text()), arm)
        if seed not in found:
            raise AdapterError(f"best seed {seed} for '{arm}' but {ckpt_path(arm, seed)} is missing "
                               f"(found seeds {sorted(found)})")
        return found[seed], f"seed {seed} ({why})"
    if len(found) == 1:
        (seed, path), = found.items()
        return path, f"only checkpoint present (seed {seed}); no p1_results.json"
    raise AdapterError(f"no p1_results.json and {len(found)} checkpoints for '{arm}': {sorted(found)}")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _import_callable(dotted: str):
    mod, _, name = dotted.partition(":")
    if not mod or not name:
        raise AdapterError(f"builder must be '<module>:<callable>', got {dotted!r}")
    return getattr(importlib.import_module(mod), name)


def _p1_module(obj: dict):
    import p1_train  # paper/phase_c/p1_train.py (adds paper/ to sys.path for model/qlayer)

    if obj["arm"] not in ARM_PREFIX.values() and obj["arm"] != "mlp30":
        raise AdapterError(f"unknown P1 arm {obj['arm']!r}")
    module = p1_train.ChunkAdapter(obj["arm"])
    module.load_state_dict(obj["model_state_dict"], strict=True)
    return module


def load_adapter(path: Path, device: str = "cpu"):
    """Return (module, call) per the layouts in the module docstring."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and all(k in obj for k in P1_KEYS):
        module = _p1_module(obj)
        call = {"stateful_prev": True, "batched": True, "returns": "adapted",
                "meta": {k: obj.get(k) for k in ("arm", "seed", "lr", "step", "params", "k_steps")},
                "valid": obj.get("valid")}
    elif isinstance(obj, torch.nn.Module):
        module, call = obj, {"stateful_prev": False, "batched": True, "returns": "adapted"}
    elif isinstance(obj, dict) and "builder" in obj and "state_dict" in obj:
        module = _import_callable(obj["builder"])(**obj.get("config", {}))
        module.load_state_dict(obj["state_dict"])
        call = {"stateful_prev": False, "batched": True, "returns": "adapted", **obj.get("call", {})}
    else:
        keys = sorted(obj)[:12] if isinstance(obj, dict) else type(obj).__name__
        raise AdapterError(f"{path.name}: unrecognized checkpoint layout ({keys}); see physiq_adapter.py")
    if call["returns"] not in ("adapted", "residual"):
        raise AdapterError(f"call.returns must be 'adapted' or 'residual', got {call['returns']!r}")
    return module.eval().requires_grad_(False).float().to(device), call


class StatefulAdapter:
    """Callable for wrap_final_step_x0: x0 -> adapted x0 (same dtype), replaying
    the P1 prev-chain per sample. Call reset(batch) before every video/batch."""

    def __init__(self, module, call: dict):
        self.module, self.call = module, call
        self.batch, self.calls, self.prev = 1, 0, [None]

    def reset(self, batch_size: int = 1) -> None:
        self.batch, self.calls, self.prev = batch_size, 0, [None] * batch_size

    def __call__(self, x0):
        import torch

        with torch.autocast("cuda", enabled=False):
            x = x0.detach().float()
            if self.call["stateful_prev"]:
                b = self.calls % self.batch
                prev = self.prev[b] if self.prev[b] is not None else torch.zeros_like(x)
                out = self.module(x[None], prev[None])[0]
                self.prev[b] = out.detach()
            else:
                out = self.module(x[None])[0] if self.call["batched"] else self.module(x)
                if self.call["returns"] == "residual":
                    out = x + out
        self.calls += 1
        if tuple(out.shape) != tuple(x0.shape):
            raise AdapterError(f"adapter returned {tuple(out.shape)} for input {tuple(x0.shape)}")
        return out.to(x0.dtype)

    @property
    def info(self) -> dict:
        gate = getattr(self.module, "alpha", None)
        return {"module": type(self.module).__name__,
                "n_params": sum(p.numel() for p in self.module.parameters()),
                "stateful_prev": self.call["stateful_prev"], "meta": self.call.get("meta"),
                "valid": self.call.get("valid"),
                "gate_abs_max": float(gate.detach().abs().max()) if gate is not None else None}


def load_stateful_adapter(path: Path, device: str = "cpu") -> StatefulAdapter:
    module, call = load_adapter(path, device)
    return StatefulAdapter(module, call)


def check_adapter(arm: str, override: Path | None = None, device: str = "cpu") -> dict:
    """Dry run: load, replay two chunks on random latents, report shape/change/gate."""
    import torch

    path, why = select_ckpt(arm, override)
    adapter = load_stateful_adapter(path, device)
    adapter.reset(1)
    torch.manual_seed(0)
    x0, x1 = torch.randn(LATENT_SHAPE, device=device), torch.randn(LATENT_SHAPE, device=device)
    out0, out1 = adapter(x0), adapter(x1)          # chunk 0 (prev = zeros), chunk 1 (prev = out0)
    return {"arm": arm, "ckpt": str(path), "why": why, "out_shape": list(out0.shape),
            "rel_change_chunk0": float((out0 - x0).norm() / x0.norm()),
            "rel_change_chunk1": float((out1 - x1).norm() / x1.norm()), **adapter.info}
