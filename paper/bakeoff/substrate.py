"""Shared kmix30 bank substrate for the structure-search bake-off (AG-2c).

Builds the same architecture as `model.build_model('kmix30', in_ch=4)` —
identical submodule names (lift/pre/mid/post/proj), so a GPU-sprint checkpoint
of that model loads 1:1 via `load_checkpoint` — but with configurable
(width, modes, n_qubits, registers, k_bank, q_layers) so the CPU smoke scale
(n=6 qubits, 2 layers) uses the same code path.

Adaptation surface (everything else frozen):
  geometry     — per-register trainable atom positions (QuantumBlock.pos)
  detuning     — per-site static detuning offsets (QuantumBlock.base)
  conditioner  — detuning conditioner linear maps (QuantumBlock.cond)
  router       — KMixtureRegister router logits (weight + bias)

Sealed constraint evaluator (hard-enforced, never penalties): 5 um min
spacing / 38 um FOV expressed in simulation units where 1 unit = 10 um
(qlayer clamps pair distances at MIN_DIST=0.5), i.e. FOV_RADIUS=3.8.

Routing is patched to deterministic argmax with a straight-through softmax
estimator (the class docstring's "argmax register at inference" deploy mode):
the stock Gumbel sampling would make SPSA's +/- evals and evolution fitness
evals stochastic in routing, breaking eval parity.
"""

import dataclasses
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from _bootstrap import add_paths

add_paths()

from model import Bottleneck, FNOBlock  # noqa: E402
from qlayer import KMixtureRegister  # noqa: E402

MIN_DIST = 0.5    # simulation units (= 5 um); matches qlayer distance clamp
FOV_RADIUS = 3.8  # simulation units (= 38 um FOV radius)
STRUCTURE_GROUPS = ("geometry", "detuning", "conditioner", "router")


@dataclasses.dataclass(frozen=True)
class SubstrateConfig:
    in_ch: int = 4
    width: int = 24
    modes: int = 12
    n_qubits: int = 10
    registers: int = 3
    k_bank: int = 4
    q_layers: int = 4
    gate_init: float = 0.1  # mid.alpha init for RANDOM substrates (zero = dead gate)


FULL = SubstrateConfig()
SMOKE = SubstrateConfig(width=12, modes=6, n_qubits=6, registers=3, k_bank=4,
                        q_layers=2)


class BakeoffFNO(nn.Module):
    """Mirror of model.build_model's inner FNO (state-dict compatible)."""

    def __init__(self, cfg: SubstrateConfig):
        super().__init__()
        w, m = cfg.width, cfg.modes
        self.lift = nn.Conv2d(cfg.in_ch, w, 1)
        self.pre = nn.Sequential(FNOBlock(w, m, m), FNOBlock(w, m, m))
        core = KMixtureRegister(w, cfg.n_qubits, cfg.registers, cfg.k_bank,
                                cfg.q_layers)
        self.mid = Bottleneck(w, core, cfg.registers * cfg.n_qubits)
        self.post = nn.Sequential(FNOBlock(w, m, m), FNOBlock(w, m, m))
        self.proj = nn.Sequential(nn.Conv2d(w, 64, 1), nn.GELU(),
                                  nn.Conv2d(64, cfg.in_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre(self.lift(x))
        z = self.mid(z)
        return self.proj(self.post(z))


def patch_deterministic_routing(kmix: KMixtureRegister) -> None:
    """Instance-level forward: argmax routing + straight-through softmax."""

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.router(h)
        p = torch.softmax(logits, dim=-1)
        idx = logits.argmax(dim=-1)
        hard = F.one_hot(idx, len(self.banks)).to(h.dtype)
        w = hard + p - p.detach()  # deterministic ST estimator for the router
        out = torch.zeros(
            h.shape[0], self.banks[0].regs[0].n * len(self.banks[0].regs),
            device=h.device, dtype=h.dtype)
        for k, bank in enumerate(self.banks):
            m = idx == k
            if m.any():
                out[m] = bank(h[m])
        sel = (w * hard).sum(-1)
        return out * sel.unsqueeze(-1)

    kmix.forward = types.MethodType(forward, kmix)


def _extract_state_dict(obj) -> dict:
    if isinstance(obj, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError(
        "unrecognized checkpoint container; expected a raw state dict or a "
        "dict with 'model_state_dict'/'model'/'state_dict'")


def load_checkpoint(model: nn.Module, path: str) -> None:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(_extract_state_dict(obj))
    if missing or unexpected:  # load_state_dict(strict=True) raises anyway
        raise ValueError(f"checkpoint mismatch: {missing=} {unexpected=}")


def save_checkpoint(model: nn.Module, path: str, **meta) -> None:
    torch.save({"model_state_dict": model.state_dict(), **meta}, path)


def build_substrate(cfg: SubstrateConfig, seed: int, checkpoint: str | None
                    = None, device: str = "cpu") -> nn.Module:
    """Random-init (seeded) or checkpoint-loaded substrate, fully frozen,
    deterministic routing patched in, eval mode."""
    torch.manual_seed(seed)
    model = BakeoffFNO(cfg)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
    else:
        # zero-init gate is correct for from-scratch training but makes the
        # quantum path a no-op; random-init substrates open it slightly so
        # structure adaptation has finite signal (trained ckpts keep theirs).
        with torch.no_grad():
            model.mid.alpha.fill_(cfg.gate_init)
    patch_deterministic_routing(model.mid.core)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def structure_params(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    kmix = model.mid.core
    geo, det, cond = [], [], []
    for bank in kmix.banks:
        for reg in bank.regs:
            geo.append(reg.pos)
            det.append(reg.base)
            cond.extend([reg.cond.weight, reg.cond.bias])
    return {"geometry": geo, "detuning": det, "conditioner": cond,
            "router": [kmix.router.weight, kmix.router.bias]}


def set_adaptable(model: nn.Module, groups) -> list[nn.Parameter]:
    """Freeze everything, then enable requires_grad for the given groups.
    Returns the enabled parameter list (stable order)."""
    for p in model.parameters():
        p.requires_grad_(False)
    table = structure_params(model)
    out: list[nn.Parameter] = []
    for g in groups:
        if g not in table:
            raise ValueError(f"unknown structure group {g!r}; "
                             f"valid: {STRUCTURE_GROUPS}")
        for p in table[g]:
            p.requires_grad_(True)
            out.append(p)
    return out


# ---------------------------------------------------------------- constraints

def repair_positions(pos: torch.Tensor, min_dist: float = MIN_DIST,
                     fov: float = FOV_RADIUS, iters: int = 3) -> torch.Tensor:
    """Sealed-evaluator repair: returns a NEW tensor satisfying the min-dist
    floor and FOV sphere (centered on the register centroid). Deterministic."""
    p = pos.detach().clone()
    n = p.shape[0]
    for _ in range(iters):
        d = torch.cdist(p, p)
        coincident = (d < 1e-6) & ~torch.eye(n, dtype=torch.bool,
                                             device=p.device)
        if coincident.any():  # deterministic split of stacked atoms
            jit = torch.linspace(-1, 1, n, device=p.device).unsqueeze(-1)
            p = p + 1e-3 * jit * torch.ones_like(p)
            d = torch.cdist(p, p)
        for i in range(n):
            for j in range(i + 1, n):
                dist = d[i, j].item()
                if dist < min_dist:
                    direction = (p[j] - p[i]) / max(dist, 1e-6)
                    shift = 0.525 * (min_dist - dist)  # symmetric + 5% margin
                    p[i] = p[i] - direction * shift
                    p[j] = p[j] + direction * shift
        center = p.mean(0)
        r = p - center
        nrm = r.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        p = center + r * (fov / nrm).clamp(max=1.0)
    return p


def repair_model_geometry(model: nn.Module) -> None:
    with torch.no_grad():
        for p in structure_params(model)["geometry"]:
            p.copy_(repair_positions(p))


def canonicalize(pos: torch.Tensor) -> torch.Tensor:
    """Gauge canonicalization (2D): atom0 at origin, atom1 on +x, reflect so
    atom2 has y>=0. Used for genome dedup/descriptors, not live params."""
    p = pos.detach().clone() - pos.detach()[0]
    if p.shape[0] > 1:
        ang = torch.atan2(p[1, 1], p[1, 0])
        c, s = torch.cos(-ang), torch.sin(-ang)
        rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        p = p @ rot.T
    if p.shape[0] > 2 and p[2, 1] < 0:
        p = p * torch.tensor([1.0, -1.0], device=p.device)
    return p
