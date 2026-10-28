"""Input-conditioned quantum configuration on the existing classical interface.

No new classical backbone. A bounded lattice makes pair separation a geometric
constraint rather than a clamp claim. Full device pulse feasibility is not claimed.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from model import Bottleneck, MLPCore, _count
from qlayer import ConditionedGeometryBlock, QuantumBlock


@dataclass(frozen=True)
class RegisterConfig:
    width: int = 10
    registers: int = 3
    layers: int = 4
    geometry: str = "legacy"
    readout: str = "z"
    encoding_rms: float = 0.0
    spacing: float = 1.0
    tau: float = 0.3
    omega: float = 0.5

    def __post_init__(self):
        if not (1 <= self.width <= 10 and 1 <= self.registers and self.width * self.registers <= 30):
            raise ValueError("Require 1..10 qubits/register and at most 30 total")
        if not (1 <= self.layers <= 8):
            raise ValueError("Require 1..8 layers")
        if self.geometry not in ("legacy", "bounded_static", "bounded_dynamic"):
            raise ValueError("Unknown geometry")
        if self.readout not in ("z", "cond2"):
            raise ValueError("Search supports z or bounded-complexity cond2 readout")
        for name in ("encoding_rms", "spacing", "tau", "omega"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"Nonfinite {name}")
        if not 0 <= self.encoding_rms <= 1:
            raise ValueError("Encoding RMS must be in [0,1]")
        if not (0.6 <= self.spacing <= 1.5 and 0.05 <= self.tau <= 1 and 0.1 <= self.omega <= 1):
            raise ValueError("Configuration outside declared search bounds")


class BoundedGeometry(ConditionedGeometryBlock):
    """Fixed lattice plus globally/input-conditioned bounded displacements.

    Each displacement has norm <= sqrt(2)*cap. Therefore every pair is at
    least spacing-2*sqrt(2)*cap > MIN_DIST apart for every input/weight value.
    """
    def __init__(self, d_in, cfg):
        super().__init__(d_in, cfg.width, cfg.layers, dims=2, readout=cfg.readout)
        side = math.ceil(math.sqrt(cfg.width))
        grid = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), -1)
        self.register_buffer("lattice", grid.reshape(-1, 2)[:cfg.width].float() * cfg.spacing)
        self.pos0 = nn.Parameter(torch.zeros(cfg.width, 2))
        self.displacement_cap = 0.95 * (cfg.spacing - self.MIN_DIST) / (2 * math.sqrt(2))
        if cfg.geometry == "bounded_static":
            self.geo = None

    def positions(self, h):
        shift = self.pos0.unsqueeze(0).expand(h.shape[0], -1, -1)
        if self.geo is not None:
            shift = shift + self.geo(h).reshape(h.shape[0], self.n, 2)
        return self.lattice.unsqueeze(0) + self.displacement_cap * torch.tanh(shift)

    def _energies(self, h):
        distances = torch.cdist(self.positions(h), self.positions(h))
        pairs = distances[:, self.iu[0], self.iu[1]]
        return pairs.pow(-6) @ self.pairbits.T


class ConfigurableCore(nn.Module):
    def __init__(self, d_in, cfg):
        super().__init__()
        self.config = cfg
        self.regs = nn.ModuleList([
            QuantumBlock(d_in, cfg.width, cfg.layers, "rydberg", readout=cfg.readout)
            if cfg.geometry == "legacy" else BoundedGeometry(d_in, cfg)
            for _ in range(cfg.registers)
        ])
        with torch.no_grad():
            for reg in self.regs:
                if cfg.encoding_rms:
                    reg.cond.weight.normal_(std=cfg.encoding_rms / math.sqrt(d_in))
                reg.tau.fill_(cfg.tau)
                reg.omega.fill_(cfg.omega)
                if cfg.geometry == "legacy":
                    reg.pos.mul_(cfg.spacing)

    def forward(self, h):
        return torch.cat([reg(h) for reg in self.regs], -1)


def make_adapter(ch, cfg, arm):
    if arm == "mlp":
        # Parameter counting must not change the classical initialization when
        # two quantum configurations have the same matched classical shape.
        with torch.random.fork_rng(devices=[]):
            reference = ConfigurableCore(ch, cfg)
    else:
        reference = ConfigurableCore(ch, cfg)
    output_width = cfg.width * cfg.registers
    target = _count(reference)
    if arm == "quantum":
        core = reference
    elif arm == "mlp":
        core = MLPCore(ch, output_width, target)
    else:
        raise ValueError(arm)
    adapter = Bottleneck(ch, core, output_width)
    return adapter, {"config": asdict(cfg), "core_params": _count(core),
                     "reference_core_params": target, "adapter_params": _count(adapter),
                     "parameter_gap": _count(core) - target}


def build_official(arm, cfg, seed, model_dir):
    from phase_b.cnext_hybrid import NeckWithBottleneck, load_official
    torch.manual_seed(seed)
    net, upstream = load_official(str(model_dir))
    meta = {"upstream_config": upstream, "arm": arm, "seed": seed}
    if arm != "vanilla":
        ch = upstream["init_features"] * 2 ** upstream["stages"]
        # Same adapter RNG for each trial, independent of backbone construction.
        torch.manual_seed(seed)
        adapter, details = make_adapter(ch, cfg, arm)
        net.neck = NeckWithBottleneck(net.neck, adapter)
        meta.update(details)
    meta["total_params"] = _count(net)
    return net, meta
