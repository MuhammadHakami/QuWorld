"""Strategy interface, eval-unit budget meter, parameter-vector utilities."""

import dataclasses
import pathlib

import torch

from _bootstrap import add_paths

add_paths()


class EvalMeter:
    """Eval-unit accounting: 1 unit = one forward pass of the full model on
    one standard batch; a backward pass costs 2 (design brief: fwd=1, bwd=2).
    Strategies must `try_charge` BEFORE spending compute."""

    def __init__(self, budget: float, fwd_cost: float = 1.0,
                 bwd_cost: float = 2.0):
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.budget = float(budget)
        self.fwd_cost, self.bwd_cost = fwd_cost, bwd_cost
        self._used = 0.0
        self.n_fwd = 0
        self.n_bwd = 0

    def cost(self, fwd: int = 0, bwd: int = 0) -> float:
        return fwd * self.fwd_cost + bwd * self.bwd_cost

    def try_charge(self, fwd: int = 0, bwd: int = 0) -> bool:
        c = self.cost(fwd, bwd)
        if self._used + c > self.budget + 1e-9:
            return False
        self._used += c
        self.n_fwd += fwd
        self.n_bwd += bwd
        return True

    @property
    def used(self) -> float:
        return self._used

    @property
    def remaining(self) -> float:
        return self.budget - self._used

    @property
    def exhausted(self) -> bool:
        return self.remaining < self.fwd_cost


@dataclasses.dataclass(frozen=True)
class StrategyConfig:
    batch: int = 16
    # gradient
    push_k: int = 2          # pushforward depth (stop-grad inner steps)
    lr_geo: float = 0.05
    lr_other: float = 0.01
    # spsa
    spsa_c: float = 0.05
    spsa_cal_probes: int = 25
    spsa_target_step: float = 0.1
    # eprop
    lam_tr: float = 0.9      # eligibility trace decay
    eta_det: float = 0.02
    eta_geo: float = 0.005
    lp_gamma: float = 0.9    # F_gamma low-pass on the three-factor product
    check_every: int = 50    # SPSA sign-probe cadence (steps)
    # evolution
    n_emitters: int = 3
    emitter_batch: int = 12
    op_batch: int = 6        # operator-portfolio mutants per generation
    sigma0: float = 0.19     # 0.05 * FOV (38 um) in sim units
    # rls
    rls_alpha: float = 10.0  # FORCE P(0) = I / alpha
    rls_pixels: int = 256    # pixels sampled per RLS step


FULL_STRATEGY = StrategyConfig()
SMOKE_STRATEGY = StrategyConfig(batch=8, spsa_cal_probes=5, check_every=20,
                                emitter_batch=4, op_batch=4, rls_pixels=128)


def flatten_params(params) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


def assign_params(params, vec: torch.Tensor) -> None:
    with torch.no_grad():
        i = 0
        for p in params:
            n = p.numel()
            p.copy_(vec[i:i + n].view_as(p))
            i += n
    if i != vec.numel():
        raise ValueError(f"vector length {vec.numel()} != params size {i}")


class Strategy:
    """One bake-off arm. Subclasses set `name`, `seed_offset`, `groups` and
    implement adapt(segment, meter, probe)."""

    name = "base"
    seed_offset = 0
    groups: tuple = ()

    def __init__(self, model, cfg: StrategyConfig, seed: int,
                 device: str = "cpu", outdir: str | None = None):
        self.model = model
        self.cfg = cfg
        self.seed = seed
        self.device = device
        self.outdir = pathlib.Path(outdir) if outdir else None
        self.gen = torch.Generator().manual_seed(7919 * seed
                                                 + self.seed_offset)
        self.samples_seen = 0

    def adapt(self, segment, meter: EvalMeter, probe) -> None:
        raise NotImplementedError

    def diagnostics(self) -> dict:
        return {}


class FrozenStrategy(Strategy):
    """No-adapt arm — the forgetting floor."""

    name = "frozen"
    seed_offset = 60

    def adapt(self, segment, meter, probe) -> None:
        probe(meter.used, self.samples_seen)
