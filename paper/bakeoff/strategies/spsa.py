"""Arm (d): SPSA — hardware-standard control (Spall gains, Qiskit-style
calibration).

g_hat = [L(th + c_k D) - L(th - c_k D)] / (2 c_k) * D,  D ~ Rademacher +/-1
th <- th - a_k g_hat;  a_k = a/(A+k+1)^0.602, c_k = c/(k+1)^0.101,
A = 10% of max iters. Calibration: probe pairs at c estimate mean |g_hat|,
then a is set so the FIRST step magnitude ~= spsa_target_step (design brief
says 2pi/10 for pure angle params; our vector mixes positions [sim units] and
detunings [rad], so 0.1 — deviation documented in the final report).

Cost: 2 fwd per step (dimension-independent — the point of the control);
calibration probes are charged. Adapts geometry + per-site detunings +
router (NOT the detuning conditioner matrices: the brief's SPSA vector is the
~200-1000-dim structure vector).

Also exports `spsa_probe`, reused by e-prop's sign-check kill rule.
"""

import torch

from _bootstrap import add_paths

add_paths()

from strategies.base import EvalMeter, Strategy, assign_params  # noqa: E402
from strategies.base import flatten_params  # noqa: E402
from substrate import repair_model_geometry, set_adaptable  # noqa: E402
from substrate import structure_params  # noqa: E402


def rademacher(n: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randint(0, 2, (n,), generator=gen).float() * 2.0 - 1.0


def spsa_probe(params, loss_fn, c: float, gen: torch.Generator
               ) -> tuple[torch.Tensor, float, float]:
    """Two-eval SPSA gradient estimate; restores params. Caller charges
    2 fwd units."""
    vec = flatten_params(params)
    delta = rademacher(vec.numel(), gen)
    assign_params(params, vec + c * delta)
    lp = float(loss_fn())
    assign_params(params, vec - c * delta)
    lm = float(loss_fn())
    assign_params(params, vec)
    return (lp - lm) / (2.0 * c) * delta, lp, lm


class SPSAStrategy(Strategy):
    name = "spsa"
    seed_offset = 40
    groups = ("geometry", "detuning", "router")

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.a_gain = float("nan")
        self.diverged = False
        self.init_loss = float("nan")
        self.last_loss = float("nan")

    def adapt(self, segment, meter: EvalMeter, probe) -> None:
        cfg = self.cfg
        set_adaptable(self.model, ())  # everything frozen; no-grad updates
        table = structure_params(self.model)
        params = [p for g in self.groups for p in table[g]]
        it = segment.batches(cfg.batch, 1)
        state = {}

        def loss_fn() -> float:
            with torch.no_grad():
                pred = self.model(state["x"])
                return (pred - state["y"]).square().mean().item()

        def next_batch() -> None:
            x, ys = next(it)
            self.samples_seen += x.shape[0]
            state["x"] = x.to(self.device)
            state["y"] = ys[0].to(self.device)

        # ---- calibration (charged): mean |g| at fixed c on one held batch
        next_batch()
        mags, first_lp = [], None
        for _ in range(cfg.spsa_cal_probes):
            if not meter.try_charge(fwd=2):
                break
            g, lp, _ = spsa_probe(params, loss_fn, cfg.spsa_c, self.gen)
            mags.append(g.abs().mean().item())
            first_lp = lp if first_lp is None else first_lp
        mean_mag = max(sum(mags) / max(len(mags), 1), 1e-9)
        max_iters = max(1, int(meter.remaining // meter.cost(fwd=2)))
        big_a = 0.1 * max_iters
        self.a_gain = (cfg.spsa_target_step / mean_mag
                       * (big_a + 1.0) ** 0.602)
        self.init_loss = first_lp if first_lp is not None else float("nan")

        k = 0
        while meter.try_charge(fwd=2):
            next_batch()
            a_k = self.a_gain / (big_a + k + 1.0) ** 0.602
            c_k = cfg.spsa_c / (k + 1.0) ** 0.101
            g, lp, lm = spsa_probe(params, loss_fn, c_k, self.gen)
            vec = flatten_params(params) - a_k * g
            assign_params(params, vec)
            repair_model_geometry(self.model)
            self.last_loss = 0.5 * (lp + lm)
            bad = (not torch.isfinite(torch.tensor(self.last_loss))
                   or (self.init_loss == self.init_loss
                       and self.last_loss > 20.0 * max(self.init_loss, 1e-9)))
            if bad:
                self.diverged = True
                break
            k += 1
            probe(meter.used, self.samples_seen)

    def diagnostics(self) -> dict:
        return {"a_gain": self.a_gain, "calibration_diverged": self.diverged,
                "init_loss": self.init_loss, "last_loss": self.last_loss}
