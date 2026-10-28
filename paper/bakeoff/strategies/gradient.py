"""Arm (a): gradient oracle — Adam on L_push (stop-grad pushforward).

Adjoint reference / oracle ceiling: full backprop through the statevector
sim, which analog hardware cannot execute (no parameter-shift for
multi-eigenvalue generators, no mid-evolution gate insertion). Update = Adam
(cosine lr, geometry on its own lr) on L_push: autoregress k steps with
stop-grad inner steps, backprop the LAST step only (Brandstetter 2202.03376).

Cost/step: k fwd (k-1 no-grad + 1 grad) + 1 bwd = k + 2 units.
Kill rule fields: nan_events (>=2 => flag), loses-to-RLS check in runner.
"""

import torch

from _bootstrap import add_paths

add_paths()

from strategies.base import EvalMeter, Strategy  # noqa: E402
from substrate import repair_model_geometry, set_adaptable  # noqa: E402
from substrate import structure_params  # noqa: E402


class GradientStrategy(Strategy):
    name = "gradient"
    seed_offset = 10
    groups = ("geometry", "detuning", "conditioner", "router")

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.nan_events = 0
        self.last_loss = float("nan")

    def adapt(self, segment, meter: EvalMeter, probe) -> None:
        cfg = self.cfg
        k = cfg.push_k
        params = set_adaptable(self.model, self.groups)
        geo = set(id(p) for p in structure_params(self.model)["geometry"])
        opt = torch.optim.Adam([
            {"params": [p for p in params if id(p) in geo],
             "lr": cfg.lr_geo},
            {"params": [p for p in params if id(p) not in geo],
             "lr": cfg.lr_other}])
        est_steps = max(1, int(meter.remaining // meter.cost(fwd=k, bwd=1)))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=est_steps)
        it = segment.batches(cfg.batch, k)
        while meter.try_charge(fwd=k, bwd=1):
            x, ys = next(it)
            self.samples_seen += x.shape[0]
            x = x.to(self.device)
            pred = x
            with torch.no_grad():
                for y in ys[:-1]:
                    pred = self.model(pred)
            pred = self.model(pred)
            loss = (pred - ys[-1].to(self.device)).square().mean()
            if not torch.isfinite(loss):
                self.nan_events += 1
                opt.zero_grad(set_to_none=True)
                probe(meter.used, self.samples_seen)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            repair_model_geometry(self.model)
            self.last_loss = loss.item()
            probe(meter.used, self.samples_seen)
        for p in params:
            p.requires_grad_(False)

    def diagnostics(self) -> dict:
        return {"nan_events": self.nan_events, "last_loss": self.last_loss}
