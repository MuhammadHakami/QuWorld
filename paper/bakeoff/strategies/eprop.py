"""Arm (c): e-prop / three-factor online rule over Hamiltonian parameters.

Factor 1 (measured-data-only eligibility trace, per register):
    e_ij^t = lam_tr * e_ij^{t-1} + C_ij^t,
    C_ij = <Z_i Z_j> - <Z_i><Z_j>   (connected correlator, Kubo surrogate
                                     for d<Z_i>/d(delta_j))
Factor 2 (learning signal): fixed random broadcast vector B (random e-prop,
    Bellec 2020): L_j = sum_i B_i e_ij.
Factor 3: broadcast scalar TD error delta_td from the frozen world model
    (one-step prediction error minus an EMA baseline).

Updates (F_gamma low-pass on the product):
    detunings:  d(delta_j) = -eta_det * delta_td * sum_i B_i e_ij
                (applied to base[:, j, 0] across all layers)
    geometry:   chain through V_ij: dV/dr = -6 C6 / r^7 (C6=1 in sim units);
                G_ij = delta_td * (B_i+B_j)/2 * e_ij treated as dL/dV_ij,
                force on atom i = sum_j G_ij * dV/dr_ij * unit(r_i - r_j),
                then sealed 5um repair + FOV clamp.

Only banks routed on the current batch get trace updates (their correlators
are the ones "measured" this step). Cost: 1 fwd = 1 unit per step; the ZZ
readout re-simulation is NOT charged (hardware measures those observables in
the same shots — zero marginal cost, see qreadout.py). Sign-check: every
check_every steps an SPSA probe (2 fwd, charged) on the detuning vector;
cosine(update, -probe_grad) logged for the kill rule.

Known biases carried per the design brief: equal-time correlator != retarded
susceptibility; router-mediated credit ignored (DFA covers it approximately).
"""

import torch

from _bootstrap import add_paths

add_paths()

from qreadout import moments  # noqa: E402
from strategies.base import EvalMeter, Strategy  # noqa: E402
from strategies.spsa import rademacher  # noqa: E402
from substrate import repair_positions, set_adaptable  # noqa: E402


class EPropStrategy(Strategy):
    name = "eprop"
    seed_offset = 30
    groups = ("geometry", "detuning")

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cosines: list[float] = []
        self.baseline = None
        self._captured_h = {}

    # ---------------------------------------------------------- det helpers
    def _regs(self):
        kmix = self.model.mid.core
        return [(k, r, reg) for k, bank in enumerate(kmix.banks)
                for r, reg in enumerate(bank.regs)]

    def _det_vector(self) -> torch.Tensor:
        return torch.cat([reg.base.detach()[:, :, 0].reshape(-1)
                          for _, _, reg in self._regs()])

    def _det_assign(self, vec: torch.Tensor) -> None:
        with torch.no_grad():
            i = 0
            for _, _, reg in self._regs():
                m = reg.base[:, :, 0].numel()
                reg.base[:, :, 0] = vec[i:i + m].view_as(reg.base[:, :, 0])
                i += m

    def _det_probe(self, loss_fn, c: float) -> torch.Tensor:
        """Two-eval SPSA gradient estimate over the per-site detuning vector
        (base[..., 0] only); restores params. Caller charges 2 fwd."""
        vec = self._det_vector()
        delta = rademacher(vec.numel(), self.gen)
        self._det_assign(vec + c * delta)
        lp = loss_fn()
        self._det_assign(vec - c * delta)
        lm = loss_fn()
        self._det_assign(vec)
        return (lp - lm) / (2.0 * c) * delta

    # -------------------------------------------------------------- adapt
    def adapt(self, segment, meter: EvalMeter, probe) -> None:
        cfg = self.cfg
        set_adaptable(self.model, ())  # no autograd anywhere in this arm
        kmix = self.model.mid.core
        regs = self._regs()
        n = regs[0][2].n
        bvec = {(k, r): torch.randn(n, generator=self.gen) / n ** 0.5
                for k, r, _ in regs}
        trace = {(k, r): torch.zeros(n, n) for k, r, _ in regs}
        lp_det = {(k, r): torch.zeros(n) for k, r, _ in regs}
        lp_geo = {(k, r): torch.zeros(n, reg.pos.shape[1])
                  for k, r, reg in regs}

        hook = kmix.register_forward_pre_hook(
            lambda mod, args: self._captured_h.__setitem__("h", args[0]))
        it = segment.batches(cfg.batch, 1)
        step = 0
        try:
            while meter.try_charge(fwd=1):
                x, ys = next(it)
                self.samples_seen += x.shape[0]
                x, y = x.to(self.device), ys[0].to(self.device)
                with torch.no_grad():
                    pred = self.model(x)
                    td = (pred - y).square().mean().item()
                self.baseline = td if self.baseline is None else self.baseline
                delta_td = td - self.baseline
                self.baseline = 0.9 * self.baseline + 0.1 * td
                h = self._captured_h["h"]
                with torch.no_grad():
                    idx = kmix.router(h).argmax(dim=-1)
                for k, r, reg in regs:
                    mask = idx == k
                    if not mask.any():
                        continue
                    with torch.no_grad():
                        _, _, conn = moments(reg, h[mask])
                    key = (k, r)
                    trace[key] = (cfg.lam_tr * trace[key]
                                  + conn.mean(0))
                    b = bvec[key]
                    lsig = b @ trace[key]                       # (n,)
                    u_det = -cfg.eta_det * delta_td * lsig
                    lp_det[key] = (cfg.lp_gamma * lp_det[key]
                                   + (1 - cfg.lp_gamma) * u_det)
                    g_ij = (delta_td * 0.5 * (b.unsqueeze(1) + b.unsqueeze(0))
                            * trace[key])                       # (n, n)
                    with torch.no_grad():
                        pos = reg.pos.detach()
                        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
                        dist = diff.norm(dim=-1).clamp(min=0.5)
                        dvdr = -6.0 * dist.pow(-7)
                        dldr = (g_ij * dvdr
                                * (1 - torch.eye(n))) / dist    # (n, n)
                        force = (dldr.unsqueeze(-1) * diff).sum(1)
                        u_geo = -cfg.eta_geo * force
                        lp_geo[key] = (cfg.lp_gamma * lp_geo[key]
                                       + (1 - cfg.lp_gamma) * u_geo)
                        reg.base[:, :, 0] += lp_det[key].unsqueeze(0)
                        reg.pos += lp_geo[key]
                        reg.pos.copy_(repair_positions(reg.pos))
                step += 1
                if step % cfg.check_every == 0 and meter.try_charge(fwd=2):
                    self._sign_check(x, y, lp_det, regs)
                probe(meter.used, self.samples_seen)
        finally:
            hook.remove()

    def _sign_check(self, x, y, lp_det, regs) -> None:
        """cosine(applied detuning update, -SPSA probe grad) on this batch."""

        def loss_fn() -> float:
            with torch.no_grad():
                return (self.model(x) - y).square().mean().item()

        g = self._det_probe(loss_fn, self.cfg.spsa_c)
        layers = regs[0][2].layers
        u = torch.cat([lp_det[(k, r)].repeat(layers) for k, r, _ in regs])
        denom = u.norm() * g.norm()
        if denom > 1e-12:
            self.cosines.append(float((u @ (-g)) / denom))

    def diagnostics(self) -> dict:
        n_checks = len(self.cosines)
        mean_cos = (sum(self.cosines) / n_checks) if n_checks else float("nan")
        anti = (sum(1 for c in self.cosines if c < 0) / n_checks
                if n_checks else float("nan"))
        return {"probe_cosine_mean": mean_cos, "anti_frac": anti,
                "n_sign_checks": n_checks}
