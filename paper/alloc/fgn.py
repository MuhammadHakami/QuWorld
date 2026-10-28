"""Noise-conditioned input-dependent configuration (doc 35 §17), after FGN / WeatherNext 3 (arXiv 2609.03582).

A code xi ~ N(0, I_ZD) per ensemble member conditions only the co-processor's configuration (FiLM of the features
entering the core, the per-atom offset network's hidden layer, the per-input tether logits); every path is zero-init.
Scores are exact in the orthonormal band-wise POD coefficient space of the ADR-C caches:
    ||X_m - y|| per field = sqrt(E_new,m / D)          (the per-field VRMSE of member m)
    ||X_m - X_m'|| per field = ||c_m - c_m'|| / sqrt(D)
fair energy score = mean_m ||X_m - y|| - 1 / (2 M (M-1)) sum_{m != m'} ||X_m - X_m'||, averaged over fields.
"""
from __future__ import annotations

import torch
import torch.nn as nn

ZD = 32


class NoiseFiLM(nn.Module):
    """x * (1 + gamma(xi)) + beta(xi), broadcast over atoms; zero-init, so xi has no effect at step 0."""

    def __init__(self, f: int):
        super().__init__()
        self.lin = nn.Linear(ZD, 2 * f); nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.lin(xi)[:, None].chunk(2, -1)
        return x * (1 + gamma) + beta


def zero_linear(fin: int, fout: int) -> nn.Linear:
    lin = nn.Linear(fin, fout); nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
    return lin


def _scale(d, rows, ref: bool) -> torch.Tensor:
    """Normaliser: the window's own target variance D(y) (VRMSE-style; observation-weighted, NOT proper for the energy
    score) or, with ref, a fixed per-field TRAIN scale d.Dref (independent of y: the energy score stays proper)."""
    return d.Dref.expand(len(rows), -1) if ref else d.D[rows]


def field_err(d, rows, corr, alpha, eps: float = 0.0, ref: bool = False) -> torch.Tensor:
    """Per-window, per-field error sqrt(E_new / D), shape (B, C)."""
    a, E = d.resid(rows, alpha); c = corr.double()
    gain = (a.square() - (a - c).square()).reshape(len(rows), d.C, d.M).sum(-1)
    return ((E - gain).clamp_min(0) / _scale(d, rows, ref) + eps).sqrt()


def field_dist(d, rows, c1, c2, eps: float = 0.0, ref: bool = False) -> torch.Tensor:
    """Per-window, per-field distance between two members, shape (B, C)."""
    diff = (c1 - c2).double().reshape(len(rows), d.C, d.M).square().sum(-1)
    return (diff / _scale(d, rows, ref) + eps).sqrt()


def fair_es(d, rows, corrs, alpha, eps: float = 0.0, ref: bool = False) -> torch.Tensor:
    """Fair energy score per window (mean over fields) for members `corrs` (list of (B, C*M)); ref: proper scale."""
    M = len(corrs)
    e = torch.stack([field_err(d, rows, c, alpha, eps, ref) for c in corrs]).mean(0)
    if M == 1:
        return e.mean(1)
    pair = sum(field_dist(d, rows, corrs[i], corrs[j], eps, ref) for i in range(M) for j in range(i + 1, M))
    return (e - pair / (M * (M - 1))).mean(1)


@torch.no_grad()
def evaluate_prob(d, model, route, al, M: int, seed: int, tau: float = 1.0) -> dict:
    """TEST/VALID probabilistic report with M fixed noise draws per window; tau scales members about their mean."""
    model.eval(); gen = torch.Generator(device=d.a.device).manual_seed(seed)
    es, esp, vmean, vmem, spread2, skill2 = [], [], [], [], [], []
    for i in range(0, d.N, 64):
        rows = torch.arange(i, min(i + 64, d.N), device=d.a.device); x, s = d.phi(rows); alpha = al(d, rows)
        corrs = [model(x, s, route, torch.randn(len(rows), ZD, device=d.a.device, generator=gen))[0] for _ in range(M)]
        if tau != 1.0:
            mu = torch.stack(corrs).mean(0); corrs = [mu + tau * (c - mu) for c in corrs]
        es.append(fair_es(d, rows, corrs, alpha))
        if getattr(d, "Dref", None) is not None:                          # doc 35 §30: proper ES (fixed TRAIN scale)
            esp.append(fair_es(d, rows, corrs, alpha, ref=True))
        cbar = torch.stack(corrs).mean(0)
        eb = field_err(d, rows, cbar, alpha); vmean.append(eb.mean(1)); skill2.append(eb.square().mean(1))
        vmem.append(field_err(d, rows, corrs[0], alpha).mean(1))
        dev = torch.stack([field_dist(d, rows, c, cbar).square() for c in corrs]).sum(0) / (M - 1)
        spread2.append(dev.mean(1))
    model.train()
    cat = lambda t: torch.cat(t).double()
    ssr = (cat(spread2).mean().sqrt() / cat(skill2).mean().sqrt() * ((M + 1) / M) ** 0.5).item()
    return {"M": M, "es": cat(es).mean().item(), "es_proper": cat(esp).mean().item() if esp else None,
            "vrmse_mean": cat(vmean).mean().item(),
            "vrmse_member": cat(vmem).mean().item(), "ssr": ssr}


@torch.no_grad()
def mode_rms(d, model, route, al) -> torch.Tensor:
    """Per-mode RMS of the residual left after the deterministic correction, over d (TRAIN); shape (C*M,)."""
    model.eval(); acc = torch.zeros(d.C * d.M, device=d.a.device, dtype=torch.float64)
    for i in range(0, d.N, 64):
        rows = torch.arange(i, min(i + 64, d.N), device=d.a.device); x, s = d.phi(rows)
        a, _ = d.resid(rows, al(d, rows)); c = model(x, s, route)[0].double()
        acc += (a - c).square().sum(0)
    model.train()
    return (acc / d.N).sqrt()


def xscale(d, rows) -> torch.Tensor:
    """Causal per-window, per-mode scale: std of the last input frame x_t of the mode's field, shape (B, C*M)."""
    sx = d.g[rows, 5 * d.C:6 * d.C].float().abs() + 1e-4                   # g block: [mean, std] of x_t per field
    return sx.repeat_interleave(d.M, 1)


@torch.no_grad()
def mode_rms_rel(d, model, route, al) -> torch.Tensor:
    """Per-mode RMS over d of the post-correction residual divided by the causal window scale, shape (C*M,)."""
    model.eval(); acc = torch.zeros(d.C * d.M, device=d.a.device, dtype=torch.float64)
    for i in range(0, d.N, 64):
        rows = torch.arange(i, min(i + 64, d.N), device=d.a.device); x, s = d.phi(rows)
        a, _ = d.resid(rows, al(d, rows)); c = model(x, s, route)[0].double()
        acc += ((a - c) / xscale(d, rows).double()).square().sum(0)
    model.train()
    return (acc / d.N).sqrt()


@torch.no_grad()
def evaluate_dressed(d, model, route, al, sigma, scale: float, M: int, seed: int, routed: bool, rel: bool = False,
                     ref: bool = False) -> float:
    """Fair energy score of the deterministic correction dressed with Gaussian noise (routed support or all modes)."""
    model.eval(); gen = torch.Generator(device=d.a.device).manual_seed(seed); es = []
    for i in range(0, d.N, 64):
        rows = torch.arange(i, min(i + 64, d.N), device=d.a.device); x, s = d.phi(rows); alpha = al(d, rows)
        c, aux = model(x, s, route)
        sig = sigma[None] * xscale(d, rows) if rel else sigma[None].expand(len(rows), -1)       # (B, C*M)
        members = []
        for _ in range(M):
            if routed:
                idx = aux["idx"]; eps = torch.randn(idx.shape, device=c.device, generator=gen)
                members.append(c.scatter_add(1, idx, (scale * sig.gather(1, idx) * eps).to(c.dtype)))
            else:
                eps = torch.randn(c.shape, device=c.device, generator=gen)
                members.append(c + (scale * sig * eps).to(c.dtype))
        es.append(fair_es(d, rows, members, alpha, ref=ref))
    model.train()
    return torch.cat(es).double().mean().item()


def dress_report(tr, va, te, model, route, al, sigma_from=None, rel: bool = False) -> dict:
    """§17 control: VALID-selected dressing scale for routed-support and all-mode noise; TEST scored once. sigma_from:
    the split whose post-correction residual RMS sets the per-mode shape (TRAIN by default; VALID for variant (c))."""
    if rel:                                                                  # variant (e): window-scaled, causal
        sigma = mode_rms_rel(tr, model, route, al).float(); scales = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
    else:
        sigma = mode_rms(sigma_from if sigma_from is not None else tr, model, route, al).float(); scales = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    out = {}
    for routed in (True, False):
        grid = {s: evaluate_dressed(va, model, route, al, sigma, s, 8, 21, routed, rel) for s in scales}
        best = min(grid, key=grid.get)
        out["routed" if routed else "all"] = {"valid_grid": grid, "scale": best,
                                               "test_es": evaluate_dressed(te, model, route, al, sigma, best, 16, 12345, routed, rel)}
        if getattr(te, "Dref", None) is not None:                            # doc 35 §30: selected and scored properly
            gp = {s: evaluate_dressed(va, model, route, al, sigma, s, 8, 21, routed, rel, ref=True) for s in scales}
            bp = min(gp, key=gp.get)
            out["routed" if routed else "all"].update(scale_proper=bp, test_es_proper=evaluate_dressed(
                te, model, route, al, sigma, bp, 16, 12345, routed, rel, ref=True))
    return out
