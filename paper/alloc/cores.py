"""Cores for the routed co-processor (33-allocation-strategy-study.md).

Every tile core maps a group of n allocated tiles, features f (G, n, F) and tile coordinates (G, n, 4) = (pos_x, pos_y,
pos_z, level), to a per-tile readout o (G, n, 3). The co-processor turns o into a coefficient correction.

  ryd    Rydberg-analog register: atom i sits at the tile (pos / lambda, + level * dz along z), so the interaction graph is
         set per input by the allocation; detunings from f_i (weights shared across atoms); L layers of
         [RZ(detuning) RX(omega) per atom, exp(-i tau sum V_ij n_i n_j)]; readout <Z_i>, <X_i>, <Y_i>.
  ryd2d  the same register with every atom in one plane (z = 0): no cross-scale / signed-geometry couplings.
  gutz   the same parameters propagated as a Gutzwiller product state: interactions kept, entanglement forbidden.
  mlp    DeepSets MLP per group, [f_i, mean_group f] -> tanh(3), hidden width matched to the register's parameter count
         (`scale` multiplies the budget: mlpX4 is the 4x capacity reference).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _bits(n: int) -> torch.Tensor:
    idx = torch.arange(2 ** n)
    return torch.stack([(idx >> (n - 1 - i)) & 1 for i in range(n)], -1).float()


def _rx(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c + 0j, -1j * s], -1), torch.stack([-1j * s, c + 0j], -1)], -2).to(torch.complex64)


def _apply_1q(state: torch.Tensor, g: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Per-sample 2x2 gate g (G, 2, 2) on qubit k (qubit 0 = most significant) of state (G, 2^n)."""
    G = state.shape[0]
    s = state.reshape(G, 2 ** k, 2, 2 ** (n - k - 1))
    return torch.einsum("gij,gajb->gaib", g, s).reshape(G, 2 ** n)


class TileRegister(nn.Module):
    MIN_DIST = 0.5

    def __init__(self, f: int, n: int = 10, layers: int = 4, mode: str = "ryd"):
        super().__init__()
        assert mode in ("ryd", "ryd2d", "gutz", "rydlin")
        self.n, self.layers, self.mode = n, layers, mode
        self.w = nn.Linear(f, layers)
        nn.init.normal_(self.w.weight, std=0.1); nn.init.zeros_(self.w.bias)
        self.base = nn.Parameter(0.02 * torch.randn(layers, n))
        self.omega = nn.Parameter(torch.full((layers,), 0.5)); self.tau = nn.Parameter(torch.full((layers,), 0.3))
        self.log_lam = nn.Parameter(torch.full((3,), math.log(4.0)))   # pixels (or frames) per blockade radius, per axis
        self.dz = nn.Parameter(torch.tensor(1.0))                      # scale-plane separation
        bits = _bits(n)
        iu = torch.triu_indices(n, n, offset=1)
        self.register_buffer("zdiag", 1.0 - 2.0 * bits)
        self.register_buffer("pairbits", bits[:, iu[0]] * bits[:, iu[1]])
        self.register_buffer("iu", iu)
        self.register_buffer("flip", torch.stack([torch.arange(2 ** n) ^ (1 << (n - 1 - i)) for i in range(n)]))

    def positions(self, coords: torch.Tensor) -> torch.Tensor:
        pos = coords[..., :3] / self.log_lam.exp()
        if self.mode == "ryd2d":
            return torch.cat([pos[..., :2], torch.zeros_like(pos[..., 2:])], -1)
        return pos + torch.stack([torch.zeros_like(coords[..., 3])] * 2 + [coords[..., 3] * self.dz], -1)

    def forward(self, f: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        G, n = f.shape[0], self.n
        if self.mode == "rydlin":                                                   # unbounded angle encoding (re-uploading)
            det = self.base.T.unsqueeze(0) + self.w(f)
        else:
            det = self.base.T.unsqueeze(0) + math.pi * torch.tanh(self.w(f))        # (G, n, L)
        if not self.training and getattr(self, "det_sigma", 0):                    # doc 35 §12: detuning miscalibration
            det = det * (1 + self.det_sigma * torch.randn_like(det))
        if not self.training and getattr(self, "pos_sigma", 0):                    # doc 35 §12: atom position jitter
            coords = torch.cat([coords[..., :2] + self.pos_sigma * torch.randn_like(coords[..., :2]), coords[..., 2:]], -1)  # in-plane
        pos = self.positions(coords)
        V = torch.cdist(pos, pos).clamp(min=self.MIN_DIST).pow(-6)
        if self.mode == "gutz":
            return self._mean_field(det, V)
        eng = (V[:, self.iu[0], self.iu[1]] @ self.pairbits.T).to(torch.complex64)   # (G, 2^n)
        state = torch.zeros(G, 2 ** n, dtype=torch.complex64, device=f.device); state[:, 0] = 1.0
        for l in range(self.layers):
            gx = _rx(self.omega[l])
            for k in range(n):
                ph = torch.exp(-0.5j * det[:, k, l].to(torch.complex64))
                d = torch.stack([ph, ph.conj()], -1).unsqueeze(-1)
                state = _apply_1q(state, d * gx, k, n)
            state = state * torch.exp(-1j * self.tau[l].to(torch.complex64) * eng)
        z = state.abs().square() @ self.zdiag                                        # (G, n)
        cross = state.conj().unsqueeze(1) * state[:, self.flip]                      # (G, n, 2^n): psi*_b psi_{b^i}
        s = cross.sum(-1)
        sign = self.zdiag.T                                                           # (n, 2^n): +1 if bit i = 0
        y = (cross * sign.unsqueeze(0)).sum(-1)
        out = torch.stack([z, s.real, y.imag], -1)                                   # <Z>, <X>, <Y>
        if __import__("os").environ.get("QW_ZZ"):                                   # doc 35 §24 H7: neighbour correlator
            zz = torch.einsum("gb,bi,bj->gij", state.abs().square(), self.zdiag, self.zdiag)          # <Z_i Z_j>
            Vn = V - torch.diag_embed(torch.diagonal(V, dim1=1, dim2=2)); Vn = Vn / Vn.sum(-1, keepdim=True).clamp_min(1e-9)
            out = torch.cat([out, (Vn * zz).sum(-1, keepdim=True)], -1)             # sum_j Vn_ij <Z_i Z_j>
        hw = getattr(self, "hw", None)
        if hw and not self.training:                     # doc 35 §28c: hardware readout (contrast, SPAM, shots)
            from handoff import readout_noise
            return readout_noise(out, **hw)
        shots = getattr(self, "shots", 0)
        if shots and not self.training:                  # finite-shot readout (doc 35 §11): N shots per basis setting
            p = ((1 + out) / 2).clamp(0, 1)
            out = 2 * torch.distributions.Binomial(total_count=shots, probs=p).sample() / shots - 1
        return out

    def _mean_field(self, det, V) -> torch.Tensor:
        G, n = det.shape[:2]
        V = V - torch.diag_embed(torch.diagonal(V, dim1=1, dim2=2))
        amp = torch.zeros(G, n, 2, dtype=torch.complex64, device=det.device); amp[..., 0] = 1.0
        for l in range(self.layers):
            ph = torch.exp(-0.5j * det[..., l].to(torch.complex64))
            d = torch.stack([ph, ph.conj()], -1).unsqueeze(-1)
            amp = torch.einsum("gnij,gnj->gni", d * _rx(self.omega[l]), amp)
            occ = amp[..., 1].abs().square()
            theta = self.tau[l] * torch.einsum("gij,gj->gi", V, occ)
            amp = torch.stack([amp[..., 0], amp[..., 1] * torch.exp(-1j * theta.to(torch.complex64))], -1)
        a0, a1 = amp[..., 0], amp[..., 1]
        z = a0.abs().square() - a1.abs().square()
        c = a0.conj() * a1
        return torch.stack([z, 2 * c.real, 2 * c.imag], -1)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class GroupMLP(nn.Module):
    def __init__(self, f: int, target: int, act: str = "gelu"):
        super().__init__()
        h = max(1, round((target - 3) / (2 * f + 4)))
        self.net = nn.Sequential(nn.Linear(2 * f, h), Sin() if act == "sin" else nn.GELU(), nn.Linear(h, 3))

    def forward(self, f: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        g = f.mean(1, keepdim=True).expand_as(f)
        out = torch.tanh(self.net(torch.cat([f, g], -1)))
        hw = getattr(self, "hw", None)
        if hw and not self.training:                     # doc 35 §28c: the same readout model on the matched control
            from handoff import readout_noise
            return readout_noise(out, **hw)
        return out


class Pair(nn.Module):
    """Division of labor: two cores on the same allocated atoms, readouts concatenated (G, n, 6). `hyb` = register +
    matched MLP; `mlp2` = two independent matched MLPs (the control with the same total budget and structure)."""

    def __init__(self, a: nn.Module, b: nn.Module):
        super().__init__()
        self.a, self.b = a, b

    def forward(self, f, coords):
        return torch.cat([self.a(f, coords), self.b(f, coords)], -1)


def make_tile_core(kind: str, f: int, n: int = 10, layers: int = 4) -> nn.Module:
    if kind == "hyb":
        return Pair(TileRegister(f, n, layers, "ryd"), make_tile_core("mlp", f, n, layers))
    if kind == "mlp2":
        return Pair(make_tile_core("mlp", f, n, layers), make_tile_core("mlp", f, n, layers))
    if kind in ("ryd", "ryd2d", "gutz", "rydlin"):
        return TileRegister(f, n, layers, kind)
    if kind == "mlpsin":                                  # classical control with a trigonometric (analog) bias
        return GroupMLP(f, n_params(TileRegister(f, n, layers)), act="sin")
    if kind.startswith("mlp"):
        scale = int(kind[4:]) if kind.startswith("mlpX") else 1
        return GroupMLP(f, scale * n_params(TileRegister(f, n, layers)))
    raise ValueError(kind)
