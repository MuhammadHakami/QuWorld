"""Spectral-Routed Register (SRR) — the register writes where the trunk's error is, not into the pooled DC mode.

Design: 32-dynamic-register-configuration.md §5; formal backbone: formal/QuotaScaling/SpectralRouting.lean.

    p = trunk(x)                                  digital part (fine-tuned as in Phase B)
    d = p - x_t                                   the trunk's own predicted change (available at inference)
    tiles = orthonormal Haar detail coefficients of d, one tile = (level, orientation, y, x) with all C channels
    route: top-K tiles by channel-summed energy of d (zero-parameter router; wins at K <= 300, §4.4)
    features f_i: Haar coefficients of d, p, x_t and x_t - x_{t-1} at the tile + log-energy + level/orientation one-hot
    core over the K tiles, split into n_reg groups of n_per by Morton order (a group = one front):
        ryd   Rydberg register per group; atom i at (cx_i / lam, cy_i / lam, level_i * dz): geometry set by the tile,
              2 global scales learned (non-parameterised placement); detuning from f_i with weights shared across atoms
        ryd2d same, z = 0 (planar control: no signed / cross-scale couplings)
        gutz  same parameters, mean-field product state (interactions kept, entanglement forbidden)
        mlp   DeepSets MLP per group ([f_i, mean_group f]) parameter-matched to the register
        lin   no core (z_i = 1): the classical routed linear corrector
    write-back: c_i = alpha * z_i * (M f_i) in the orthonormal Haar basis (alpha zero-init, so the model starts as the
    trunk exactly) -> inverse Haar -> y = p + correction.
    route "rand": K uniformly random tiles (is *where* the atoms go what matters?).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

LEVELS = 5


def haar2d(x: torch.Tensor, levels: int = LEVELS):
    """Orthonormal multi-level 2-D Haar DWT: x (B,C,H,W) -> (approx, [(LH,HL,HH) per level, fine -> coarse])."""
    details, a = [], x
    for _ in range(levels):
        p, q, r, s = a[..., 0::2, 0::2], a[..., 0::2, 1::2], a[..., 1::2, 0::2], a[..., 1::2, 1::2]
        details.append(((p - q + r - s) / 2, (p + q - r - s) / 2, (p - q - r + s) / 2))
        a = (p + q + r + s) / 2
    return a, details


def ihaar2d(a: torch.Tensor, details) -> torch.Tensor:
    """Exact inverse of haar2d."""
    for lh, hl, hh in reversed(details):
        p = (a + lh + hl + hh) / 2; q = (a - lh + hl - hh) / 2
        r = (a + lh - hl - hh) / 2; s = (a - lh - hl + hh) / 2
        B, C, h, w = a.shape
        out = a.new_empty(B, C, 2 * h, 2 * w)
        out[..., 0::2, 0::2], out[..., 0::2, 1::2], out[..., 1::2, 0::2], out[..., 1::2, 1::2] = p, q, r, s
        a = out
    return a


class TileBank:
    """Flat tile indexing of all detail coefficients: level, orientation, y, x (channels kept together)."""

    def __init__(self, H: int, W: int, levels: int, device):
        lv, orient, cy, cx, sizes = [], [], [], [], []
        for j in range(levels):
            h, w, step = H >> (j + 1), W >> (j + 1), 2 ** (j + 1)
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            for o in range(3):
                lv.append(torch.full((h * w,), j)); orient.append(torch.full((h * w,), o))
                cy.append(((yy.flatten() + 0.5) * step)); cx.append(((xx.flatten() + 0.5) * step))
            sizes.append((h, w))
        self.level = torch.cat(lv).to(device); self.orient = torch.cat(orient).to(device)
        self.cy = torch.cat(cy).float().to(device); self.cx = torch.cat(cx).float().to(device)
        self.sizes, self.levels, self.n = sizes, levels, int(self.level.numel())
        ky, kx = self.cy.long() // 2, self.cx.long() // 2           # Morton key of the tile centre (fronts stay together)
        key = torch.zeros_like(ky)
        for b in range(12):
            key |= ((kx >> b) & 1) << (2 * b) | ((ky >> b) & 1) << (2 * b + 1)
        self.morton = key

    def flatten(self, details) -> torch.Tensor:
        """[(LH,HL,HH)] -> (B, n_tiles, C)."""
        parts = []
        for lvl in details:
            for c in lvl:
                parts.append(c.flatten(2).transpose(1, 2))            # (B, h*w, C)
        return torch.cat(parts, dim=1)

    def unflatten(self, flat: torch.Tensor, C: int):
        """(B, n_tiles, C) -> [(LH,HL,HH)] with this bank's shapes."""
        out, i, B = [], 0, flat.shape[0]
        for h, w in self.sizes:
            lvl = []
            for _ in range(3):
                lvl.append(flat[:, i:i + h * w].transpose(1, 2).reshape(B, C, h, w)); i += h * w
            out.append(tuple(lvl))
        return out


def _bits(n: int) -> torch.Tensor:
    idx = torch.arange(2 ** n)
    return torch.stack([(idx >> (n - 1 - i)) & 1 for i in range(n)], -1).float()


def _rx(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c + 0j, -1j * s], -1), torch.stack([-1j * s, c + 0j], -1)], -2).to(torch.complex64)


class TileRegister(nn.Module):
    """Rydberg-analog register over a group of n tiles, geometry per input from the tiles' positions.
    z (B, n) = <Z_i>. `mode`: ryd | ryd2d | gutz."""

    MIN_DIST = 0.5

    def __init__(self, f: int, n: int, layers: int = 4, mode: str = "ryd"):
        super().__init__()
        self.n, self.layers, self.mode = n, layers, mode
        self.w = nn.Linear(f, layers)                                # detuning map shared across atoms
        nn.init.normal_(self.w.weight, std=0.1); nn.init.zeros_(self.w.bias)
        self.base = nn.Parameter(0.02 * torch.randn(layers, n))
        self.omega = nn.Parameter(torch.full((layers,), 0.5)); self.tau = nn.Parameter(torch.full((layers,), 0.3))
        self.log_lam = nn.Parameter(torch.tensor(math.log(4.0)))     # pixels per blockade radius
        self.dz = nn.Parameter(torch.tensor(1.0))                    # scale-plane separation
        bits = _bits(n)
        iu = torch.triu_indices(n, n, offset=1)
        self.register_buffer("zdiag", 1.0 - 2.0 * bits)
        self.register_buffer("pairbits", bits[:, iu[0]] * bits[:, iu[1]])
        self.register_buffer("iu", iu)

    def couplings(self, cx, cy, lvl) -> torch.Tensor:
        """(G, n) tile coordinates -> (G, n, n) V_ij = max(r, r_min)^-6."""
        lam = self.log_lam.exp()
        z = torch.zeros_like(cx) if self.mode == "ryd2d" else lvl.float() * self.dz
        pos = torch.stack([cx / lam, cy / lam, z], -1)
        return torch.cdist(pos, pos).clamp(min=self.MIN_DIST).pow(-6)

    def forward(self, f, cx, cy, lvl) -> torch.Tensor:
        G, n = f.shape[0], self.n
        det = self.base.T.unsqueeze(0) + math.pi * torch.tanh(self.w(f))          # (G, n, L)
        V = self.couplings(cx, cy, lvl)
        if self.mode == "gutz":
            return self._mean_field(det, V)
        eng = (V[:, self.iu[0], self.iu[1]] @ self.pairbits.T).to(torch.complex64)  # (G, 2^n)
        state = torch.zeros(G, 2 ** n, dtype=torch.complex64, device=f.device); state[:, 0] = 1.0
        for l in range(self.layers):
            gx = _rx(self.omega[l])
            for k in range(n):
                ph = torch.exp(-0.5j * det[:, k, l].to(torch.complex64))
                d = torch.stack([ph, ph.conj()], -1).unsqueeze(-1)                  # RZ as row scaling (G, 2, 1)
                state = _apply_1q(state, d * gx, k, n)
            state = state * torch.exp(-1j * self.tau[l].to(torch.complex64) * eng)
        return state.abs().square() @ self.zdiag                                   # (G, n)

    def _mean_field(self, det, V) -> torch.Tensor:
        G, n = det.shape[:2]
        V = V - torch.diag_embed(torch.diagonal(V, dim1=1, dim2=2))
        amp = torch.zeros(G, n, 2, dtype=torch.complex64, device=det.device); amp[..., 0] = 1.0
        for l in range(self.layers):
            ph = torch.exp(-0.5j * det[..., l].to(torch.complex64))
            d = torch.stack([ph, ph.conj()], -1).unsqueeze(-1)                      # (G, n, 2, 1)
            amp = torch.einsum("gnij,gnj->gni", d * _rx(self.omega[l]), amp)
            occ = amp[..., 1].abs().square()
            theta = self.tau[l] * torch.einsum("gij,gj->gi", V, occ)
            amp = torch.stack([amp[..., 0], amp[..., 1] * torch.exp(-1j * theta.to(torch.complex64))], -1)
        return amp[..., 0].abs().square() - amp[..., 1].abs().square()


def _apply_1q(state: torch.Tensor, g: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Apply per-sample 2x2 gate g (G, 2, 2) to qubit k (qubit 0 = MSB) of state (G, 2^n)."""
    G = state.shape[0]
    s = state.reshape(G, 2 ** k, 2, 2 ** (n - k - 1))
    return torch.einsum("gij,gajb->gaib", g, s).reshape(G, 2 ** n)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


class GroupMLP(nn.Module):
    """DeepSets per group: z_i = tanh(MLP([f_i, mean_group f])), hidden width chosen to match `target` params."""

    def __init__(self, f: int, target: int):
        super().__init__()
        h = max(1, round((target - 1) / (2 * f + 2)))
        self.net = nn.Sequential(nn.Linear(2 * f, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, f, cx, cy, lvl):
        g = f.mean(1, keepdim=True).expand_as(f)
        return torch.tanh(self.net(torch.cat([f, g], -1))).squeeze(-1)


class SRR(nn.Module):
    """Trunk + spectral-routed correction. `core` in {lin, mlp, ryd, ryd2d, gutz}; `route` in {trunk, rand}."""

    def __init__(self, trunk: nn.Module, C: int, H: int, W: int, n_hist: int, core: str = "ryd", route: str = "trunk",
                 n_reg: int = 3, n_per: int = 10, layers: int = 4, levels: int = LEVELS):
        super().__init__()
        self.trunk, self.C, self.n_hist, self.core_kind, self.route = trunk, C, n_hist, core, route
        self.n_reg, self.n_per, self.K, self.levels = n_reg, n_per, n_reg * n_per, levels
        self.bank_shape = (H, W)
        self.F = 4 * C + 1 + levels + 3
        self.norm = nn.LayerNorm(4 * C, elementwise_affine=False)
        if core in ("ryd", "ryd2d", "gutz"):
            self.core = TileRegister(self.F, n_per, layers, core)
        elif core == "mlp":
            self.core = GroupMLP(self.F, n_params(TileRegister(self.F, n_per, layers)))
        elif core == "lin":
            self.core = None
        else:
            raise ValueError(core)
        self.M = nn.Linear(self.F, C)
        self.alpha = nn.Parameter(torch.zeros(C))                   # zero-init: starts exactly as the trunk
        self._bank = None

    def bank(self, device) -> TileBank:
        if self._bank is None or self._bank.level.device != torch.device(device):
            self._bank = TileBank(*self.bank_shape, self.levels, device)
        return self._bank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.trunk(x)
        C, B = self.C, x.shape[0]
        xt = x[:, (self.n_hist - 1) * C:]; xp = x[:, (self.n_hist - 2) * C:(self.n_hist - 1) * C]
        bank = self.bank(x.device)
        coef = lambda f: bank.flatten(haar2d(f, self.levels)[1])            # (B, N, C)
        cd = coef(p - xt)
        energy = cd.detach().square().sum(-1)                              # (B, N) router score
        if self.route == "rand":
            idx = torch.rand(energy.shape, device=x.device).topk(self.K, dim=1).indices
        else:
            idx = energy.topk(self.K, dim=1).indices
        order = bank.morton[idx].argsort(dim=1)                            # group the chosen tiles by locality
        idx = idx.gather(1, order)
        take = lambda t: t.gather(1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
        lvl, ori = bank.level[idx], bank.orient[idx]
        feats = torch.cat([take(cd), take(coef(p)), take(coef(xt)), take(coef(xt - xp))], -1)
        f = torch.cat([self.norm(feats), energy.gather(1, idx).clamp_min(1e-12).log().unsqueeze(-1) / 10,
                       nn.functional.one_hot(lvl, self.levels).float(), nn.functional.one_hot(ori, 3).float()], -1)
        if self.core is None:
            z = torch.ones(B, self.K, device=x.device)
        else:
            g = lambda t: t.reshape(B * self.n_reg, self.n_per, *t.shape[2:])
            z = self.core(g(f), g(bank.cx[idx]), g(bank.cy[idx]), g(lvl)).reshape(B, self.K)
        corr = self.alpha * z.unsqueeze(-1) * self.M(f)                    # (B, K, C) Haar-coefficient corrections
        flat = torch.zeros(B, bank.n, C, device=x.device, dtype=p.dtype).scatter(1, idx.unsqueeze(-1).expand(-1, -1, C), corr.to(p.dtype))
        H, W = self.bank_shape
        a0 = torch.zeros(B, C, H >> self.levels, W >> self.levels, device=x.device, dtype=p.dtype)
        return p + ihaar2d(a0, bank.unflatten(flat, C))
