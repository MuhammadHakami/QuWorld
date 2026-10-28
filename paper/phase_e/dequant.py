"""Phase-E dequantization surrogates for trained Rydberg MultiRegister cores.

Eval-time surgical replacements for the QuantumBlock registers inside a
CNextU-net hybrid checkpoint (paper/phase_b/cnext_hybrid.py). Checkpoint
weights are NEVER modified; wrappers recompute the register readout from the
same trained parameters under a surrogate physics:

- ProductStateReadout : entanglement removed — (a) blockade energies zeroed,
  or (b) geometry frozen to far-separated atoms (spacing >> blockade radius).
- ChiTruncatedReadout : Schmidt decomposition of the FINAL state across the
  5|5 qubit bipartition truncated to rank chi before <Z> readout (MPS-style
  bond-dimension surrogate, B4).
- ShotReadout         : exact <Z_i> replaced by s-shot binomial estimates
  (hardware-deployability, finite-shot readout).

Everything is pure torch; cudaq is never imported here (LLVM clash — any
CUDA-Q referee runs live in subprocess scripts, see bench/cudaq_check.py).
"""

import math
import pathlib
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn

_PAPER = pathlib.Path(__file__).resolve().parent.parent
if str(_PAPER) not in sys.path:
    sys.path.insert(0, str(_PAPER))
from qlayer import apply_1q, rx_gate, rz_phases  # noqa: E402

FAR_FACTOR = 100.0  # geometry scale for the far-separated ablation


def blockade_energy(qb, pos=None) -> torch.Tensor:
    """(2^n,) diagonal interaction energy; mirrors QuantumBlock._blockade_energy
    but accepts arbitrary positions (used for the far-geometry ablation)."""
    pos = qb.pos if pos is None else pos
    d = torch.cdist(pos, pos) + torch.eye(qb.n, device=pos.device)
    v = d.clamp(min=0.5).pow(-6)
    v = v - torch.diag(torch.diag(v))
    return 0.5 * ((qb.bitsf @ v) * qb.bitsf).sum(-1)


def rydberg_state(qb, h: torch.Tensor, eng=None) -> torch.Tensor:
    """Final statevector (B, 2^n) of a rydberg QuantumBlock; exact mirror of
    QuantumBlock.forward's rydberg branch, with an optional energy override."""
    if qb.ansatz != "rydberg":
        raise ValueError(f"expected rydberg ansatz, got {qb.ansatz}")
    b, n = h.shape[0], qb.n
    ang = qb.base.unsqueeze(0) + math.pi * torch.tanh(
        qb.cond(h).reshape(b, qb.layers, n, 2))
    state = torch.zeros(b, 2**n, dtype=torch.complex64, device=h.device)
    state[:, 0] = 1.0
    if eng is None:
        eng = qb._blockade_energy()
    for l in range(qb.layers):
        gx = rx_gate(qb.omega[l].expand(1))[0]
        for k in range(n):
            d = rz_phases(ang[:, l, k, 0]).unsqueeze(-1)
            state = apply_1q(state, d * gx.unsqueeze(0), k, n)
        state = state * torch.exp(-1j * qb.tau[l].to(torch.complex64) *
                                  eng.to(torch.complex64))
    return state


def readout(qb, state: torch.Tensor) -> torch.Tensor:
    """<Z_i> readout from a statevector: (B, 2^n) -> (B, n)."""
    return state.abs().square() @ qb.zdiag


class ProductStateReadout(nn.Module):
    """B5 ablation: entanglement surgically disabled, drive+detunings kept.

    mode='zero' : blockade interaction energies -> 0 exactly.
    mode='far'  : trained geometry scaled by FAR_FACTOR (spacing >> blockade
                  radius; energies ~ factor^-6 of trained values).
    Either way the evolution factorizes into single-qubit rotations, so the
    final state is a product state (Schmidt rank 1 across every bipartition).
    """

    def __init__(self, qb, mode: str, far_factor: float = FAR_FACTOR):
        super().__init__()
        if mode not in ("zero", "far"):
            raise ValueError(f"unknown product-state mode {mode}")
        self.qb, self.mode, self.far_factor = qb, mode, far_factor

    def energy(self) -> torch.Tensor:
        if self.mode == "zero":
            return torch.zeros(2**self.qb.n, device=self.qb.pos.device)
        return blockade_energy(self.qb, self.qb.pos * self.far_factor)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return readout(self.qb, rydberg_state(self.qb, h, eng=self.energy()))


class ChiTruncatedReadout(nn.Module):
    """B4 surrogate: rank-chi Schmidt truncation of the final state.

    The (B, 2^n) state is reshaped to (B, 2^(n/2), 2^(n/2)) — qubits [0..n/2)
    vs [n/2..n) — SVD'd, truncated to the top-chi singular values,
    renormalized to unit norm, and read out. chi = 2^(n/2) is exact.
    """

    def __init__(self, qb, chi: int):
        super().__init__()
        if not 1 <= chi <= 2 ** (qb.n // 2):
            raise ValueError(f"chi {chi} out of range for n={qb.n}")
        self.qb, self.chi = qb, chi

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        state = rydberg_state(self.qb, h)
        half = 2 ** (self.qb.n // 2)
        m = state.reshape(-1, half, half)
        u, s, vh = torch.linalg.svd(m, full_matrices=False)
        st = torch.zeros_like(s)
        st[:, : self.chi] = s[:, : self.chi]
        st = st / st.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        mt = (u * st.unsqueeze(-2).to(u.dtype)) @ vh
        return readout(self.qb, mt.reshape(state.shape))


class ShotReadout(nn.Module):
    """Finite-shot readout: each exact <Z_i> replaced by a binomial estimate
    from `shots` measurement shots of the marginal P(z_i = +1)."""

    def __init__(self, qb, shots: int):
        super().__init__()
        if shots < 1:
            raise ValueError(f"shots must be >= 1, got {shots}")
        self.qb, self.shots = qb, shots

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.qb(h)
        p = ((z + 1.0) * 0.5).clamp(0.0, 1.0)
        k = torch.binomial(torch.full_like(p, float(self.shots)), p)
        return 2.0 * k / self.shots - 1.0


class ConstantReadout(nn.Module):
    """Control: register readout replaced by zeros — kills ALL information
    flow through the quantum core at eval (trained W_out bias path kept).
    Separates 'readout content is surrogate-able' from 'readout is unused'."""

    def __init__(self, qb):
        super().__init__()
        self.qb = qb

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.zeros(h.shape[0], self.qb.n, device=h.device)


@contextmanager
def gate_disabled(model):
    """Control: zero the bottleneck's alpha gate — the adapter contributes
    nothing to the field at eval (restored on exit; disk weights untouched)."""
    alpha = model.neck.bottleneck.alpha
    saved = alpha.data.clone()
    alpha.data.zero_()
    try:
        yield model
    finally:
        alpha.data.copy_(saved)


@contextmanager
def surrogate_registers(model, wrap):
    """Temporarily replace every register in the hybrid's MultiRegister core
    with wrap(register); restores the originals on exit (weights untouched)."""
    core = model.neck.bottleneck.core
    originals = list(core.regs)
    core.regs = nn.ModuleList([wrap(qb) for qb in originals])
    try:
        yield model
    finally:
        core.regs = nn.ModuleList(originals)


@torch.no_grad()
def verify_state_replication(core, dev: str, atol: float = 2e-5) -> float:
    """Assert rydberg_state reproduces each register's own forward readout."""
    h = torch.randn(4, core.regs[0].cond.in_features, device=dev)
    worst = 0.0
    for qb in core.regs:
        diff = (readout(qb, rydberg_state(qb, h)) - qb(h)).abs().max().item()
        worst = max(worst, diff)
    if worst > atol:
        raise AssertionError(f"state replication mismatch: {worst:.2e}")
    return worst


@torch.no_grad()
def capture_latents(model, valid, n_latents: int = 256, stride_t: int = 3,
                    dev: str = "cuda", n_hist: int = 4) -> torch.Tensor:
    """Pooled neck latents h (the exact input to the quantum core) from real
    valid-split windows; returns (n_latents, 672) on CPU."""
    grabbed = []
    core = model.neck.bottleneck.core
    hook = core.register_forward_pre_hook(
        lambda _m, args: grabbed.append(args[0].detach().cpu()))
    model.eval()
    n, t = valid.shape[0], valid.shape[1]
    try:
        for t0 in range(n_hist - 1, t - 1, stride_t):
            idx = torch.arange(t0 - n_hist + 1, t0 + 1)
            x = valid[:, idx].reshape(n, -1, *valid.shape[-2:]).to(dev)
            model(x)
            if sum(g.shape[0] for g in grabbed) >= n_latents:
                break
    finally:
        hook.remove()
    h = torch.cat(grabbed)
    if h.shape[0] < n_latents:
        raise RuntimeError(f"only captured {h.shape[0]} latents")
    return h[:n_latents]


@torch.no_grad()
def schmidt_singular_values(qb, h: torch.Tensor, eng=None,
                            chunk: int = 64) -> torch.Tensor:
    """(B, 2^(n/2)) singular values of the final state across the 5|5 cut."""
    dev = qb.pos.device
    half = 2 ** (qb.n // 2)
    outs = []
    for i in range(0, h.shape[0], chunk):
        s = rydberg_state(qb, h[i:i + chunk].to(dev), eng=eng)
        outs.append(torch.linalg.svdvals(s.reshape(-1, half, half)).cpu())
    return torch.cat(outs)


def entropy_bits(sv: torch.Tensor) -> torch.Tensor:
    """Entanglement entropy in bits from Schmidt singular values (B, r)."""
    p = sv.square().clamp_min(1e-12)
    p = p / p.sum(-1, keepdim=True)
    return -(p * p.log2()).sum(-1)


@torch.no_grad()
def readout_rms_delta(core, wrap, h: torch.Tensor, chunk: int = 64) -> float:
    """RMS over samples and qubits of (surrogate - exact) 30-dim core readout."""
    dev = next(core.parameters()).device
    sq = []
    for i in range(0, h.shape[0], chunk):
        hc = h[i:i + chunk].to(dev)
        exact = core(hc)
        sur = torch.cat([wrap(qb)(hc) for qb in core.regs], dim=-1)
        sq.append((sur - exact).square().mean(dim=-1).cpu())
    return torch.cat(sq).mean().sqrt().item()


@torch.no_grad()
def min_pair_dist(qb) -> float:
    """Smallest trained inter-atom distance (blockade-radius units)."""
    d = torch.cdist(qb.pos, qb.pos)
    mask = ~torch.eye(qb.n, dtype=torch.bool, device=d.device)
    return d[mask].min().item()
