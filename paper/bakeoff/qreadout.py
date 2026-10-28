"""Z / ZZ observable readout for Rydberg QuantumBlocks (e-prop factor 1).

Re-simulates a block's statevector to expose the basis probabilities the
block's forward() discards (qlayer.py is not modified). On hardware these are
the same shots the block already measures — <Z_i> and <Z_i Z_j> come for free
— so the re-simulation is NOT charged to the eval-unit budget (documented in
runner.py; the design brief prices an e-prop step at 1 unit).

`rydberg_probs` mirrors QuantumBlock.forward's rydberg branch exactly
(detuning angles = base + pi*tanh(cond(h)), global RX drive, diagonal
blockade phase from the block's own trainable positions).
"""

import numpy as np
import torch

from _bootstrap import add_paths

add_paths()

from qlayer import QuantumBlock, apply_1q, rx_gate, rz_phases  # noqa: E402


def rydberg_probs(block: QuantumBlock, h: torch.Tensor) -> torch.Tensor:
    """(B, 2^n) basis probabilities after the block's analog evolution."""
    if block.ansatz != "rydberg":
        raise ValueError("ZZ readout is defined for the rydberg ansatz only")
    b, n = h.shape[0], block.n
    ang = block.base.unsqueeze(0) + np.pi * torch.tanh(
        block.cond(h).reshape(b, block.layers, n, 2))
    state = torch.zeros(b, 2 ** n, dtype=torch.complex64, device=h.device)
    state[:, 0] = 1.0
    eng = block._blockade_energy()
    for layer in range(block.layers):
        gx = rx_gate(block.omega[layer].expand(1))[0]
        for k in range(n):
            d = rz_phases(ang[:, layer, k, 0]).unsqueeze(-1)
            state = apply_1q(state, d * gx.unsqueeze(0), k, n)
        phase = torch.exp(-1j * block.tau[layer].to(torch.complex64) *
                          eng.to(torch.complex64))
        state = state * phase
    return state.abs().square()


def moments(block: QuantumBlock, h: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(<Z_i>, <Z_i Z_j>, connected correlator C_ij), shapes
    (B, n), (B, n, n), (B, n, n)."""
    probs = rydberg_probs(block, h)
    z = probs @ block.zdiag                                   # (B, n)
    zz = torch.einsum("bs,si,sj->bij", probs, block.zdiag, block.zdiag)
    conn = zz - z.unsqueeze(-1) * z.unsqueeze(-2)
    return z, zz, conn
