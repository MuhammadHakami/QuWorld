"""Behavioural tests for the Gutzwiller (mean-field) Rydberg control.

The control keeps the blockade interactions but forbids entanglement: the
register evolves as a product state under the mean-field decoupling
V_ij n_i n_j -> V_ij (<n_j> n_i + <n_i> n_j - <n_i><n_j>).
"""
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import qlayer  # noqa: E402

D_IN, N, LAYERS = 8, 4, 3


def _pair(seed: int = 0, spread: float = 1.0):
    """Exact-Rydberg and mean-field blocks sharing identical weights."""
    torch.manual_seed(seed)
    exact = qlayer.QuantumBlock(D_IN, N, LAYERS, "rydberg")
    with torch.no_grad():
        exact.cond.weight.normal_(std=0.3)   # exercise input-conditioning
        exact.pos.mul_(spread)
    mf = qlayer.QuantumBlock(D_IN, N, LAYERS, "rydberg_mf")
    mf.load_state_dict(exact.state_dict())
    return exact, mf


def test_mf_state_dict_matches_rydberg_parameterisation():
    exact, mf = _pair()
    shapes = lambda m: {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert shapes(mf) == shapes(exact)


def test_mf_equals_exact_when_interactions_vanish():
    exact, mf = _pair(spread=1e3)            # atoms far apart: V ~ 1e-18
    h = torch.randn(5, D_IN)
    torch.testing.assert_close(mf(h), exact(h), atol=1e-5, rtol=0)


def test_mf_interactions_change_readout():
    _, close = _pair(spread=1.0)
    _, far = _pair(spread=1e3)               # same weights, only V differs
    h = torch.randn(5, D_IN)
    assert not torch.allclose(close(h), far(h), atol=1e-4)


def test_mf_two_qubit_phase_matches_hand_computation():
    torch.manual_seed(0)
    mf = qlayer.QuantumBlock(D_IN, 2, 2, "rydberg_mf")
    with torch.no_grad():
        mf.base.zero_()                      # zero detunings (cond is zero-init)
        mf.pos.copy_(torch.tensor([[0.0, 0.0], [0.7, 0.0]]))
        mf.omega.copy_(torch.tensor([0.9, 1.3]))
        mf.tau.copy_(torch.tensor([0.8, 0.5]))
    out = mf(torch.zeros(1, D_IN))[0]

    def rx(w):
        c, s = math.cos(w / 2), math.sin(w / 2)
        return np.array([[c, -1j * s], [-1j * s, c]])

    v = 0.7 ** -6
    a = rx(0.9) @ np.array([1.0, 0.0])
    occ = abs(a[1]) ** 2                     # both qubits identical by symmetry
    a = a * np.array([1.0, np.exp(-1j * 0.8 * v * occ)])  # local MF phase
    a = rx(1.3) @ a                          # layer-2 phase is diagonal: no effect on <Z>
    z = abs(a[0]) ** 2 - abs(a[1]) ** 2
    torch.testing.assert_close(out, torch.tensor([z, z], dtype=torch.float32),
                               atol=1e-5, rtol=0)


def test_mf_batch_samples_are_independent():
    _, mf = _pair()
    h = torch.randn(3, D_IN)
    single = torch.cat([mf(h[i:i + 1]) for i in range(3)])
    torch.testing.assert_close(mf(h), single, atol=1e-6, rtol=0)


def test_mf_gradients_reach_geometry():
    _, mf = _pair()
    mf(torch.randn(4, D_IN)).sum().backward()
    assert mf.pos.grad is not None and mf.pos.grad.abs().sum() > 0


def test_multiregister_accepts_mean_field_ansatz():
    reg = qlayer.MultiRegister(D_IN, 3, 2, 2, "rydberg_mf")
    assert reg(torch.randn(2, D_IN)).shape == (2, 6)


def test_mf_first_order_response_in_tau_matches_exact():
    """At tau=0 the exact and mean-field first-order responses of the readouts
    coincide for product states; this pins the phase sign and the pair factor."""
    exact, mf = _pair()
    for m in (exact, mf):
        with torch.no_grad():
            m.tau.zero_()
    h = torch.randn(4, D_IN)
    exact(h).sum().backward()
    mf(h).sum().backward()
    torch.testing.assert_close(mf.tau.grad, exact.tau.grad, atol=1e-5, rtol=0)


@pytest.mark.parametrize("n,layers", [(1, 1), (1, 3), (3, 1)])
def test_mf_equals_exact_in_degenerate_configurations(n, layers):
    """n=1 has no pairs; a single layer ends in a diagonal phase that cannot
    move <Z>: in both cases mean-field and exact must agree with interactions on."""
    torch.manual_seed(1)
    exact = qlayer.QuantumBlock(D_IN, n, layers, "rydberg")
    with torch.no_grad():
        exact.cond.weight.normal_(std=0.3)
    mf = qlayer.QuantumBlock(D_IN, n, layers, "rydberg_mf")
    mf.load_state_dict(exact.state_dict())
    h = torch.randn(3, D_IN)
    torch.testing.assert_close(mf(h), exact(h), atol=1e-5, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_mf_runs_on_cuda():
    _, mf = _pair()
    out = mf.cuda()(torch.randn(2, D_IN, device="cuda"))
    assert out.device.type == "cuda" and torch.isfinite(out).all()
