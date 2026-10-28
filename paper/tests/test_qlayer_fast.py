"""QW_QFAST=1 layered register path (doc 35 §43h): each layer's single-qubit gates applied as two batched
(2^h x 2^h) matmuls must give the same circuit as the per-qubit path - probabilities, readout and gradients -
for every ansatz / gating / link and odd qubit counts."""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import qlayer  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _close(a, b, tol=1e-4):
    """Whole-tensor relative error: fp32 rounding gives up to ~3e-5 here; in complex128 the two paths agree to 1e-13."""
    return ((a - b).norm() / a.norm().clamp(min=1e-12)).item() < tol


def _both(monkeypatch, module, h, init_state=None):
    out = []
    for flag in ("", "1"):
        monkeypatch.setenv("QW_QFAST", flag)
        module.zero_grad(set_to_none=True)
        y = module.probs(h, init_state) if init_state is not None else module(h)
        y.square().sum().backward()
        out.append((y.detach(), {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}))
    return out


@pytest.mark.parametrize("ansatz,gate,n", [("rydberg", True, 10), ("rydberg", False, 7), ("hea", False, 5), ("rydberg", True, 2)])
def test_block_matches_per_qubit_path(monkeypatch, ansatz, gate, n):
    torch.manual_seed(0)
    blk = qlayer.QuantumBlock(16, n, 3, ansatz, gate=gate).to(DEV)
    with torch.no_grad():
        blk.cond.weight.normal_(std=0.3)
        if gate:
            blk.gate_lin.weight.normal_(std=0.3)
    h = torch.randn(6, 16, device=DEV)
    (y0, g0), (y1, g1) = _both(monkeypatch, blk, h)
    assert torch.allclose(y0, y1, atol=2e-6)
    assert g0.keys() == g1.keys()
    for k in g0:
        assert _close(g0[k], g1[k]), k


def test_probs_normalised_and_init_state(monkeypatch):
    torch.manual_seed(1)
    blk = qlayer.QuantumBlock(16, 6, 2, "rydberg", gate=True).to(DEV)
    with torch.no_grad():
        blk.cond.weight.normal_(std=0.3)
    h = torch.randn(4, 16, device=DEV)
    s0 = torch.randn(4, 2**6, dtype=torch.complex64, device=DEV); s0 = s0 / s0.abs().square().sum(1, keepdim=True).sqrt()
    (p0, _), (p1, _) = _both(monkeypatch, blk, h, init_state=s0)
    assert torch.allclose(p0, p1, atol=2e-6)
    assert torch.allclose(p1.sum(1), torch.ones(4, device=DEV), atol=1e-5)


@pytest.mark.parametrize("link", ["independent", "cascade"])
def test_multiregister_matches(monkeypatch, link):
    torch.manual_seed(2)
    reg = qlayer.MultiRegister(32, 10, 3, layers=2, ansatz="rydberg", link=link, gate=True).to(DEV)
    with torch.no_grad():
        for r in reg.regs:
            r.cond.weight.normal_(std=0.1); r.gate_lin.weight.normal_(std=0.1)
    h = torch.randn(5, 32, device=DEV)
    (y0, g0), (y1, g1) = _both(monkeypatch, reg, h)
    assert torch.allclose(y0, y1, atol=2e-6)
    for k in g0:
        assert _close(g0[k], g1[k]), k
