"""FW-A readout families (Sep 19 sweep): z (default) | diag (trainable diagonal observables) | cond2
(input-conditioned 1+2-local Z-strings). Contracts: identical to `z` at init, trainable, state-dict
compatible with pre-FW-A checkpoints under `z`, parameter counts as documented, product-state path."""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import qlayer  # noqa: E402

D_IN, N, LAYERS, B = 8, 5, 2, 4
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _block(mode, ansatz="rydberg", seed=0):
    torch.manual_seed(seed)
    blk = qlayer.QuantumBlock(D_IN, N, LAYERS, ansatz, readout=mode)
    with torch.no_grad():
        blk.cond.weight.normal_(std=0.3)
    return blk.to(DEV)


def _h(seed=1):
    torch.manual_seed(seed)
    return torch.randn(B, D_IN, device=DEV)


@pytest.mark.parametrize("mode", ["diag", "cond2"])
@pytest.mark.parametrize("ansatz", ["rydberg", "hea"])
def test_identical_to_z_at_init(mode, ansatz):
    ref, blk = _block("z", ansatz), _block(mode, ansatz)
    blk.load_state_dict(ref.state_dict(), strict=False)   # shared circuit weights; readout params stay zero
    h = _h()
    assert torch.allclose(blk(h), ref(h), atol=1e-6)
    assert blk(h).shape == (B, N)


def test_z_state_dict_is_unchanged():
    keys = set(_block("z").state_dict().keys())
    assert not any(k.startswith("readout.") for k in keys), keys


def test_param_counts():
    n_z = qlayer._count(_block("z")) if hasattr(qlayer, "_count") else sum(p.numel() for p in _block("z").parameters())
    n_diag = sum(p.numel() for p in _block("diag").parameters())
    n_cond = sum(p.numel() for p in _block("cond2").parameters())
    P = N + N * (N - 1) // 2
    assert n_diag - n_z == 2**N * N
    assert n_cond - n_z == (D_IN * 16 + 16) + (16 * N * P + N * P)


@pytest.mark.parametrize("mode", ["diag", "cond2"])
def test_readout_trains(mode):
    blk = _block(mode)
    h = _h()
    target = torch.zeros(B, N, device=DEV)
    opt = torch.optim.Adam(blk.readout.parameters(), lr=0.05)
    l0 = ((blk(h) - target) ** 2).mean()
    for _ in range(10):
        opt.zero_grad()
        loss = ((blk(h) - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert ((blk(h) - target) ** 2).mean() < l0
    g = [p.grad for p in blk.readout.parameters()]
    assert all(t is not None and t.abs().sum() > 0 for t in g)


def test_cond2_matches_exact_on_product_state():
    """Gutzwiller control under cond2: <Z_i Z_j> = <Z_i><Z_j> path equals the full formula for a product state."""
    blk = _block("cond2", "rydberg_mf")
    with torch.no_grad():
        blk.readout.hyper[-1].weight.normal_(std=0.2)
    h = _h()
    z = blk._forward_mean_field(blk.base.unsqueeze(0) + torch.pi * torch.tanh(blk.cond(h).reshape(B, LAYERS, N, 2)))
    out = blk.readout.from_product_state(z, h)
    # rebuild the product-state basis distribution explicitly and use the full-state readout
    p1 = (1 - z) / 2                                                  # P(bit=1) per qubit
    bits = qlayer._bits(N).to(DEV)
    probs = torch.stack([(bits * p1[b] + (1 - bits) * (1 - p1[b])).prod(-1) for b in range(B)])
    assert torch.allclose(blk.readout(probs, h), out, atol=1e-5)


def test_diag_not_defined_for_mean_field():
    with pytest.raises(NotImplementedError):
        _block("diag", "rydberg_mf")(_h())


def test_env_selects_mode(monkeypatch):
    monkeypatch.setenv("QREADOUT", "cond2")
    assert qlayer.readout_mode() == "cond2"
    monkeypatch.setenv("QREADOUT", "bogus")
    with pytest.raises(ValueError):
        qlayer.readout_mode()
