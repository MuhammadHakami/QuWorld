import pathlib
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "paper/alloc"))
mhd_divfree = pytest.importorskip("mhd_divfree")


def spectral_div_share(B):                              # B (3, n, n, n) physical units
    n = B.shape[-1]; k = torch.fft.fftfreq(n) * n; k[n // 2] = 0.0; K = torch.stack(torch.meshgrid(k, k, k, indexing="ij"))
    Bh = torch.fft.fftn(B, dim=(1, 2, 3)); L = K * (K * Bh).sum(0) / (K ** 2).sum(0).clamp_min(1)
    return float((L.abs() ** 2).sum() / (Bh.abs() ** 2).sum())


def test_projection_removes_divergence_keeps_other_channels_and_is_idempotent():
    torch.manual_seed(0)
    m = mhd_divfree.DivFreeB(nn.Identity(), n=16)
    x = torch.randn(2, 7, 16, 16, 16)
    y = m(x)
    assert torch.equal(y[:, 0], x[:, 0]) and torch.equal(y[:, 4:], x[:, 4:])
    for b in range(2):
        assert spectral_div_share(x[b, 1:4] * m.std[0] + m.mean[0]) > 0.1
        assert spectral_div_share(y[b, 1:4] * m.std[0] + m.mean[0]) < 1e-10
    assert torch.allclose(m(y), y, atol=1e-5)


def test_divergence_free_input_is_unchanged():
    n = 16; g = torch.arange(n) * 2 * torch.pi / n
    X, Y, Z = torch.meshgrid(g, g, g, indexing="ij")
    B = torch.stack([torch.sin(Y), torch.sin(Z), torch.sin(X)])          # div B = 0
    m = mhd_divfree.DivFreeB(nn.Identity(), n=n)
    x = torch.zeros(1, 7, n, n, n); x[0, 1:4] = (B - m.mean[0]) / m.std[0]
    assert torch.allclose(m(x), x, atol=1e-5)


def test_shift_average_of_an_equivariant_model_is_the_model():
    torch.manual_seed(1)
    conv = nn.Conv3d(2, 2, 3, padding=1, padding_mode="circular", bias=False)     # exactly shift-equivariant
    x = torch.randn(1, 2, 8, 8, 8)
    avg = mhd_divfree.ShiftAverage(conv, shifts=((0, 0, 0), (3, 5, 1), (7, 2, 6)))
    assert torch.allclose(avg(x), conv(x), atol=1e-5)


def test_field_grouped_summary_weights_density_by_a_third():
    v = torch.tensor([[0.3, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]])
    s = mhd_divfree.summary(v)
    assert s["per_channel"] == pytest.approx(1.2 / 7) and s["field_grouped"] == pytest.approx((0.3 + 0.1 + 0.2) / 3)
