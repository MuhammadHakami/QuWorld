import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "paper/alloc"))
import handoff as H  # noqa: E402


def coalitions(fn):
    import itertools
    return {frozenset(S): fn(frozenset(S)) for k in range(4) for S in itertools.combinations(H.PLAYERS, k)}


def test_shapley_is_efficient_symmetric_and_zero_for_a_null_player():
    v = coalitions(lambda S: 2.0 * ("classical" in S) + 2.0 * ("core" in S) + 1.0 * ("classical" in S and "core" in S))
    phi = H.shapley(v)
    assert sum(phi.values()) == pytest.approx(v[frozenset(H.PLAYERS)] - v[frozenset()])
    assert phi["classical"] == pytest.approx(phi["core"]) == pytest.approx(2.5)
    assert phi["gate"] == pytest.approx(0.0)


def test_shapley_matches_marginals_for_an_additive_game():
    w = {"gate": 0.3, "classical": 1.2, "core": -0.1}
    phi = H.shapley(coalitions(lambda S: sum(w[p] for p in S)))
    assert all(phi[p] == pytest.approx(w[p]) for p in w)


def test_noise_free_readout_is_the_identity():
    out = torch.rand(4, 10, 3) * 2 - 1
    assert torch.allclose(H.readout_noise(out, 0, (0.0, 0.0), 1.0), out, atol=1e-6)


def test_spam_and_contrast_act_as_expected_in_expectation():
    out = torch.tensor([[[0.6, 0.5, -0.4]]])
    got = H.readout_noise(out, 0, (0.01, 0.03), 0.9)
    for ch, v in enumerate((0.6, 0.45, -0.36)):                    # contrast on X and Y only
        p1 = (1 - v) / 2 * (1 - 0.03) + (1 + v) / 2 * 0.01
        assert float(got[0, 0, ch]) == pytest.approx(1 - 2 * p1, abs=1e-6)


def test_shot_noise_is_unbiased_with_binomial_variance():
    torch.manual_seed(0)
    out = torch.full((20000, 10, 3), 0.4)
    got = H.readout_noise(out, 100, (0.0, 0.0), 1.0)
    assert float(got.mean()) == pytest.approx(0.4, abs=2e-3)
    assert float(got.var()) == pytest.approx((1 - 0.4 ** 2) / 100, rel=0.05)


def test_group_trust_is_per_group_and_rises_with_snr():
    o = torch.cat([torch.full((2, 10, 3), 0.05), torch.full((2, 10, 3), 0.8)], 1)
    t = H.group_trust(o, 10, 1000, 0.0, 1.0)
    assert t.shape == (2, 20, 1) and bool((t >= 0).all() and (t <= 1).all())
    assert float(t[0, 0]) < float(t[0, 10])
    assert torch.equal(t[:, :10], t[:, :1].expand(-1, 10, -1))
    assert H.trust_factor(None, o, 10) == 1.0 and H.trust_factor({"kind": "const", "t": 0.25}, o, 10) == 0.25
