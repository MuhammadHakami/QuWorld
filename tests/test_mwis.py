import itertools
import random

import pytest
import torch

from paper.qmethods import mwis as M


def weight(x, w):
    return float((x.float() * w.cpu()).sum())


def random_graph(n, p, seed):
    rng = random.Random(seed)
    edges = [(i, j) for i, j in itertools.combinations(range(n), 2) if rng.random() < p]
    w = torch.tensor([0.2 + rng.random() for _ in range(n)])
    return edges, w


def test_path_graph_greedy_misses_the_optimum():
    edges, w = [(0, 1), (1, 2)], torch.tensor([1.0, 1.5, 1.0])
    assert M.exact(3, edges, w).tolist() == [1, 0, 1]
    assert M.greedy(3, edges, w).tolist() == [0, 1, 0]
    assert M.rydberg(3, edges, w, shots=50, generator=torch.Generator().manual_seed(0)).tolist() == [1, 0, 1]
    assert M.anneal(3, edges, w, shots=20, generator=torch.Generator().manual_seed(0)).tolist() == [1, 0, 1]


def test_exact_matches_brute_force():
    for seed in range(5):
        edges, w = random_graph(8, 0.3, seed)
        best = max((x for x in itertools.product((0, 1), repeat=8) if M.is_independent(torch.tensor(x), edges)),
                   key=lambda x: weight(torch.tensor(x), w))
        assert weight(M.exact(8, edges, w), w) == pytest.approx(weight(torch.tensor(best), w))


@pytest.mark.parametrize("name", ["rydberg", "greedy", "exact", "anneal"])
def test_every_selector_returns_an_independent_set_no_heavier_than_exact(name):
    for seed in range(4):
        edges, w = random_graph(10, 0.35, seed)
        kw = {"generator": torch.Generator().manual_seed(seed)} if name in ("rydberg", "anneal") else {}
        x = M.SELECTORS[name](10, edges, w, **kw)
        assert M.is_independent(x, edges)
        assert weight(x, w) <= weight(M.exact(10, edges, w), w) + 1e-6


def test_blockade_evolution_is_unitary_and_stays_in_the_subspace():
    edges, w = random_graph(9, 0.4, 7)
    psi, ok = M.rydberg_state(9, edges, w, M.Sweep(T=10.0, steps=50))
    assert float(psi.abs().square().sum()) == pytest.approx(1.0, abs=1e-4)
    assert float(psi[~ok].abs().max()) == 0.0


def test_sweep_concentrates_on_the_optimum_and_shots_find_it():
    edges, w = random_graph(8, 0.3, 3)                                     # best two sets differ by only 4% in weight
    opt = M.exact(8, edges, w)
    index = int(sum(int(b) << (7 - i) for i, b in enumerate(opt)))
    psi, ok = M.rydberg_state(8, edges, w, M.Sweep(T=80.0, steps=800))
    assert float(psi[index].abs().square()) > 10 / int(ok.sum())         # well above uniform over independent sets
    assert M.rydberg(8, edges, w, shots=100, generator=torch.Generator().manual_seed(1)).tolist() == opt.tolist()


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        M.exact(3, [(0, 3)], torch.ones(3))
    with pytest.raises(ValueError):
        M.greedy(2, [], torch.tensor([1.0, -1.0]))
    with pytest.raises(ValueError):
        M.exact(M.MAX_N + 1, [], torch.ones(M.MAX_N + 1))


def test_subspace_batch_matches_the_full_space_evolution():
    graphs = [M.Graph(n, tuple(e), w) for n, (e, w) in ((9, random_graph(9, 0.4, 11)), (6, random_graph(6, 0.3, 12)),
                                                        (8, random_graph(8, 0.5, 13)))]
    sweep = M.Sweep(T=12.0, steps=60)
    for g, (ids, p) in zip(graphs, M.rydberg_probs_batch(graphs, sweep)):
        psi, ok = M.rydberg_state(g.n, g.edges, g.w, sweep)
        full = psi.abs().square()
        assert torch.allclose(p, full[ids], atol=1e-5)
        assert float(full[~ok].sum()) == 0.0


def test_batched_selectors_return_independent_sets():
    graphs = [M.Graph(n, tuple(e), w) for n, (e, w) in ((10, random_graph(10, 0.3, 21)), (7, random_graph(7, 0.4, 22)))]
    for fn in (M.rydberg_batch, M.anneal_batch):
        for g, x in zip(graphs, fn(graphs, shots=30, generator=torch.Generator().manual_seed(0))):
            assert x.shape == (g.n,) and M.is_independent(x, g.edges)
            assert weight(x, g.w) <= weight(M.exact(g.n, g.edges, g.w), g.w) + 1e-6


def test_exact_eps_samples_near_optimal_independent_sets():
    edges, w = random_graph(10, 0.3, 5)
    best = weight(M.exact(10, edges, w), w)
    g = torch.Generator().manual_seed(0)
    picks = [M.exact_eps(10, edges, w, 0.05, g) for _ in range(30)]
    assert all(M.is_independent(x, edges) and weight(x, w) >= 0.95 * best - 1e-6 for x in picks)
    assert weight(M.exact_eps(10, edges, w, 0.0, g), w) == pytest.approx(best)
