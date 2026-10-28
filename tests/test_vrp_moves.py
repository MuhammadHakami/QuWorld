import itertools
import random

import numpy as np
import pytest

from paper.qmethods import vrp_moves as V


def make(seed, n_cust=30, n_depot=3, open_route=False, limit=None, capacity=1.0):
    rng = np.random.default_rng(seed)
    inst = V.instance_from_arrays(rng.random((n_depot + n_cust, 2)), rng.uniform(0.05, 0.2, n_cust), capacity, n_depot,
                                  limit=limit, open_route=open_route)
    order = list(rng.permutation(np.arange(n_depot, n_depot + n_cust)))
    routes, cur, lo = [], [], 0.0
    for c in order:                                              # random capacity-feasible routes, random depots
        if lo + inst.demand[c] > capacity:
            routes.append(V.Route(int(rng.integers(n_depot)), tuple(cur))); cur, lo = [], 0.0
        cur.append(int(c)); lo += inst.demand[c]
    routes.append(V.Route(int(rng.integers(n_depot)), tuple(cur)))
    return inst, tuple(routes)


def random_independent_set(moves, edges, rng):
    nb = {i: set() for i in range(len(moves))}
    for i, j in edges:
        nb[i].add(j); nb[j].add(i)
    chosen = set()
    for i in rng.sample(range(len(moves)), len(moves)):
        if not nb[i] & chosen and rng.random() < 0.6:
            chosen.add(i)
    return sorted(chosen)


def test_split_join_roundtrip_and_multidepot_cost():
    inst, routes = make(0)
    assert V.split(V.join(routes), inst.n_depot) == routes
    r = routes[0]
    manual = inst.D[r.depot, r.customers[0]] + sum(inst.D[u, v] for u, v in zip(r.customers[:-1], r.customers[1:])) \
        + inst.D[r.customers[-1], r.depot]
    assert V.route_length(inst, r) == pytest.approx(manual)


def test_split_skips_empty_routes_and_padding():
    assert V.split([0, 5, 6, 0, 0, 1, 7, 0, 0], 3) == (V.Route(0, (5, 6)), V.Route(1, (7,)))
    with pytest.raises(ValueError):
        V.split([5, 0], 3)


@pytest.mark.parametrize("open_route,limit", [(False, None), (True, None), (False, 3.0), (True, 3.0)])
def test_independent_moves_add_exactly_and_stay_feasible(open_route, limit):
    rng = random.Random(1)
    for seed in range(6):
        inst, routes = make(seed, open_route=open_route, limit=None)
        if limit is not None:                                       # limit that the current routes satisfy
            inst = V.Instance(inst.D, inst.demand, inst.capacity, max(V.route_length(inst, r) for r in routes) + 0.05,
                              inst.open_route, inst.n_depot)
        moves = V.candidates(inst, routes, 60)
        assert moves and all(m.gain > 0 for m in moves)
        edges = V.conflict_edges(moves)
        for _ in range(10):
            s = random_independent_set(moves, edges, rng)
            new = V.apply(routes, [moves[i] for i in s])
            assert V.cost(inst, new) == pytest.approx(V.cost(inst, routes) - sum(moves[i].gain for i in s), abs=1e-9)
            assert V.feasible(inst, new)
            assert sorted(c for r in new for c in r.customers) == sorted(c for r in routes for c in r.customers)


def test_every_single_move_gain_is_exact():
    inst, routes = make(3, open_route=True)
    for m in V.candidates(inst, routes, 200):
        assert V.cost(inst, V.apply(routes, [m])) == pytest.approx(V.cost(inst, routes) - m.gain, abs=1e-9)


def test_conflicting_moves_are_rejected():
    inst, routes = make(4)
    moves = V.candidates(inst, routes, 60)
    i, j = V.conflict_edges(moves)[0]
    with pytest.raises(ValueError):
        V.apply(routes, [moves[i], moves[j]])


def test_improve_is_monotone():
    inst, routes = make(5)
    greedy = lambda moves, edges: [0]                               # best single move per round
    _, hist = V.improve(inst, routes, greedy, rounds=15, max_n=20)
    assert all(b < a for a, b in zip(hist[:-1], hist[1:]))


@pytest.mark.parametrize("open_route", [False, True])
def test_nearest_neighbour_serves_everyone_feasibly(open_route):
    inst, _ = make(6, open_route=open_route)
    inst = V.Instance(inst.D, inst.demand, inst.capacity, 2.5, inst.open_route, inst.n_depot)
    routes = V.nearest_neighbour(inst)
    assert sorted(c for r in routes for c in r.customers) == list(range(inst.n_depot, len(inst.D)))
    assert V.feasible(inst, routes)


def test_improve_batch_matches_improve_one_by_one():
    pairs = [make(s) for s in (7, 8, 9)]
    best1 = lambda moves, edges: [0]
    one = [V.improve(inst, r, best1, rounds=5, max_n=10)[0] for inst, r in pairs]
    many, calls = V.improve_batch([i for i, _ in pairs], [r for _, r in pairs], lambda jobs: [[0] for _ in jobs], 5, 10)
    assert many == one and calls == 15
