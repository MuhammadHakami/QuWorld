"""Route-improvement moves for the MWIS move-selection co-processor (doc 35 §28, M1), with RouteFinder's multi-depot cost
(routefinder/envs/mtdvrp: a route leaves a depot and returns to that same depot unless routes are open; depot to depot
costs nothing).

A solution is a tuple of Routes. Candidate moves are intra-route 2-opt and inter-route relocate, each with its exact cost
change and capacity / distance-limit feasibility. Two moves conflict when applying both would not add their cost changes
exactly (2-opt segments on one route that overlap or touch; any move sharing a route with a relocate). Every independent
set of the conflict graph can therefore be applied at once, and its gain is the sum of the vertex weights.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class Instance:
    D: np.ndarray                 # (N, N) Euclidean distances; nodes 0..n_depot-1 are depots
    demand: np.ndarray            # (N,) linehaul demand, 0 at depots
    capacity: float
    limit: float | None           # route distance limit (L variants), None if absent
    open_route: bool
    n_depot: int


@dataclass(frozen=True)
class Route:
    depot: int
    customers: tuple[int, ...]


@dataclass(frozen=True)
class Move:
    kind: str                     # "2opt" | "relocate"
    routes: tuple[int, ...]       # route indices touched
    a: int                        # 2opt: first reversed position (1-based in the path); relocate: position in route A
    b: int                        # 2opt: last reversed position; relocate: insertion slot in route B
    gain: float                   # cost decrease (> 0)


def instance_from_arrays(locs, demand_linehaul, capacity, n_depot, limit=None, open_route=False) -> Instance:
    locs = np.asarray(locs, dtype=np.float64)
    D = np.sqrt(((locs[:, None, :] - locs[None, :, :]) ** 2).sum(-1))
    dem = np.zeros(len(locs)); dem[n_depot:] = np.asarray(demand_linehaul, dtype=np.float64)[-(len(locs) - n_depot):]
    return Instance(D, dem, float(capacity), None if limit is None else float(limit), bool(open_route), int(n_depot))


def split(actions, n_depot: int) -> tuple[Route, ...]:
    routes, depot, cur = [], None, []
    for a in (int(v) for v in actions):
        if a < n_depot:
            if depot is not None and cur:
                routes.append(Route(depot, tuple(cur)))
            depot, cur = a, []
        else:
            if depot is None:
                raise ValueError("action sequence must start at a depot")
            cur.append(a)
    if depot is not None and cur:
        routes.append(Route(depot, tuple(cur)))
    return tuple(routes)


def join(routes: tuple[Route, ...]) -> list[int]:
    out: list[int] = []
    for r in routes:
        out += [r.depot, *r.customers]
    return out + [routes[0].depot if routes else 0]


def _path(inst: Instance, r: Route) -> list[int]:
    return [r.depot, *r.customers] + ([] if inst.open_route else [r.depot])


def route_length(inst: Instance, r: Route) -> float:
    p = _path(inst, r)
    return float(sum(inst.D[u, v] for u, v in zip(p[:-1], p[1:])))


def cost(inst: Instance, routes: tuple[Route, ...]) -> float:
    return sum(route_length(inst, r) for r in routes)


def load(inst: Instance, r: Route) -> float:
    return float(inst.demand[list(r.customers)].sum())


def feasible(inst: Instance, routes: tuple[Route, ...]) -> bool:
    for r in routes:
        if load(inst, r) > inst.capacity + 1e-6:
            return False
        if inst.limit is not None and route_length(inst, r) > inst.limit + 1e-6:
            return False
    return True


def nearest_neighbour(inst: Instance) -> tuple[Route, ...]:
    """Classical constructor (the no-policy baseline for the credit split): each route starts at the depot nearest to the
    unvisited customer closest to any depot and extends to the nearest customer that keeps it feasible."""
    D, n_dep = inst.D, inst.n_depot
    todo, routes = set(range(n_dep, len(D))), []
    while todo:
        c0 = min(todo, key=lambda c: D[:n_dep, c].min())
        depot = int(D[:n_dep, c0].argmin())
        cur, lo, length, route = depot, 0.0, 0.0, []
        while True:
            back = lambda c: 0.0 if inst.open_route else D[c, depot]
            ok = [c for c in todo if lo + inst.demand[c] <= inst.capacity + 1e-6
                  and (inst.limit is None or length + D[cur, c] + back(c) <= inst.limit + 1e-6)]
            if not ok:
                break
            c = min(ok, key=lambda c: D[cur, c])
            route.append(c); todo.discard(c); lo += inst.demand[c]; length += D[cur, c]; cur = c
        if not route:
            raise ValueError(f"customer {c0} cannot be served by a single route")
        routes.append(Route(depot, tuple(route)))
    return tuple(routes)


def two_opt_moves(inst: Instance, routes: tuple[Route, ...]) -> list[Move]:
    D, out = inst.D, []
    for i, r in enumerate(routes):
        p, k = _path(inst, r), len(r.customers)
        for a, b in itertools.combinations(range(1, k + 1), 2):
            nxt = p[b + 1] if b + 1 < len(p) else None
            delta = D[p[a - 1], p[b]] - D[p[a - 1], p[a]] + (D[p[a], nxt] - D[p[b], nxt] if nxt is not None else 0.0)
            if delta < -EPS:                                  # shorter route: the distance limit stays satisfied
                out.append(Move("2opt", (i,), a, b, -float(delta)))
    return out


def relocate_moves(inst: Instance, routes: tuple[Route, ...]) -> list[Move]:
    D, out = inst.D, []
    lengths = [route_length(inst, r) for r in routes]
    loads = [load(inst, r) for r in routes]
    for ia, ra in enumerate(routes):
        ca = ra.customers
        for pos, c in enumerate(ca):
            prev = ra.depot if pos == 0 else ca[pos - 1]
            nxt = ca[pos + 1] if pos + 1 < len(ca) else (None if inst.open_route else ra.depot)
            d_rem = (D[prev, nxt] - D[prev, c] - D[c, nxt]) if nxt is not None else -D[prev, c]
            for ib, rb in enumerate(routes):
                if ib == ia or loads[ib] + inst.demand[c] > inst.capacity + 1e-6:
                    continue
                cb, best = rb.customers, None
                for q in range(len(cb) + 1):
                    p_ = rb.depot if q == 0 else cb[q - 1]
                    n_ = cb[q] if q < len(cb) else (None if inst.open_route else rb.depot)
                    d_ins = D[p_, c] + (D[c, n_] - D[p_, n_] if n_ is not None else 0.0)
                    if inst.limit is not None and lengths[ib] + d_ins > inst.limit + 1e-6:
                        continue
                    if best is None or d_ins < best[1]:
                        best = (q, d_ins)
                if best is not None and d_rem + best[1] < -EPS:
                    out.append(Move("relocate", (ia, ib), pos, best[0], -float(d_rem + best[1])))
    return out


def candidates(inst: Instance, routes: tuple[Route, ...], max_n: int) -> list[Move]:
    moves = two_opt_moves(inst, routes) + relocate_moves(inst, routes)
    return sorted(moves, key=lambda m: -m.gain)[:max_n]


def _clash(m: Move, o: Move) -> bool:
    if not set(m.routes) & set(o.routes):
        return False
    if m.kind == "2opt" and o.kind == "2opt":                 # same route: must be disjoint and not touch
        return not (m.b + 1 < o.a or o.b + 1 < m.a)
    return True


def conflict_edges(moves: list[Move]) -> list[tuple[int, int]]:
    return [(i, j) for i, j in itertools.combinations(range(len(moves)), 2) if _clash(moves[i], moves[j])]


def apply(routes: tuple[Route, ...], moves: list[Move]) -> tuple[Route, ...]:
    """Apply an independent set of moves (checked) and drop routes left empty."""
    for m, o in itertools.combinations(moves, 2):
        if _clash(m, o):
            raise ValueError("moves conflict")
    cust = [list(r.customers) for r in routes]
    for m in moves:
        if m.kind == "2opt":
            (i,) = m.routes
            cust[i][m.a - 1:m.b] = cust[i][m.a - 1:m.b][::-1]
    for m in moves:
        if m.kind == "relocate":
            ia, ib = m.routes
            c = routes[ia].customers[m.a]
            cust[ia].remove(c)
            cust[ib].insert(m.b, c)
    return tuple(Route(r.depot, tuple(c)) for r, c in zip(routes, cust) if c)


def improve_batch(insts: list[Instance], sols: list[tuple[Route, ...]],
                  select: Callable[[list[tuple[list[Move], list[tuple[int, int]]]]], list[list[int]]],
                  rounds: int, max_n: int) -> tuple[list[tuple[Route, ...]], int]:
    """`improve` for many instances with one batched selector call per round; returns solutions and selector calls."""
    sols, calls = list(sols), 0
    for _ in range(rounds):
        todo = [(k, candidates(insts[k], sols[k], max_n)) for k in range(len(sols))]
        todo = [(k, m) for k, m in todo if m]
        if not todo:
            break
        picks = select([(m, conflict_edges(m)) for _, m in todo]); calls += len(todo)
        for (k, m), chosen in zip(todo, picks):
            sols[k] = apply(sols[k], [m[i] for i in chosen])
    return sols, calls


def improve(inst: Instance, routes: tuple[Route, ...], select: Callable[[list[Move], list[tuple[int, int]]], list[int]],
            rounds: int, max_n: int) -> tuple[tuple[Route, ...], list[float]]:
    """`rounds` selector calls; each applies the selected independent set of the top-`max_n` improving moves."""
    history = [cost(inst, routes)]
    for _ in range(rounds):
        moves = candidates(inst, routes, max_n)
        if not moves:
            break
        chosen = select(moves, conflict_edges(moves))
        routes = apply(routes, [moves[i] for i in chosen])
        history.append(cost(inst, routes))
    return routes, history
