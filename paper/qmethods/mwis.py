"""Maximum-weight independent set (MWIS) selectors for the move-selection co-processor (doc 35 §28, M1).

A conflict graph (n <= 22 vertices, edge list, weights w > 0) goes in; every selector returns an independent set as a
0/1 tensor. Only `rydberg` is quantum; the other three are its matched classical controls.

  rydberg  adiabatic sweep on an ideal Rydberg-blockade array, H(t) = Omega(t)/2 sum_i X_i - Delta(t) sum_i w_i n_i / w_max,
           restricted to independent sets (the blockade subspace: P X_i P only couples two independent sets that differ
           in vertex i), exact state vector, symmetric Trotter steps; `shots` bitstrings are drawn from the final state
           and the heaviest one is kept, as on hardware.
  greedy   heaviest vertex first, its neighbours removed.
  exact    heaviest independent set by enumeration (the ceiling every selector is measured against).
  anneal   simulated annealing over independent sets (flip in, flip out), `shots` parallel chains of 20 n moves,
           best one kept.

The *_batch functions run many graphs in one pass (one GPU launch sequence per round of the routing loop): the Rydberg
state is stored on the independent sets only, graphs stacked block-diagonally; annealing pads graphs with isolated
zero-weight vertices that are never sampled. Bit convention: vertex i is bit (n - 1 - i) of the basis index.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

MAX_N = 22
MAX_STATES = 1 << 23                   # independent sets per rydberg_batch chunk (memory bound)


@dataclass(frozen=True)
class Sweep:
    """Adiabatic schedule in units of 1/Omega_max: Omega = sin^2(pi t / T), Delta ramps from -delta to +delta."""
    T: float = 40.0
    steps: int = 400
    delta: float = 3.0


@dataclass(frozen=True)
class Graph:
    n: int
    edges: tuple[tuple[int, int], ...]
    w: torch.Tensor


def _check(n: int, edges, w: torch.Tensor) -> None:
    if not 1 <= n <= MAX_N:
        raise ValueError(f"n = {n} outside [1, {MAX_N}]")
    if w.shape != (n,) or bool((w <= 0).any()):
        raise ValueError("weights must be a positive vector of length n")
    for i, j in edges:
        if not (0 <= i < n and 0 <= j < n and i != j):
            raise ValueError(f"bad edge ({i}, {j})")


def _bit(idx: torch.Tensor, i: int, n: int) -> torch.Tensor:
    return (idx >> (n - 1 - i)) & 1


def independent_mask(n: int, edges, device="cpu") -> torch.Tensor:
    idx = torch.arange(2 ** n, device=device)
    ok = torch.ones(2 ** n, dtype=torch.bool, device=device)
    for i, j in edges:
        ok &= ~((_bit(idx, i, n) == 1) & (_bit(idx, j, n) == 1))
    return ok


def set_weight(idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    n = w.shape[0]
    tot = torch.zeros(idx.shape, dtype=w.dtype, device=idx.device)
    for i in range(n):
        tot = tot + _bit(idx, i, n).to(w.dtype) * w[i]
    return tot


def to_vector(index: int, n: int) -> torch.Tensor:
    return torch.tensor([(index >> (n - 1 - i)) & 1 for i in range(n)], dtype=torch.long)


def is_independent(x: torch.Tensor, edges) -> bool:
    return all(not (int(x[i]) and int(x[j])) for i, j in edges)


def exact(n: int, edges, w: torch.Tensor) -> torch.Tensor:
    _check(n, edges, w)
    idx = torch.arange(2 ** n, device=w.device)
    val = torch.where(independent_mask(n, edges, w.device), set_weight(idx, w), torch.full_like(w[:1], -1.0).expand(2 ** n))
    return to_vector(int(val.argmax()), n)


def exact_eps(n: int, edges, w: torch.Tensor, eps: float = 0.01, generator: torch.Generator | None = None) -> torch.Tensor:
    """Control for sampling-driven diversity (doc 35 §28 addendum): an independent set drawn uniformly among those within
    (1 - eps) of the optimum weight. eps = 0 recovers `exact` up to ties."""
    _check(n, edges, w)
    idx = torch.arange(2 ** n, device=w.device)
    val = torch.where(independent_mask(n, edges, w.device), set_weight(idx, w), torch.full_like(w[:1], -1.0).expand(2 ** n))
    cand = torch.nonzero(val >= (1 - eps) * val.max()).flatten()
    pick = torch.randint(len(cand), (1,), device=w.device, generator=generator)
    return to_vector(int(cand[pick]), n)


def greedy(n: int, edges, w: torch.Tensor) -> torch.Tensor:
    _check(n, edges, w)
    nbrs = [set() for _ in range(n)]
    for i, j in edges:
        nbrs[i].add(j); nbrs[j].add(i)
    x, blocked = torch.zeros(n, dtype=torch.long), set()
    for v in torch.argsort(w.cpu(), descending=True).tolist():
        if v not in blocked:
            x[v] = 1; blocked |= nbrs[v] | {v}
    return x


def _schedule(sweep: Sweep):
    dt = sweep.T / sweep.steps
    for s in range(sweep.steps):
        t = (s + 0.5) * dt
        yield dt, math.sin(math.pi * t / sweep.T) ** 2, sweep.delta * (2 * t / sweep.T - 1)


def rydberg_state(n: int, edges, w: torch.Tensor, sweep: Sweep = Sweep()) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation on the full 2^n space: final state and the independent-set mask."""
    _check(n, edges, w)
    idx = torch.arange(2 ** n, device=w.device)
    ok = independent_mask(n, edges, w.device)
    energy = set_weight(idx, w / w.max())
    pairs = [(idx ^ (1 << (n - 1 - i)), ok & ok[idx ^ (1 << (n - 1 - i))]) for i in range(n)]
    psi = torch.zeros(2 ** n, dtype=torch.complex64, device=w.device); psi[0] = 1.0   # empty set: ground state at Delta < 0
    for dt, omega, delta in _schedule(sweep):
        half = torch.exp(0.5j * delta * dt * energy).to(torch.complex64)             # exp(-i H_diag dt/2), H_diag = -Delta E
        psi = psi * half
        c, sn = math.cos(0.5 * omega * dt), math.sin(0.5 * omega * dt)
        for partner, both in pairs:
            psi = torch.where(both, c * psi - 1j * sn * psi[partner], psi)
        psi = psi * half
    return psi, ok


def _subspace(g: Graph) -> tuple[torch.Tensor, list[torch.Tensor]]:
    dev = g.w.device
    ids = torch.nonzero(independent_mask(g.n, g.edges, dev)).flatten()               # sorted: ids[0] = 0, the empty set
    lookup = torch.full((2 ** g.n,), -1, dtype=torch.long, device=dev)
    lookup[ids] = torch.arange(len(ids), device=dev)
    return ids, [lookup[ids ^ (1 << (g.n - 1 - i))] for i in range(g.n)]


def rydberg_probs_batch(graphs: list[Graph], sweep: Sweep = Sweep()) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Blockade-subspace evolution of every graph at once; returns (independent sets, probabilities) per graph."""
    for g in graphs:
        _check(g.n, g.edges, g.w)
    dev = graphs[0].w.device
    blocks = [_subspace(g) for g in graphs]
    sizes = [len(ids) for ids, _ in blocks]
    offs = torch.tensor([0] + sizes[:-1], device=dev).cumsum(0)
    n_max = max(g.n for g in graphs)
    energy = torch.cat([set_weight(ids, g.w / g.w.max()) for g, (ids, _) in zip(graphs, blocks)])
    partner = []
    for i in range(n_max):
        parts = [torch.where(p[i] >= 0, p[i] + int(o), -1) if i < g.n else torch.full((m,), -1, device=dev)
                 for g, (_, p), o, m in zip(graphs, blocks, offs.tolist(), sizes)]
        partner.append(torch.cat(parts))
    valid = [p >= 0 for p in partner]; pidx = [p.clamp(min=0) for p in partner]
    psi = torch.zeros(sum(sizes), dtype=torch.complex64, device=dev); psi[offs] = 1.0
    for dt, omega, delta in _schedule(sweep):
        half = torch.exp(0.5j * delta * dt * energy).to(torch.complex64)
        psi = psi * half
        c, sn = math.cos(0.5 * omega * dt), math.sin(0.5 * omega * dt)
        for v, p in zip(valid, pidx):
            psi = torch.where(v, c * psi - 1j * sn * psi[p], psi)
        psi = psi * half
    probs = psi.abs().square().real.split(sizes)
    return [(ids, pr) for (ids, _), pr in zip(blocks, probs)]


def rydberg_batch(graphs: list[Graph], shots: int = 100, sweep: Sweep = Sweep(),
                  generator: torch.Generator | None = None) -> list[torch.Tensor]:
    out, chunk, states = [], [], 0
    def flush():
        for g, (ids, p) in zip(chunk, rydberg_probs_batch(chunk, sweep)):
            draws = ids[torch.multinomial(p / p.sum(), shots, replacement=True, generator=generator)]
            out.append(to_vector(int(draws[set_weight(draws, g.w).argmax()]), g.n))
    for g in graphs:
        est = 2 ** g.n                                     # upper bound on this graph's independent sets
        if chunk and states + est > MAX_STATES:
            flush(); chunk, states = [], 0
        chunk.append(g); states += est
    if chunk:
        flush()
    return out


def anneal_batch(graphs: list[Graph], shots: int = 100, moves: int | None = None,
                 generator: torch.Generator | None = None) -> list[torch.Tensor]:
    """`shots` Metropolis chains per graph over independent sets; flipping a vertex in evicts its neighbours."""
    for g in graphs:
        _check(g.n, g.edges, g.w)
    dev, B = graphs[0].w.device, len(graphs)
    n = max(g.n for g in graphs)
    A = torch.zeros(B, n, n, device=dev); W = torch.zeros(B, n, device=dev)
    for b, g in enumerate(graphs):
        for i, j in g.edges:
            A[b, i, j] = A[b, j, i] = 1.0
        W[b, :g.n] = g.w
    nb = torch.tensor([g.n for g in graphs], device=dev)
    moves = moves or 20 * n
    t0 = W.max(1).values; decay = 0.01 ** (torch.arange(moves, device=dev) / max(moves - 1, 1))
    x = torch.zeros(B, shots, n, device=dev)
    best, best_val = x.clone(), torch.zeros(B, shots, device=dev)
    V = (torch.rand(moves, B, shots, device=dev, generator=generator) * nb[None, :, None]).long()
    U = torch.rand(moves, B, shots, device=dev, generator=generator)
    bi = torch.arange(B, device=dev)[:, None]
    for m in range(moves):
        v, temp = V[m], (t0 * decay[m])[:, None]
        inside = x.gather(-1, v[..., None]).squeeze(-1) > 0
        evict = A[bi, v] * x                                                # (B, S, n): neighbours of v in the set
        wv = W.gather(1, v)
        gain = torch.where(inside, -wv, wv - (evict * W[:, None, :]).sum(-1))
        accept = (gain >= 0) | (U[m] < torch.exp(gain / temp))
        onehot = torch.nn.functional.one_hot(v, n).float()
        new = torch.where(inside[..., None], x * (1 - onehot), x * (1 - evict) * (1 - onehot) + onehot)
        x = torch.where(accept[..., None], new, x)
        val = (x * W[:, None, :]).sum(-1)
        better = val > best_val
        best = torch.where(better[..., None], x, best); best_val = torch.maximum(val, best_val)
    top = best_val.argmax(1)
    return [best[b, top[b], :g.n].round().long().cpu() for b, g in enumerate(graphs)]


def rydberg(n: int, edges, w: torch.Tensor, shots: int = 100, sweep: Sweep = Sweep(),
            generator: torch.Generator | None = None) -> torch.Tensor:
    return rydberg_batch([Graph(n, tuple(edges), w)], shots, sweep, generator)[0]


def anneal(n: int, edges, w: torch.Tensor, shots: int = 100, moves: int | None = None,
           generator: torch.Generator | None = None) -> torch.Tensor:
    return anneal_batch([Graph(n, tuple(edges), w)], shots, moves, generator)[0]


SELECTORS = {"rydberg": rydberg, "greedy": greedy, "exact": exact, "anneal": anneal, "exact_eps": exact_eps}
