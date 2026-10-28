"""Sub-study B of 33-allocation-strategy-study.md: analog vs digital inductive-bias probe for the tile cores.

    .venv/bin/python paper/alloc/bias_probe.py --target analog --omega 4 --core ryd --seed 1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import gpucap  # noqa: E402
from cores import _apply_1q, _bits, _rx, make_tile_core, n_params  # noqa: E402

gpucap.cap_from_env()
DEV = torch.device("cuda")
OUT = HERE.parents[1] / "runs" / "alloc" / "bias_probe"
N, F = 10, 4


def data(n: int, target: str, omega: float, seed: int):
    g = torch.Generator().manual_seed(1000 + seed)
    f = torch.rand(n, N, F, generator=g) * 2 - 1
    sites = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4), torch.arange(2), indexing="ij"), -1).reshape(-1, 3).float()
    pick = torch.stack([torch.randperm(len(sites), generator=g)[:N] for _ in range(n)])
    coords = torch.cat([sites[pick] * 2.0, torch.zeros(n, N, 1)], -1)                    # (n, N, 4): x, y, z, level=0
    gu = torch.Generator().manual_seed(7)
    u, v = torch.randn(F, generator=gu), torch.randn(F, generator=gu)
    u, v = u / u.norm(), v / v.norm()
    a, b = f @ u, (f @ v).mean(1, keepdim=True)
    if target == "analog":
        y = torch.sin(omega * a) * torch.cos(omega * b)
    elif target == "digital":
        y = torch.sign(a) * torch.sign(b)
    elif target == "parity":
        y = torch.sign(a) * torch.sign(torch.roll(a, -1, 1))
    elif target == "quantum":                            # doc 35 §21: interacting Rydberg quench driven by the inputs
        y = quantum_teacher(2.0 * a, sites[pick] * 2.0 / 2.2)
    else:
        raise ValueError(target)
    x = torch.cat([f, coords[..., :3] / 8.0], -1)                                         # every core sees the sites
    return x.to(DEV), coords.to(DEV), y.to(DEV)


@torch.no_grad()
def quantum_teacher(delta: torch.Tensor, pos: torch.Tensor, steps: int = 30, dt: float = 0.1, omega: float = 1.5) -> torch.Tensor:
    """<Z_i> after a Trotterized quench (doc 35 §21): per step, per-atom Rz(delta_i dt) Rx(omega dt), then the phase
    exp(-i dt sum_{i<j} V_ij n_i n_j) with V_ij = |r_i - r_j|^-6 (clamped at 0.5). delta (n, N), pos (n, N, 3)."""
    out = []
    bits = _bits(N).to(DEV); iu = torch.triu_indices(N, N, offset=1, device=DEV)
    pairbits = bits[:, iu[0]] * bits[:, iu[1]]; zdiag = 1.0 - 2.0 * bits
    for i in range(0, len(delta), 2000):
        d, p = delta[i:i + 2000].to(DEV), pos[i:i + 2000].to(DEV); G = len(d)
        V = torch.cdist(p, p).clamp(min=0.5).pow(-6)
        eng = (V[:, iu[0], iu[1]] @ pairbits.T).to(torch.complex64)
        phase = torch.exp(-1j * dt * eng)
        gx = _rx(torch.tensor(omega * dt, device=DEV)).expand(G, 2, 2)
        state = torch.zeros(G, 2 ** N, dtype=torch.complex64, device=DEV); state[:, 0] = 1.0
        for _ in range(steps):
            for k in range(N):
                ph = torch.exp(-0.5j * (d[:, k] * dt).to(torch.complex64))
                rz = torch.stack([ph, ph.conj()], -1).unsqueeze(-1)
                state = _apply_1q(state, rz * gx, k, N)
            state = state * phase
        out.append((state.abs().square() @ zdiag).cpu())
    return torch.cat(out)


class GNNProbe(nn.Module):
    """doc 35 §21 classical competitor with the teacher's exact pair interactions (ADR-C GNNCore, one round)."""

    def __init__(self, F, target):
        super().__init__()
        h = max(2, round((target - 3) / (3 * F + 10)))
        self.inp = nn.Linear(F, h); self.msg = nn.Linear(h, h); self.out = nn.Linear(2 * h, 3)

    def forward(self, x, coords):
        p = coords[..., :3] / 2.2
        V = torch.cdist(p, p).clamp(min=0.5).pow(-6); V = V - torch.diag_embed(torch.diagonal(V, dim1=1, dim2=2))
        h = torch.tanh(self.inp(x)); A = V / V.sum(-1, keepdim=True).clamp_min(1e-9)
        return torch.tanh(self.out(torch.cat([h, torch.tanh(self.msg(A @ h))], -1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="analog"); ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--core", default="ryd"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    tag = f"{a.target}{'' if a.target != 'analog' else f'_w{a.omega:g}'}_{a.core}_s{a.seed}"
    out = OUT / f"{tag}.json"
    if out.exists():
        return
    tr, va, te = data(20000, a.target, a.omega, a.seed), data(5000, a.target, a.omega, a.seed + 100), \
        data(10000, a.target, a.omega, a.seed + 200)
    torch.manual_seed(a.seed)
    if a.core == "mlp+":                                  # next hidden width up: never fewer parameters than the register
        import math
        from cores import GroupMLP, TileRegister
        f = F + 3; h = math.ceil((n_params(TileRegister(f)) - 3) / (2 * f + 4))
        core = GroupMLP(f, h * (2 * f + 4) + 3).to(DEV)
    elif a.core in ("gnn", "gnnX4"):
        from cores import TileRegister
        core = GNNProbe(F + 3, (4 if a.core == "gnnX4" else 1) * n_params(TileRegister(F + 3))).to(DEV)
    else:
        core = make_tile_core(a.core, F + 3).to(DEV)
    head = nn.Linear(3, 1).to(DEV)
    params = list(core.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    pred = lambda x, c: head(core(x, c)).squeeze(-1)

    @torch.no_grad()
    def r2(split):
        x, c, y = split
        p = torch.cat([pred(x[i:i + 1000], c[i:i + 1000]) for i in range(0, len(x), 1000)])
        return 1 - ((p - y) ** 2).sum().item() / ((y - y.mean()) ** 2).sum().item()

    g = torch.Generator(device="cpu").manual_seed(a.seed)
    best, state, t0 = -1e9, None, time.time()
    for step in range(1, a.steps + 1):
        b = torch.randint(0, 20000, (256,), generator=g).to(DEV)
        loss = (pred(tr[0][b], tr[1][b]) - tr[2][b]).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0:
            v = r2(va)
            if v > best:
                best, state = v, ([p.detach().clone() for p in params])
    with torch.no_grad():
        for p, s in zip(params, state):
            p.copy_(s)
    res = {"target": a.target, "omega": a.omega, "core": a.core, "seed": a.seed, "valid_r2": best, "test_r2": r2(te),
           "params_core": n_params(core), "seconds": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(f"[probe] {tag}: TEST R2 {res['test_r2']:.4f} (core {res['params_core']} params, {res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
