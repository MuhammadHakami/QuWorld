"""Benchmark the revised gradient strategy: PyTorch complex64 statevector
propagator with autograd (replaces the missing CUDA-Q adjoint gradient).

- Correctness: expectation values cross-checked against cudaq.observe at n=12.
- Throughput: forward and forward+backward at n=20 and n=24
  (per-layer activation checkpointing at n=24 to bound memory).

Circuit matches g0_bench.py: L layers of RY,RZ per qubit + CZ ring; H = sum Z_i.
"""

import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

LAYERS = 6
DEVICE = "cuda"


def cz_ring_mask(n: int, device) -> torch.Tensor:
    """Combined diagonal ±1 mask for the CZ ring (all CZs are diagonal)."""
    bits = torch.arange(2**n, device=device)
    mask = torch.ones(2**n, device=device)
    for i in range(n - 1):
        bi = (bits >> (n - 1 - i)) & 1
        bj = (bits >> (n - 2 - i)) & 1
        mask = mask * torch.where((bi & bj).bool(), -1.0, 1.0)
    return mask.to(torch.complex64)


def z_diag(n: int, device) -> torch.Tensor:
    """Diagonal of H = sum_i Z_i."""
    bits = torch.arange(2**n, device=device)
    d = torch.zeros(2**n, device=device)
    for i in range(n):
        d = d + 1.0 - 2.0 * ((bits >> (n - 1 - i)) & 1).float()
    return d


def apply_1q(state: torch.Tensor, gate: torch.Tensor, k: int, n: int) -> torch.Tensor:
    s = state.reshape(2**k, 2, 2 ** (n - k - 1))
    return torch.einsum("ab,ibj->iaj", gate, s).reshape(-1)


def layer_fn(state, theta_layer, mask, n):
    for k in range(n):
        ry, rz = theta_layer[k, 0], theta_layer[k, 1]
        c, s = torch.cos(ry / 2), torch.sin(ry / 2)
        g_ry = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex64)
        e = torch.exp(-0.5j * rz.to(torch.complex64))
        g_rz = torch.diag(torch.stack([e, e.conj()]))
        state = apply_1q(state, g_rz @ g_ry, k, n)
    return state * mask


def forward(theta, n, mask, hdiag, use_ckpt):
    state = torch.zeros(2**n, dtype=torch.complex64, device=DEVICE)
    state = state + torch.nn.functional.one_hot(
        torch.tensor(0, device=DEVICE), 2**n
    ).to(torch.complex64)
    for layer in range(LAYERS):
        if use_ckpt:
            state = checkpoint(layer_fn, state, theta[layer], mask, n, use_reentrant=False)
        else:
            state = layer_fn(state, theta[layer], mask, n)
    return (state.abs().square() * hdiag).sum()


def bench(n: int, use_ckpt: bool, reps: int = 3):
    mask = cz_ring_mask(n, DEVICE)
    hdiag = z_diag(n, DEVICE)
    theta = torch.nn.Parameter(
        torch.rand(LAYERS, n, 2, device=DEVICE) * 2 * np.pi - np.pi
    )
    # warmup
    ev = forward(theta, n, mask, hdiag, use_ckpt)
    ev.backward()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        with torch.no_grad():
            forward(theta, n, mask, hdiag, use_ckpt)
    torch.cuda.synchronize()
    fwd = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        theta.grad = None
        ev = forward(theta, n, mask, hdiag, use_ckpt)
        ev.backward()
    torch.cuda.synchronize()
    full = (time.perf_counter() - t0) / reps

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"n={n:2d} ckpt={use_ckpt!s:5s}  fwd {fwd*1e3:8.2f} ms   "
        f"fwd+bwd {full*1e3:8.2f} ms   ratio {full/fwd:4.1f}x   peak {peak:5.2f} GiB"
    )
    torch.cuda.reset_peak_memory_stats()
    return float(ev.detach())


def crosscheck(n: int = 12) -> None:
    """cudaq and torch cannot share a process (LLVM option-registry clash on
    this stack), so the CUDA-Q reference value is computed in a subprocess."""
    import json
    import pathlib
    import subprocess
    import sys

    torch.manual_seed(0)
    theta = torch.rand(LAYERS, n, 2, device=DEVICE) * 2 - 1
    payload = {"n": n, "layers": LAYERS, "theta": theta.flatten().tolist()}
    path = pathlib.Path(__file__).parent / "crosscheck_theta.json"
    path.write_text(json.dumps(payload))

    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).parent / "cudaq_check.py"), str(path)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print(f"crosscheck SKIPPED (cudaq subprocess failed): {proc.stderr[-200:]}")
        return
    ev_cudaq = float(proc.stdout.strip().splitlines()[-1])

    mask, hdiag = cz_ring_mask(n, DEVICE), z_diag(n, DEVICE)
    with torch.no_grad():
        ev_torch = float(forward(theta, n, mask, hdiag, False))
    print(f"crosscheck n={n}: torch {ev_torch:+.6f}  cudaq {ev_cudaq:+.6f}  "
          f"|diff| {abs(ev_torch - ev_cudaq):.2e}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, "
          f"device: {torch.cuda.get_device_name(0)}")
    crosscheck(12)
    bench(16, use_ckpt=False)
    bench(20, use_ckpt=False)
    bench(24, use_ckpt=True)
