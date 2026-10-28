"""G0 gate benchmark: CUDA-Q statevector throughput on DGX Spark (GB10).

Times forward expectation-value evaluation of a hardware-efficient ansatz
(the shape of our adapter PQC: L layers of RY/RZ rotations + CZ ring)
at increasing qubit counts, on the GPU statevector target.
"""

import sys
import time

import cudaq
import numpy as np

QUBIT_SIZES = [12, 16, 20, 24, 26, 28]
LAYERS = 6
WARMUP_REPS = 1
TIMED_REPS = 3


def build_kernel(n: int, layers: int):
    kernel, params = cudaq.make_kernel(list)
    q = kernel.qalloc(n)
    idx = 0
    for _ in range(layers):
        for i in range(n):
            kernel.ry(params[idx], q[i])
            idx += 1
            kernel.rz(params[idx], q[i])
            idx += 1
        for i in range(n - 1):
            kernel.cz(q[i], q[i + 1])
    return kernel, idx


def hamiltonian(n: int):
    ham = cudaq.spin.z(0)
    for i in range(1, n):
        ham += cudaq.spin.z(i)
    return ham


def main() -> int:
    print(f"cudaq version: {cudaq.__version__}")
    try:
        cudaq.set_target("nvidia")
        print("target: nvidia (GPU statevector, fp32)")
    except RuntimeError as exc:
        print(f"FAILED to set nvidia target: {exc}")
        return 1

    print(f"gradient classes: {[g for g in dir(cudaq.gradients) if not g.startswith('_')]}")

    rng = np.random.default_rng(0)
    results = []
    for n in QUBIT_SIZES:
        kernel, n_params = build_kernel(n, LAYERS)
        ham = hamiltonian(n)
        theta = rng.uniform(-np.pi, np.pi, n_params).tolist()
        try:
            for _ in range(WARMUP_REPS):
                cudaq.observe(kernel, ham, theta)
            t0 = time.perf_counter()
            for _ in range(TIMED_REPS):
                ev = cudaq.observe(kernel, ham, theta).expectation()
            dt = (time.perf_counter() - t0) / TIMED_REPS
            results.append((n, n_params, dt, ev))
            print(f"n={n:2d}  params={n_params:4d}  forward observe: {dt*1e3:9.2f} ms  <H>={ev:+.4f}")
        except (RuntimeError, MemoryError) as exc:
            print(f"n={n:2d}  FAILED: {exc}")
            break

    # MPS backend availability check (for the chi-sweep dequantization ablation)
    try:
        cudaq.set_target("tensornet-mps")
        kernel, n_params = build_kernel(20, LAYERS)
        theta = rng.uniform(-np.pi, np.pi, n_params).tolist()
        t0 = time.perf_counter()
        cudaq.observe(kernel, hamiltonian(20), theta)
        print(f"tensornet-mps target: OK (20q observe {time.perf_counter()-t0:.2f}s)")
    except RuntimeError as exc:
        print(f"tensornet-mps target: FAILED: {exc}")

    print("\nCSV: n,params,forward_ms")
    for n, p, dt, _ in results:
        print(f"{n},{p},{dt*1e3:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
