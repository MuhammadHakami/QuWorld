"""Compute the CUDA-Q reference expectation value for the crosscheck.

Runs in its own process: cudaq and torch cannot coexist in one process on
this stack (LLVM CommandLine option registry clash). Prints <H> on the last
stdout line.
"""

import json
import sys

import cudaq


def main() -> int:
    payload = json.loads(open(sys.argv[1]).read())
    n, layers, flat = payload["n"], payload["layers"], payload["theta"]

    cudaq.set_target("nvidia")
    kernel, params = cudaq.make_kernel(list)
    q = kernel.qalloc(n)
    idx = 0
    for _ in range(layers):
        for i in range(n):
            kernel.ry(params[idx], q[i]); idx += 1
            kernel.rz(params[idx], q[i]); idx += 1
        for i in range(n - 1):
            kernel.cz(q[i], q[i + 1])
    ham = cudaq.spin.z(0)
    for i in range(1, n):
        ham += cudaq.spin.z(i)
    print(cudaq.observe(kernel, ham, flat).expectation())
    return 0


if __name__ == "__main__":
    sys.exit(main())
