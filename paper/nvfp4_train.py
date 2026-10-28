"""NVFP4 TRAINING linear (all three GEMMs in NVFP4 on sm_12x via torch scaled_mm; paper/nvfp4.py quantizer, round-to-nearest) and a
fwd + bwd microbenchmark against bf16 at given shapes. Forward y = Q(x) Q(W)^T; dX = Q(dY) Q(W^T)^T over N; dW = Q(dY^T) Q(X^T)^T over
the token axis. No stochastic rounding / Hadamard transform (the full NVFP4 pretraining recipe adds both for gradient bias) - this is
the throughput ceiling test: if it is not faster than bf16 at the model's real shapes, NVFP4 training cannot pay off there.

    python paper/nvfp4_train.py            (VisionHOPE-T stage shapes at batch 16, pooled low-pass input)
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from nvfp4 import quantize_nvfp4

ST, SW = F.ScalingType, F.SwizzleType


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [M, K] @ b [N, K]^T with both operands quantized to NVFP4 along K -> bf16 [M, N]."""
    ad, asc, ag = quantize_nvfp4(a); bd, bsc, bg = quantize_nvfp4(b)
    return F.scaled_mm(ad, bd.t(), [asc, ag], [ST.BlockWise1x16, ST.TensorWise], [bsc, bg], [ST.BlockWise1x16, ST.TensorWise],
                       swizzle_a=[SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE], swizzle_b=[SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE], output_dtype=torch.bfloat16)


class NVFP4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):                                                     # x [M, K], w [N, K]
        ctx.save_for_backward(x, w); return _mm(x, w)

    @staticmethod
    def backward(ctx, dy):                                                      # dy [M, N]
        x, w = ctx.saved_tensors
        dx = _mm(dy.contiguous(), w.t().contiguous())                            # [M, N] @ [K, N]^T -> [M, K]
        dw = _mm(dy.t().contiguous(), x.t().contiguous())                        # [N, M] @ [K, M]^T -> [N, K]
        return dx, dw


def bench(M: int, K: int, N: int, reps: int = 20) -> dict:
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, requires_grad=True) * 0.05
    w = w.detach().requires_grad_(True); out = {}
    for name, f in (("bf16", lambda: (x @ w.t())), ("nvfp4", lambda: NVFP4LinearFn.apply(x, w))):
        for _ in range(3):
            f().float().pow(2).mean().backward()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(reps):
            f().float().pow(2).mean().backward()
        torch.cuda.synchronize(); out[name] = (time.time() - t0) / reps * 1e3
    with torch.no_grad():
        ref = (x @ w.t()).float(); q = NVFP4LinearFn.apply(x, w).float(); out["rel_l2"] = ((q - ref).norm() / ref.norm()).item()
    return out


if __name__ == "__main__":
    for tag, M, K, N in (("stage1 FFN up", 16 * 3916, 64, 256), ("stage2 FFN up", 16 * 979, 128, 512), ("stage3 FFN up", 16 * 270, 256, 1024),
                         ("stage4 FFN up", 16 * 69, 512, 2048), ("stage1 1x1 proj", 16 * 3916, 64, 32), ("big (LLM-like)", 8192, 4096, 4096)):
        r = bench(M, K, N)
        print(f"[nvfp4] {tag:>16} M {M:>6} K {K:>4} N {N:>4}: bf16 {r['bf16']:.2f} ms  nvfp4 {r['nvfp4']:.2f} ms  "
              f"speedup x{r['bf16'] / r['nvfp4']:.2f}  fwd rel-L2 {r['rel_l2']:.3f}", flush=True)
