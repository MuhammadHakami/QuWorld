"""Transformer Engine NVFP4 (NVFP4BlockScaling: fused quantization, RHT, 2D weight scaling) vs bf16: te.Linear fwd + bwd at VisionHOPE-T
stage shapes (batch 16, pooled low-pass input) + one large GEMM; relative L2 of the NVFP4 forward vs bf16."""
import time

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling, Float8CurrentScaling

ac = getattr(te, "autocast", None)


def ctx(recipe):
    return ac(enabled=True, recipe=recipe) if ac else te.fp8_autocast(enabled=True, fp8_recipe=recipe)


def bench(M, K, N, recipe, reps=20):
    lin = te.Linear(K, N, bias=True, params_dtype=torch.bfloat16).cuda(); x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    def run():
        if recipe is None:
            y = lin(x)
        else:
            with ctx(recipe):
                y = lin(x)
        y.float().pow(2).mean().backward(); return y
    for _ in range(3):
        run()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        run()
    torch.cuda.synchronize(); return (time.time() - t0) / reps * 1e3, lin, x


for tag, M, K, N in (("stage1 FFN", 16 * 3916, 64, 256), ("stage2 FFN", 16 * 979, 128, 512), ("stage3 FFN", 16 * 272, 256, 1024),
                     ("stage4 FFN", 16 * 80, 512, 2048), ("big", 8192, 4096, 4096)):
    t_bf, lin, x = bench(M, K, N, None); row = [f"bf16 {t_bf:.2f} ms"]
    for name, rec in (("fp8", Float8CurrentScaling()), ("nvfp4", NVFP4BlockScaling())):
        try:
            t, _, _ = bench(M, K, N, rec)
            with torch.no_grad():
                ref = lin(x).float()
                with ctx(rec):
                    q = lin(x).float()
            row.append(f"{name} {t:.2f} ms (x{t_bf / t:.2f}, rel-L2 {((q - ref).norm() / ref.norm()).item():.3f})")
        except Exception as e:
            row.append(f"{name} FAIL {str(e)[:80]}")
    print(f"[te] {tag:>11} M {M:>6} K {K:>4} N {N:>4}: " + " | ".join(row), flush=True)
