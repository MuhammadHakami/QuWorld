"""NVFP4 weight-only linear layers on native PyTorch (no torchao / modelopt: neither
ships an aarch64 wheel for this GB10).

Recipe = the NVFP4 case of torch's own `scaled_mm` op tests (torch 2.13, sm_12x):
  data   : float4_e2m1fn_x2 (two e2m1 values per byte), [rows, K // 2]
  scales : per 16-element block along K, float8_e4m3fn, in the cuBLAS 32x4x4
           swizzled block layout (`to_blocked`), plus one tensor-wise fp32 global
           scale -> ScalingType.BlockWise1x16 + TensorWise.
Weights are quantized once at load (static); activations are quantized per call
(dynamic, same recipe). Bias is added in bf16 by scaled_mm.

`quantize_linears_(module, ...)` swaps every nn.Linear whose in_features is a
multiple of 16 (the DiT's 5120/13824 all qualify) for an NVFP4Linear in place and
returns a report; `fidelity_check()` measures the relative L2 error of one layer
against bf16 so every use records its delta (user directive Sep 21 2026: NVFP4
wherever supported, same precision across arms, fidelity recorded).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK = 16
F4_MAX = 6.0                                    # e2m1 max
F8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)   # 448


_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])   # code i -> magnitude
_E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])   # decision boundaries


def _f32_to_e2m1_unpacked(x: torch.Tensor) -> torch.Tensor:
    """fp32 (already clamped to +-6) -> e2m1 codes (uint8, one value per byte, bits 0-3):
    sign bit 3, magnitude code bits 0-2. Round to nearest representable value; an
    exact midpoint goes to the even code of the two neighbours (IEEE ties-to-even)."""
    mags = x.abs()
    vals = _E2M1_VALUES.to(x.device)
    mids = _E2M1_MIDPOINTS.to(x.device)
    code = torch.searchsorted(mids, mags, right=True)                  # #midpoints strictly below mags -> nearest code
    on_mid = (mags.unsqueeze(-1) == mids).any(-1)                      # exact tie: searchsorted picked the upper code
    code = torch.where(on_mid & (code % 2 == 1), code - 1, code)       # tie -> even code
    return (code.to(torch.uint8) | ((x < 0).to(torch.uint8) << 3))


def _pack_uint4(u8: torch.Tensor) -> torch.Tensor:
    shape = u8.shape
    u8 = u8.reshape(-1)
    return (u8[1::2] << 4 | u8[::2]).view(*shape[:-1], shape[-1] // 2)


def _to_blocked(m: torch.Tensor) -> torch.Tensor:
    """cuBLAS 32x4x4 block-scaling layout (SwizzleType.SWIZZLE_32_4_4); (H, W) ->
    flat (32*ceil(H/128)) x (16*ceil(W/4)). Same construction as torch's test helper."""
    rows, cols = m.shape
    nr, nc = -(-rows // 128), -(-cols // 4)
    if (rows, cols) != (nr * 128, nc * 4):
        pad = torch.zeros((nr * 128, nc * 4), device=m.device, dtype=m.dtype); pad[:rows, :cols] = m; m = pad
    blocks = m.view(nr, 128, nc, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def quantize_nvfp4(x: torch.Tensor, global_scale: torch.Tensor | None = None):
    """x: [rows, K] (bf16/fp32, K % 16 == 0) -> (data fp4x2 [rows, K//2],
    blocked fp8 scales, global fp32 scale [1]).  Global scale chosen so the largest
    per-block scale fits e4m3 (the standard NVFP4 two-level scaling)."""
    assert x.dim() == 2 and x.shape[1] % BLOCK == 0, x.shape
    xf = x.float()
    rows, K = xf.shape
    blocks = xf.view(rows, K // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1)                                   # [rows, K/16]
    if global_scale is None:
        global_scale = (amax.max() / (F4_MAX * F8_MAX)).clamp_min(1e-12).reshape(1)
    s_f32 = amax / (F4_MAX * global_scale)                             # per-block, in fp8 units
    s_f8 = s_f32.clamp(min=1e-12, max=F8_MAX).to(torch.float8_e4m3fn)
    s_eff = s_f8.float() * global_scale                                # decoded scale
    q = blocks / s_eff.unsqueeze(-1).clamp_min(1e-30)
    q = q.clamp(-F4_MAX, F4_MAX).reshape(rows, K)
    data = _pack_uint4(_f32_to_e2m1_unpacked(q)).view(torch.float4_e2m1fn_x2)
    return data, _to_blocked(s_f8), global_scale.float()


class NVFP4Linear(nn.Module):
    """Drop-in for nn.Linear (weight-only static NVFP4, dynamic NVFP4 activations)."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.detach()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        data, scales, gs = quantize_nvfp4(w.to(w.device))
        # scaled_mm wants mat_b as [K//2, N] fp4 with scales [K//16, N]: keep W^T layout
        self.register_buffer("w_t", data)                                   # [N, K//2] fp4x2
        self.register_buffer("w_scales", scales)
        self.register_buffer("w_gscale", gs)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().to(torch.bfloat16), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xd, xs, xg = quantize_nvfp4(x2)
        ST, SW = F.ScalingType, F.SwizzleType
        out = F.scaled_mm(
            xd, self.w_t.t(),
            [xs, xg], [ST.BlockWise1x16, ST.TensorWise], [self.w_scales, self.w_gscale], [ST.BlockWise1x16, ST.TensorWise],
            swizzle_a=[SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE], swizzle_b=[SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE],
            bias=self.bias, output_dtype=torch.bfloat16)
        return out.reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, nvfp4 weight ({self.w_t.numel() * 2 / 1e6:.1f} M values)"


def quantize_linears_(module: nn.Module, min_in: int = BLOCK, skip: tuple[str, ...] = ()) -> dict:
    """Replace eligible nn.Linear children in place. Returns a report."""
    n_done, n_skipped, bytes_before, bytes_after = 0, 0, 0, 0
    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if child.in_features % BLOCK or child.in_features < min_in or any(s in name for s in skip):
            n_skipped += 1
            continue
        parent = module
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        q = NVFP4Linear(child)
        bytes_before += child.weight.numel() * child.weight.element_size()
        bytes_after += q.w_t.numel() + q.w_scales.numel()
        setattr(parent, parts[-1], q)
        n_done += 1
    return {"quantized": n_done, "skipped": n_skipped,
            "weight_gb_before": round(bytes_before / 1e9, 2), "weight_gb_after": round(bytes_after / 1e9, 2)}


@torch.no_grad()
def fidelity_check(in_features: int = 5120, out_features: int = 13824, tokens: int = 4096, seed: int = 0) -> dict:
    """Relative L2 error and timing of one NVFP4 linear vs its bf16 source on the GPU."""
    import time
    torch.manual_seed(seed)
    lin = nn.Linear(in_features, out_features).cuda().to(torch.bfloat16)
    x = torch.randn(1, tokens, in_features, device="cuda", dtype=torch.bfloat16)
    ref = lin(x).float()
    q = NVFP4Linear(lin)
    torch.cuda.synchronize(); t = time.time(); out = q(x); torch.cuda.synchronize(); t_q = time.time() - t
    t = time.time(); lin(x); torch.cuda.synchronize(); t_bf = time.time() - t
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    return {"rel_l2": rel, "ms_nvfp4": t_q * 1e3, "ms_bf16": t_bf * 1e3, "shape": (in_features, out_features, tokens)}
