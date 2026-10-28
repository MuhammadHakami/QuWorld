"""Profile J1QPIpi training steps (PK config, batch 4096, 598 inputs) on random data: CUDA kernel time by category."""
import sys, time, json, collections, torch
sys.path.insert(0, "paper/strat/fusion"); sys.path.insert(0, "paper")
import hpo_qpikan as H
PK = {"block": "mlp", "width": 4096, "depth": 4, "drop": 0.1, "lr": 0.0025322139193052723, "w_pi": 0.0, "q_layers": 4, "q_in": 128, "link": "cascade", "alpha_mult": 1.0, "alpha_init": 0.1}
fin, out, B = 598, 135, 4096
net = H.build(PK, fin, out); opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
X = torch.randn(B * 4, fin, device="cuda"); Y = torch.randn(B * 4, out, device="cuda")
def step():
    i = torch.randint(0, len(X), (B,), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = net(X[i])
    loss = (o.float() - Y[i]).square().mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
for _ in range(5): step()
torch.cuda.synchronize(); t0 = time.time()
for _ in range(20): step()
torch.cuda.synchronize(); dt = (time.time() - t0) / 20; print(f"[prof] wall per step {dt * 1e3:.1f} ms")
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    for _ in range(10): step()
    torch.cuda.synchronize()
cat = collections.Counter(); n = collections.Counter(); tot = 0
for e in prof.key_averages():
    t = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
    if t <= 0 or e.key.startswith("aten::") or e.key.startswith("autograd") or e.key.startswith("Optimizer") or "ProfilerStep" in e.key: continue
    k = e.key.lower(); c = ("gemm" if any(s in k for s in ("gemm", "cutlass", "sm90", "sm100", "sm120", "nvjet", "cublas", "matmul")) else
                           "complex/elementwise" if any(s in k for s in ("elementwise", "vectorized", "unrolled", "complex", "mul", "add", "exp", "sin", "cos")) else
                           "reduce" if "reduce" in k else "other")
    cat[c] += t; n[c] += e.count; tot += t
gpu_ms = tot / 1e3 / 10
print(f"[prof] GPU kernel time per step {gpu_ms:.1f} ms of wall {dt * 1e3:.1f} ms (busy {gpu_ms / (dt * 1e3):.0%})")
for c, t in cat.most_common(): print(f"[prof]   {c:22s} {t / tot:5.1%}  kernels/step {n[c] / 10:.0f}")
top = sorted(((e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total), e.key[:70], e.count) for e in prof.key_averages() if not e.key.startswith(("aten::", "autograd", "Optimizer", "ProfilerStep")))[-6:]
for t, k, c in reversed(top): print(f"[prof]   top {t / 1e3 / 10:7.2f} ms/step  x{c / 10:.0f}  {k}")
