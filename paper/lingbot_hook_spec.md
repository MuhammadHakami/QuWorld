---
title: LingBot-World v2 hook spec (frozen Ant Group backbone + 30 simulated qubits)
date: 2026-07-27
project: quworld
tags: [lingbot, hook, integration]
status: traced-from-code
---

# Hook Spec — frozen LingBot-World v2 + quantum bottleneck

User directive (2026-07-27): headline system = **ready frozen world model from Ant
Group** = `robbyant/lingbot-world-v2-14b-causal-fast` (verified released 2026-07-09,
CC BY-NC-SA 4.0) **integrated with 30 simulated qubits**.

## Verified from code (baseline/lingbot-world-v2)

- Backbone: Wan2.1-family **causal DiT** — `wan/modules/model_causal.py:WanModelCausal`
  (+ `model_fast.py` for the causal-fast checkpoint), VAE `vae2_1.py` (16-ch latents).
- Rollout: `wan/image2video.py:generate()` — chunked autoregressive denoise,
  `chunk_size=3` latent frames (code default; CLI exposes `--chunk_size`, README
  examples use 4), `--local_attn_size 18`, `--sink_size 6`, self+cross KV caches
  initialized per run (`_initialize_self_kv_cache`, image2video.py:627).
- Conditioning `y` = concat(mask 4ch, VAE latent 16ch) = 20ch; DiT input at
  patch-embed = 36ch (image2video.py:262-268).

## Tap point

In the `generate()` chunk loop: the **denoised chunk latent** `(16, chunk, lat_h,
lat_w)` at the moment a chunk finishes denoising and BEFORE it is written into the
conditioning/KV path of the next chunk. One adapter call per chunk.

Adapter: attention-pool chunk latent → h → **MultiRegister quantum block, 3×10q =
30 simulated qubits** (paper/qlayer.py:MultiRegister) → zero-init gated residual
added to the chunk latent. Backbone stays frozen (bf16); adapter fp32.

## 30-qubit ladder (measured constraints, [[01-hardware-budget]])

1. **mr30 (3×10q product register)** — trainable now (~ms/sample); hardware-honest
   (30 atoms, 3 far-separated patches, blockade-decoupled).
2. **Entangled 30q via bounded-χ MPS propagator** — next; exact at low
   entanglement; exports to one connected 30-atom register.
3. **Exact 30q statevector** — inference/showcase only (8.6 GB state; ~minutes per
   gradient — not trainable).

## Training route (per [[03-pipeline]])

P0: encode The Well fields (3-ch renders → VAE) once; run frozen backbone
teacher-forced, cache chunk latents + predictions. P1: backbone-free adapter
training on cached tensors (the sprint-loop FNO results select the ansatz first —
see results.tsv). P2: closed-loop rollout eval through the real frozen pipeline.

Open item: causal-fast checkpoint distills the denoise to few steps — confirm
step count + guidance in `model_fast.py` before caching (affects P0 cost only).
