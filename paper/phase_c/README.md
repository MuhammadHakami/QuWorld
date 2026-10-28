# Phase-C P0 — frozen LingBot-World v2 latent cache over The Well renders

Pipeline (per `paper/lingbot_hook_spec.md` and `08-scoreboard.md` H2a):
encode TRL2D physics as video once, replay the frozen 14B causal-fast DiT
teacher-forced, cache the chunk-boundary latents, then train adapters
backbone-free on the cached tensors (P1) and close the loop through the
real pipeline (P2).

```
phaseb_cache.pt (81 traj, 101x4x128x384, z-scored)
  └─ 1. render_fields.py   [CPU, ~min]   -> data/phase_c/renders/  (~10 GB uint8)
       └─ 2. vae_encode.py [GPU <20 GB]  -> data/phase_c/latents/  (~1.7 GB fp32)
            └─ 3. teacher_cache.py [GPU] -> data/phase_c/teacher/  (~1.6 GB fp16)
                 └─ P1 adapter training (quantum vs LoRA vs MLP, H2a)
```

## Steps

### 1. `render_fields.py` — fields → RGB video (CPU, safe anytime)

- Fixed linear+clip 4→3 channel map (orthonormal rows, null = common
  mode, pseudo-inverse = transpose — "invertible-ish", roundtrip error
  reported by `--smoke`). Density painted along a viridis-like
  (R↑, G↑, B↓) direction. Pure torch.
- 128×384 → bilinear 278×832 (aspect kept to 0.24%) → pad to 480×832.
- Output: uint8 `(3, 101, 480, 832)` per trajectory, ~120 MB × 81 ≈ 10 GB.

```bash
uv run python paper/phase_c/render_fields.py --smoke   # 2 traj + checks
uv run python paper/phase_c/render_fields.py           # all 81, ~minutes CPU
```

### 2. `vae_encode.py` — Wan2.1 VAE encode (GPU, small)

- Loads ONLY `Wan2.1_VAE.pth` (485 MB fp32); never touches the DiT/T5.
- `(3,101,480,832)` → `(16,26,60,104)` normalized latents + the
  first-frame conditioning latent (`y` path, image2video.py:599-607).
- bf16 autocast default (`--dtype fp32` for bit-parity with the released
  pipeline, which runs the VAE fp32).
- Per-trajectory atomic writes + skip-existing = checkpoint/resume.

```bash
uv run python paper/phase_c/vae_encode.py --dry-run    # CPU, no weights materialized
uv run python paper/phase_c/vae_encode.py              # GPU
```

**Memory**: 0.49 GB weights + causal 4-frame micro-chunk activations —
expected peak 3–6 GB (asserted < 20 GB per trajectory).
**Time on GB10**: ~5–15 s/traj → **< 0.5 GPU-h** for all 81.

### 3. `teacher_cache.py` — teacher-forced chunk replay (GPU, big) — SKELETON

Replays `_generate_causal_fast` (image2video.py:643-709) with the frozen
DiT: 4 denoise forwards per chunk (timesteps `scheduler.timesteps[[0,250,
500,750]]`, no CFG) + 1 KV-update forward where **ground-truth** VAE
latents are written into the cache (teacher forcing; `--kv-source pred`
gives the free-running variant). Caches per chunk: `x0_pred` (tap-point
latent, adapter input) and `gt` (adapter target).

Resolve the inline `TODO(verify)` markers on a free GPU first
(1-trajectory pilot): static-camera Plücker shapes, prompt choice,
shift value (5.0 py-default vs 10.0 CLI vs 3.0 480p recommendation),
KV-cache mutation check.

**Memory (GB10, 121 GB unified)**: DiT bf16 ≈ 28.5 GB (74 GB on disk,
mixed precision) + self-KV cache ≈ 23 GB (`local_attn_size=18`:
40 layers × 2 × 28080 × 5120 × 2 B) + activations/cross-KV ≈ 3 GB +
umt5 encode transient (CPU with `--t5-cpu`, default) → **~55–65 GB**.
Fits; do not co-schedule with Phase-B training.

**Time on GB10**: per chunk ≈ 5 forwards × (~130 TFLOP dense +
~110 TFLOP attention) ≈ 1.2 PFLOP; 8 chunks/traj ≈ 9.6 PFLOP. At a
sustained 50–100 bf16 TFLOP/s: 1.5–4 min/traj → **2–5.5 GPU-h** for 81
trajectories (+ ~10 min load). Measure trajectory 0 first and re-project;
the run resumes per trajectory if interrupted.

## Data geometry (fixed by the code, not by us)

| quantity | value | source |
|---|---|---|
| video | 3×101×480×832, [-1,1] | vae_stride (4,8,8), wan_i2v_A14B.py:16 |
| latent | 16×26×60×104 (use first 24) | 26=1+(101−1)/4; 24 = lcm-fit for chunk 3 and 4 |
| frame_seqlen | 1560 tokens | image2video.py:619 |
| chunks | 8 × (16,3,60,104) per traj | chunk_size 3 (py default; CLI default is 4) |
| latent↔physics time | lat 0 = t₀; lat i = t∈[4i−3, 4i] | causal VAE, vae2_1.py:519-532 |

## Adapter-training handoff (P1, scoreboard H2a)

H2a pass rule: **quantum < matched LoRA AND matched MLP** on chunk-latent
rollout VRMSE + invariant drift, 3 seeds (1337/1338/1339), p<0.05,
identical LR-grid tuning budget (the 2607.09905 bar).

- **Training set**: `data/phase_c/teacher/*.pt` — pairs
  (`x0_pred[j]`, `gt[j]`) per chunk. Adapter learns the residual
  correction at the tap point: `x0_adapted = x0_pred + gate ⊙
  Adapter(x0_pred)`, zero-init gate (backbone-equivalent at init).
- **Arms** (all consume the same pooled feature, attention-pool over
  16×3×60×104 → h):
  1. `mr30ryd` — `paper/qlayer.py:MultiRegister` (3×10q Rydberg-analog,
     wave-1.5 winner, fp32),
  2. LoRA — rank-matched low-rank map on the same pooled feature
     (param count matched to the quantum head),
  3. `mlp30` — matched-width MLP (Phase-B control, param-matched).
- **Objective**: TD/rollout-aware (k=4) per the wave-1.5 finding that the
  quantum edge is invisible under one-step MSE; teacher-forced chunks
  chain into k-step rollouts by feeding `x0_adapted` through the cached
  next-chunk prediction (backbone-free surrogate), exact rollout in P2.
- **Metrics**: chunk-latent VRMSE vs `gt`; invariant drift measured after
  VAE-decode + `render_fields` pseudo-inverse back to fields (mass/energy
  proxies on the recovered density/pressure).
- **P2**: winning adapter re-inserted at image2video.py:702 (before the
  KV-update forward at :705-709) for closed-loop rollout through the real
  frozen pipeline.

## Ledger

- No GPU work has been run from this directory yet; smoke (`render_fields
  --smoke`) and `vae_encode --dry-run` are CPU-only.
- Hook-spec discrepancies found while tracing (line numbers): see
  "Verified code path" header of `teacher_cache.py`; summary: (a) CLI
  `--chunk_size` default is 4 (generate.py:93-96) — README examples don't
  set it, python API default is 3; (b) spec's "36ch at patch-embed,
  image2video.py:262-268" points at the prewarm comment — the real concat
  is image2video.py:599-608 + model_fast.py:547-548; (c) causal-fast
  effective timesteps are [0,250,500,750] (generate() default,
  image2video.py:393) — `_generate_causal_fast`'s [0,179,358,679]
  (image2video.py:431) is shadowed, and the open item on step count is
  resolved: 4 steps, no CFG, 5 forwards/chunk total.
