"""Phase-C P0, step 3: teacher-forced chunk-latent cache through the frozen
LingBot-World v2 causal-fast DiT.  Pilot-verified on GB10 (2026-09-19).

Per paper/lingbot_hook_spec.md, the adapter tap point is the *denoised chunk
latent* x0 (16, chunk_size, 60, 104) at the moment a chunk finishes
denoising, BEFORE the KV-cache update forward writes it into the next
chunk's conditioning path. This script replays the frozen backbone over the
VAE-encoded TRL2D renders, teacher-forced (ground-truth latents fed to the
KV cache), and caches per chunk:
    x0_pred  — the backbone's denoised chunk latent (adapter input),
    gt       — the ground-truth VAE chunk latent   (adapter target).

Verified code path (baseline/lingbot-world-v2, traced 2026-08-17)
=================================================================
Model load (mirrored in load_backbone()):
  - WanModelFast.from_pretrained(checkpoint_dir, subfolder='transformers',
    torch_dtype=torch.bfloat16, local_attn_size, sink_size)
        wan/image2video.py:127-132; subfolder name 'transformers' from
        wan/configs/wan_i2v_A14B.py:30 (fast_checkpoint); class at
        wan/modules/model_fast.py:356.
  - eval().requires_grad_(False), then .to(device)
        image2video.py:342 + 140-145 (_configure_model, no FSDP/SP here).

Denoise schedule (causal_fast — resolves the hook spec's open item):
  - FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1,
    use_dynamic_shifting=False)                    image2video.py:147-150
  - scheduler.set_timesteps(1000, shift=shift); timesteps =
    scheduler.timesteps[timesteps_index]           image2video.py:530-531
  - ACTUAL steps: 4 denoise forwards per chunk, NO CFG. The effective
    timesteps_index is generate()'s default [0, 250, 500, 750]
    (image2video.py:393) — _generate_causal_fast's own default
    [0, 179, 358, 679] (image2video.py:431) is dead code because
    generate() always forwards its kwarg (image2video.py:410-422).
  - shift: python-API default 5.0 (image2video.py:394); the CLI defaults
    to cfg.sample_shift=10.0 (generate.py:48-49, wan_i2v_A14B.py:33);
    docstring recommends 3.0 at 480p (image2video.py:451). Resolved by
    the pilot — see "Runtime verification" below.

Per-chunk loop being replayed (image2video.py:643-709):
  - latent chunks:      noise.split(chunk_size, dim=1)      :644
  - conditioning chunks: y.split(chunk_size, dim=1)          :645
  - plucker chunks:     c2ws_plucker_emb.split(.., dim=2)    :646
  - 4x denoise: noise_pred = model(x=[cur], t=timestep, **kw) :679-682
      x0 = _convert_flow_pred_to_x0(noise_pred, cur, t, sched) :688-693
      re-noise to next timestep via scheduler.add_noise        :695-697
  - TAP POINT: final x0 appended                              :702
  - KV update: model(x=[x0], t=0, **kw)  <-- teacher forcing
      replaces x0 with the GT chunk latent here               :705-709

Forward signature (model_fast.py:485-499): x=[Tensor(16,cs,60,104)], t=(1,),
context=[Tensor(L,4096)], seq_len=4680 (cs=3), y=[Tensor(20,cs,60,104)],
dit_cond_dict={'c2ws_plucker_emb': tuple}, kv_cache, crossattn_cache,
current_start=chunk_id*cs*1560, max_attention_size, frame_seqlen=1560,
cross_attn_first_call. y = concat(mask 4ch, cond latent 16ch) -> 20ch;
x||y concatenated at model_fast.py:548 -> in_dim=36 at patch embed
(model_fast.py:551, config.json in_dim=36).

KV caches (image2video.py:616-635; ctors :992-1005, :1008-1020):
  self:  per layer k/v (1, kv_size, 40, 128) bf16, kv_size =
         frame_seqlen * local_attn_size = 1560*18 = 28080 (or *lat_f if
         local_attn_size=-1). 40 layers -> ~23 GB bf16 at local 18.
  cross: per layer k/v (1, 512, 40, 128) + is_init int32 tensor.
  Writes/eviction happen inside CausalWanSelfAttention.forward
  (model_fast.py:120-162): every forward writes the current chunk's
  roped k / v into slots [local_end-4680, local_end) and then fill_()s
  global_end_index / local_end_index (:161-162). The 4 denoise forwards
  of a chunk therefore overwrite the same slots (current_end ==
  global_end -> else-branch :151-155), and the final t=0 update forward
  overwrites them once more with the CLEAN latent's k/v — which is
  exactly where teacher forcing substitutes the GT chunk. Eviction with
  sink (:133-150) first triggers at chunk 6 (frames 18-20 > window 18).

Geometry for our data: renders 480x832 -> lat 60x104, frame_seqlen =
60*104/4 = 1560 (image2video.py:619). VAE gives 26 latent frames from
T=101; we use lat_f = 24 (largest multiple of chunk_size 3 and 4) ->
8 chunks of 3. GT latents and cond latents come from vae_encode.py
outputs sliced [:, :24] (exact — the VAE is temporally causal).

Runtime verification (GB10 pilot, 2026-09-19) — resolutions of the former
TODO(verify) markers; each is also asserted at runtime where cheap:
  (1) Environment: easydict 1.13 / diffusers 0.40 / transformers 5.15 are
      in the quworld env; flash_attn is NOT (aarch64). Self-attention goes
      through wan.modules.attention.attention (model_fast.py:159) which
      falls back to torch SDPA (attention.py:147-177). Cross-attention
      calls flash_attention directly (model_fast.py:208, imported at :19)
      and would hit `assert FLASH_ATTN_2_AVAILABLE` (attention.py:111);
      context_lens is always None (model_fast.py:572), so rebinding that
      name to attention() (install_attention_fallback) is semantically
      exact (no padding mask to lose). diffusers' low_cpu_mem_usage shard
      loader needs `accelerate` (modeling_utils.py:1059-1063); without it
      from_pretrained materialises the full fp32 model (74 GB) — run with
      `uv run --with accelerate` (ephemeral overlay, project env untouched).
  (2) Checkpoint: transformers/ holds 8 fp32 safetensors shards (74.18 GB,
      18.54 G params incl. the camera-control layers) and NO config.json;
      diffusers' load_config(subfolder='transformers') falls back to the
      root config.json (in_dim 36, dim 5120, 40 layers/heads). bf16 DiT is
      therefore ~37 GB resident, not the 28.5 GB in README.md.
  (3) Plucker / static camera: get_plucker_embeddings (cam_utils.py:95-129)
      returns (lat_f, 480, 832, 6); identity c2ws through
      compute_relative_poses(framewise=True) (cam_utils.py:56-76) stay
      identity (max_norm == 0 -> no translation normalisation, :69-75), so
      rays_o == 0 and rays_d == per-pixel unit directions from the
      intrinsics; the two rearranges (image2video.py:579-586) give
      (1, 384, lat_f, 60, 104), consumed at model_fast.py:583-599 as
      (1, f*30*52, 1536) -> patch_embedding_wancamctrl (in 6*64*4=1536).
      Intrinsics row = examples/03/intrinsics.npy[0] (fx,fy,cx,cy =
      415.53, 415.69, 415.78, 239.78) — identical to the static camera the
      Physics-IQ harness writes (physiq_runner.py INTRINSICS_ROW); at
      480x832 get_Ks_transformed (cam_utils.py:132-160) is the identity.
  (4) Mask: build_mask == image2video.py:520-527 -> (4, lat_f, 60, 104).
  (5) KV-cache mutation: asserted in --pilot (kv_verify): after the chunk-0
      update forward global_end_index == local_end_index == 4680 on every
      layer, slots [:4680) are non-zero and [4680:) are zero, cross
      is_init == 1; zeroing the chunk-0 slots changes chunk-1's first
      noise_pred (relative L1 difference reported in the pilot log), and
      after the last chunk local_end_index == kv_size (window full) with
      sink slots [:9360) unchanged since chunk 2. Numbers: see
      runs/phase_c/pilot_report.txt.
  (6) Prompt / shift: chosen by the pilot grid shift {3,5,10} x prompt
      {physics, empty} on train_000 (teacher-forced chunk RMSE vs GT);
      the selected values are the argparse defaults below and are logged
      into every output's meta.

ENVIRONMENT: run through `uv run --with accelerate python ...`
(see (1)); T5 on CPU is the default (--t5-cpu) and is encoded BEFORE the
DiT is loaded, with the context cached under runs/phase_c/t5_context/.
Memory on GB10 (unified): DiT bf16 ~37 GB + self-KV ~23 GB + activations
~2 GB -> ~62 GB GPU-side; the T5 transient (~23 GB CPU) never overlaps.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import importlib.util
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
BASELINE = ROOT / "baseline" / "lingbot-world-v2"
sys.path.insert(0, str(BASELINE))

WEIGHTS = ROOT / "data" / "lingbot-world-v2-14b-causal-fast"
LATENT_DIR = ROOT / "data" / "phase_c" / "latents"
OUT_DIR = ROOT / "data" / "phase_c" / "teacher"
RUNS_DIR = ROOT / "runs" / "phase_c"
T5_CACHE_DIR = RUNS_DIR / "t5_context"
STATUS_FILE = RUNS_DIR / "phase_c_status"
INTRINSICS = BASELINE / "examples" / "03" / "intrinsics.npy"

LAT_H, LAT_W = 60, 104
PIX_H, PIX_W = LAT_H * 8, LAT_W * 8          # 480, 832
LAT_F_USED = 24                              # 26 available; multiple of chunk_size
FRAME_SEQLEN = LAT_H * LAT_W // 4            # 1560, image2video.py:619
NUM_LAYERS, NUM_HEADS, HEAD_DIM = 40, 40, 128
TEXT_LEN = 512
CONTEXT_DIM = 4096                           # umt5-xxl width, text_dim
PLUCKER_CH = 6 * 8 * 8                       # 384 = control_dim * 8 * 8
GPU_BUDGET_GB = 80.0
PILOT_SHIFTS = (3.0, 5.0, 10.0)
PILOT_PROMPTS = ("physics", "empty")

# Fixed physics prompt reused for every trajectory (T5 encoded once).
PROMPTS = {
    "physics": ("Two fluid layers mixing turbulently, dense cold gas below "
                "and hot diffuse gas above, swirling filaments and shear "
                "instabilities, scientific visualization, fixed camera"),
    "empty": "",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--latents", type=pathlib.Path, default=LATENT_DIR)
    ap.add_argument("--out", type=pathlib.Path, default=OUT_DIR)
    ap.add_argument("--ckpt", type=pathlib.Path, default=WEIGHTS)
    ap.add_argument("--chunk-size", type=int, default=3,
                    help="python-API default 3 (image2video.py:390); "
                         "CLI default is 4 (generate.py:93-96)")
    ap.add_argument("--local-attn-size", type=int, default=18,
                    help="deployment setting, run_fast.sh")
    ap.add_argument("--sink-size", type=int, default=6)
    ap.add_argument("--timesteps-index", type=int, nargs="+",
                    default=[0, 250, 500, 750],
                    help="generate() default (image2video.py:393)")
    ap.add_argument("--shift", type=float, default=5.0,
                    help="pilot-selected (runs/phase_c/pilot_report.txt); "
                         "5.0 py default / 10.0 CLI / 3.0 480p docstring")
    ap.add_argument("--prompt-name", choices=tuple(PROMPTS), default="physics",
                    help="pilot-selected (runs/phase_c/pilot_report.txt)")
    ap.add_argument("--kv-source", choices=("teacher", "pred"),
                    default="teacher",
                    help="teacher: GT latent into KV update (P0 spec); "
                         "pred: x0 as in generate() (free-running)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--t5-cpu", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="run umt5-xxl on CPU (image2video.py:547-548 "
                         "path); ~23 GB RAM transient, before the DiT load")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subset", type=pathlib.Path, default=None,
                    help="text file of trajectory stems to cache (pre-"
                         "registered subset, e.g. runs/phase_c/teacher_subset.txt)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="one-trajectory grid (shift x prompt) + KV-cache "
                         "checks; writes runs/phase_c/pilot_report.txt, "
                         "outputs under runs/phase_c/pilot/ (not --out)")
    ap.add_argument("--kv-verify", action="store_true",
                    help="run the KV-cache mutation assertions (slow-ish)")
    ap.add_argument("--status-file", type=pathlib.Path, default=STATUS_FILE)
    return ap.parse_args()


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def status(path: pathlib.Path, msg: str) -> None:
    """Append one monitor-readable line; also echo to stdout."""
    line = f"{now()} teacher {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line + "\n")


def system_used_gb() -> float:
    """Unified-memory view: MemTotal - MemAvailable (GPU allocs included)."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k] = int(v.strip().split()[0])
    return (info["MemTotal"] - info["MemAvailable"]) / 1e6


def mem_line() -> str:
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    return (f"cuda alloc {alloc:.1f} GB, peak alloc {peak:.1f} GB, peak "
            f"reserved {reserved:.1f} GB, system used {system_used_gb():.1f} GB")


def check_budget() -> None:
    reserved = torch.cuda.max_memory_reserved() / 1e9
    if reserved > GPU_BUDGET_GB:
        raise RuntimeError(f"GPU reserved {reserved:.1f} GB > {GPU_BUDGET_GB} GB")


# --------------------------------------------------------------------------
# environment shims (resolution (1))
# --------------------------------------------------------------------------

def install_attention_fallback() -> str:
    """flash_attn is absent on this aarch64 env. Self-attention already uses
    wan.modules.attention.attention (model_fast.py:159) whose SDPA fallback
    is attention.py:147-177. Cross-attention calls flash_attention directly
    (model_fast.py:208; name imported at :19) with context_lens=None
    (model_fast.py:572) -> rebinding it to attention() is exact."""
    from wan.modules import attention as attn_mod
    from wan.modules import model_fast
    if attn_mod.FLASH_ATTN_2_AVAILABLE or attn_mod.FLASH_ATTN_3_AVAILABLE:
        return "flash_attn"
    model_fast.flash_attention = attn_mod.attention
    return "torch-sdpa (flash_attn absent; cross-attn rebound, model_fast.py:208)"


def require_accelerate() -> None:
    """diffusers' shard-by-shard loader (low_cpu_mem_usage) needs accelerate
    (diffusers/models/modeling_utils.py:1059-1063); without it the full
    fp32 model (74 GB) is materialised on CPU first."""
    if importlib.util.find_spec("accelerate") is None:
        raise RuntimeError("accelerate missing: launch with "
                           "`uv run --with accelerate python ...`")


# --------------------------------------------------------------------------
# model + conditioning loading
# --------------------------------------------------------------------------

def load_backbone(ckpt_dir: pathlib.Path, local_attn_size: int,
                  sink_size: int, device: str):
    """Frozen causal-fast DiT, mirrors image2video.py:127-132 + 140-145."""
    require_accelerate()
    from wan.modules.model_fast import WanModelFast     # model_fast.py:356
    model = WanModelFast.from_pretrained(
        str(ckpt_dir),
        subfolder="transformers",                       # wan_i2v_A14B.py:30
        torch_dtype=torch.bfloat16,
        local_attn_size=local_attn_size,
        sink_size=sink_size)
    model.eval().requires_grad_(False)                  # image2video.py:342
    model = model.to(device)                            # image2video.py:145
    n_params = sum(p.numel() for p in model.parameters())
    dtypes = {str(p.dtype) for p in model.parameters()}
    print(f"DiT loaded: {n_params / 1e9:.2f} G params, dtypes {dtypes}, "
          f"in_dim {model.config.in_dim}, layers {model.config.num_layers}, "
          f"local_attn_size {model.local_attn_size}, sink_size {sink_size}")
    if n_params * 2 / 1e9 > GPU_BUDGET_GB / 2:
        raise RuntimeError("DiT alone exceeds half the GPU budget")
    return model


def prompt_cache_path(prompt: str) -> pathlib.Path:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    return T5_CACHE_DIR / f"ctx_{digest}.pt"


def encode_prompts(ckpt_dir: pathlib.Path, names: tuple[str, ...],
                   t5_cpu: bool, device: str) -> dict[str, torch.Tensor]:
    """One-off T5 contexts, mirrors image2video.py:110-117 + 541-548.
    Disk-cached so resumed runs never re-load umt5-xxl."""
    T5_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    contexts: dict[str, torch.Tensor] = {}
    missing = []
    for name in names:
        path = prompt_cache_path(PROMPTS[name])
        if path.exists():
            contexts[name] = torch.load(path, weights_only=True).to(device)
            print(f"T5 context '{name}' from cache {path.name}: "
                  f"{tuple(contexts[name].shape)}")
        else:
            missing.append(name)
    if missing:
        from wan.modules.t5 import T5EncoderModel
        t5_device = torch.device("cpu") if t5_cpu else torch.device(device)
        t0 = time.time()
        text_encoder = T5EncoderModel(
            text_len=TEXT_LEN, dtype=torch.bfloat16, device=t5_device,
            checkpoint_path=str(ckpt_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(ckpt_dir / "google" / "umt5-xxl"))
        print(f"T5 loaded on {t5_device} in {time.time() - t0:.0f}s")
        for name in missing:
            t0 = time.time()
            ctx = text_encoder([PROMPTS[name]], t5_device)[0].cpu()
            tmp = prompt_cache_path(PROMPTS[name]).with_suffix(".tmp")
            torch.save(ctx, tmp)
            tmp.replace(prompt_cache_path(PROMPTS[name]))
            contexts[name] = ctx.to(device)
            print(f"T5 context '{name}': {tuple(ctx.shape)} {ctx.dtype} "
                  f"in {time.time() - t0:.1f}s")
        del text_encoder                                # free ~11 GB
        import gc
        gc.collect()
    for name, ctx in contexts.items():
        if ctx.dim() != 2 or ctx.shape[1] != CONTEXT_DIM or ctx.shape[0] > TEXT_LEN:
            raise RuntimeError(f"context '{name}' bad shape {tuple(ctx.shape)}")
    return contexts


def static_camera_plucker(lat_f: int, device: str, dtype: torch.dtype,
                          verbose: bool = False) -> torch.Tensor:
    """Identity-pose (static camera) Plucker embedding for the physics
    renders — TRL2D has no camera motion. Mirrors image2video.py:551-586
    with c2ws = repeated identity extrinsics (resolution (3))."""
    import numpy as np
    from einops import rearrange
    from wan.utils.cam_utils import (compute_relative_poses,
                                     get_Ks_transformed,
                                     get_plucker_embeddings)
    c2ws = torch.eye(4).repeat(lat_f, 1, 1)             # latent-rate frames
    c2ws_rel = compute_relative_poses(c2ws, framewise=True)   # :572
    if not torch.equal(c2ws_rel, c2ws):
        raise RuntimeError("identity poses did not stay identity")

    ks = torch.from_numpy(np.load(INTRINSICS)).float()  # (N, 4) fx,fy,cx,cy
    ks = get_Ks_transformed(ks, height_org=480, width_org=832,
                            height_resize=PIX_H, width_resize=PIX_W,
                            height_final=PIX_H, width_final=PIX_W)[0]  # :554-561
    ks = ks.repeat(lat_f, 1)                             # :573

    emb = get_plucker_embeddings(c2ws_rel.to(device), ks.to(device),
                                 PIX_H, PIX_W)           # :578
    if tuple(emb.shape) != (lat_f, PIX_H, PIX_W, 6):
        raise RuntimeError(f"plucker raw shape {tuple(emb.shape)}")
    if emb[..., :3].abs().max().item() != 0.0:
        raise RuntimeError("static camera should have zero ray origins")
    emb = rearrange(emb, "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
                    c1=8, c2=8)[None]                    # :579-585
    emb = rearrange(emb, "b (f h w) c -> b c f h w",
                    f=lat_f, h=LAT_H, w=LAT_W).to(dtype)  # :586
    if tuple(emb.shape) != (1, PLUCKER_CH, lat_f, LAT_H, LAT_W):
        raise RuntimeError(f"plucker shape {tuple(emb.shape)}")
    if verbose:
        # channel layout (c c1 c2): c in 0..5 outermost, so rays_d = c 3..5
        # -> channels [192:384) as 3 blocks of 64 sub-pixel offsets.
        rays_d = emb.float()[0, 3 * 64:].view(3, 64, lat_f, LAT_H, LAT_W)
        print(f"plucker: {tuple(emb.shape)} {emb.dtype}, intrinsics "
              f"{ks[0].tolist()}, |rays_d| mean "
              f"{rays_d.norm(dim=0).mean().item():.3f} (expect 1.0)")
    return emb


def build_mask(lat_f: int, device: str) -> torch.Tensor:
    """First-frame conditioning mask, verbatim from image2video.py:520-527."""
    f = (lat_f - 1) * 4 + 1
    msk = torch.ones(1, f, LAT_H, LAT_W, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, LAT_H, LAT_W)
    return msk.transpose(1, 2)[0]                        # (4, lat_f, 60, 104)


def convert_flow_pred_to_x0(flow_pred, xt, timestep, scheduler):
    """Verbatim copy of WanI2VCausal._convert_flow_pred_to_x0
    (image2video.py:363-383; does not use self): x0 = x_t - sigma_t*pred."""
    original_dtype = flow_pred.dtype
    flow_pred, xt, sigmas, timesteps = map(
        lambda x: x.double().to(flow_pred.device),
        [flow_pred, xt, scheduler.sigmas, scheduler.timesteps])
    timestep_id = torch.argmin((timesteps - timestep).abs())
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
    return (xt - sigma_t * flow_pred).to(original_dtype)


def init_kv_caches(local_attn_size: int, lat_f: int, device: str):
    """Self + cross KV caches, mirrors image2video.py:616-635 with the
    ctors at :992-1005 and :1008-1020."""
    kv_frames = local_attn_size if local_attn_size > -1 else lat_f
    kv_size = FRAME_SEQLEN * kv_frames
    self_shape = [1, kv_size, NUM_HEADS, HEAD_DIM]
    cross_shape = [1, TEXT_LEN, NUM_HEADS, HEAD_DIM]
    mk = lambda shape: {                                  # noqa: E731
        "k": torch.zeros(shape, dtype=torch.bfloat16, device=device),
        "v": torch.zeros(shape, dtype=torch.bfloat16, device=device)}
    self_kv = [{**mk(self_shape),
                "global_end_index": torch.tensor([0], dtype=torch.long,
                                                 device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long,
                                                device=device)}
               for _ in range(NUM_LAYERS)]
    cross_kv = [{**mk(cross_shape),
                 "is_init": torch.tensor(0, dtype=torch.int32, device=device)}
                for _ in range(NUM_LAYERS)]
    return self_kv, cross_kv, kv_size


# --------------------------------------------------------------------------
# KV-cache verification helpers (resolution (5), --kv-verify / --pilot)
# --------------------------------------------------------------------------

def kv_check_after_chunk0(self_kv, cross_kv, n_tokens: int) -> str:
    for layer, kv in enumerate(self_kv):
        g, l = kv["global_end_index"].item(), kv["local_end_index"].item()
        if g != n_tokens or l != n_tokens:
            raise RuntimeError(f"layer {layer}: end indices {g},{l} != {n_tokens}")
        if kv["k"][:, :n_tokens].abs().sum().item() == 0:
            raise RuntimeError(f"layer {layer}: written K slots are zero")
        if kv["k"][:, n_tokens:].abs().sum().item() != 0:
            raise RuntimeError(f"layer {layer}: unwritten K slots non-zero")
        if cross_kv[layer]["is_init"].item() != 1:
            raise RuntimeError(f"layer {layer}: cross is_init not set")
    return (f"kv-check chunk0: all {len(self_kv)} layers global_end=local_end="
            f"{n_tokens}, slots [:{n_tokens}) non-zero, rest zero, cross is_init=1")


def kv_ablation(model, self_kv, n_tokens: int, forward_kwargs: dict,
                x_in, t_in, reference: torch.Tensor) -> str:
    """Zero the chunk-0 slots, repeat the same forward, compare, restore."""
    saved = [(kv["k"][:, :n_tokens].clone(), kv["v"][:, :n_tokens].clone())
             for kv in self_kv]
    idx = [(kv["global_end_index"].item(), kv["local_end_index"].item())
           for kv in self_kv]
    for kv in self_kv:
        kv["k"][:, :n_tokens].zero_()
        kv["v"][:, :n_tokens].zero_()
    ablated = model(x=[x_in], t=t_in, cross_attn_first_call=False,
                    **forward_kwargs)[0]
    for kv, (k, v) in zip(self_kv, saved):
        kv["k"][:, :n_tokens].copy_(k)
        kv["v"][:, :n_tokens].copy_(v)
    idx_after = [(kv["global_end_index"].item(), kv["local_end_index"].item())
                 for kv in self_kv]
    if idx != idx_after:
        raise RuntimeError("repeat forward moved the cache indices")
    rel = ((ablated - reference).abs().mean() /
           reference.abs().mean().clamp_min(1e-8)).item()
    if rel < 1e-3:
        raise RuntimeError(f"chunk-1 forward ignores chunk-0 cache (rel {rel:.2e})")
    return (f"kv-ablation: zeroing chunk-0 K/V changes chunk-1 first "
            f"noise_pred by rel L1 {rel:.3f}; indices unchanged by repeat forward")


def kv_check_final(self_kv, kv_size: int, n_chunks: int, cs: int,
                   sink_snapshot: torch.Tensor | None, sink_tokens: int) -> str:
    g_expect = n_chunks * cs * FRAME_SEQLEN
    for layer, kv in enumerate(self_kv):
        g, l = kv["global_end_index"].item(), kv["local_end_index"].item()
        if g != g_expect or l != min(kv_size, g_expect):
            raise RuntimeError(f"layer {layer}: final indices {g},{l}")
    msg = f"kv-check final: global_end={g_expect}, local_end={min(kv_size, g_expect)}"
    if sink_snapshot is not None:
        same = torch.equal(self_kv[0]["k"][:, :sink_tokens], sink_snapshot)
        if not same:
            raise RuntimeError("sink slots changed after eviction")
        msg += f", sink [:{sink_tokens}) of layer 0 unchanged through eviction"
    return msg


# --------------------------------------------------------------------------
# per-trajectory teacher-forced replay
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ReplayConfig:
    chunk_size: int
    local_attn_size: int
    sink_size: int
    timesteps_index: tuple[int, ...]
    shift: float
    kv_source: str
    prompt_name: str
    kv_verify: bool = False


def cache_trajectory(model, scheduler, context: torch.Tensor,
                     plucker: torch.Tensor, cfg: ReplayConfig,
                     gt_latent: torch.Tensor, cond_latent: torch.Tensor,
                     seed: int, device: str) -> dict:
    """Replay image2video.py:643-709 teacher-forced over one trajectory.

    gt_latent, cond_latent: (16, LAT_F_USED, 60, 104) fp32 (vae_encode.py
    outputs sliced [:, :LAT_F_USED]).
    Returns {"x0_pred": (n_chunks,16,cs,60,104), "gt": same, "meta": ...}.
    """
    cs = cfg.chunk_size
    if LAT_F_USED % cs:
        raise ValueError(f"LAT_F_USED={LAT_F_USED} not divisible by {cs}")
    n_chunks = LAT_F_USED // cs
    n_tokens = cs * FRAME_SEQLEN                         # 4680
    max_seq_len = cs * LAT_H * LAT_W // 4                # image2video.py:501-503

    scheduler.set_timesteps(1000, shift=cfg.shift)       # image2video.py:530
    timesteps = scheduler.timesteps[list(cfg.timesteps_index)]

    seed_g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(16, LAT_F_USED, LAT_H, LAT_W, dtype=torch.float32,
                        generator=seed_g, device=device)  # image2video.py:511-518

    msk = build_mask(LAT_F_USED, device)
    y_full = torch.concat([msk, cond_latent.to(device)])  # image2video.py:608

    self_kv, cross_kv, kv_size = init_kv_caches(cfg.local_attn_size,
                                                LAT_F_USED, device)
    latents_chunk = noise.split(cs, dim=1)               # image2video.py:644
    condition_chunk = y_full.split(cs, dim=1)            # :645
    plucker_chunk = plucker.split(cs, dim=2)             # :646
    gt_chunks = gt_latent.to(device).split(cs, dim=1)

    x0_preds, chunk_rmse, chunk_secs, notes = [], [], [], []
    sink_snapshot, sink_tokens = None, cfg.sink_size * FRAME_SEQLEN
    cross_attn_initialized = False
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        for chunk_id in range(n_chunks):
            t_chunk = time.time()
            current_latent = latents_chunk[chunk_id]
            kwargs = {                                    # image2video.py:658-668
                "context": [context],
                "seq_len": max_seq_len,
                "y": [condition_chunk[chunk_id]],
                "dit_cond_dict": {
                    "c2ws_plucker_emb": plucker_chunk[chunk_id].chunk(1, dim=0)},
                "kv_cache": self_kv,
                "crossattn_cache": cross_kv,
                "current_start": chunk_id * cs * FRAME_SEQLEN,
                "max_attention_size": kv_size,
                "frame_seqlen": FRAME_SEQLEN,
            }
            # ---- 4-step denoise (image2video.py:673-700) ----
            for timestep_idx in range(len(timesteps)):
                timestep = torch.stack([timesteps[timestep_idx]]).to(device)
                noise_pred = model(
                    x=[current_latent], t=timestep,
                    cross_attn_first_call=not cross_attn_initialized,
                    **kwargs)[0]                           # image2video.py:679-682
                cross_attn_initialized = True
                if cfg.kv_verify and chunk_id == 1 and timestep_idx == 0:
                    notes.append(kv_ablation(model, self_kv, n_tokens, kwargs,
                                             current_latent, timestep, noise_pred))
                x0 = convert_flow_pred_to_x0(
                    noise_pred, current_latent, timesteps[timestep_idx],
                    scheduler)                             # :688-693
                if timestep_idx < len(timesteps) - 1:
                    current_latent = scheduler.add_noise(   # :695-697
                        x0, torch.randn(x0.shape, generator=seed_g,
                                        device=device, dtype=x0.dtype),
                        timesteps[timestep_idx + 1])

            x0_preds.append(x0.float().cpu())              # TAP (:702)
            gt_j = gt_chunks[chunk_id].float()
            chunk_rmse.append((x0.float() - gt_j).pow(2).mean().sqrt().item())

            # ---- KV update forward (:705-709), TEACHER-FORCED ----
            kv_input = gt_j if cfg.kv_source == "teacher" else x0
            timestep0 = torch.stack([timesteps[-1] * 0.0]).to(device)
            model(x=[kv_input], t=timestep0,
                  cross_attn_first_call=False, **kwargs)
            torch.cuda.synchronize()
            chunk_secs.append(time.time() - t_chunk)

            if cfg.kv_verify and chunk_id == 0:
                notes.append(kv_check_after_chunk0(self_kv, cross_kv, n_tokens))
            if cfg.kv_verify and chunk_id == 2 and cfg.sink_size > 0:
                sink_snapshot = self_kv[0]["k"][:, :sink_tokens].clone()
        if cfg.kv_verify:
            notes.append(kv_check_final(self_kv, kv_size, n_chunks, cs,
                                        sink_snapshot, sink_tokens))

    gt_std = [g.float().std().item() for g in gt_chunks]
    return {
        "x0_pred": torch.stack(x0_preds).to(torch.float16),
        "gt": torch.stack([g.cpu().to(torch.float16) for g in gt_chunks]),
        "meta": {
            "chunk_size": cs, "n_chunks": n_chunks, "lat_f_used": LAT_F_USED,
            "timesteps_index": list(cfg.timesteps_index), "shift": cfg.shift,
            "local_attn_size": cfg.local_attn_size, "kv_size": kv_size,
            "sink_size": cfg.sink_size, "kv_source": cfg.kv_source,
            "seed": seed, "prompt_name": cfg.prompt_name,
            "prompt": PROMPTS[cfg.prompt_name],
            "timesteps": timesteps.cpu(),
            "chunk_rmse": chunk_rmse, "gt_std": gt_std,
            "chunk_secs": chunk_secs, "kv_notes": notes,
        },
    }


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------

def load_latent(path: pathlib.Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-file load (no mmap), sliced to LAT_F_USED; shapes asserted."""
    d = torch.load(path, weights_only=True)
    gt = d["latent"][:, :LAT_F_USED].float()
    cond = d["cond_latent"][:, :LAT_F_USED].float()
    exp = (16, LAT_F_USED, LAT_H, LAT_W)
    if tuple(gt.shape) != exp or tuple(cond.shape) != exp:
        raise RuntimeError(f"{path.name}: latent shapes {tuple(gt.shape)} "
                           f"{tuple(cond.shape)} != {exp}")
    return gt, cond


def save_atomic(path: pathlib.Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def summarize(result: dict) -> str:
    m = result["meta"]
    mean_rmse = sum(m["chunk_rmse"]) / len(m["chunk_rmse"])
    return (f"rmse mean {mean_rmse:.4f} per-chunk "
            f"{[round(r, 3) for r in m['chunk_rmse']]} | secs "
            f"{sum(m['chunk_secs']):.1f} ({[round(s, 1) for s in m['chunk_secs']]})")


def setup(args, prompt_names: tuple[str, ...]):
    """T5 first (CPU transient), then the DiT; scheduler + plucker."""
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    scheduler = FlowUniPCMultistepScheduler(                # image2video.py:147-150
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    print("attention backend:", install_attention_fallback())
    t0 = time.time()
    contexts = encode_prompts(args.ckpt, prompt_names, args.t5_cpu, args.device)
    print(f"prompts ready in {time.time() - t0:.0f}s; {mem_line()}")
    t0 = time.time()
    model = load_backbone(args.ckpt, args.local_attn_size, args.sink_size,
                          args.device)
    torch.cuda.synchronize()
    print(f"DiT ready in {time.time() - t0:.0f}s; {mem_line()}")
    plucker = static_camera_plucker(LAT_F_USED, args.device, torch.bfloat16,
                                    verbose=True)
    check_budget()
    return scheduler, contexts, model, plucker


def make_cfg(args, shift: float, prompt_name: str, kv_verify: bool) -> ReplayConfig:
    return ReplayConfig(chunk_size=args.chunk_size,
                        local_attn_size=args.local_attn_size,
                        sink_size=args.sink_size,
                        timesteps_index=tuple(args.timesteps_index),
                        shift=shift, kv_source=args.kv_source,
                        prompt_name=prompt_name, kv_verify=kv_verify)


def run_pilot(args, latent_files: list[pathlib.Path]) -> int:
    """Trajectory-0 grid over shift x prompt with KV checks on the first run."""
    pilot_dir = RUNS_DIR / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    status(args.status_file, f"pilot start traj={latent_files[0].stem} "
                             f"shifts={PILOT_SHIFTS} prompts={PILOT_PROMPTS}")
    scheduler, contexts, model, plucker = setup(args, PILOT_PROMPTS)
    gt, cond = load_latent(latent_files[0])
    rows, first = [], True
    for shift in PILOT_SHIFTS:
        scheduler.set_timesteps(1000, shift=shift)
        print(f"shift {shift}: effective timesteps "
              f"{scheduler.timesteps[list(args.timesteps_index)].tolist()}")
        for prompt_name in PILOT_PROMPTS:
            cfg = make_cfg(args, shift, prompt_name, kv_verify=first)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            result = cache_trajectory(model, scheduler, contexts[prompt_name],
                                      plucker, cfg, gt, cond, args.seed,
                                      args.device)
            secs = time.time() - t0
            check_budget()
            out = pilot_dir / f"{latent_files[0].stem}_shift{shift:g}_{prompt_name}.pt"
            save_atomic(out, result)
            m = result["meta"]
            mean_rmse = sum(m["chunk_rmse"]) / len(m["chunk_rmse"])
            rows.append((shift, prompt_name, mean_rmse, secs, m["chunk_rmse"]))
            line = (f"pilot shift={shift:g} prompt={prompt_name} {summarize(result)} "
                    f"| wall {secs:.1f}s | {mem_line()}")
            status(args.status_file, line)
            for note in m["kv_notes"]:
                status(args.status_file, note)
            first = False
    best = min(rows, key=lambda r: r[2])
    report = [f"# teacher_cache pilot {now()} on {latent_files[0].stem}",
              f"# grid: shift {PILOT_SHIFTS} x prompt {PILOT_PROMPTS}; "
              f"timesteps_index {args.timesteps_index}; chunk {args.chunk_size}; "
              f"local_attn {args.local_attn_size}; sink {args.sink_size}; "
              f"kv_source {args.kv_source}; seed {args.seed}",
              "shift\tprompt\tmean_chunk_rmse\twall_s\tper_chunk_rmse"]
    for shift, prompt_name, mean_rmse, secs, per in rows:
        report.append(f"{shift:g}\t{prompt_name}\t{mean_rmse:.5f}\t{secs:.1f}\t"
                      f"{[round(r, 4) for r in per]}")
    report.append(f"# gt std per chunk: {[round(s, 3) for s in m['gt_std']]}")
    report.append(f"# lowest mean chunk RMSE: shift={best[0]:g} prompt={best[1]}")
    report.append(f"# memory: {mem_line()}")
    (RUNS_DIR / "pilot_report.txt").write_text("\n".join(report) + "\n")
    status(args.status_file, f"pilot done -> runs/phase_c/pilot_report.txt; "
                             f"best shift={best[0]:g} prompt={best[1]}")
    return 0


def run_full(args, latent_files: list[pathlib.Path]) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in latent_files[:args.limit]
            if args.overwrite or not (args.out / p.name).exists()]
    status(args.status_file, f"full start: {len(todo)} pending of "
                             f"{len(latent_files)} | shift={args.shift:g} "
                             f"prompt={args.prompt_name} kv_source={args.kv_source}")
    if not todo:
        status(args.status_file, "full done: nothing pending")
        return 0
    scheduler, contexts, model, plucker = setup(args, (args.prompt_name,))
    cfg = make_cfg(args, args.shift, args.prompt_name, kv_verify=args.kv_verify)
    t_all, n_done = time.time(), 0
    for path in todo:
        i = latent_files.index(path)
        gt, cond = load_latent(path)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        result = cache_trajectory(model, scheduler, contexts[args.prompt_name],
                                  plucker, cfg, gt, cond, args.seed + i,
                                  args.device)
        secs = time.time() - t0
        check_budget()
        save_atomic(args.out / path.name, result)
        n_done += 1
        remaining = (len(todo) - n_done) * (time.time() - t_all) / n_done
        status(args.status_file,
               f"traj {path.stem} ({n_done}/{len(todo)}) x0_pred "
               f"{tuple(result['x0_pred'].shape)} {summarize(result)} | wall "
               f"{secs:.1f}s | eta {remaining / 60:.1f} min | {mem_line()}")
        for note in result["meta"]["kv_notes"]:
            status(args.status_file, note)
    status(args.status_file, f"full done: {n_done} trajectories in "
                             f"{(time.time() - t_all) / 60:.1f} min -> {args.out}")
    return 0


def main() -> int:
    args = parse_args()
    latent_files = sorted(args.latents.glob("*.pt"))
    if not latent_files:
        print(f"FAIL: no latents in {args.latents} — run vae_encode.py first")
        return 1
    if args.subset is not None:
        wanted = [s for s in args.subset.read_text().split() if s]
        by_stem = {p.stem: p for p in latent_files}
        missing = [s for s in wanted if s not in by_stem]
        if missing:
            print(f"FAIL: subset stems without latents: {missing[:5]}")
            return 1
        latent_files = [by_stem[s] for s in wanted]     # keep listed order
        print(f"subset: {len(latent_files)} trajectories from {args.subset}")
    try:
        return run_pilot(args, latent_files) if args.pilot else run_full(args, latent_files)
    except Exception as exc:                              # noqa: BLE001
        status(args.status_file, f"ERROR {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
