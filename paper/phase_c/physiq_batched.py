"""Batched (B>1) causal-fast generation for the Physics-IQ runs.

`WanI2VCausal._generate_causal_fast` (image2video.py:430-720) is single-
sample: one image, one prompt, one noise tensor. `WanModelFast.forward`
(model_fast.py:485-631) is batch-agnostic everywhere except three lines that
silently assume B == 1:
  - `t = t.expand(t.size(0), seq_lens)` / `.unflatten(0, (bt, seq_lens))`
    use the seq_lens *tensor* as an int and broadcast (B,) -> (B, L), which
    only works for B == 1;
  - the Plucker embeddings are concatenated along the token axis
    (`torch.cat(..., dim=1)` -> [1, B*L, C]) and later broadcast against
    x of shape [B, L, C] in the blocks.
`patched_forward` is a faithful copy with those lines fixed; it is installed
only when batch > 1, so the B == 1 path stays the released code path.

Noise is drawn per sample from its own seeded generator (initial noise and
the re-noising draws in image2video.py:695-697), so batch composition never
changes a scenario's noise stream: batch-2 output for a scenario is the
batch-1 output up to kernel numerics. `stage_batchtest` in physiq_runner.py
verifies exactly that (chunk-0 x0 relative L2) plus the memory bound.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from pathlib import Path

MEMINFO = Path("/proc/meminfo")


# --------------------------------------------------------------------------
# model forward with the B>1 fixes (copy of model_fast.py:485-631)
# --------------------------------------------------------------------------

def patched_forward(self, x, t, context, seq_len, y=None, dit_cond_dict=None,
                    kv_cache=None, crossattn_cache=None, current_start=0,
                    max_attention_size=1_000_000, frame_seqlen=None,
                    cross_attn_first_call=None):
    import torch
    from wan.modules import model_fast as mf

    if self.model_type == 'i2v':
        assert y is not None
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)
    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat(x)

    # time embeddings — B>1 fix: one shared token count, explicit broadcast
    n_tok = int(seq_lens.max())
    assert bool((seq_lens == n_tok).all()), "batched samples must share the geometry"
    if t.dim() == 1:
        t = t[:, None].expand(t.size(0), n_tok)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            mf.sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, n_tok)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = self.text_embedding(torch.stack([
        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))

    if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
        emb = [mf.rearrange(i, '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                            c1=self.patch_size[0], c2=self.patch_size[1], c3=self.patch_size[2])
               for i in dit_cond_dict["c2ws_plucker_emb"]]
        emb = torch.cat(emb, dim=0)  # B>1 fix: [B, L, C] (was dim=1 -> [1, B*L, C])
        emb = self.patch_embedding_wancamctrl(emb)
        hidden = self.c2ws_hidden_states_layer2(mf.torch_F.silu(self.c2ws_hidden_states_layer1(emb)))
        dit_cond_dict = dict(dit_cond_dict)
        dit_cond_dict["c2ws_plucker_emb"] = emb + hidden

    kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs,
                  context=context, context_lens=context_lens, dit_cond_dict=dit_cond_dict,
                  max_attention_size=max_attention_size, frame_seqlen=frame_seqlen,
                  cross_attn_first_call=cross_attn_first_call)
    for block_index, block in enumerate(self.blocks):
        kwargs.update({"kv_cache": kv_cache[block_index],
                       "crossattn_cache": crossattn_cache[block_index],
                       "current_start": current_start})
        x = block(x, **kwargs)
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def install_batched_forward(pipe) -> None:
    """Bind patched_forward on the pipe's DiT instance (instance-level; idempotent)."""
    import types

    if getattr(pipe.model, "_physiq_batched", False):
        return
    pipe.model.forward = types.MethodType(patched_forward, pipe.model)
    pipe.model._physiq_batched = True


# --------------------------------------------------------------------------
# batched sampling loop (mirror of image2video.py:430-720 with B samples)
# --------------------------------------------------------------------------

def _context_for(pipe, prompt: str):
    """pipe._t5_cache lookup with the pipeline's own fallback (image2video.py:534-548)."""
    import torch

    key = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    if key not in pipe._t5_cache:
        if pipe.text_encoder is None:
            raise RuntimeError("prompt not in T5 cache and text encoder dropped: run --stage t5cache")
        if pipe.t5_cpu:
            ctx = pipe.text_encoder([prompt], torch.device('cpu'))
            ctx = [u.to(pipe.device) for u in ctx]
        else:
            pipe.text_encoder.model.to(pipe.device)
            ctx = pipe.text_encoder([prompt], pipe.device)
        pipe._t5_cache[key] = ctx
    return pipe._t5_cache[key][0]


def _camera_embedding(pipe, action_path: str, chunk_size: int, lat_f: int, lat_h: int, lat_w: int, h: int, w: int):
    """Static-camera Plucker embedding [1, C, lat_f, lat_h, lat_w] (image2video.py:551-582)."""
    import numpy as np
    import torch
    from einops import rearrange
    from wan.utils.cam_utils import (compute_relative_poses, get_Ks_transformed,
                                     get_plucker_embeddings, interpolate_camera_poses)

    c2ws = np.load(os.path.join(action_path, "poses.npy"))
    Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
    Ks = get_Ks_transformed(Ks, height_org=480, width_org=832, height_resize=h,
                            width_resize=w, height_final=h, width_final=w)[0]
    n = len(c2ws)
    n_lat = int((n - 1) // 4) + 1
    n_lat = int(n_lat - (n_lat % chunk_size))
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, n - 1, n), src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3], tgt_indices=np.linspace(0, n - 1, n_lat))
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True).to(pipe.device)
    Ks = Ks.repeat(len(c2ws_infer), 1).to(pipe.device)
    emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
    emb = rearrange(emb, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))
    emb = rearrange(emb[None, ...], 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w)
    return emb.to(pipe.param_dtype)


def generate_batched(pipe, prompts, imgs, action_path, chunk_size, max_area, frame_num,
                     timesteps_index, shift, seeds, max_sequence_length=512):
    """B videos (C, F, H, W) in [-1, 1] for B (prompt, image, seed) triples."""
    import numpy as np
    import torch
    import torchvision.transforms.functional as TF

    B = len(prompts)
    assert B == len(imgs) == len(seeds) and B >= 1
    install_batched_forward(pipe)
    dev = pipe.device
    n_poses = len(np.load(os.path.join(action_path, "poses.npy")))
    frame_num = min(((frame_num - 1) // 4) * 4 + 1, ((n_poses - 1) // 4) * 4 + 1)

    imgs_t = [TF.to_tensor(im).sub_(0.5).div_(0.5).to(dev) for im in imgs]
    assert len({tuple(u.shape) for u in imgs_t}) == 1, "batched images must share one size"
    h, w = imgs_t[0].shape[1:]
    aspect = h / w
    lat_h = round(np.sqrt(max_area * aspect) // pipe.vae_stride[1] // pipe.patch_size[1] * pipe.patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect) // pipe.vae_stride[2] // pipe.patch_size[2] * pipe.patch_size[2])
    h, w = lat_h * pipe.vae_stride[1], lat_w * pipe.vae_stride[2]
    lat_f = (frame_num - 1) // pipe.vae_stride[0] + 1
    lat_f = int(lat_f - (lat_f % chunk_size))
    F = (lat_f - 1) * 4 + 1
    max_seq_len = chunk_size * lat_h * lat_w // (pipe.patch_size[1] * pipe.patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / pipe.sp_size)) * pipe.sp_size
    pipe._cross_attn_initialized = False

    gens = [torch.Generator(device=dev).manual_seed(int(s)) for s in seeds]
    noise = torch.stack([torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32, generator=g, device=dev)
                         for g in gens])                                   # [B, 16, lat_f, H, W]

    msk = torch.ones(1, F, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]  # [4, lat_f, H, W]

    pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=shift)
    timesteps = pipe.scheduler.timesteps[list(timesteps_index)]
    contexts = [_context_for(pipe, p) for p in prompts]
    cam = _camera_embedding(pipe, action_path, chunk_size, lat_f, lat_h, lat_w, h, w)

    ys = []
    for im in imgs_t:
        first = torch.nn.functional.interpolate(im[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1)
        y = pipe.vae.encode([torch.concat([first, torch.zeros(3, F - 1, h, w)], dim=1).to(dev)])[0]
        ys.append(torch.concat([msk, y]))                                   # [20, lat_f, H, W]

    cfg = pipe.model.config
    frame_seqlen = int(lat_h * lat_w // 4)
    kv_size = frame_seqlen * pipe.local_attn_size if pipe.local_attn_size > -1 else frame_seqlen * lat_f
    head_dim = cfg.dim // cfg.num_heads
    self_kv = pipe._initialize_self_kv_cache(
        num_layers=cfg.num_layers, shape=[B, kv_size, cfg.num_heads // pipe.sp_size, head_dim],
        dtype=pipe.pipe_dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=cfg.num_layers, shape=[B, max_sequence_length, cfg.num_heads, head_dim],
        dtype=pipe.pipe_dtype, device=dev)

    with torch.amp.autocast('cuda', dtype=pipe.param_dtype), torch.no_grad():
        latent_chunks = noise.split(chunk_size, dim=2)
        cond_chunks = [y.split(chunk_size, dim=1) for y in ys]
        cam_chunks = cam.split(chunk_size, dim=2)
        preds = []
        for chunk_id in range(len(latent_chunks)):
            current = latent_chunks[chunk_id]                              # [B, 16, cs, H, W]
            kwargs = {
                'context': contexts, 'seq_len': max_seq_len,
                'y': [cond_chunks[b][chunk_id] for b in range(B)],
                'dit_cond_dict': {"c2ws_plucker_emb": tuple(cam_chunks[chunk_id] for _ in range(B))},
                'kv_cache': self_kv, 'crossattn_cache': cross_kv,
                'current_start': chunk_id * chunk_size * frame_seqlen,
                'max_attention_size': kv_size, 'frame_seqlen': frame_seqlen,
            }
            for ti in range(len(timesteps)):
                t = timesteps[ti]
                flow = pipe.model(x=[current[b] for b in range(B)], t=torch.stack([t] * B).to(dev),
                                  cross_attn_first_call=not pipe._cross_attn_initialized, **kwargs)
                pipe._cross_attn_initialized = True
                x0 = torch.stack([pipe._convert_flow_pred_to_x0(flow_pred=flow[b], xt=current[b],
                                                                timestep=t, scheduler=pipe.scheduler)
                                  for b in range(B)])
                if ti == len(timesteps) - 1:
                    break
                eps = torch.stack([torch.randn(x0[b].shape, generator=gens[b], device=dev, dtype=x0.dtype)
                                   for b in range(B)])
                current = pipe.scheduler.add_noise(x0, eps, timesteps[ti + 1])
            preds.append(x0)
            pipe.model(x=[x0[b] for b in range(B)], t=torch.stack([timesteps[-1] * 0.0] * B).to(dev),
                       cross_attn_first_call=False, **kwargs)
        latents = torch.cat(preds, dim=2)                                   # [B, 16, lat_f, H, W]
        videos = pipe.vae.decode([latents[b] for b in range(B)])
    return videos


# --------------------------------------------------------------------------
# measurement helpers for the batch test
# --------------------------------------------------------------------------

def system_used_gb() -> float:
    """MemTotal - MemAvailable from /proc/meminfo (GB, 1e9)."""
    vals = {}
    for line in MEMINFO.read_text().splitlines():
        k, _, rest = line.partition(":")
        if k in ("MemTotal", "MemAvailable"):
            vals[k] = int(rest.split()[0]) * 1024
    return (vals["MemTotal"] - vals["MemAvailable"]) / 1e9


class SystemMemorySampler:
    """Background sampler of system memory in use; `.peak_gb` after stop()."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.peak_gb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak_gb = max(self.peak_gb, system_used_gb())
            time.sleep(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self.peak_gb = max(self.peak_gb, system_used_gb())


def rel_l2(a, b) -> float:
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def psnr_uint8(a, b) -> float:
    """PSNR (dB) between two uint8 arrays of equal shape."""
    import numpy as np

    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
