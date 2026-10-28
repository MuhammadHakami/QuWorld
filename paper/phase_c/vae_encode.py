"""Phase-C P0, step 2: encode rendered TRL2D videos with the frozen Wan2.1 VAE.

Loads ONLY the VAE from data/lingbot-world-v2-14b-causal-fast/
(Wan2.1_VAE.pth, 485 MB) — never the 14B DiT or the umt5 text encoder.

Code path (baseline/lingbot-world-v2):
  - class `Wan2_1_VAE`            wan/modules/vae2_1.py:617
      builds `WanVAE_` via `_video_vae` (vae2_1.py:590-614): model is
      constructed on the meta device (vae2_1.py:606-607) and the
      checkpoint is loaded with `assign=True` (vae2_1.py:611-612).
      Config: dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2,
      temperal_downsample=[False,True,True] (vae2_1.py:595-602).
  - encode API: `Wan2_1_VAE.encode(videos)` (vae2_1.py:645-653), videos
      a list of [3, T, H, W] tensors in [-1, 1]; T must be 4k+1
      (WanVAE_.encode iterates 1 + (T-1)//4 causal steps, vae2_1.py:519).
  - returned latents are already in the DiT's normalized latent space:
      (mu - mean) * (1/std) per 16 channels (vae2_1.py:534-538 with the
      channel stats at vae2_1.py:627-637) — the same space as `noise`
      and `x0` inside `generate()` (wan/image2video.py:511-518, 688-693).
  - the pipeline constructs this VAE with dtype=torch.float (fp32
      autocast; wan/image2video.py:121-123, default at vae2_1.py:622).
      We default to bf16 (halves activation memory; use --dtype fp32
      for bit-parity with the released pipeline).

Shapes: (3, 101, 480, 832) render  ->  (16, 26, 60, 104) latent
        (vae_stride (4,8,8), wan/configs/wan_i2v_A14B.py:16;
         26 = 1 + (101-1)//4, 60 = 480/8, 104 = 832/8).

Also saves the first-frame conditioning latent used by the DiT's `y`
input: encode(concat([frame0, zeros(3, T-1, H, W)])) exactly as
image2video.py:599-607. Because the VAE is temporally causal, encoding
with T=101 and slicing latent frames [:24] later equals encoding with
T=93 — teacher_cache.py relies on this.

Per-trajectory checkpoint/resume: output files are written atomically
(.tmp -> rename) and existing outputs are skipped, so the job can be
killed and relaunched at any time.

Memory envelope: fp32 weights 0.49 GB + bf16 activations (4-frame causal
micro-chunks internally) — measured peak expected ~3-6 GB, far below the
20 GB budget; per-trajectory peak is printed and asserted < 20 GB.

GPU run (DO NOT run while GPU is busy):
    uv run python paper/phase_c/vae_encode.py
CPU dry-run (no weight materialization, no GPU):
    uv run python paper/phase_c/vae_encode.py --dry-run
"""

import argparse
import importlib.util
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
BASELINE = ROOT / "baseline" / "lingbot-world-v2"
VAE_MODULE = BASELINE / "wan" / "modules" / "vae2_1.py"


def load_vae_module():
    """Import vae2_1.py directly by file path. The module is standalone
    (torch + einops only); going through `import wan...` would execute
    wan/__init__.py, which pulls easydict/diffusers/flash_attn — deps of
    the DiT, not of the VAE, and absent from this env by design."""
    spec = importlib.util.spec_from_file_location("wan_vae2_1", VAE_MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

WEIGHTS = ROOT / "data" / "lingbot-world-v2-14b-causal-fast"
VAE_PTH = WEIGHTS / "Wan2.1_VAE.pth"           # wan_i2v_A14B.py:15
RENDER_DIR = ROOT / "data" / "phase_c" / "renders"
OUT_DIR = ROOT / "data" / "phase_c" / "latents"
Z_DIM = 16
VAE_STRIDE = (4, 8, 8)                          # wan_i2v_A14B.py:16
PEAK_BUDGET_GB = 20.0

# must mirror _video_vae's cfg (vae2_1.py:595-602) with z_dim=16
VAE_CFG = dict(dim=96, z_dim=Z_DIM, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
               attn_scales=[], temperal_downsample=[False, True, True],
               dropout=0.0)


def expected_latent_shape(t: int, h: int, w: int) -> tuple[int, int, int, int]:
    if (t - 1) % 4 != 0:
        raise ValueError(f"T must be 4k+1 for the causal VAE, got {t}")
    if h % VAE_STRIDE[1] or w % VAE_STRIDE[2]:
        raise ValueError(f"H,W must be divisible by 8, got {h}x{w}")
    return (Z_DIM, 1 + (t - 1) // VAE_STRIDE[0],
            h // VAE_STRIDE[1], w // VAE_STRIDE[2])


def load_render(path: pathlib.Path) -> torch.Tensor:
    d = torch.load(path, weights_only=True, mmap=True)
    video = d["video"]
    if video.dtype != torch.uint8 or video.dim() != 4 or video.shape[0] != 3:
        raise ValueError(f"{path}: bad render tensor {video.dtype} "
                         f"{tuple(video.shape)}")
    return video.float() / 127.5 - 1.0            # (3, T, H, W) in [-1, 1]


def dry_run() -> int:
    """CPU-only verification: build the VAE on the meta device (zero
    memory) and check state-dict keys/shapes against the checkpoint
    (mmap'd, tensor data untouched). No GPU, no materialization."""
    vae2_1 = load_vae_module()

    if not VAE_PTH.exists():
        print(f"FAIL: checkpoint missing: {VAE_PTH}")
        return 1

    with torch.device("meta"):                     # mirrors vae2_1.py:606-607
        model = vae2_1.WanVAE_(**VAE_CFG)          # vae2_1.py:482
    model_sd = model.state_dict()
    ckpt_sd = torch.load(VAE_PTH, map_location="cpu", weights_only=True,
                         mmap=True)

    missing = sorted(set(model_sd) - set(ckpt_sd))
    unexpected = sorted(set(ckpt_sd) - set(model_sd))
    mismatched = sorted(k for k in set(model_sd) & set(ckpt_sd)
                        if tuple(model_sd[k].shape) != tuple(ckpt_sd[k].shape))

    print(f"model params/buffers: {len(model_sd)} | checkpoint tensors: "
          f"{len(ckpt_sd)}")
    print(f"missing in ckpt: {len(missing)} | unexpected in ckpt: "
          f"{len(unexpected)} | shape mismatches: {len(mismatched)}")
    for label, keys in (("missing", missing), ("unexpected", unexpected),
                        ("mismatch", mismatched)):
        for k in keys[:8]:
            print(f"  {label}: {k}")

    n_params = sum(v.numel() for v in ckpt_sd.values())
    print(f"checkpoint params: {n_params / 1e6:.1f} M "
          f"({n_params * 4 / 1e9:.2f} GB fp32)")
    print(f"latent shape for (3,101,480,832): "
          f"{expected_latent_shape(101, 480, 832)}")

    renders = sorted(RENDER_DIR.glob("*.pt")) if RENDER_DIR.exists() else []
    print(f"renders present: {len(renders)} in {RENDER_DIR}")

    ok = not missing and not unexpected and not mismatched
    print("DRY-RUN " + ("OK: state dict matches WanVAE_ exactly." if ok
                        else "FAILED: state-dict mismatch (see above)."))
    return 0 if ok else 1


def build_vae(device: str, dtype: torch.dtype):
    """Materialize the VAE exactly as the pipeline does
    (image2video.py:121-123), with our dtype for the encode autocast."""
    vae2_1 = load_vae_module()
    return vae2_1.Wan2_1_VAE(z_dim=Z_DIM, vae_pth=str(VAE_PTH),  # vae2_1.py:617
                             dtype=dtype, device=device)


def encode_one(vae, video: torch.Tensor, device: str) -> dict:
    """video (3, T, H, W) in [-1,1] -> {"latent", "cond_latent"} fp32 CPU."""
    t, h, w = video.shape[1], video.shape[2], video.shape[3]
    exp = expected_latent_shape(t, h, w)
    with torch.no_grad():
        z = vae.encode([video.to(device)])[0]                 # (16, lat_f, lat_h, lat_w)
        # first-frame conditioning input, mirrors image2video.py:599-607
        cond_in = torch.cat([video[:, :1], torch.zeros(3, t - 1, h, w)],
                            dim=1)
        z_cond = vae.encode([cond_in.to(device)])[0]
    for name, lat in (("latent", z), ("cond_latent", z_cond)):
        if tuple(lat.shape) != exp:
            raise RuntimeError(f"{name} shape {tuple(lat.shape)} != {exp}")
    return {"latent": z.cpu().float(), "cond_latent": z_cond.cpu().float()}


def encode_all(args) -> int:
    renders = sorted(args.renders.glob("*.pt"))
    if not renders:
        print(f"FAIL: no renders in {args.renders} — run render_fields.py first")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    vae = build_vae(args.device, dtype)
    print(f"VAE loaded on {args.device} (autocast {args.dtype})")

    n_done = 0
    for path in renders[:args.limit]:
        out = args.out / path.name
        if out.exists() and not args.overwrite:
            print(f"skip {path.stem} (exists)")
            continue
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        latents = encode_one(vae, load_render(path), args.device)
        latents["meta"] = {"traj": path.stem, "dtype": args.dtype,
                           "space": "wan-normalized (vae2_1.py:534-538)",
                           "vae_pth": str(VAE_PTH)}
        tmp = out.with_suffix(".tmp")
        torch.save(latents, tmp)
        tmp.replace(out)
        msg = f"{path.stem}: latent {tuple(latents['latent'].shape)}"
        if args.device.startswith("cuda"):
            peak = torch.cuda.max_memory_allocated() / 1e9
            msg += f", peak {peak:.2f} GB"
            if peak > PEAK_BUDGET_GB:
                raise RuntimeError(f"peak GPU {peak:.2f} GB > "
                                   f"{PEAK_BUDGET_GB} GB budget")
        print(msg)
        n_done += 1
    print(f"done: {n_done} encoded -> {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU key/shape verification only (no GPU)")
    ap.add_argument("--renders", type=pathlib.Path, default=RENDER_DIR)
    ap.add_argument("--out", type=pathlib.Path, default=OUT_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run()
    return encode_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
