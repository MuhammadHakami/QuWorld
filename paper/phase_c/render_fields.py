"""Phase-C P0, step 1: render TRL2D field frames as 3-channel RGB videos.

Maps The Well `turbulent_radiative_layer_2D` trajectories — normalized
(T, 4, 128, 384) field stacks (density, pressure, vx, vy) from
data/phaseb_cache.pt (built by paper/phase_b/data_ram.py) — into
480x832 RGB video tensors in [-1, 1] (stored uint8), the input format of
the LingBot-World v2 / Wan2.1 VAE (baseline/lingbot-world-v2/wan/
modules/vae2_1.py; pipeline preprocessing at wan/image2video.py:485).

Field -> RGB mapping (fixed, linear + clip, documented, invertible-ish)
----------------------------------------------------------------------
Pure torch, no matplotlib. RGB = clip(GAIN * M @ fields, -1, 1) with a
3x4 matrix M constructed (deterministically, at import) from:

  field-space basis, all orthogonal to the discarded common mode
  n = (1,1,1,1)/2 of the four z-scored fields:
    u1 = (3,-1,-1,-1)/sqrt(12)   density-dominant component
    u2 = (0, 2,-1,-1)/sqrt(6)    pressure-dominant component
    u3 = (0, 0, 1,-1)/sqrt(2)    shear component (vx - vy)

  channel loadings (columns orthonormalized in order via QR, so the
  density direction is kept exactly):
    density  -> (R,G,B) = (0.70, 0.60, -0.39)  viridis-like: brightens
                red+green, darkens blue as density rises
    pressure -> (0.10, -0.50, 0.60)            green<->blue opponent
    shear    -> (0.00, 0.00, 1.00)             blue

M has orthonormal rows, so the pseudo-inverse is M^T: fields can be
recovered as M^T @ rgb / GAIN up to (a) the common-mode component
(null space, spread evenly over the 4 fields), (b) clipping of >~3.3
sigma tails, (c) uint8 quantization. --smoke reports the measured
roundtrip error on real data.

Spatial mapping 128x384 -> 480x832 (aspect preserved with padding)
------------------------------------------------------------------
bilinear resize to 278x832 (width fits exactly, scale 832/384 = 13/6;
height 128 * 13/6 = 277.33 rounded to 278 => 0.24% anisotropy), then
pad 101 rows of mid-gray (0.0 in [-1,1]) top and bottom to 480.

Output: data/phase_c/renders/{split}_{idx:03d}.pt, each a dict
  {"video": uint8 (3, T, 480, 832), "meta": {...}}   (~120 MB/traj)
Dequantize with video.float() / 127.5 - 1.

CPU-only. Full run: `uv run python paper/phase_c/render_fields.py`
Smoke (2 trajectories + verification): `... --smoke`
"""

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "paper" / "phase_b"))
import data_ram  # noqa: E402  (paper/phase_b/data_ram.py)

OUT_DIR = ROOT / "data" / "phase_c" / "renders"
FIELDS = ("density", "pressure", "vx", "vy")
SRC_H, SRC_W = 128, 384
RESIZE_H, RESIZE_W = 278, 832
TARGET_H, TARGET_W = 480, 832
PAD_TOP = (TARGET_H - RESIZE_H) // 2          # 101
PAD_BOTTOM = TARGET_H - RESIZE_H - PAD_TOP    # 101
GAIN = 0.30


def build_field_rgb_matrix() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (M, null): M (3,4) with orthonormal rows, null (4,) unit
    vector spanning ker(M) — the discarded common mode."""
    u1 = torch.tensor([3.0, -1.0, -1.0, -1.0]) / 12.0 ** 0.5
    u2 = torch.tensor([0.0, 2.0, -1.0, -1.0]) / 6.0 ** 0.5
    u3 = torch.tensor([0.0, 0.0, 1.0, -1.0]) / 2.0 ** 0.5
    basis = torch.stack([u1, u2, u3])                       # (3 basis, 4 fields)

    loadings = torch.tensor([                                # columns = basis
        [0.70, 0.10, 0.00],                                  # R
        [0.60, -0.50, 0.00],                                 # G
        [-0.39, 0.60, 1.00],                                 # B
    ])
    q, r = torch.linalg.qr(loadings)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)       # deterministic signs
    m = q @ basis                                            # (3 rgb, 4 fields)

    null = torch.ones(4) / 2.0
    _validate_matrix(m, null)
    return m, null


def _validate_matrix(m: torch.Tensor, null: torch.Tensor) -> None:
    gram_err = (m @ m.T - torch.eye(3)).abs().max().item()
    null_err = (m @ null).abs().max().item()
    if gram_err > 1e-5 or null_err > 1e-5:
        raise AssertionError(
            f"field->RGB matrix broken: gram_err={gram_err:.2e} "
            f"null_err={null_err:.2e}")


def fields_to_rgb(fields: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """(T, 4, H, W) z-scored fields -> (T, 3, H, W) in [-1, 1]."""
    if fields.dim() != 4 or fields.shape[1] != 4:
        raise ValueError(f"expected (T,4,H,W), got {tuple(fields.shape)}")
    rgb = torch.einsum("cf,tfhw->tchw", m, fields) * GAIN
    return rgb.clamp(-1.0, 1.0)


def rgb_to_fields(rgb: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Approximate inverse (pinv = M^T since rows are orthonormal)."""
    return torch.einsum("cf,tchw->tfhw", m, rgb) / GAIN


def upscale_and_pad(rgb: torch.Tensor) -> torch.Tensor:
    """(T, 3, 128, 384) -> (T, 3, 480, 832), aspect kept, gray padding."""
    if rgb.shape[-2:] != (SRC_H, SRC_W):
        raise ValueError(f"expected {SRC_H}x{SRC_W} frames, got {tuple(rgb.shape)}")
    up = F.interpolate(rgb, size=(RESIZE_H, RESIZE_W), mode="bilinear",
                       align_corners=False)
    return F.pad(up, (0, 0, PAD_TOP, PAD_BOTTOM), value=0.0)


def quantize(video: torch.Tensor) -> torch.Tensor:
    """[-1,1] float -> uint8 (dequantize: x/127.5 - 1)."""
    return ((video + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)


def render_traj(fields: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, float]:
    """(T, 4, 128, 384) -> (uint8 video (3, T, 480, 832), clip fraction)."""
    rgb_raw = torch.einsum("cf,tfhw->tchw", m, fields) * GAIN
    clip_frac = (rgb_raw.abs() > 1.0).float().mean().item()
    video = upscale_and_pad(rgb_raw.clamp(-1.0, 1.0))
    return quantize(video).permute(1, 0, 2, 3).contiguous(), clip_frac


def make_meta(name: str, t: int, m: torch.Tensor, clip_frac: float) -> dict:
    return {
        "traj": name,
        "field_order": FIELDS,
        "src_shape": (t, 4, SRC_H, SRC_W),
        "resize_hw": (RESIZE_H, RESIZE_W),
        "pad_top_bottom": (PAD_TOP, PAD_BOTTOM),
        "gain": GAIN,
        "matrix": m,                       # RGB = clip(GAIN * matrix @ fields)
        "clip_frac": clip_frac,
        "dequant": "video.float() / 127.5 - 1",
        "value_range": (-1.0, 1.0),
    }


def save_atomic(path: pathlib.Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def iter_trajs(train: torch.Tensor, valid: torch.Tensor):
    for i in range(train.shape[0]):
        yield f"train_{i:03d}", train[i]
    for i in range(valid.shape[0]):
        yield f"valid_{i:03d}", valid[i]


def smoke_report(fields: torch.Tensor, video_u8: torch.Tensor,
                 m: torch.Tensor, null: torch.Tensor) -> None:
    """Verify shapes and measure honest roundtrip error (clip+uint8
    included, at source resolution — resize excluded by design)."""
    t = fields.shape[0]
    assert video_u8.shape == (3, t, TARGET_H, TARGET_W), video_u8.shape
    assert video_u8.dtype == torch.uint8

    rgb_q = quantize(fields_to_rgb(fields, m)).float() / 127.5 - 1.0
    recon = rgb_to_fields(rgb_q, m)
    # component of the fields recoverable in principle (null mode removed)
    proj = fields - torch.einsum("f,tfhw->thw", null, fields).unsqueeze(1) \
        * null.view(1, 4, 1, 1)
    err = recon - fields
    print(f"  roundtrip rmse per field (vs raw z-fields): "
          f"{[round(err[:, i].pow(2).mean().sqrt().item(), 4) for i in range(4)]}")
    print(f"  roundtrip rmse per field (vs recoverable projection): "
          f"{[round((recon - proj)[:, i].pow(2).mean().sqrt().item(), 4) for i in range(4)]}")
    print(f"  field std: {[round(fields[:, i].std().item(), 3) for i in range(4)]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=pathlib.Path, default=OUT_DIR)
    ap.add_argument("--smoke", action="store_true",
                    help="render only train_000 + valid_000 and verify")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N trajectories (resume-friendly)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    m, null = build_field_rgb_matrix()
    print("field->RGB matrix (rows R,G,B; cols density,pressure,vx,vy):")
    print(m)
    print(f"null (lost) direction: {null.tolist()}  gain={GAIN}")

    train, valid = data_ram.load()
    print(f"cache: train {tuple(train.shape)}  valid {tuple(valid.shape)}")
    args.out.mkdir(parents=True, exist_ok=True)

    todo = list(iter_trajs(train, valid))
    if args.smoke:
        todo = [todo[0], todo[train.shape[0]]]          # train_000, valid_000
    if args.limit is not None:
        todo = todo[:args.limit]

    n_done = 0
    for name, traj in todo:
        path = args.out / f"{name}.pt"
        if path.exists() and not args.overwrite:
            print(f"skip {name} (exists)")
            continue
        fields = traj.float()                            # (T, 4, 128, 384)
        video, clip_frac = render_traj(fields, m)
        save_atomic(path, {"video": video, "meta": make_meta(
            name, fields.shape[0], m, clip_frac)})
        mb = path.stat().st_size / 1e6
        print(f"{name}: video {tuple(video.shape)} uint8, "
              f"clip {clip_frac * 100:.3f}%, {mb:.1f} MB")
        if args.smoke:
            smoke_report(fields, video, m, null)
        n_done += 1

    print(f"done: {n_done} rendered -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
