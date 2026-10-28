"""CPU-only video I/O and Physics-IQ contract helpers (ffmpeg/ffprobe on PATH).

Torch-free on purpose: the trim / verify / score stages and the unit tests
run without the GPU environment. The contract mirrored here is the official
scorer's `validate_generations` (baseline/physics-iq/physiq/run_physics_iq.py
:168-200): exactly 198 `.mp4` files, sorted names prefixed 0001_..0198_,
each with ffprobe `format=duration` == 5 s within abs_tol 1e-3.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

FPS = 16                 # cfg.sample_fps (wan/configs/shared_config.py:17)
TRIM_FRAMES = 80         # 80 / 16 = 5.000 s
TARGET_DURATION_S = 5.0
DURATION_TOL_S = 1e-3    # run_physics_iq.py:196 abs_tol
N_VIDEOS = 198
NAME_RE = re.compile(
    r"^(\d{4})_(perspective-(?:left|center|right))_(trimmed-[A-Za-z0-9\-]+)\.mp4$"
)
X264_ARGS = (
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
    "-an", "-movflags", "+faststart",
)
TMP_SUFFIX = ".tmp.mp4"


class VideoToolError(RuntimeError):
    """ffmpeg/ffprobe missing or failed."""


@dataclass(frozen=True)
class Probe:
    duration_s: float
    nb_frames: int
    fps: float
    width: int
    height: int

    @property
    def duration_ok(self) -> bool:
        return math.isclose(self.duration_s, TARGET_DURATION_S, abs_tol=DURATION_TOL_S)


def require_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise VideoToolError(f"not on PATH: {', '.join(missing)}")


def _run(cmd: list[str], stdin_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, input=stdin_bytes, capture_output=True)
    if res.returncode != 0:
        raise VideoToolError(
            f"{cmd[0]} failed ({res.returncode}): {res.stderr.decode(errors='replace').strip()[-2000:]}"
        )
    return res


def _parse_rate(text: str) -> float:
    num, _, den = text.partition("/")
    return float(num) / float(den or 1)


def probe(path: Path | str) -> Probe:
    """ffprobe the first video stream; duration is `format=duration` (scorer's field)."""
    res = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ])
    info = json.loads(res.stdout.decode())
    stream = info["streams"][0]
    return Probe(
        duration_s=float(info["format"]["duration"]),
        nb_frames=int(stream["nb_read_frames"]),
        fps=_parse_rate(stream["r_frame_rate"]),
        width=int(stream["width"]),
        height=int(stream["height"]),
    )


def write_mp4_rgb(frames, path: Path | str, fps: int = FPS) -> Probe:
    """Encode uint8 RGB frames (N, H, W, 3) to H.264 mp4 atomically; returns its probe."""
    import numpy as np  # local: keep module importable without numpy for name checks

    arr = np.ascontiguousarray(frames)
    if arr.dtype != np.uint8 or arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected uint8 (N,H,W,3), got {arr.dtype} {arr.shape}")
    n, h, w, _ = arr.shape
    if h % 2 or w % 2:
        raise ValueError(f"yuv420p needs even frame size, got {w}x{h}")
    dst = Path(path)
    tmp = dst.with_name(dst.name + TMP_SUFFIX)
    _run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", *X264_ARGS, str(tmp),
    ], stdin_bytes=arr.tobytes())
    info = probe(tmp)
    if info.nb_frames != n:
        tmp.unlink(missing_ok=True)
        raise VideoToolError(f"wrote {info.nb_frames} frames, expected {n}: {dst.name}")
    os.replace(tmp, dst)
    return info


def trim_to_frames(src: Path | str, dst: Path | str,
                   n_frames: int = TRIM_FRAMES, fps: int = FPS) -> Probe:
    """Re-encode the first `n_frames` frames at `fps` (80 @ 16 fps = 5.000 s), verified."""
    dst = Path(dst)
    tmp = dst.with_name(dst.name + TMP_SUFFIX)
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-frames:v", str(n_frames), "-r", str(fps), *X264_ARGS, str(tmp),
    ])
    info = probe(tmp)
    expected_s = n_frames / fps
    if info.nb_frames != n_frames or not math.isclose(info.duration_s, expected_s, abs_tol=DURATION_TOL_S):
        tmp.unlink(missing_ok=True)
        raise VideoToolError(
            f"trim produced {info.nb_frames} frames / {info.duration_s:.4f} s, "
            f"expected {n_frames} / {expected_s:.3f} s: {dst.name}"
        )
    os.replace(tmp, dst)
    return info


def synthetic_mp4(path: Path | str, n_frames: int, fps: int = FPS,
                  size: tuple[int, int] = (64, 64)) -> Probe:
    """Test helper: ffmpeg `testsrc` clip with exactly `n_frames` frames."""
    w, h = size
    dst = Path(path)
    _run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate={fps}",
        "-frames:v", str(n_frames), *X264_ARGS, str(dst),
    ])
    return probe(dst)


def list_mp4s(folder: Path | str) -> tuple[Path, ...]:
    return tuple(sorted(p for p in Path(folder).iterdir() if p.name.endswith(".mp4")))


def clear_tmp_files(folder: Path | str) -> tuple[Path, ...]:
    """Remove interrupted `*.tmp.mp4` leftovers (they end in .mp4 and would break the scorer)."""
    removed = tuple(p for p in Path(folder).glob(f"*{TMP_SUFFIX}"))
    for p in removed:
        p.unlink()
    return removed


def validate_contract(folder: Path | str, n_videos: int = N_VIDEOS,
                      check_media: bool = True) -> tuple[str, ...]:
    """Return problems (empty tuple == the folder satisfies the scorer contract).

    Mirrors run_physics_iq.py: rename_generated_videos rejects any non-.mp4
    entry (:98), validate_generations enforces count, 0001_.. prefixes and the
    5 s duration (:168-200); our own extra checks are the name pattern,
    80 frames and 16 fps.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return (f"not a directory: {folder}",)
    entries = sorted(folder.iterdir())
    problems = [f"non-mp4 entry (scorer raises): {p.name}" for p in entries if not p.name.endswith(".mp4")]
    mp4s = [p for p in entries if p.name.endswith(".mp4")]
    if len(mp4s) != n_videos:
        problems.append(f"found {len(mp4s)} videos but expected {n_videos}")
    for i, p in enumerate(mp4s, start=1):
        if not p.name.startswith(f"{i:04d}_"):
            problems.append(f"{p.name} does not start with expected prefix {i:04d}_")
        if not NAME_RE.match(p.name):
            problems.append(f"name does not match {{id}}_<perspective>_<scenario>.mp4: {p.name}")
    if check_media:
        problems.extend(_media_problems(mp4s))
    return tuple(problems)


def _media_problems(mp4s: list[Path]) -> list[str]:
    out = []
    for p in mp4s:
        try:
            info = probe(p)
        except (VideoToolError, KeyError, ValueError, IndexError) as e:
            out.append(f"unreadable {p.name}: {e}")
            continue
        if not info.duration_ok:
            out.append(f"{p.name} is {info.duration_s} s but needs to be 5 s")
        if info.nb_frames != TRIM_FRAMES:
            out.append(f"{p.name} has {info.nb_frames} frames, expected {TRIM_FRAMES}")
        if not math.isclose(info.fps, FPS, abs_tol=1e-6):
            out.append(f"{p.name} is {info.fps} fps, expected {FPS}")
    return out
