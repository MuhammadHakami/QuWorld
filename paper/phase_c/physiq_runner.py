"""Phase-C Physics-IQ harness — stages: download | actiondir | t5cache |
adaptercheck | generate | batchtest | trim | verify | score | compare |
project | plan (all support --dry-run; generate/batchtest are the GPU stages).

Scores frozen LingBot-World v2 arms {vanilla, quantum (mr30ryd), lora} on
the Physics-IQ benchmark (66 scenarios x 3 perspectives = 198 i2v
generations per arm) with the official scorer — the benchmark's first
quantum entries (11-master-plan.md Phase C; 15-challenge-ladder.md
"Physics-IQ" row). Companion modules: physiq_video.py (ffmpeg I/O +
scorer contract), physiq_t5.py (CPU T5 context cache), physiq_adapter.py
(P1 checkpoint contract), physiq_batched.py (B>1 sampling), physiq_compare.py
(paired 198-row comparison, runs in the scorer env).

Verified benchmark facts (baseline/physics-iq, cloned 2026-08-17)
=================================================================
- Scorer entry point: physiq/run_physics_iq.py (own uv env: `uv sync` in
  baseline/physics-iq; needs ffprobe on PATH). `--original_physics_iq`
  selects the original GT. Both scores (final_score_origround /
  final_score_view) land in <run>_metrics.json.
- Inputs per arm-run: EXACTLY 198 mp4s named `{0001..0198}_<perspective>_
  <scenario>.mp4`, each EXACTLY 5.000 s (run_physics_iq.py:168-200,
  abs_tol=1e-3 via ffprobe); nothing else in the folder (:98). Scoring is
  all-or-nothing; generation is resumable per scenario.
- Conditioning for i2v: switch-frames/*.jpg (3840x2160), 198 take-1 rows of
  the descriptions CSV give the prompt + output name (README.md D1).

Verified LingBot-World v2 call path (baseline/lingbot-world-v2)
===============================================================
- Pipe: wan.WanI2VCausal(config=WAN_CONFIGS['i2v-A14B'], checkpoint_dir,
  infer_mode='causal_fast', local_attn_size=18, sink_size=6 per run_fast.sh);
  pipe.generate(prompt, img, action_path, chunk_size, max_area, frame_num,
  timesteps_index, shift, seed) (image2video.py:387-422).
- action_path REQUIRED (image2video.py:477-482): poses.npy (N,4,4) +
  intrinsics.npy (N,4 in the 480x832 reference frame, :551-561). Static
  camera = identity poses -> identity relatives (:572).
- GEOMETRY (image2video.py:487-500): lat_h = round(sqrt(max_area*ar)//8//2*2).
  The `//8` floors, and for ANY image of exact aspect 480:832 (even a
  native 480x832 one) the float sqrt lands a hair below 480 -> lat_h 58 ->
  output 832x464. The P1 adapters and the teacher cache live at lat
  60x104 (480x832 renders), so every arm generates at that geometry: the
  switch frame is center-cropped to aspect 480:832 (3840x2160 -> 3744x2160,
  2.5% of the width) and max_area carries a +1000 margin, which yields
  exactly lat 60x104 -> 480x832, frame_seqlen 1560 (verified numerically,
  test_physiq_contract.py). Identical for all arms; the scorer resizes
  generated videos to 960x540 anyway.
- Frame count: frame_num=81 -> lat_f=21, chunk_size=3 keeps 21 (chunk 4
  would truncate to 20 -> 77 frames < 5 s). 81 frames @16 fps -> trimmed
  to 80 = 5.000 s. fps = cfg.sample_fps = 16 (shared_config.py:17).
- Attention on this aarch64 box (no flash_attn): self-attention already
  falls back to torch SDPA (attention.py:147-179); cross-attention calls
  flash_attention() directly (model_fast.py:208) and asserts flash_attn
  (attention.py:111) -> patch_attention_fallback() rebinds it to
  attention(). context_lens is always None (model_fast.py:572), so no key
  mask is lost: SDPA is numerically equivalent (max abs diff 1.5e-7 fp32).
- T5: umt5-xxl on the Grace CPU costs ~43 s/prompt and 11 GB RSS; contexts
  are cached on disk by physiq_t5 and the pipe is built WITHOUT loading
  T5 (T5EncoderModel stubbed during construction).
- Video output: wan.utils.utils.save_video needs imageio (absent) and
  swallows exceptions (utils.py:119) -> frames go through an ffmpeg
  rawvideo pipe (physiq_video.write_mp4_rgb) instead.
- Adapter tap point (P2): the final-timestep x0 of each chunk, appended at
  image2video.py:702 and fed to the KV-update forward at :705-709.
  wrap_final_step_x0() wraps pipe._convert_flow_pred_to_x0 at runtime (no
  source edits) and adapts only that x0.

Resumable per scenario: atomic writes + skip-existing everywhere.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import datetime as dt
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physiq_adapter  # noqa: E402
import physiq_t5  # noqa: E402
from physiq_video import (  # noqa: E402
    FPS, N_VIDEOS, TRIM_FRAMES, clear_tmp_files, list_mp4s, require_tools,
    trim_to_frames, validate_contract, write_mp4_rgb,
)

REPO = Path(__file__).resolve().parents[2]
PHYSIQ_REPO = REPO / "baseline" / "physics-iq"
LINGBOT_REPO = REPO / "baseline" / "lingbot-world-v2"
LINGBOT_WEIGHTS = REPO / "data" / "lingbot-world-v2-14b-causal-fast"
DATA_ROOT = REPO / "data" / "phase_c" / "physiq"
GT_ORIGINAL = DATA_ROOT / "physics-IQ-benchmark"
MANIFEST = DATA_ROOT / "bulk_manifest.json"
ACTION_DIR = DATA_ROOT / "static_action"
RUNS_ROOT = DATA_ROOT / "generated_videos"
TRIMMED_ROOT = DATA_ROOT / "generated_videos_5s"
RESULTS_ROOT = DATA_ROOT / "results"
PHASE_C_RUNS = REPO / "runs" / "phase_c"
STATUS_FILE = PHASE_C_RUNS / "phase_c_status"
BATCH_MODE_FILE = PHASE_C_RUNS / "physiq_batch_mode"

GCS_BUCKET = "physics-iq-benchmark"
ARMS = ("vanilla", "quantum", "lora")
N_SCENARIO_VIEWS = N_VIDEOS  # 66 scenarios x 3 perspectives (take-1 rows)

# Generation constants — sources in the module docstring.
FRAME_NUM = 81
CHUNK_SIZE = 3
CROP_ASPECT = (480, 832)          # (h, w) aspect the switch frame is cropped to
MAX_AREA = 480 * 832 + 1000       # +1000: float-boundary margin, see GEOMETRY
LAT_H, LAT_W = 60, 104            # -> 480x832 output, frame_seqlen 1560
LOCAL_ATTN_SIZE = 18              # run_fast.sh
SINK_SIZE = 6                     # run_fast.sh
TIMESTEPS_INDEX = (0, 250, 500, 750)  # generate() default, image2video.py:393
SHIFT = 3.0                       # 480p recommendation (image2video.py:451); --shift overrides
BASE_SEED = 1337
INTRINSICS_ROW = (415.5298, 415.6922, 415.77786, 239.77779)  # examples/03, 480x832 frame
BATCHTEST_MEM_LIMIT_GB = 105.0    # coordinator's bound on system memory for batch=2
BATCHTEST_REL_L2_MAX = 0.05       # chunk-0 x0 agreement batch-2 vs batch-1
BATCHTEST_MIN_SPEEDUP = 1.05      # batch=2 must be >=5% faster per video


@dataclass(frozen=True)
class Scenario:
    """One take-1 scenario x perspective row of the descriptions CSV."""

    video_id: str        # "0001"
    prompt: str
    out_name: str        # "0001_perspective-left_trimmed-....mp4"
    switch_frame: Path


def fail(msg: str) -> None:
    raise SystemExit(f"physiq_runner: {msg}")


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def append_status(msg: str) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "a") as f:
        f.write(f"{now_iso()} physiq: {msg}\n")


def load_scenarios(descriptions_csv: Path) -> tuple[Scenario, ...]:
    """Parse the 198 take-1 rows; validate count and switch-frame presence."""
    if not descriptions_csv.exists():
        fail(f"descriptions file missing: {descriptions_csv}")
    sf_dir = GT_ORIGINAL / "switch-frames"
    if not sf_dir.exists():
        fail(f"switch-frames not downloaded yet: {sf_dir} (run --stage download)")
    scenarios = []
    with open(descriptions_csv, newline="") as f:
        for row in csv.DictReader(f):
            if "take-1" not in row["scenario"]:
                continue
            vid, view, _, scen = row["scenario"].split("_")
            sf = sf_dir / f"{vid}_switch-frames_anyFPS_{view}_{scen.replace('.mp4', '.jpg')}"
            if not sf.exists():
                fail(f"missing switch frame: {sf}")
            scenarios.append(Scenario(vid, row["description"], row["generated_video_name"], sf))
    if len(scenarios) != N_SCENARIO_VIEWS:
        fail(f"expected {N_SCENARIO_VIEWS} take-1 rows, got {len(scenarios)}")
    return tuple(scenarios)


def scenario_seed(run_id: int, video_id: str) -> int:
    """Deterministic per (run, scenario-view): identical noise across arms (paired)."""
    return BASE_SEED + run_id * 10_000 + int(video_id)


def crop_box(width: int, height: int, aspect: tuple[int, int] = CROP_ASPECT) -> tuple[int, int, int, int]:
    """Center crop box (left, top, right, bottom) to the (h, w) aspect; pure."""
    ah, aw = aspect
    target_w = round(height * aw / ah)
    if target_w <= width:
        left = (width - target_w) // 2
        return (left, 0, left + target_w, height)
    target_h = round(width * ah / aw)
    top = (height - target_h) // 2
    return (0, top, width, top + target_h)


def latent_geometry(height: int, width: int, max_area: int = MAX_AREA) -> tuple[int, int]:
    """Replica of image2video.py:487-494 (vae_stride 8, patch 2): (lat_h, lat_w)."""
    import numpy as np

    ar = height / width
    lat_h = round(np.sqrt(max_area * ar) // 8 // 2 * 2)
    lat_w = round(np.sqrt(max_area / ar) // 8 // 2 * 2)
    return int(lat_h), int(lat_w)


def load_conditioning_image(path: Path):
    """Switch frame -> RGB PIL image center-cropped to CROP_ASPECT (see GEOMETRY)."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    cropped = im.crop(crop_box(*im.size))
    geom = latent_geometry(cropped.size[1], cropped.size[0])
    if geom != (LAT_H, LAT_W):
        fail(f"{path.name}: cropped {cropped.size} -> lat {geom}, expected {(LAT_H, LAT_W)}")
    return cropped


# --------------------------------------------------------------------------
# Stage: download (CPU/network; anonymous HTTPS — gcloud NOT required)
# --------------------------------------------------------------------------

def stage_download(dry_run: bool) -> None:
    """Bulk-download the original-benchmark subset in bulk_manifest.json (md5-checked)."""
    if not MANIFEST.exists():
        fail(f"manifest missing: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text())
    todo = [it for pref in manifest["prefixes"].values() for it in pref["files"]
            if not it["name"].endswith("/")]
    done = sum(1 for it in todo if _present(it))
    print(f"[download] {done}/{len(todo)} files present; total {sum(it['size'] for it in todo) / 1e9:.2f} GB")
    if dry_run:
        return
    for it in todo:
        if not _present(it):
            _download_one(it)


def _present(it: dict) -> bool:
    dst = GT_ORIGINAL / it["name"]
    return dst.exists() and dst.stat().st_size == it["size"]


def _download_one(it: dict) -> None:
    dst = GT_ORIGINAL / it["name"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://storage.googleapis.com/{GCS_BUCKET}/{urllib.parse.quote(it['name'])}"
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    if it.get("md5") and base64.b64encode(hashlib.md5(data).digest()).decode() != it["md5"]:
        fail(f"md5 mismatch: {it['name']}")
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    print(f"[download] {it['name']} ({it['size'] / 1e6:.1f} MB)")


# --------------------------------------------------------------------------
# Stage: actiondir (CPU) — static-camera action_path for all generations
# --------------------------------------------------------------------------

def stage_actiondir(dry_run: bool) -> None:
    """Write identity poses.npy + constant intrinsics.npy (image2video.py:477,551)."""
    print(f"[actiondir] static camera: {FRAME_NUM} identity c2w poses + intrinsics {INTRINSICS_ROW} -> {ACTION_DIR}")
    if dry_run:
        return
    import numpy as np

    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ACTION_DIR / "poses.npy", np.tile(np.eye(4, dtype=np.float32), (FRAME_NUM, 1, 1)))
    np.save(ACTION_DIR / "intrinsics.npy", np.tile(np.asarray(INTRINSICS_ROW, dtype=np.float32), (FRAME_NUM, 1)))
    print(f"[actiondir] wrote poses ({FRAME_NUM},4,4) + intrinsics ({FRAME_NUM},4)")


# --------------------------------------------------------------------------
# Stage: t5cache (GPU when PHYSIQ_T5_DEVICE=cuda, else CPU/GPU-free) and adaptercheck (CPU)
# --------------------------------------------------------------------------

def stage_t5cache(prompts: str, dry_run: bool) -> None:
    if physiq_t5.encoder_device_name() != "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # before any torch import: GPU-free unless PHYSIQ_T5_DEVICE=cuda
    unique = tuple(dict.fromkeys(s.prompt for s in load_scenarios(descriptions_csv(prompts))))
    missing = physiq_t5.missing_prompts(unique)
    print(f"[t5cache] {len(unique)} unique prompts, {len(missing)} missing -> {physiq_t5.CACHE_DIR}")
    if dry_run or not missing:
        return
    physiq_t5.encode_missing(unique)


def stage_adaptercheck(arm: str, adapter_ckpt: Path | None) -> None:
    if arm == "vanilla":
        print("[adaptercheck] vanilla: identity, nothing to load")
        return
    try:
        info = physiq_adapter.check_adapter(arm, adapter_ckpt, device="cpu")
    except physiq_adapter.AdapterError as e:
        fail(f"adapter check failed for {arm}: {e}")
    out = PHASE_C_RUNS / f"physiq_{arm}_adapter.json"
    out.write_text(json.dumps(info, indent=2))
    print(f"[adaptercheck] {json.dumps(info)}")
    append_status(f"{arm} adapter OK: {info['module']} ({info['n_params']} params, gate |alpha|max "
                  f"{info['gate_abs_max']}) from {Path(info['ckpt']).name} [{info['why']}], "
                  f"rel_change on random latent {info['rel_change_chunk0']:.3g}, valid {info.get('valid')}")


# --------------------------------------------------------------------------
# Stage: generate (GPU)
# --------------------------------------------------------------------------

def check_arm_supported(arm: str, adapter_ckpt: Path | None = None) -> None:
    """Fail before the model load if the arm's adapter checkpoint cannot be resolved."""
    if arm == "vanilla":
        return
    try:
        physiq_adapter.select_ckpt(arm, adapter_ckpt)
    except physiq_adapter.AdapterError as e:
        raise NotImplementedError(f"arm '{arm}' not runnable yet: {e}") from e


def wrap_final_step_x0(pipe, adapt_fn, timesteps_index=TIMESTEPS_INDEX):
    """Adapter hook: replace the FINAL-timestep x0 of every chunk by adapt_fn(x0).

    That x0 is the tensor appended at image2video.py:702 and fed to the
    KV-update forward at :705-709, so adapting it changes both the output
    video and the cache seen by later chunks. Intermediate denoise steps are
    untouched. On first call the effective timesteps are asserted pairwise
    distinct so the final-step test cannot misfire mid-chunk. Returns pipe.
    """
    original = pipe._convert_flow_pred_to_x0
    state = {"checked": False}

    def wrapped(flow_pred, xt, timestep, scheduler):
        effective = scheduler.timesteps[list(timesteps_index)]
        if not state["checked"]:
            vals = [float(t) for t in effective]
            if len(set(vals)) != len(vals):
                raise RuntimeError(f"effective timesteps not pairwise distinct: {vals}")
            state["checked"] = True
        x0 = original(flow_pred=flow_pred, xt=xt, timestep=timestep, scheduler=scheduler)
        return adapt_fn(x0) if float(timestep) == float(effective[-1]) else x0

    pipe._convert_flow_pred_to_x0 = wrapped
    pipe._physiq_hook_original = original
    return pipe


def unwrap_final_step_x0(pipe):
    if hasattr(pipe, "_physiq_hook_original"):
        pipe._convert_flow_pred_to_x0 = pipe._physiq_hook_original
        del pipe._physiq_hook_original
    return pipe


def install_arm_adapter(pipe, arm: str, adapter_ckpt: Path | None = None):
    """(pipe, info): vanilla = identity (released path, no wrapper); else the P1
    adapter (physiq_adapter contract) hooked at the final-step x0."""
    if arm == "vanilla":
        return pipe, {"arm": arm, "adapter": None}
    path, why = physiq_adapter.select_ckpt(arm, adapter_ckpt)
    adapter = physiq_adapter.load_stateful_adapter(path, device=str(pipe.device))
    pipe = wrap_final_step_x0(pipe, adapter)
    pipe._physiq_adapter = adapter          # reset(B) before every video/batch (_generate_group)
    return pipe, {"arm": arm, "adapter": str(path), "why": why, **adapter.info}


def patch_attention_fallback() -> str:
    """Make cross-attention work without flash_attn (see module docstring)."""
    from wan.modules import attention as attn_mod
    from wan.modules import model_fast

    if attn_mod.FLASH_ATTN_2_AVAILABLE or attn_mod.FLASH_ATTN_3_AVAILABLE:
        return "flash_attn available: no patch"
    model_fast.flash_attention = attn_mod.attention
    return "flash_attn absent: model_fast.flash_attention -> attention() (torch SDPA)"


def lingbot_ckpt_dir() -> Path:
    required = ("transformers", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl")
    missing = [r for r in required if not (LINGBOT_WEIGHTS / r).exists()]
    if missing:
        fail(f"LingBot weights incomplete at {LINGBOT_WEIGHTS}: missing {missing}")
    return LINGBOT_WEIGHTS


def build_pipeline(prompts_needed) -> tuple:
    """WanI2VCausal (causal-fast, bf16 DiT on cuda:0) with T5 contexts from the
    disk cache; umt5-xxl is never loaded (T5EncoderModel stubbed)."""
    n_encoded = physiq_t5.encode_missing(prompts_needed)  # CPU, only if the cache is incomplete
    sys.path.insert(0, str(LINGBOT_REPO))
    import wan
    from wan import image2video
    from wan.configs import WAN_CONFIGS

    note = patch_attention_fallback()
    real_t5 = image2video.T5EncoderModel
    image2video.T5EncoderModel = lambda **kw: None
    try:
        pipe = wan.WanI2VCausal(config=WAN_CONFIGS["i2v-A14B"], checkpoint_dir=str(lingbot_ckpt_dir()),
                                device_id=0, rank=0, t5_cpu=True, local_attn_size=LOCAL_ATTN_SIZE,
                                sink_size=SINK_SIZE, infer_mode="causal_fast")
    finally:
        image2video.T5EncoderModel = real_t5
    pipe._t5_cache.update(physiq_t5.load_contexts(prompts_needed, pipe.device))
    dropped = drop_checkpoint_page_cache()
    qnote = apply_dit_quant(pipe)
    return pipe, f"{note}; T5 contexts from disk cache ({n_encoded} freshly encoded); ckpt page cache dropped ({dropped} files); {qnote}"


DIT_QUANT = os.environ.get("PHYSIQ_DIT_QUANT", "bf16")   # bf16 | nvfp4 (user directive Sep 21 2026: NVFP4 wherever supported)


def apply_dit_quant(pipe) -> str:
    """Weight-only NVFP4 on every eligible nn.Linear of the frozen DiT (paper/nvfp4.py,
    native torch scaled_mm on this Blackwell GB10). Same precision for every arm; the
    P1 adapters were trained on bf16 teacher latents, so `--stage quanttest` records the
    x0 / video delta vs bf16 before an arm runs under it."""
    if DIT_QUANT == "bf16":
        return "DiT bf16"
    if DIT_QUANT != "nvfp4":
        fail(f"PHYSIQ_DIT_QUANT={DIT_QUANT!r}: expected bf16 or nvfp4")
    import torch
    sys.path.insert(0, str(REPO / "paper"))
    import nvfp4
    # Scope: the transformer blocks' attention/FFN linears only (~99% of the weights). The
    # conditioning path (time_embedding/time_projection run under an fp32 autocast and assert
    # fp32; text_embedding, camera injectors, patch embed, head) stays in native precision.
    with torch.no_grad():
        rep = nvfp4.quantize_linears_(pipe.model.blocks, skip=("cam_", "modulation"))
    torch.cuda.empty_cache()
    note = f"DiT NVFP4: {rep['quantized']} linears ({rep['weight_gb_before']} GB bf16 -> {rep['weight_gb_after']} GB), {rep['skipped']} skipped"
    print(f"[build] {note}", flush=True)
    return note


def drop_checkpoint_page_cache() -> int:
    """Sep 20 23:44 reset: after load, the ~65 GB of fp32 safetensors shards stayed in
    the page cache (system used 88 GB with 38 GB on the GPU and a 2 GB RSS); the first
    video's 36 GB of CUDA working memory then put the box at 123 GB and the kernel could
    not reclaim fast enough. Ask the kernel to drop those file pages now (advisory,
    per-file, no root)."""
    import os
    n = 0
    for f in sorted(LINGBOT_WEIGHTS.rglob("*")):
        if not f.is_file() or f.suffix not in (".safetensors", ".pth"):
            continue
        try:
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            n += 1
        except OSError as e:
            print(f"[build] fadvise failed for {f.name}: {e}", flush=True)
    return n


def video_to_uint8(video):
    """(C, N, H, W) in [-1, 1] -> uint8 numpy (N, H, W, C), same mapping as save_video."""
    import torch

    x = (video.detach().float().clamp(-1, 1) + 1) / 2 * 255
    return x.to("cpu").to(dtype=torch.uint8).permute(1, 2, 3, 0).numpy()


def _mem_stats() -> dict:
    import torch

    from physiq_batched import system_used_gb

    vm = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            vm[line.split(":")[0]] = int(line.split()[1]) / 1e6
    return {"peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            "rss_gb": round(vm.get("VmRSS", 0), 2), "rss_peak_gb": round(vm.get("VmHWM", 0), 2),
            "system_used_gb": round(system_used_gb(), 1)}


def timing_path(arm: str) -> Path:
    return PHASE_C_RUNS / f"physiq_{arm}_timing.jsonl"


def log_timing(arm: str, record: dict) -> None:
    timing_path(arm).parent.mkdir(parents=True, exist_ok=True)
    with open(timing_path(arm), "a") as f:
        f.write(json.dumps({"ts": now_iso(), "arm": arm, **record}) + "\n")


def batch_mode() -> int:
    """Decided by --stage batchtest (runs/phase_c/physiq_batch_mode); default 1."""
    try:
        return int(BATCH_MODE_FILE.read_text().strip())
    except (OSError, ValueError):
        return 1


def stage_generate(arm: str, prompts: str, run_id: int, limit: int, dry_run: bool,
                   shift: float, batch: int, adapter_ckpt: Path | None) -> None:
    """198 i2v generations for one arm. Per-scenario resumable."""
    scenarios = load_scenarios(descriptions_csv(prompts))[: limit or None]
    out_dir = RUNS_ROOT / run_name(arm, prompts, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    removed = clear_tmp_files(out_dir)
    pending = [s for s in scenarios if not (out_dir / s.out_name).exists()]
    print(f"[generate:{arm}] {len(scenarios) - len(pending)} done, {len(pending)} pending, batch {batch} "
          f"({len(removed)} tmp leftovers removed) -> {out_dir}")
    if dry_run:
        for s in pending[:3]:
            print(f"  would generate {s.out_name}: seed={scenario_seed(run_id, s.video_id)} shift={shift} "
                  f"crop={crop_box(3840, 2160)} prompt={s.prompt[:50]!r}...")
        return
    check_arm_supported(arm, adapter_ckpt)
    require_tools()
    if not (ACTION_DIR / "poses.npy").exists():
        fail("run --stage actiondir first")
    if not pending:
        print(f"[generate:{arm}] nothing to do")
        return

    t0 = time.perf_counter()
    pipe, note = build_pipeline([s.prompt for s in pending])
    pipe, adapter_info = install_arm_adapter(pipe, arm, adapter_ckpt)
    load_s = time.perf_counter() - t0
    log_timing(arm, {"event": "pipeline_loaded", "load_s": round(load_s, 1), "attention": note,
                     "adapter": adapter_info, **_mem_stats()})
    append_status(f"{arm}: pipeline loaded in {load_s:.0f}s ({note}); adapter {adapter_info.get('adapter')}; "
                  f"{len(pending)} pending, batch {batch}")
    _write_meta(out_dir, arm, prompts, run_id, shift, batch, note, adapter_info)

    groups = [pending[i:i + batch] for i in range(0, len(pending), batch)]
    n_done = 0
    for group in groups:
        for record in _generate_group(pipe, group, out_dir, run_id, shift):
            n_done += 1
            log_timing(arm, record)
            print(f"[generate:{arm}] {n_done}/{len(pending)} {record['out_name']} gen {record['gen_s']}s "
                  f"(batch {record['batch']}) write {record['write_s']}s peak_alloc {record['peak_alloc_gb']} GB "
                  f"sys {record['system_used_gb']} GB", flush=True)
    append_status(f"{arm}: generation batch finished ({len(pending)} videos) -> {out_dir}")


def _generate_group(pipe, group: list[Scenario], out_dir: Path, run_id: int, shift: float,
                    capture=None) -> list[dict]:
    """Generate 1 (released path) or B>1 (physiq_batched) scenarios; write + time them."""
    import torch

    from physiq_batched import generate_batched

    torch.cuda.reset_peak_memory_stats()
    adapter = getattr(pipe, "_physiq_adapter", None)
    if adapter is not None:
        adapter.reset(len(group))           # P1 prev-chain starts from zeros at chunk 0
    imgs = [load_conditioning_image(s.switch_frame) for s in group]
    seeds = [scenario_seed(run_id, s.video_id) for s in group]
    t1 = time.perf_counter()
    if len(group) == 1:
        videos = [pipe.generate(group[0].prompt, imgs[0], action_path=str(ACTION_DIR), chunk_size=CHUNK_SIZE,
                                max_area=MAX_AREA, frame_num=FRAME_NUM, timesteps_index=list(TIMESTEPS_INDEX),
                                shift=shift, seed=seeds[0], offload_model=False)]
    else:
        videos = generate_batched(pipe, [s.prompt for s in group], imgs, str(ACTION_DIR), CHUNK_SIZE, MAX_AREA,
                                  FRAME_NUM, TIMESTEPS_INDEX, shift, seeds)
    torch.cuda.synchronize()
    gen_s = (time.perf_counter() - t1) / len(group)
    records = []
    for s, video in zip(group, videos):
        t2 = time.perf_counter()
        info = write_mp4_rgb(video_to_uint8(video), out_dir / s.out_name, FPS)
        records.append({"out_name": s.out_name, "video_id": s.video_id, "batch": len(group),
                        "t5_cache_hit": True, "gen_s": round(gen_s, 1), "write_s": round(time.perf_counter() - t2, 1),
                        "frames": info.nb_frames, "w": info.width, "h": info.height, **_mem_stats()})
    if capture is not None:
        capture.extend(videos)
    return records


def _write_meta(out_dir: Path, arm: str, prompts: str, run_id: int, shift: float, batch: int,
                note: str, adapter_info: dict) -> None:
    meta = {"arm": arm, "prompts": prompts, "run_id": run_id, "shift": shift, "batch": batch,
            "chunk_size": CHUNK_SIZE, "frame_num": FRAME_NUM, "fps": FPS, "max_area": MAX_AREA,
            "crop_aspect": CROP_ASPECT, "latent_hw": [LAT_H, LAT_W], "output_wh": [LAT_W * 8, LAT_H * 8],
            "timesteps_index": list(TIMESTEPS_INDEX), "local_attn_size": LOCAL_ATTN_SIZE, "sink_size": SINK_SIZE,
            "seed_policy": f"{BASE_SEED} + run_id*10000 + int(video_id)", "attention": note,
            "adapter": adapter_info, "weights": str(LINGBOT_WEIGHTS), "git_rev": _git_rev(), "written": now_iso()}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# --------------------------------------------------------------------------
# Stage: batchtest (GPU) — decide batch=2 vs 1 (coordinator criteria)
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _chunk_recorder(pipe):
    """Record every final-step chunk x0 (per sample, in order) via the hook."""
    recorded = []
    wrap_final_step_x0(pipe, lambda x0: recorded.append(x0.detach().clone()) or x0)
    try:
        yield recorded
    finally:
        unwrap_final_step_x0(pipe)


def stage_batchtest(prompts: str, run_id: int, shift: float, dry_run: bool) -> None:
    """2-scenario test: batch-2 vs batch-1 agreement (chunk-0 x0), peak system
    memory (< BATCHTEST_MEM_LIMIT_GB) and per-video speedup -> physiq_batch_mode."""
    from physiq_batched import SystemMemorySampler, psnr_uint8, rel_l2, system_used_gb

    pair = list(load_scenarios(descriptions_csv(prompts))[:2])
    out_dir = RUNS_ROOT / "batchtest"
    print(f"[batchtest] scenarios {[s.video_id for s in pair]} -> {out_dir}; system used now {system_used_gb():.1f} GB")
    if dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe, note = build_pipeline([s.prompt for s in pair])
    b1_dir, b2_dir = out_dir / "b1", out_dir / "b2"
    b1_dir.mkdir(exist_ok=True)
    b2_dir.mkdir(exist_ok=True)
    with _chunk_recorder(pipe) as x0_b1:
        rec1 = _generate_group(pipe, pair[:1], b1_dir, run_id, shift)
    with SystemMemorySampler() as mem, _chunk_recorder(pipe) as x0_b2:
        rec2 = _generate_group(pipe, pair, b2_dir, run_id, shift)
    from physiq_video import probe  # noqa: F401  (contract helpers already imported)
    import numpy as np

    v1 = _read_frames(b1_dir / pair[0].out_name)
    v2 = _read_frames(b2_dir / pair[0].out_name)
    result = {
        "chunk0_rel_l2": rel_l2(x0_b2[0], x0_b1[0]),            # sample 0, chunk 0
        "last_chunk_rel_l2": rel_l2(x0_b2[-2], x0_b1[-1]),       # sample 0, last chunk
        "psnr_db": psnr_uint8(v1, v2) if v1.shape == v2.shape else None,
        "b1_s_per_video": rec1[0]["gen_s"], "b2_s_per_video": rec2[0]["gen_s"],
        "b2_peak_system_gb": round(mem.peak_gb, 1), "b2_peak_alloc_gb": rec2[0]["peak_alloc_gb"],
        "limit_gb": BATCHTEST_MEM_LIMIT_GB, "attention": note, "ts": now_iso(),
    }
    result["speedup"] = result["b1_s_per_video"] / max(result["b2_s_per_video"], 1e-9)
    result["batch_mode"] = 2 if (result["chunk0_rel_l2"] <= BATCHTEST_REL_L2_MAX
                                 and result["b2_peak_system_gb"] < BATCHTEST_MEM_LIMIT_GB
                                 and result["speedup"] >= BATCHTEST_MIN_SPEEDUP) else 1
    (PHASE_C_RUNS / "physiq_batchtest.json").write_text(json.dumps(result, indent=2))
    BATCH_MODE_FILE.write_text(f"{result['batch_mode']}\n")
    print(f"[batchtest] {json.dumps(result)}")
    append_status(f"batchtest: chunk0 relL2 {result['chunk0_rel_l2']:.3g}, PSNR {result['psnr_db']}, "
                  f"b1 {result['b1_s_per_video']}s vs b2 {result['b2_s_per_video']}s/video "
                  f"(x{result['speedup']:.2f}), peak system {result['b2_peak_system_gb']} GB "
                  f"(limit {BATCHTEST_MEM_LIMIT_GB}) -> batch_mode {result['batch_mode']}")


def _read_frames(path: Path):
    """Decode an mp4 to uint8 (N, H, W, 3) via ffmpeg rawvideo (for PSNR)."""
    import numpy as np

    from physiq_video import probe

    info = probe(path)
    res = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True)
    return np.frombuffer(res.stdout, dtype=np.uint8).reshape(info.nb_frames, info.height, info.width, 3)


# --------------------------------------------------------------------------
# Stage: quanttest (GPU) — NVFP4 DiT vs bf16 DiT on one scenario (chunk x0 agreement,
# video PSNR, time, peak system memory). Writes runs/phase_c/physiq_quanttest.json.
# --------------------------------------------------------------------------

def stage_quanttest(prompts: str, run_id: int, shift: float, dry_run: bool) -> None:
    global DIT_QUANT
    from physiq_batched import SystemMemorySampler, psnr_uint8, rel_l2, system_used_gb
    import torch

    one = list(load_scenarios(descriptions_csv(prompts))[:1])
    out_dir = RUNS_ROOT / "quanttest"
    print(f"[quanttest] scenario {one[0].video_id} -> {out_dir}; system used now {system_used_gb():.1f} GB")
    if dry_run:
        return
    result = {"ts": now_iso(), "scenario": one[0].video_id}
    x0 = {}
    for mode in ("bf16", "nvfp4"):
        d = out_dir / mode
        d.mkdir(parents=True, exist_ok=True)
        saved = d / "x0.pt"
        if saved.exists() and (d / one[0].out_name).exists():          # resumable: reuse a finished pass
            prev = torch.load(saved)
            x0[mode], result[mode] = prev["x0"], prev["record"]
            print(f"[quanttest] {mode}: reusing saved pass ({result[mode]})", flush=True)
            continue
        DIT_QUANT = mode
        pipe, note = build_pipeline([s.prompt for s in one])
        with SystemMemorySampler() as mem, _chunk_recorder(pipe) as rec_x0:
            rec = _generate_group(pipe, one, d, run_id, shift)
        x0[mode] = [t.float().cpu() for t in rec_x0]
        result[mode] = {"gen_s": rec[0]["gen_s"], "peak_alloc_gb": rec[0]["peak_alloc_gb"],
                        "peak_system_gb": round(mem.peak_gb, 1), "note": note}
        torch.save({"x0": x0[mode], "record": result[mode]}, saved)
        # full teardown before the next load: the second from_pretrained materialised a bf16 DiT on top
        # of the first pipeline (00:18 Sep 21: MemAvailable 8 GB during the NVFP4 pass's load)
        pipe.model = None; pipe.vae = None; pipe._t5_cache.clear()
        del pipe, rec_x0, rec
        import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        print(f"[quanttest] {mode} done; system used after teardown {system_used_gb():.1f} GB", flush=True)
    v1 = _read_frames(out_dir / "bf16" / one[0].out_name)
    v2 = _read_frames(out_dir / "nvfp4" / one[0].out_name)
    result["chunk0_rel_l2"] = rel_l2(x0["nvfp4"][0], x0["bf16"][0])
    result["last_chunk_rel_l2"] = rel_l2(x0["nvfp4"][-1], x0["bf16"][-1])
    result["psnr_db"] = psnr_uint8(v1, v2) if v1.shape == v2.shape else None
    result["speedup"] = result["bf16"]["gen_s"] / max(result["nvfp4"]["gen_s"], 1e-9)
    result["system_gb_saved"] = round(result["bf16"]["peak_system_gb"] - result["nvfp4"]["peak_system_gb"], 1)
    (PHASE_C_RUNS / "physiq_quanttest.json").write_text(json.dumps(result, indent=2))
    print(f"[quanttest] {json.dumps(result)}")
    append_status(f"quanttest: NVFP4 vs bf16 chunk0 relL2 {result['chunk0_rel_l2']:.3g}, last {result['last_chunk_rel_l2']:.3g}, "
                  f"PSNR {result['psnr_db']}, {result['bf16']['gen_s']}s -> {result['nvfp4']['gen_s']}s/video, "
                  f"peak system {result['bf16']['peak_system_gb']} -> {result['nvfp4']['peak_system_gb']} GB")


# --------------------------------------------------------------------------
# Stage: trim / verify (CPU; ffmpeg) — 81 frames -> 80 frames = 5.000 s
# --------------------------------------------------------------------------

def stage_trim(arm: str, prompts: str, run_id: int, dry_run: bool) -> None:
    name = run_name(arm, prompts, run_id)
    src_dir, dst_dir = RUNS_ROOT / name, TRIMMED_ROOT / name
    if not src_dir.exists():
        fail(f"no generated videos at {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    clear_tmp_files(src_dir)
    clear_tmp_files(dst_dir)
    vids = list_mp4s(src_dir)
    pending = [v for v in vids if not (dst_dir / v.name).exists()]
    print(f"[trim:{arm}] {len(vids)} raw videos, {len(pending)} pending -> {dst_dir}")
    if dry_run:
        return
    require_tools()
    for v in pending:
        info = trim_to_frames(v, dst_dir / v.name, TRIM_FRAMES, FPS)
        print(f"[trim:{arm}] {v.name}: {info.nb_frames} frames, {info.duration_s:.3f} s")
    print(f"[trim:{arm}] {len(list_mp4s(dst_dir))}/{N_SCENARIO_VIEWS} trimmed")


def stage_verify(arm: str, prompts: str, run_id: int) -> None:
    folder = TRIMMED_ROOT / run_name(arm, prompts, run_id)
    problems = validate_contract(folder)
    for p in problems[:20]:
        print(f"[verify:{arm}] {p}")
    if problems:
        fail(f"{len(problems)} contract problem(s) in {folder}")
    print(f"[verify:{arm}] OK: {folder} satisfies the scorer contract ({N_SCENARIO_VIEWS} x 5.000 s)")


# --------------------------------------------------------------------------
# Stage: score / compare (CPU; official scorer in its own uv env)
# --------------------------------------------------------------------------

def scorer_cmd(script: str, *args: str) -> list[str]:
    return ["uv", "run", "--no-sync", "python", script, *args]


def stage_score(arms: tuple[str, ...], prompts: str, run_id: int, dry_run: bool, n_process: int) -> None:
    """Official scorer on the trimmed folders (complete 198 x 5.000 s each; validated first)."""
    folders = {a: TRIMMED_ROOT / run_name(a, prompts, run_id) for a in arms}
    for a, f in folders.items():
        print(f"[score] {a}: {len(list_mp4s(f)) if f.exists() else 0}/{N_SCENARIO_VIEWS} trimmed videos in {f}")
    cmd = scorer_cmd("physiq/run_physics_iq.py", "--input_folders", *[str(f) for f in folders.values()],
                     "--output_folder", str(RESULTS_ROOT), "--descriptions_file", str(descriptions_csv(prompts)),
                     "--benchmark_base_folder", str(DATA_ROOT), "--original_physics_iq", "--n_process", str(n_process))
    print("[score] command:\n  cd", PHYSIQ_REPO, "&&", " ".join(cmd))
    if dry_run:
        return
    _check_scorer_env()
    for a, f in folders.items():
        problems = validate_contract(f)
        if problems:
            fail(f"{a}: {len(problems)} contract problem(s), first: {problems[0]}")
    append_status(f"scoring {list(arms)} with the official scorer (n_process={n_process})")
    res = subprocess.run(cmd, cwd=PHYSIQ_REPO, env={**os.environ, "MPLBACKEND": "Agg"})
    if res.returncode != 0:
        append_status(f"SCORER FAILED rc={res.returncode}")
        fail(f"scorer exited with {res.returncode}")
    for a, f in folders.items():
        s = collect_scores(a, f.name)
        print(f"[score] {a}: final_score_origround={s['final_score_origround']} final_score_view={s['final_score_view']}")


def _check_scorer_env() -> None:
    require_tools()
    if not (PHYSIQ_REPO / ".venv" / "bin" / "python").exists():
        fail(f"scorer env missing: run `uv sync` in {PHYSIQ_REPO}")


def collect_scores(arm: str, run: str) -> dict:
    """Parse <run>_metrics.json into runs/phase_c/physiq_<arm>_scores.json; touch DONE."""
    results_dir = RESULTS_ROOT / "physics-IQ-benchmark" / "results"
    metrics_json = results_dir / f"{run}_metrics.json"
    if not metrics_json.exists():
        fail(f"scorer produced no metrics for {run}: {metrics_json}")
    values = json.loads(metrics_json.read_text())
    out = PHASE_C_RUNS / f"physiq_{arm}_scores.json"
    scores = {"arm": arm, "run": run, "scored_at": now_iso(),
              "final_score_origround": values.get("final_score_origround"),
              "final_score_view": values.get("final_score_view"), "final_score_view_pct": _pct(values.get("final_score_view")),
              "final_score_orig": values.get("final_score_orig"), "final_score_orig_pct": _pct(values.get("final_score_orig")),
              "metrics_json": str(metrics_json), "results_csv": str(results_dir / f"{run}.csv"),
              "scores_json": str(out), "all_metrics": values}
    out.write_text(json.dumps(scores, indent=2))
    (PHASE_C_RUNS / f"physiq_{arm}_DONE").write_text(f"{now_iso()} {out}\n")
    append_status(f"{arm} SCORED: final_score_origround={scores['final_score_origround']} "
                  f"final_score_view={scores['final_score_view']} (verified formula on original GT) -> {out}")
    return scores


def _pct(v) -> float | None:
    return round(round(v, 4) * 100, 2) if isinstance(v, (int, float)) else None


def stage_compare(arms: tuple[str, ...], prompts: str, run_id: int, dry_run: bool) -> None:
    """Paired 198-row comparison of arm results CSVs (physiq_compare.py, scorer env)."""
    if len(arms) < 2:
        fail("--arms needs >= 2 arms to compare")
    results_dir = RESULTS_ROOT / "physics-IQ-benchmark" / "results"
    csvs = [results_dir / f"{run_name(a, prompts, run_id)}.csv" for a in arms]
    out = PHASE_C_RUNS / f"physiq_compare_{'_vs_'.join(arms)}.json"
    cmd = scorer_cmd(str(Path(__file__).with_name("physiq_compare.py")), "--arms", *arms, "--csvs",
                     *[str(c) for c in csvs], "--out", str(out))
    print("[compare] command:\n  cd", PHYSIQ_REPO, "&&", " ".join(cmd))
    if dry_run:
        return
    for c in csvs:
        if not c.exists():
            fail(f"results CSV missing (score that arm first): {c}")
    res = subprocess.run(cmd, cwd=PHYSIQ_REPO, env={**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(PHYSIQ_REPO)})
    if res.returncode != 0:
        fail(f"compare exited with {res.returncode}")
    summary = json.loads(out.read_text())["summary"]
    append_status(f"paired comparison {' vs '.join(arms)}: {summary}")


# --------------------------------------------------------------------------
# Stage: project (CPU) — pilot timing -> full-run projection
# --------------------------------------------------------------------------

def project_wallclock(records: list[dict], remaining_videos: int, remaining_unique_prompts: int,
                      load_s: float) -> dict:
    """Pure projection: baseline = median of T5-cache-hit videos (all videos hit
    once physiq_t5 has cached the prompts), t5_miss_s = mean(miss) - baseline."""
    videos = [r for r in records if "gen_s" in r]
    if not videos:
        raise ValueError("no per-video timing records")
    per = [r["gen_s"] + r["write_s"] for r in videos]
    hits = [r["gen_s"] + r["write_s"] for r in videos if r.get("t5_cache_hit")]
    misses = [r["gen_s"] + r["write_s"] for r in videos if not r.get("t5_cache_hit")]
    baseline_s = statistics.median(hits or per)
    t5_miss_s = max(statistics.fmean(misses) - baseline_s, 0.0) if hits and misses else 0.0
    total_s = remaining_videos * baseline_s + remaining_unique_prompts * t5_miss_s + load_s
    return {"n_timed": len(videos), "baseline_s": baseline_s, "t5_miss_s": t5_miss_s,
            "mean_s": statistics.fmean(per), "min_s": min(per), "max_s": max(per),
            "peak_alloc_gb": max(r.get("peak_alloc_gb") or 0 for r in videos),
            "peak_rss_gb": max(r.get("rss_peak_gb") or 0 for r in videos),
            "peak_system_gb": max(r.get("system_used_gb") or 0 for r in videos),
            "load_s": load_s, "remaining_videos": remaining_videos,
            "remaining_unique_prompts": remaining_unique_prompts, "hours": total_s / 3600}


def stage_project(arm: str, prompts: str, run_id: int) -> None:
    path = timing_path(arm)
    if not path.exists():
        fail(f"no timing log yet: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    loads = [r for r in records if r.get("event") == "pipeline_loaded"]
    out_dir = RUNS_ROOT / run_name(arm, prompts, run_id)
    pending = [s for s in load_scenarios(descriptions_csv(prompts)) if not (out_dir / s.out_name).exists()]
    try:
        p = project_wallclock(records, len(pending), len({s.prompt for s in pending}),
                              loads[-1]["load_s"] if loads else 0.0)
    except ValueError as e:
        fail(str(e))
    eta = (dt.datetime.now() + dt.timedelta(hours=p["hours"])).strftime("%Y-%m-%d %H:%M")
    msg = (f"{arm}: {p['n_timed']} timed videos, baseline {p['baseline_s']:.0f}s/video (mean {p['mean_s']:.0f}, "
           f"min {p['min_s']:.0f}, max {p['max_s']:.0f}) + {p['t5_miss_s']:.0f}s per uncached prompt; peak CUDA alloc "
           f"{p['peak_alloc_gb']:.1f} GB, peak RSS {p['peak_rss_gb']:.1f} GB, peak system {p['peak_system_gb']:.1f} GB; "
           f"load {p['load_s']:.0f}s; remaining {p['remaining_videos']} videos ~ {p['hours']:.1f} h (ETA {eta})")
    print(f"[project] {msg}")
    append_status(f"projection {msg}")


# --------------------------------------------------------------------------
# helpers + CLI
# --------------------------------------------------------------------------

def run_name(arm: str, prompts: str, run_id: int) -> str:
    return f"lingbot-world-v2-{arm}-{prompts}-run_{run_id:02d}"


def descriptions_csv(prompts: str) -> Path:
    if prompts == "op":
        return PHYSIQ_REPO / "descriptions" / "descriptions_original.csv"
    return PHYSIQ_REPO / "descriptions" / "best_practice" / "descriptions_base.csv"


def stage_plan(prompts: str) -> None:
    scenarios = load_scenarios(descriptions_csv(prompts))
    print(f"[plan] {len(scenarios)} scenario-views; {len({s.prompt for s in scenarios})} unique prompts; "
          f"example: {scenarios[0].out_name}")
    print(f"[plan] switch frame crop {crop_box(3840, 2160)} -> lat {latent_geometry(2160, 3744)} -> "
          f"{LAT_W * 8}x{LAT_H * 8}, {FRAME_NUM} frames @ {FPS} fps, chunk {CHUNK_SIZE}, shift {SHIFT}, "
          f"timesteps_index {TIMESTEPS_INDEX}, batch_mode {batch_mode()}")
    print("[plan] see paper/phase_c/PHYSIQ.md for GPU-hour + storage budget")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", required=True,
                   choices=["download", "actiondir", "t5cache", "adaptercheck", "generate", "batchtest", "quanttest",
                            "trim", "verify", "score", "compare", "project", "plan"])
    p.add_argument("--arm", default="vanilla", choices=ARMS)
    p.add_argument("--arms", nargs="+", default=["vanilla"], choices=ARMS, help="score/compare stages")
    p.add_argument("--prompts", default="op", choices=["op", "bpp"])
    p.add_argument("--run-id", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="cap scenarios (pilot)")
    p.add_argument("--shift", type=float, default=SHIFT)
    p.add_argument("--batch", type=int, default=None, help="videos per DiT pass (default: physiq_batch_mode or 1)")
    p.add_argument("--adapter-ckpt", type=Path, default=None, help="override the P1 checkpoint (quantum/lora)")
    p.add_argument("--n-process", type=int, default=4, help="scorer workers (~9 GB each)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    a.batch = a.batch or batch_mode()
    return a


def main() -> None:
    a = parse_args()
    if a.stage == "download":
        stage_download(a.dry_run)
    elif a.stage == "actiondir":
        stage_actiondir(a.dry_run)
    elif a.stage == "t5cache":
        stage_t5cache(a.prompts, a.dry_run)
    elif a.stage == "adaptercheck":
        stage_adaptercheck(a.arm, a.adapter_ckpt)
    elif a.stage == "generate":
        stage_generate(a.arm, a.prompts, a.run_id, a.limit, a.dry_run, a.shift, a.batch, a.adapter_ckpt)
    elif a.stage == "batchtest":
        stage_batchtest(a.prompts, a.run_id, a.shift, a.dry_run)
    elif a.stage == "quanttest":
        stage_quanttest(a.prompts, a.run_id, a.shift, a.dry_run)
    elif a.stage == "trim":
        stage_trim(a.arm, a.prompts, a.run_id, a.dry_run)
    elif a.stage == "verify":
        stage_verify(a.arm, a.prompts, a.run_id)
    elif a.stage == "score":
        stage_score(tuple(a.arms), a.prompts, a.run_id, a.dry_run, a.n_process)
    elif a.stage == "compare":
        stage_compare(tuple(a.arms), a.prompts, a.run_id, a.dry_run)
    elif a.stage == "project":
        stage_project(a.arm, a.prompts, a.run_id)
    else:
        stage_plan(a.prompts)


if __name__ == "__main__":
    main()
