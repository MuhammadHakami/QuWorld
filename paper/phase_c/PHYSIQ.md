# Phase-C Physics-IQ — protocol, budget, and ladder slot

Scores the frozen LingBot-World v2 causal-fast pipeline, three arms
{vanilla, LoRA, quantum (mr30ryd)}, on the Physics-IQ benchmark with the
official scorer — the benchmark's first quantum entries
(11-master-plan.md Phase C; minimum-for-Aug-29 set with P0-P2).

Harness: `paper/phase_c/physiq_runner.py` (stages: download / actiondir /
generate / trim / score / plan; per-scenario resumable). Benchmark repo
cloned at `baseline/physics-iq` (blobless, 130 MB).

## Protocol (sourced)

| fact | value | source |
|---|---|---|
| task granularity | 66 scenarios x 3 perspectives = **198 i2v generations per arm** (take-1 rows of the descriptions CSV) | physics-iq README.md D1; descriptions_original.csv (396 rows = x2 takes) |
| conditioning input (i2v) | one switch-frame JPG per scenario-view, 3840x2160 | README.md D1; switch-frames/ (198 files, verified locally) |
| prompt | descriptions CSV `description` column; `op` = original prompts (default here), `bpp` = best-practice base | README.md C2-C5 |
| generated-video contract | exactly 198 mp4s named `{0001..0198}_<perspective>_<scenario>.mp4`, each **exactly 5.000 s** (ffprobe, abs_tol 1e-3) | run_physics_iq.py:168-200 |
| our generation | 81 frames @ 16 fps, **832x480 (lat 60x104)**: switch frame center-cropped to aspect 480:832 (3840x2160 → 3744x2160, 2.5% of the width) + max_area 480\*832+1000 (the released `//8` floor gives 58x104 for ANY 480:832 image, even a native 480x832 one — verified numerically), so all arms run at the P1 adapters' / teacher cache's latent geometry (frame_seqlen 1560); chunk_size **3** (4 would truncate lat_f 21→20 → 77 frames < 5 s), static-camera action_path (identity poses + 480x832-frame intrinsics), then ffmpeg trim to 80 frames = 5.000 s | image2video.py:477-500; shared_config.py:17; physiq_runner.py GEOMETRY note; test_physiq_contract.py |
| scorer entry point | `physiq/run_physics_iq.py --input_folders ... --original_physics_iq` (own `uv sync` env inside baseline/physics-iq; needs ffprobe) | README.md F |
| GT auto-prep | scorer detects generated fps (16), uses pre-computed 16FPS GT + masks; generated-video masks are auto-computed (threshold diff) into `video-masks/generated/<run>/16FPS` | run_physics_iq.py:592-634; binary_mask_generator.py |
| metric composition | per scenario-view vs real take-1: spatial IoU, weighted-spatial IoU, spatiotemporal IoU, MSE, at GT/4 resolution (960x540), first 80 frames; take-1-vs-take-2 gives the physical-variance normalizer per component | calculate_and_write_metrics_to_csv.py:205-261 |
| score, original | (mean spatial/var + mean wspatial/var + mean stIoU/var)/3 − (MSE − varMSE), clipped, x100 → `final_score_origround` | calculate_iq_score.py:179-191 |
| score, verified | per scenario-view arithmetic mean of the four variance-ratio components, each clipped to [0,1], averaged, x100 → `final_score_view` | calculate_iq_score_stable.py:266-275, 396-434 |
| leaderboard context | original i2v top: Cosmos3-Super+WMReward BoN 48.9 (v2v top 64.5); verified i2v top: Cosmos3-Super 39.5; Wan2.2 (our backbone family) 38.3 original / 32.2 verified | physics-iq README leaderboards |
| runs | 1 run per arm is accepted; 4 runs (mean±std) required only to claim SOTA | README.md leaderboard rules |

Both `final_score_origround` and `final_score_view` come out of the same
`<run>_metrics.json`; we report both, using the **original** GT (fully
public GCS bucket; the ladder's Cosmos-3 63.4 reference is the original
board). The verified GT (HF, gated auto-approval) is optional later —
requires a one-time `hf auth login` (no token on this machine today).

## Data budget

Verified against the public GCS JSON API on 2026-08-17 (anonymous —
gcloud NOT required; the runner downloads over plain HTTPS with md5
checks). Frozen manifest: `data/phase_c/physiq/bulk_manifest.json`.

| tier | contents | size | status |
|---|---|---|---|
| **now (done)** | switch-frames (198 JPG) + static_action/ + manifest | 0.18 GB | downloaded, md5-verified |
| **run-day bulk** | testing 30FPS (2.65) + testing 16FPS (1.69) + real masks 16FPS (1.62) | **5.97 GB** | `python3 paper/phase_c/physiq_runner.py --stage download` |
| not needed | conditioning-videos (i2v uses switch-frames), full-videos 7.99 GB, 30FPS masks 2.57 GB, 8/24FPS | ~15 GB avoided | whole bucket is 25.2 GB |

Outputs storage: raw 81-frame videos ~0.6-1.5 GB/arm + trimmed copies
~same + generated masks ~0.3 GB/arm + scorer CSV/JSON (KB) →
**< 8 GB total for 3 arms**, under `data/phase_c/physiq/`.

## GPU budget per arm (GB10, causal-fast, 4 denoise + 1 KV forward/chunk)

Geometry per video: lat 58x104 → frame_seqlen 1508 tokens; chunk_size 3 →
4,524-token forwards, 7 chunks, KV window 18 frames (27,144 tokens).
Per forward ≈ 2·14e9·4524 ≈ 127 TFLOP dense + ≈ 100 TFLOP attention
(40 layers, d=5120) ≈ 230 TFLOP → 5 forwards x 7 chunks ≈ **8 PFLOP/video**
(+ VAE decode, small). At the 50-100 sustained bf16 TFLOP/s assumed in
phase_c/README.md: **80-160 s/video** → 198 videos =

- **4.5-9 GPU-h per arm**, 3 arms = **13-27 GPU-h** (one model load per
  arm, T5 on CPU, prompt cache warm). The master plan's ~10 h estimate is
  the optimistic end; measure 3 pilot scenarios first and re-project.
- Memory ≈ DiT bf16 28.5 GB + self-KV ~22 GB + activations ≈ **~55-60 GB**
  — fits the 121 GB GB10 alone; do not co-schedule with Phase-B training.
- Scoring is CPU-only: ~9 GB/process, `--n_process 4`, tens of minutes.

## Ladder slot (15-challenge-ladder.md, "Physics-IQ" row)

Fill "Ours" with three numbers: vanilla / LoRA / quantum
`final_score_origround` (original board; Cosmos-3 63.4 BoN v2v is the
"Current top" cell, but our comparable class is **i2v single-sample**:
Wan2.2 38.3, Sora2 42.3). Also archive `final_score_view` for a future
verified-leaderboard PR. Gate C (master plan): quantum > LoRA on
drift/judge scores — here the judge score is the Physics-IQ score; the
paired unit for significance is the 198 scenario-view rows in each
results CSV (per-scenario component columns support a paired Wilcoxon
between arms; scorer emits per-run CSVs we can difference directly).

## Run-day checklist

1. `uv sync` inside `baseline/physics-iq` (own lockfile; quworld env
   untouched) and install ffmpeg/ffprobe (`sudo apt-get install ffmpeg`
   — **not present today**).
2. `python3 physiq_runner.py --stage download` (5.97 GB, resumable).
3. Resolve TODO(run-day)s in physiq_runner.py:
   - LingBot weights dir (`lingbot_ckpt_dir()`); flash_attn→SDPA patch if
     the aarch64 import fails (11-master-plan Phase C risk).
   - shift pilot {3.0, 5.0, 10.0} on 3 scenarios (`--limit 3`), fix one
     value for all arms.
   - intrinsics sanity for the 3840x2160 camera (static poses → identity
     relatives; only the ray embedding depends on it).
   - P1 adapter checkpoints + `install_arm_adapter` patch (assert the 4
     effective timesteps are pairwise distinct; zero-gate ⇒ vanilla-equal).
4. Per arm: `--stage generate --arm X` (resumable) → `--stage trim --arm X`.
5. `--stage score` (all arms in one call → one results dir, one barplot).
6. Copy `final_score_origround` / `final_score_view` per arm into
   15-challenge-ladder.md + 08-scoreboard.md; paired stats vs LoRA.

## Gaps found while preparing (no fixes applied)

- **ffmpeg/ffprobe/gcloud absent** — ffmpeg is a hard dependency (trim +
  scorer duration checks); gcloud is worked around (public HTTPS).
- **No HF token** — verified GT (Physics-IQ Verified) is gated; original
  GT is fully public, so the default plan does not block on this.
- **quworld env lacks cv2/moviepy/joblib** — scorer must run in
  baseline/physics-iq's own uv env (no `uv add` to quworld needed).
- **P1 adapters not trained yet** — vanilla arm can generate as soon as
  weights are local; lora/quantum arms wait on P1.

## Harness status (2026-09-19)

- Implemented + unit-tested (`paper/phase_c/test_physiq_contract.py`, 19 tests):
  `physiq_runner.py` (stages download | actiondir | t5cache | adaptercheck |
  generate | batchtest | trim | verify | score | compare | project | plan),
  `physiq_video.py` (ffmpeg writer/prober/trimmer + scorer-contract validator,
  agrees with the official `validate_generations`), `physiq_t5.py` (CPU
  umt5-xxl context cache, ~43 s/prompt on the Grace CPU, 63 unique prompts,
  shared by all arms; the pipe is built with T5 stubbed), `physiq_adapter.py`
  (P1 checkpoint contract + best-seed selection from `p1_results.json`),
  `physiq_batched.py` (B>1 sampling; `WanModelFast.forward` copied with its
  three B==1-only lines fixed, installed only when batch > 1),
  `physiq_compare.py` (paired 198-row stats in the scorer env).
- Environment facts (resolved): ffmpeg/ffprobe present (`~/.local/bin`);
  scorer env synced (`uv sync`, pandas 3, its unit tests pass); flash_attn
  absent → cross-attention rebound to torch SDPA (equivalent, no key mask in
  the released path); `save_video` not used (imageio absent, swallows errors).
- Measured throughput (teacher pilot, same DiT, 60x104, torch-SDPA): ~47 s per
  chunk (5 forwards) → **~330 s/video → ~19 h per arm** (plan above assumed
  4.5–9 h). Confirmed/revised by each arm's 3-scenario pilot
  (`--stage project`, `runs/phase_c/phase_c_status`).
- Orchestration: `runs/phase_c/physiq_orchestrator.sh` (detached) — T5 cache
  (memory-gated), wait for `TEACHER_DONE` + free GPU, then vanilla → quantum →
  lora (pilot → projection → [batch-2 test] → full run → trim → verify →
  score → paired compare), sentinels `physiq_<arm>_DONE` /
  `BLOCKED_physiq_<arm>` / `physiq_lora_SKIPPED` (deadline 2026-09-22 08:00).
