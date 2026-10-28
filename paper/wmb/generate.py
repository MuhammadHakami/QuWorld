"""WorldModelBench (i2v) video generation for the Phase-C arms — built on the Physics-IQ
runner (paper/phase_c/physiq_runner.py): same frozen LingBot-World v2 causal-fast pipeline,
same P1 adapter hook (vanilla | quantum | lora), same 480x832 geometry, seeds and mp4 writer.

Protocol (WorldModelBench README, image-to-video): image = instance["first_frame"], text =
instance["text_instruction"]; the mp4 is named by the first-frame stem. Instances come from the
pre-registered 60-instance subset (paper/wmb/subset60.json, seed 20260919, domain-stratified).
Outputs: data/phase_c/wmb/generated_videos/<arm>/<stem>.mp4 (+ meta.json), timing in
runs/wmb/gen_<arm>_timing.jsonl, sentinel runs/wmb/GEN_<arm>_DONE when every instance exists.

Usage: uv run --no-sync python paper/wmb/generate.py --stage {plan,t5cache,generate} --arm <arm>
       [--subset paper/wmb/subset60.json] [--limit N] [--batch 1] [--shift 3.0] [--run-id 1]
Resumable per instance (skip-existing, atomic writes via physiq_video.write_mp4_rgb).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper" / "phase_c"))
import physiq_runner as pr  # noqa: E402
import physiq_t5  # noqa: E402

WMB = REPO / "baseline" / "WorldModelBench"
SUBSET = REPO / "paper" / "wmb" / "subset60.json"
OUT_ROOT = REPO / "data" / "phase_c" / "wmb" / "generated_videos"
RUNS = REPO / "runs" / "wmb"


def load_instances(subset: Path, limit: int = 0) -> list[pr.Scenario]:
    data = json.loads(subset.read_text())
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{subset}: expected a non-empty JSON list of instances")
    out = []
    for i, inst in enumerate(data[: limit or None]):
        for k in ("first_frame", "text_instruction"):
            if k not in inst:
                raise SystemExit(f"instance {i} lacks {k!r}")
        frame = WMB / inst["first_frame"]
        if not frame.exists():
            raise SystemExit(f"missing first frame {frame}")
        out.append(pr.Scenario(video_id=f"{i:04d}", prompt=inst["text_instruction"], out_name=f"{frame.stem}.mp4", switch_frame=frame))
    return out


def stage_plan(scen: list[pr.Scenario], out_dir: Path) -> None:
    """CPU-only: crop every first frame to the pipeline geometry and report sizes."""
    from PIL import Image
    sizes = {}
    for s in scen:
        im = Image.open(s.switch_frame); sizes[im.size] = sizes.get(im.size, 0) + 1
        pr.load_conditioning_image(s.switch_frame)  # fails loudly if the crop does not give the 60x104 latent
    done = sum((out_dir / s.out_name).exists() for s in scen)
    print(f"[wmb:plan] {len(scen)} instances, {done} already generated in {out_dir}; frame sizes {sizes}; "
          f"unique prompts {len(set(s.prompt for s in scen))}; t5 contexts missing {len(physiq_t5.missing_prompts([s.prompt for s in scen]))}")


def stage_t5cache(scen: list[pr.Scenario]) -> None:
    n = physiq_t5.encode_missing(tuple(dict.fromkeys(s.prompt for s in scen)))
    print(f"[wmb:t5cache] encoded {n} prompts on {physiq_t5.encoder_device_name()}")


def stage_generate(scen: list[pr.Scenario], arm: str, out_dir: Path, run_id: int, shift: float, batch: int,
                   adapter_ckpt: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    removed = pr.clear_tmp_files(out_dir)
    pending = [s for s in scen if not (out_dir / s.out_name).exists()]
    print(f"[wmb:generate:{arm}] {len(scen) - len(pending)} done, {len(pending)} pending, batch {batch} "
          f"({len(removed)} tmp leftovers removed) -> {out_dir}", flush=True)
    if not pending:
        (RUNS / f"GEN_{arm}_DONE").write_text(pr.now_iso() + "\n"); return
    pr.check_arm_supported(arm, adapter_ckpt); pr.require_tools()
    if not (pr.ACTION_DIR / "poses.npy").exists():
        pr.fail("Physics-IQ static action dir missing (run physiq_runner --stage actiondir)")
    t0 = time.perf_counter()
    pipe, note = pr.build_pipeline([s.prompt for s in pending])
    pipe, adapter_info = pr.install_arm_adapter(pipe, arm, adapter_ckpt)
    pr.append_status(f"wmb {arm}: pipeline loaded in {time.perf_counter() - t0:.0f}s ({note}); adapter {adapter_info.get('adapter')}; {len(pending)} pending")
    meta = {"benchmark": "WorldModelBench i2v", "subset": str(SUBSET), "arm": arm, "run_id": run_id, "shift": shift, "batch": batch,
            "frame_num": pr.FRAME_NUM, "fps": pr.FPS, "output_wh": [pr.LAT_W * 8, pr.LAT_H * 8], "crop_aspect": pr.CROP_ASPECT,
            "prompt_field": "text_instruction", "image_field": "first_frame", "attention": note, "adapter": adapter_info,
            "weights": str(pr.LINGBOT_WEIGHTS), "git_rev": pr._git_rev(), "written": pr.now_iso()}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    RUNS.mkdir(parents=True, exist_ok=True); timing = RUNS / f"gen_{arm}_timing.jsonl"
    n_done = 0
    for i in range(0, len(pending), batch):
        for rec in pr._generate_group(pipe, pending[i:i + batch], out_dir, run_id, shift):
            n_done += 1
            with open(timing, "a") as f:
                f.write(json.dumps({"ts": pr.now_iso(), **rec}) + "\n")
            print(f"[wmb:generate:{arm}] {n_done}/{len(pending)} {rec['out_name']} gen {rec['gen_s']}s peak_alloc {rec['peak_alloc_gb']} GB sys {rec['system_used_gb']} GB", flush=True)
    if all((out_dir / s.out_name).exists() for s in scen):
        (RUNS / f"GEN_{arm}_DONE").write_text(pr.now_iso() + "\n")
        pr.append_status(f"wmb {arm}: generation complete ({len(scen)} videos) -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["plan", "t5cache", "generate"], required=True)
    ap.add_argument("--arm", default="vanilla", choices=pr.ARMS)
    ap.add_argument("--subset", type=Path, default=SUBSET); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0, help="0 = physiq batch_mode file (default 1)")
    ap.add_argument("--shift", type=float, default=pr.SHIFT); ap.add_argument("--run-id", type=int, default=1)
    ap.add_argument("--adapter-ckpt", type=Path, default=None)
    a = ap.parse_args()
    scen = load_instances(a.subset, a.limit); out_dir = OUT_ROOT / a.arm
    if a.stage == "plan":
        stage_plan(scen, out_dir)
    elif a.stage == "t5cache":
        stage_t5cache(scen)
    else:
        stage_generate(scen, a.arm, out_dir, a.run_id, a.shift, a.batch or pr.batch_mode(), a.adapter_ckpt)


if __name__ == "__main__":
    main()
