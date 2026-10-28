"""Run the official WorldModelBench evaluator (baseline/WorldModelBench/evaluation.py, unmodified) with the VILA
judge loaded in SDPA attention mode (no flash-attn on this machine) and, optionally, restricted to our
pre-registered 60-instance subset (paper/wmb/subset60.json).
Usage (inside .venv-vila, VILA_VISION_ATTN=sdpa):
  python paper/wmb/judge_run.py --video_dir <dir with <first_frame stem>.mp4> --model_name <arm> [--subset paper/wmb/subset60.json] [--save_name ...]
"""
import argparse, json, os, pathlib, shutil, sys, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[2]
WMB = ROOT / "baseline" / "WorldModelBench"
os.environ.setdefault("VILA_VISION_ATTN", "sdpa")
sys.path.insert(0, str(WMB)); os.chdir(WMB)
import evaluation as ev  # noqa: E402

JUDGE = str(ROOT / "data" / "wmb_judge" / "vila-ewm-qwen2-1.5b")


def _load_judge_sdpa(judge_path: str):
    import llava
    return llava.load(judge_path, attn_implementation="sdpa")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True); ap.add_argument("--model_name", required=True)
    ap.add_argument("--subset", default=None, help="JSON list of instances (default: all 350)")
    ap.add_argument("--save_name", default=None); ap.add_argument("--cot", action="store_true")
    a = ap.parse_args()
    ev.WorldModelEvaluator._load_judge = staticmethod(_load_judge_sdpa)
    argv = ["evaluation.py", "--judge", JUDGE, "--video_dir", str(pathlib.Path(a.video_dir).resolve()), "--model_name", a.model_name]
    if a.save_name:
        argv += ["--save_name", a.save_name]
    if a.cot:
        argv += ["--cot"]
    if a.subset:
        # the evaluator reads ./worldmodelbench.json relative to its folder: swap in the subset for this run, restore after
        orig = WMB / "worldmodelbench.json"; backup = WMB / "worldmodelbench.full.json"
        if not backup.exists():
            shutil.copy(orig, backup)
        shutil.copy(a.subset, orig)
        try:
            sys.argv = argv; ev.main()
        finally:
            shutil.copy(backup, orig)
    else:
        sys.argv = argv; ev.main()


if __name__ == "__main__":
    main()
