"""Mid-run sanity checker for sprint queues and long runs.

Usage: uv run paper/check_runs.py [round_prefix]
Scans runs/<prefix>*.log for: crashes, NaN/inf, suspicious step counts,
missing summaries (still running vs stalled), and prints a one-line verdict
per log plus an overall status. Exit code 1 if any FAIL.
"""

import glob
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "round"


def check(path: str) -> tuple[str, str]:
    txt = open(path).read()
    age_min = (time.time() - pathlib.Path(path).stat().st_mtime) / 60
    if re.search(r"nan|inf", txt.split("---")[-1], re.I) and "val_vrmse" in txt:
        return "FAIL", "NaN/inf in summary"
    if "Traceback" in txt or "Error" in txt.replace("CUDA_ERROR", ""):
        tail = txt.strip().splitlines()[-1][:80]
        return "FAIL", f"exception: {tail}"
    m = re.search(r"^val_vrmse_1:\s+(\S+)", txt, re.M)
    if not m:
        if age_min > 12:
            return "STALL", f"no summary, log idle {age_min:.0f} min"
        return "RUNNING", f"log active ({age_min:.0f} min old)"
    v1 = float(m.group(1))
    v8 = float(re.search(r"^val_vrmse_8:\s+(\S+)", txt, re.M).group(1))
    steps = int(re.search(r"^num_steps:\s+(\d+)", txt, re.M).group(1))
    if v1 != v1 or v8 != v8:  # NaN
        return "FAIL", "NaN metric"
    if v1 > 1.5:
        return "WARN", f"v1={v1:.3f} suspiciously bad (>1.5)"
    if steps < 500:
        return "WARN", f"only {steps} steps — throughput collapse?"
    return "OK", f"v1={v1:.3f} v8={v8:.3f} steps={steps}"


def main() -> int:
    fails = 0
    for f in sorted(glob.glob(str(ROOT / "runs" / f"{PREFIX}*.log"))):
        status, msg = check(f)
        fails += status == "FAIL" or status == "STALL"
        print(f"[{status:7s}] {pathlib.Path(f).stem}: {msg}")
    print("OVERALL:", "ISSUES FOUND" if fails else "healthy")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
