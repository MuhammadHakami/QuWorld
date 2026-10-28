"""End-to-end verification of the streaming TokaMark G2 loader (task 2-1).

Streams N shots (default 3, the first shots of the OFFICIAL random-split train
CSV), builds every window, and reports:

- per shot: cold-load wall time (S3), warm-reload wall time (LRU cache),
  window count, bytes fetched, and a full shape check of every window against
  the task metadata (input 5 ms / output 25 ms / stride 1 ms on each signal's
  official dt);
- totals: exact bytes moved off S3, cache directory size, and a hard
  <1 GB transfer-budget verdict (exit 1 on any failure).

Run:  uv run --no-sync python paper/tokamark/verify_stream.py [--n 3] [--split train]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import DEFAULT_CACHE_DIR, TokamarkStreamDataset  # noqa: E402

TRANSFER_BUDGET_BYTES = 1_000_000_000  # hard cap for this verification run


def expected_window_shapes(dataset: TokamarkStreamDataset) -> dict[str, dict[str, tuple]]:
    """Expected {role: {key: shape}} for window 'values' (official squeeze rule)."""
    expected: dict[str, dict[str, tuple]] = {}
    for role in ("input", "actuator", "output"):
        expected[role] = {}
        for key, meta in dataset.task_metadata[role].items():
            ts = meta["length_in_time_stamps"]
            channels = tuple(meta["values_shape"])
            expected[role][key] = (ts,) if channels == (1,) else channels + (ts,)
    return expected


def check_windows(dataset: TokamarkStreamDataset, shot_id: int,
                  expected: dict[str, dict[str, tuple]]) -> tuple[int, list[str], dict[str, float]]:
    """Iterate ALL windows of one shot; return (count, shape_errors, nan_fraction/role)."""
    errors: list[str] = []
    nan_sums = {"input": 0.0, "actuator": 0.0, "output": 0.0}
    nan_counts = {"input": 0, "actuator": 0, "output": 0}
    n_windows = 0
    for window in dataset.iter_shot(shot_id):
        n_windows += 1
        for role in ("input", "actuator", "output"):
            for key, want in expected[role].items():
                got = window[role][key]["values"].shape
                if got != want and len(errors) < 10:
                    errors.append(f"shot {shot_id} w{window['window_index']} {role}/{key}: "
                                  f"shape {got} != expected {want}")
            if n_windows % 50 == 1:  # NaN stats on a subsample (cost control)
                values = np.concatenate(
                    [np.ravel(window[role][key]["values"]) for key in expected[role]])
                nan_sums[role] += float(np.isnan(values).mean())
                nan_counts[role] += 1
    nan_fractions = {role: (nan_sums[role] / nan_counts[role] if nan_counts[role] else float("nan"))
                     for role in nan_sums}
    return n_windows, errors, nan_fractions


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="number of shots (default 3)")
    parser.add_argument("--split", default="train", help="official split column (default train)")
    parser.add_argument("--shots", type=int, nargs="*", default=None,
                        help="explicit shot IDs (overrides --n/--split ordering)")
    args = parser.parse_args(argv)

    dataset = TokamarkStreamDataset(split=args.split, shots=args.shots,
                                    max_shots=None if args.shots else args.n)
    expected = expected_window_shapes(dataset)

    print(f"TokaMark task 2-1 streaming verification — split={args.split}, "
          f"shots={dataset.shots}")
    print(f"cache: {DEFAULT_CACHE_DIR} (cap {dataset.cache.cap_bytes / 2**30:.0f} GiB)")
    print(f"expected input shapes:    { {k.split('-')[1]: v for k, v in expected['input'].items()} }")
    print(f"expected actuator shapes: { {k.split('-')[1]: v for k, v in expected['actuator'].items()} }")
    print(f"expected output shapes:   21 channels over "
          f"{len(expected['output'])} signals, e.g. ip {expected['output']['summary-ip']}, "
          f"q95 {expected['output']['equilibrium-q95']}, "
          f"x_point_r {expected['output']['equilibrium-x_point_r']}\n")

    failures = 0
    total_windows = 0
    for shot_id in dataset.shots:
        cache_path = dataset.cache._path(shot_id)
        cache_path.unlink(missing_ok=True)  # force a genuine cold load

        bytes_before = dataset.bytes_moved
        t_cold = time.time()
        dataset.load_shot(shot_id)
        t_cold = time.time() - t_cold
        shot_bytes = dataset.bytes_moved - bytes_before

        t_warm = time.time()
        dataset.load_shot(shot_id)  # second load must come from the LRU cache
        t_warm = time.time() - t_warm
        if dataset.bytes_moved != bytes_before + shot_bytes:
            print(f"  ERROR shot {shot_id}: warm reload touched S3")
            failures += 1

        t_win = time.time()
        n_windows, shape_errors, nan_fractions = check_windows(dataset, shot_id, expected)
        t_win = time.time() - t_win
        total_windows += n_windows

        cached_mb = cache_path.stat().st_size / 1e6 if cache_path.is_file() else 0.0
        print(f"shot {shot_id}: cold {t_cold:5.1f}s ({shot_bytes / 1e6:5.2f} MB from S3) | "
              f"warm {t_warm:5.2f}s | {n_windows} windows built+checked in {t_win:4.1f}s | "
              f"cache npz {cached_mb:.2f} MB")
        print(f"   NaN fraction (subsampled windows): "
              f"input {nan_fractions['input']:.3f}, actuator {nan_fractions['actuator']:.3f}, "
              f"output {nan_fractions['output']:.3f}")
        if shape_errors:
            failures += 1
            for line in shape_errors:
                print(f"   SHAPE ERROR: {line}")
        elif n_windows == 0:
            print(f"   WARNING: shot {shot_id} produced zero windows")

    cache_total = dataset.cache.total_bytes()
    print(f"\nTOTAL: {total_windows} windows across {len(dataset.shots)} shots | "
          f"{dataset.bytes_moved / 1e6:.2f} MB moved from S3 in {dataset.counter.ngets} GETs | "
          f"cache dir {cache_total / 1e6:.2f} MB")

    if dataset.bytes_moved > TRANSFER_BUDGET_BYTES:
        print(f"FAIL: transfer {dataset.bytes_moved} B exceeds budget {TRANSFER_BUDGET_BYTES} B")
        failures += 1
    print("VERDICT:", "PASS" if failures == 0 else f"FAIL ({failures} issue(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
