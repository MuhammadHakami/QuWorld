"""Single-shot official-protocol TEST evaluation on a 2-D Well dataset: every
(trajectory, t0) window of the test split, N_HIST=4 -> next frame, ZScore
normalization, per-window mean-over-fields VRMSE (eps 1e-7) — the_well's
metric; writes per-window CSV and prints the mean vs the published number.
Usage: uv run paper/phase_b/eval_well2d.py <dataset> official|<ckpt.pt> [variant] [out.csv]
Env: BACKBONE (cnext default | tfno), BATCH (default 16), LIMIT (windows; default all) for smoke tests.
"""

import csv
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import well2d  # noqa: E402
from well2d import vrmse_rows  # noqa: E402
from well2d_backbone import ROOT, build, load_checkpoint  # noqa: E402

DEV = "cuda"


def main() -> None:
    dataset, src = sys.argv[1], sys.argv[2]
    variant = sys.argv[3] if len(sys.argv) > 3 else "vanilla"
    out_csv = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else None
    bsz, limit = int(os.environ.get("BATCH", "16")), int(os.environ.get("LIMIT", "0"))
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_FRACTION", "0.3")))
    backbone = os.environ.get("BACKBONE", "cnext")
    model = build("vanilla", dataset, backbone).to(DEV) if src == "official" else load_checkpoint(variant, dataset, src, backbone, DEV)
    model.eval()
    test = well2d.Split(dataset, "test")
    n_total = test.n * (test.T - well2d.N_HIST)
    t0, rows, pairs = time.time(), [], []
    with torch.no_grad():
        for i, (pp, x, y) in enumerate(well2d.trajectory_windows(test, bsz)):
            rows.extend(vrmse_rows(model(x.to(DEV)), y.to(DEV)).cpu().tolist()); pairs.extend(pp)
            if i % 100 == 0:
                print(f"  {len(rows)}/{n_total} windows, running mean {sum(rows)/len(rows):.4f}, {time.time()-t0:.0f}s", flush=True)
            if limit and len(rows) >= limit:
                break
    scores = torch.tensor(rows)
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["traj", "t0", "vrmse"])
            for (k, t), s in zip(pairs, rows):
                w.writerow([k, t, f"{s:.6f}"])
    pub = well2d.PUBLISHED.get(dataset, {})
    print("---")
    print(f"dataset:        {dataset}"); print(f"source:         {src} ({variant}, backbone {backbone})")
    print(f"n_windows:      {len(rows)} ({test.n} traj x {test.T - well2d.N_HIST} starts{' LIMITED' if limit else ''})")
    reference = ", ".join(f"{name} {value}" for name, value in pub.items())
    print(f"mean_vrmse:     {scores.mean():.4f}   (published references: {reference or 'not recorded'})")
    print(f"median_vrmse:   {scores.median():.4f}"); print(f"eval_seconds:   {time.time() - t0:.0f}")
    test.close()


if __name__ == "__main__":
    main()
