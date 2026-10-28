"""Bake-off runner: eval-unit-parity protocol, metrics, JSONL, kill checks.

Usage (from repo root):
  uv run paper/bakeoff/runner.py --strategy spsa --segment B --seed 1337
  uv run paper/bakeoff/runner.py --smoke          # full CPU smoke crossing

Protocol (14-structure-bakeoff.md section 2):
  - budget parity in EVAL-UNITS, not steps: 1 unit = one forward pass of the
    full model on one standard batch; a backward pass costs 2. Default
    7,500 units/segment (smoke: 200).
  - continual stream A -> B -> C (--segment all runs them in order on the
    SAME adapting model; a single --segment adapts the fresh substrate on
    that segment only).
  - controls: rls (FORCE readout-only, always-on baseline) and frozen
    (no-adapt forgetting floor) are strategies in the same harness.

Per-arm unit prices (documented deviations in the smoke report):
  gradient  k-step stop-grad pushforward: k fwd + 1 bwd = k+2 units/step
  spsa      2 fwd/step (+ charged calibration probes)
  eprop     1 fwd/step (+2 per SPSA sign-check); ZZ readout re-simulation is
            NOT charged — hardware measures those observables in the same
            shots (see qreadout.py)
  evolution 1 fwd per tier-0 weight-inheritance eval (fixed val-slice batch)
  rls       1 fwd/step

Metrics per (strategy, segment, seed) row (JSONL, results.jsonl):
  post-adaptation one-step VRMSE + rollout VRMSE@H on the segment's val
  trajectory; sample-efficiency (units/samples to reach within 5% of the
  arm's final val-slice VRMSE); forgetting (segment-A val regression vs the
  reference: A-val after A-adaptation in continual runs, else the substrate
  baseline); eval-units actually consumed; per-strategy kill-rule flags.
"""

import argparse
import dataclasses
import json
import pathlib
import time

import torch

from _bootstrap import REPO_DIR, add_paths

add_paths()

from data import vrmse  # noqa: E402
from strategies import REGISTRY  # noqa: E402
from strategies.base import FULL_STRATEGY, SMOKE_STRATEGY  # noqa: E402
from strategies.base import EvalMeter  # noqa: E402
from stream import FULL_STREAM, SEGMENTS, SMOKE_STREAM  # noqa: E402
from stream import build_segment  # noqa: E402
from substrate import FULL, SMOKE, build_substrate  # noqa: E402

SCALES = {
    "full": dict(substrate=FULL, stream=FULL_STREAM, strategy=FULL_STRATEGY,
                 units=7500.0, eval_every=250.0, rollout_h=8,
                 one_step_stride=4, rollout_stride=16),
    "smoke": dict(substrate=SMOKE, stream=SMOKE_STREAM,
                  strategy=SMOKE_STRATEGY, units=200.0, eval_every=25.0,
                  rollout_h=4, one_step_stride=8, rollout_stride=32),
}
# run order so the rls control exists before gradient's loses-to-RLS check
ORDER = ("frozen", "rls", "spsa", "eprop", "evolution", "gradient")


@torch.no_grad()
def one_step_val(model, val, device, stride, batch=16) -> float:
    idx = torch.arange(0, val.shape[0] - 1, stride)
    errs = []
    for i in range(0, len(idx), batch):
        sel = idx[i:i + batch]
        errs.append(vrmse(model(val[sel].to(device)),
                          val[sel + 1].to(device)).item())
    return sum(errs) / len(errs)


@torch.no_grad()
def rollout_val(model, val, device, horizon, stride) -> float:
    starts = torch.arange(0, val.shape[0] - 1 - horizon, stride)
    x = val[starts].to(device)
    for _ in range(horizon):
        x = model(x)
    return vrmse(x, val[starts + horizon].to(device)).item()


def make_probe(model, segment, batch, eval_every, curve, device):
    xv, yv = segment.val_slice(batch)
    xv, yv = xv.to(device), yv.to(device)
    state = {"next": 0.0}

    def probe(units: float, samples: int) -> None:
        if units + 1e-9 < state["next"]:
            return
        state["next"] = units + eval_every
        with torch.no_grad():
            v = vrmse(model(xv), yv).item()
        curve.append({"units": round(units, 2), "samples": samples,
                      "val_slice": round(v, 6)})

    probe.eval_pair = (xv, yv)
    return probe


def efficiency(curve: list, final_v: float) -> dict:
    thr = 1.05 * final_v
    for pt in curve:
        if pt["val_slice"] <= thr:
            return {"units_to_5pct": pt["units"],
                    "samples_to_5pct": pt["samples"]}
    return {"units_to_5pct": None, "samples_to_5pct": None}


def kill_checks(name: str, diag: dict, row: dict, rows: list) -> list:
    flags = []

    def is_num(v):
        return isinstance(v, (int, float)) and v == v

    if name == "gradient":
        if diag.get("nan_events", 0) >= 2:
            flags.append("nan_divergence")
        rls = [r for r in rows if r["strategy"] == "rls"
               and r["segment"] == row["segment"]
               and r["seed"] == row["seed"]]
        if rls and row["post_val_1"] >= rls[-1]["post_val_1"]:
            flags.append("loses_to_rls")
    elif name == "evolution":
        if is_num(diag.get("elite_gain_rel")) and \
                diag["elite_gain_rel"] <= 0.001:
            flags.append("no_elite_gain")
        if is_num(diag.get("coverage")) and diag["coverage"] < 0.10:
            flags.append("low_coverage")
    elif name == "eprop":
        if is_num(diag.get("probe_cosine_mean")) and \
                diag["probe_cosine_mean"] < 0.2:
            flags.append("low_probe_cosine")
        if is_num(diag.get("anti_frac")) and diag["anti_frac"] > 0.5:
            flags.append("anti_correlated")
    elif name == "spsa":
        if diag.get("calibration_diverged"):
            flags.append("calibration_divergence")
    return flags


def run_strategy(name: str, seg_names: list, seed: int, scale: str,
                 units: float, ckpt: str | None, outdir: pathlib.Path,
                 device: str, seg_cache: dict) -> list:
    sc = SCALES[scale]
    cls = REGISTRY[name]
    model = build_substrate(sc["substrate"], seed, checkpoint=ckpt,
                            device=device)

    def seg(n):
        if n not in seg_cache:
            seg_cache[n] = build_segment(n, sc["stream"])
        return seg_cache[n]

    pre_a = one_step_val(model, seg("A").val, device, sc["one_step_stride"])
    refs = {"A": pre_a}  # val after most recent adaptation touching it
    adapted: list = []
    rows = []
    for sname in seg_names:
        segment = seg(sname)
        strategy = cls(model, sc["strategy"], seed, device, str(outdir))
        pre_val = one_step_val(model, segment.val, device,
                               sc["one_step_stride"])
        meter = EvalMeter(units)
        curve: list = []
        probe = make_probe(model, segment, sc["strategy"].batch,
                           sc["eval_every"], curve, device)
        t0 = time.perf_counter()
        strategy.adapt(segment, meter, probe)
        wall = time.perf_counter() - t0
        xv, yv = probe.eval_pair
        with torch.no_grad():
            final_slice = vrmse(model(xv), yv).item()
        curve.append({"units": round(meter.used, 2),
                      "samples": strategy.samples_seen,
                      "val_slice": round(final_slice, 6)})
        post_val = one_step_val(model, segment.val, device,
                                sc["one_step_stride"])
        post_roll = rollout_val(model, segment.val, device, sc["rollout_h"],
                                sc["rollout_stride"])
        forgetting = {}
        for prev in adapted:
            now = one_step_val(model, seg(prev).val, device,
                               sc["one_step_stride"])
            forgetting[prev] = now - refs[prev]
        post_a = (one_step_val(model, seg("A").val, device,
                               sc["one_step_stride"])
                  if sname != "A" else post_val)
        diag = strategy.diagnostics()
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "scale": scale, "strategy": name, "segment": sname,
            "seed": seed, "continual_prefix": list(adapted),
            "units_budget": units, "units_used": round(meter.used, 2),
            "n_fwd": meter.n_fwd, "n_bwd": meter.n_bwd,
            "samples_seen": strategy.samples_seen,
            "pre_val_1": round(pre_val, 6), "post_val_1": round(post_val, 6),
            "post_rollout": round(post_roll, 6),
            "rollout_h": sc["rollout_h"],
            "pre_A_val": round(pre_a, 6), "post_A_val": round(post_a, 6),
            "forgetting_A": round(post_a - refs["A"], 6),
            "forgetting_prev": {k: round(v, 6)
                                for k, v in forgetting.items()},
            "final_val_slice": round(final_slice, 6),
            **efficiency(curve, final_slice),
            "wall_seconds": round(wall, 2), "curve": curve,
            "diagnostics": diag, "ckpt": ckpt,
        }
        rows.append(row)
        adapted.append(sname)
        refs[sname] = post_val
        refs["A"] = post_a if sname != "A" else post_val
    return rows


def print_table(rows: list) -> None:
    cols = ("strategy", "seg", "pre_val1", "post_val1", "roll", "forget_A",
            "units", "samples", "wall_s", "kill_flags")
    widths = (10, 3, 9, 9, 8, 9, 7, 7, 7, 30)
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        vals = (r["strategy"], r["segment"], f"{r['pre_val_1']:.4f}",
                f"{r['post_val_1']:.4f}", f"{r['post_rollout']:.4f}",
                f"{r['forgetting_A']:+.4f}", f"{r['units_used']:.0f}",
                str(r["samples_seen"]), f"{r['wall_seconds']:.1f}",
                ",".join(r["kill_flags"]) or "-")
        print("  ".join(str(v).ljust(w) for v, w in zip(vals, widths)))
    print()


def load_existing(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strategy", default="all",
                    choices=[*REGISTRY, "all"])
    ap.add_argument("--segment", default="all", choices=[*SEGMENTS, "all"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scale", default="full", choices=list(SCALES))
    ap.add_argument("--units", type=float, default=None,
                    help="eval-unit budget per segment (default per scale)")
    ap.add_argument("--ckpt", default=None,
                    help="GPU-trained kmix30 substrate checkpoint (.pt); "
                         "random-init substrate if omitted")
    ap.add_argument("--out", default=None,
                    help="output dir (default runs/bakeoff[/smoke])")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced-scale CPU crossing: all strategies x all "
                         "segments, prints the smoke table")
    args = ap.parse_args()

    if args.smoke:
        args.scale, args.strategy, args.segment = "smoke", "all", "all"
    sc = SCALES[args.scale]
    units = args.units if args.units is not None else sc["units"]
    if args.out:
        outdir = pathlib.Path(args.out)
    else:
        outdir = REPO_DIR / "runs" / "bakeoff"
        if args.scale == "smoke":
            outdir = outdir / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"

    names = [n for n in ORDER if args.strategy in ("all", n)]
    seg_names = list(SEGMENTS) if args.segment == "all" else [args.segment]
    print(f"bake-off: strategies={names} segments={seg_names} "
          f"seed={args.seed} scale={args.scale} units={units} "
          f"ckpt={args.ckpt or 'random-init'}")
    print(f"substrate={dataclasses.asdict(sc['substrate'])}")
    print(f"stream={dataclasses.asdict(sc['stream'])}")

    all_rows = load_existing(results_path)
    new_rows = []
    seg_cache: dict = {}
    for name in names:
        t0 = time.perf_counter()
        rows = run_strategy(name, seg_names, args.seed, args.scale, units,
                            args.ckpt, outdir, args.device, seg_cache)
        with open(results_path, "a") as fh:  # write per strategy so a later
            for row in rows:                 # crash cannot lose earlier rows
                row["kill_flags"] = kill_checks(name, row["diagnostics"],
                                                row, all_rows + new_rows)
                new_rows.append(row)
                fh.write(json.dumps(row) + "\n")
        print(f"  {name}: {len(rows)} segment(s) done in "
              f"{time.perf_counter() - t0:.1f}s")
    print(f"appended {len(new_rows)} rows -> {results_path}")
    print_table(new_rows)
    bad = [r for r in new_rows
           if not all(isinstance(r[k], (int, float)) and r[k] == r[k]
                      for k in ("post_val_1", "post_rollout"))]
    if bad:
        raise SystemExit(f"non-finite metrics in {len(bad)} rows")


if __name__ == "__main__":
    main()
