"""Render the Phase-E results table (mean +/- std over seeds) from
runs/phase_e/results.json. Usage: uv run paper/phase_e/summarize.py [--md]"""

import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "runs" / "phase_e" / "results.json"

ROW_ORDER = ["intact", "prod_zero", "prod_far", "chi1", "chi2", "chi4",
             "chi8", "chi16", "chi32", "shots128", "shots1024", "shots8192",
             "const0", "gate_off"]
LABELS = {
    "intact": "intact (exact statevector)",
    "prod_zero": "product-state: blockade -> 0",
    "prod_far": "product-state: far geometry (100x)",
    "chi1": "chi=1 Schmidt truncation",
    "chi2": "chi=2", "chi4": "chi=4", "chi8": "chi=8",
    "chi16": "chi=16", "chi32": "chi=32 (exact rank)",
    "shots128": "128-shot readout",
    "shots1024": "1024-shot readout",
    "shots8192": "8192-shot readout",
    "const0": "CONTROL: readout -> zeros",
    "gate_off": "CONTROL: adapter gate off",
}


def fmt(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def main() -> None:
    md = "--md" in sys.argv
    res = json.loads(RESULTS.read_text())
    seeds = sorted(res["seeds"])
    sep = " | " if md else "  "
    hdr = ["config"] + [f"s{s}" for s in seeds] + ["mean+/-std", "d_vs_intact"]
    if md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
    else:
        print(sep.join(f"{h:<34}" if i == 0 else f"{h:>10}"
                       for i, h in enumerate(hdr)))
    intact_mean = statistics.mean(
        res["seeds"][s]["rollout_vrmse8"]["intact"] for s in seeds)
    for name in ROW_ORDER:
        vals = [res["seeds"][s]["rollout_vrmse8"][name] for s in seeds
                if name in res["seeds"][s].get("rollout_vrmse8", {})]
        if not vals:
            continue
        m, sd = fmt(vals)
        cells = ([LABELS[name]] + [f"{v:.4f}" for v in vals]
                 + [f"{m:.4f}+/-{sd:.4f}", f"{m - intact_mean:+.4f}"])
        if md:
            print("| " + " | ".join(cells) + " |")
        else:
            print(sep.join(f"{c:<34}" if i == 0 else f"{c:>10}"
                           for i, c in enumerate(cells)))
    print()
    print("entanglement entropy (bits, 5|5 cut, 256 valid latents):")
    for s in seeds:
        ent = res["seeds"][s]["entanglement"]
        row = ", ".join(
            f"reg{r}: {ent[f'reg{r}']['entropy_bits']['mean']:.3f}"
            f"+/-{ent[f'reg{r}']['entropy_bits']['std']:.3f}"
            for r in range(3))
        cw1 = statistics.mean(
            ent[f"reg{r}"]["cum_schmidt_weight"]["chi1"] for r in range(3))
        print(f"  s{s}: {row}  (mean chi=1 weight {cw1:.4f})")


if __name__ == "__main__":
    main()
