"""LaTeX table for the analog-digital probe (sub-study B and exploratory B2 of 33-allocation-strategy-study.md):
mean test R^2 over seeds per (target, core); prints the registered decision for the analog edge."""
from __future__ import annotations

import collections
import json
import pathlib

import numpy as np

RES = pathlib.Path(__file__).resolve().parents[2] / "runs" / "alloc" / "bias_probe"
CORES = [("ryd", "register"), ("gutz", "mean-field"), ("mlp", "MLP"), ("mlp+", "MLP$^{+}$"), ("mlpX4", "MLP$\\times$4"),
         ("rydlin", "register, linear enc.$^*$"), ("mlpsin", "sine-MLP$^*$")]
TARGETS = [("analog", w) for w in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)] + [("digital", 0.0)]


def main():
    R = collections.defaultdict(dict)
    for f in RES.glob("*.json"):
        d = json.loads(f.read_text())
        R[(d["target"], d["omega"] if d["target"] == "analog" else 0.0, d["core"])][d["seed"]] = d["test_r2"]
    cols = [t for t in TARGETS if any(R[(t[0], t[1], c)] for c, _ in CORES)]
    head = " & ".join(f"$\\omega{{=}}{w:g}$" if t == "analog" else t for t, w in cols)
    lines = ["\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", f"core & {head} \\\\", "\\midrule"]
    for c, name in CORES:
        cells = []
        for t, w in cols:
            v = R[(t, w, c)]
            cells.append(f"{np.mean(list(v.values())):.3f}" if v else "---")
        if any(x != "---" for x in cells):
            lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}\\end{center}"]
    wins = 0
    for w in (0.5, 1.0, 2.0, 4.0, 8.0):
        q, m = R[("analog", w, "ryd")], R[("analog", w, "mlp")]
        s = sorted(set(q) & set(m))
        if len(s) == 3 and all(q[k] > m[k] for k in s):
            wins += 1
    print("\n".join(lines))
    print(f"% registered rule: register beats matched MLP on 3/3 seeds at {wins} of 5 omega (needs >= 3) -> "
          f"{'SUPPORTED' if wins >= 3 else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
