"""Compare a RouteFinder test.py log against the published Table 1 (RF-TE, n=100).

    python paper/vrp/compare_repro.py runs/routefinder/repro_rf-transformer_100.log
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent


def parse(log: str) -> dict[str, tuple[float, float]]:
    rows = re.findall(r"^(\w+) \| Cost: ([0-9.]+) \| Gap: (-?[0-9.]+)%", open(log).read(), re.M)
    return {name: (float(c), float(g)) for name, c, g in rows}


def main(log: str) -> None:
    pub = json.load(open(HERE / "published_rf_te_100.json"))
    ours = parse(log)
    print(f"{'variant':10s} {'obj ours':>9s} {'obj pub':>8s} {'Δobj %':>7s} {'gap ours':>9s} {'gap pub':>8s} {'OR-Tools':>9s}")
    d_obj, d_gap = [], []
    for v, (c, g) in ours.items():
        po, pg, orr = pub["obj"].get(v), pub["gap_pct"].get(v), pub["or_tools_gap_pct"].get(v)
        if po is None:
            continue
        d_obj.append((c - po) / po * 100); d_gap.append(g - pg)
        print(f"{v:10s} {c:9.3f} {po:8.3f} {d_obj[-1]:+7.2f} {g:8.3f}% {pg:7.3f}% {orr:8.3f}%")
    print(f"\n{len(d_obj)} variants | mean Δobj {st.mean(d_obj):+.3f}% (max |Δ| {max(map(abs, d_obj)):.3f}%) | "
          f"mean gap ours {st.mean(g for _, g in ours.values()):.3f}% vs published {st.mean(pub['gap_pct'][v] for v in ours if v in pub['gap_pct']):.3f}%")


if __name__ == "__main__":
    main(sys.argv[1])
