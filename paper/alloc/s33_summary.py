"""Doc 35 §33 / §33a: apply the registered VALID-only rules to whatever cells have landed (QW_SCREEN runs, so every
number here is VALID; TEST is never read). Writes runs/alloc/s33_summary.md.

    python paper/alloc/s33_summary.py
"""
from __future__ import annotations

import json
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
R, DIAG, OUT = ROOT / "runs/alloc/adrc_results_s33", ROOT / "runs/alloc/s33_diag.jsonl", ROOT / "runs/alloc/s33_summary.md"
ROWS = ("shear_flow__gate_vanilla_s1337", "rayleigh_benard__gatest_vanilla_s1337")
ARM = {"ryd": "adrc_ryd_gate_rs_dyn", "mlp": "adrc_mlp_gate_rs_dyn", "cls": "cls_only_gate"}


def valid(cell: str, row: str, arm: str, seed: int):
    f = R / cell / f"{row}_{ARM[arm]}_s{seed}.json"
    return json.loads(f.read_text())["test"] if f.exists() else None      # QW_SCREEN: the "test" field is VALID


def diag() -> dict:
    out = {}
    if DIAG.exists():
        for line in DIAG.read_text().splitlines():
            d = json.loads(line); out[(d["cell"], d["row"], d["arm"], d["seed"])] = d
    return out


def pct(a, b):
    return None if a is None or b is None else 100 * (a - b) / b


def fmt(v, p=5):
    return "---" if v is None else (f"{v:+.2f}%" if p == "%" else f"{v:.{p}f}")


def stage_a(D: dict) -> list[str]:
    L = ["## Stage A (current design, VALID)", "",
         "| row | seed | cls | MLP | register | reg gain/qubit vs cls | no-core Δ | swap pos Δ | swap tether Δ | fixed grid Δ | "
         "MLP swap pos Δ | bypass q50 | tether active | eff atoms |", "|" + "---|" * 14]
    causal = {}
    for row in ROWS:
        hits = 0
        for s in (1, 2, 3):
            c, m, q = (valid("A", row, a, s) for a in ("cls", "mlp", "ryd"))
            dq, dm = D.get(("A", row, "adrc_ryd", s), {}), D.get(("A", row, "adrc_mlp", s), {})
            g = None if c is None or q is None else 100 * (c - q) / c / 30
            sp = pct(dq.get("valid_swap_pos"), dq.get("valid")); stt = pct(dq.get("valid_swap_tether"), dq.get("valid"))
            hits += bool((sp or 0) >= 0.5 or (stt or 0) >= 0.5) if dq else 0
            L.append(f"| {row.split('__')[0]} | {s} | {fmt(c)} | {fmt(m)} | {fmt(q)} | {fmt(g, '%')} | "
                     f"{fmt(pct(dq.get('valid_no_core'), dq.get('valid')), '%')} | {fmt(sp, '%')} | {fmt(stt, '%')} | "
                     f"{fmt(pct(dq.get('valid_fixed_grid'), dq.get('valid')), '%')} | "
                     f"{fmt(pct(dm.get('valid_swap_pos'), dm.get('valid')), '%')} | "
                     f"{fmt((dq.get('bypass_ratio_q10_50_90') or [None] * 3)[1], 3)} | {fmt(dq.get('tether_frac_active'), 3)} | "
                     f"{fmt((dq.get('eff_atoms_mean_q10_50_90') or [None])[0], 2)} |")
        causal[row] = hits
    L += ["", "Rule (§33): per-input configuration is causal on a row only if swapping positions or tethers raises the "
          "register's VALID by >= 0.5% on >= 2/3 seeds: " + ", ".join(f"{r.split('__')[0]} {h}/3" for r, h in causal.items())]
    return L


def candidates() -> list[str]:
    L = ["", "## Stage B candidates vs the current register (VALID; lower is better)", "",
         "| cell | row | seeds | register: candidate lower / n | MLP: candidate lower / n | promote? |", "|---|---|---|---|---|---|"]
    for cell, seeds, need in (("B1det", (1, 2, 3), 2), ("B1off", (1, 2, 3), 2), ("B2loss", (1, 2), 2), ("B2tgt", (1, 2), 2),
                              ("B2both", (1, 2), 2), ("B3lam", (1, 2, 3), 2),          # §33b: register only (MLP n = 0)
                              ("C0tau0", (1, 2, 3), 2)):        # §33c: "candidate lower" here means interaction did not help
        ok_rows = []
        for row in ROWS:
            rq = [(valid(cell, row, "ryd", s), valid("A", row, "ryd", s)) for s in seeds]
            rm = [(valid(cell, row, "mlp", s), valid("A", row, "mlp", s)) for s in seeds]
            nq = sum(1 for a, b in rq if a is not None and b is not None); wq = sum(1 for a, b in rq if a is not None and b is not None and a < b)
            nm = sum(1 for a, b in rm if a is not None and b is not None); wm = sum(1 for a, b in rm if a is not None and b is not None and a < b)
            ok_rows.append(nq == len(seeds) and wq >= need)
            L.append(f"| {cell} | {row.split('__')[0]} | {len(seeds)} | {wq}/{nq} | {wm}/{nm} | |")
        L[-1] = L[-1][:-2] + ("yes |" if all(ok_rows) else "no (or incomplete) |")
    return L


def nocore(D: dict) -> list[str]:
    L = ["", "## §33a idle-core test (VALID)", "", "| row | seed | register | MLP | gap (reg-MLP) | retrained no-core | "
         "no-core - register | frozen no-core Δ |", "|---|---|---|---|---|---|---|---|"]
    for row in ROWS:
        worse = 0
        for s in (1, 2, 3):
            q, m, n = valid("A", row, "ryd", s), valid("A", row, "mlp", s), valid("Anocore", row, "ryd", s)
            gap = None if q is None or m is None else abs(q - m)
            dq = D.get(("A", row, "adrc_ryd", s), {})
            fz = None if not dq else dq.get("valid_no_core", 0) - dq.get("valid", 0)
            worse += bool(n is not None and q is not None and gap is not None and n - q > gap)
            L.append(f"| {row.split('__')[0]} | {s} | {fmt(q)} | {fmt(m)} | {fmt(gap)} | {fmt(n)} | "
                     f"{fmt(None if n is None or q is None else n - q)} | {fmt(fz)} |")
        L.append(f"| {row.split('__')[0]} | rule | core materially helps if no-core worse than register by > gap on >= 2/3 "
                 f"seeds: {worse}/3 | | | | | |")
    return L


def main() -> None:
    D = diag()
    text = "\n".join(["# §33 / §33a summary (VALID only; generated by paper/alloc/s33_summary.py)", ""] + stage_a(D)
                     + candidates() + nocore(D)) + "\n"
    OUT.write_text(text); print(text)


if __name__ == "__main__":
    main()
