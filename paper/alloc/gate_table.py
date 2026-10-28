"""Fill the evaluation-error appendix (docs 36-37) of main_iclr.tex from result JSONs.

    .venv/bin/python paper/alloc/adrc_summary.py && .venv/bin/python paper/alloc/gate_table.py

Replaces the text between %GATE-ROWS-BEGIN / %GATE-ROWS-END (table rows) and %FT-ROWS-BEGIN / %FT-ROWS-END (fine-tuning
rows). Co-processor columns use the route-supervised router (doc 35 §9) when that arm exists, else the plain router.
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEX = ROOT / "paper/manuscript/iclr2027/main_iclr.tex"
PUB = {"rayleigh_benard": "0.0415", "shear_flow": "0.0345",   # PDE-FM (arXiv:2511.21861, Table 3)
       "turbulent_radiative_layer_2D": "0.1930", "MHD_64": "0.1160"}   # Flowers 4->1; EddyFormer
NAME = {"rayleigh_benard": "rayleigh\\_benard", "shear_flow": "shear\\_flow", "turbulent_radiative_layer_2D": "TRL2D", "MHD_64": "MHD\\_64"}
TRUNK = {"mr30ryd": "register neck", "vanilla": "vanilla", "released": "released", "mlp30": "MLP neck", "q30-geo=3d": "3-D register neck", "q30-geo=3d-mlp": "3-D MLP neck"}
CORES = ["adrc_ryd", "adrc_gnn", "adrc_mlp"]
ORDER = ["shear_flow", "rayleigh_benard", "turbulent_radiative_layer_2D", "MHD_64"]


def fmt(v, sd=None):
    return "---" if v is None else f"{v:.4f}" + (f"\\,$\\pm$\\,{sd:.4f}" if sd is not None and sd >= 5e-5 else "")


def gate_rows():
    S = json.loads((ROOT / "runs/alloc/adrc_summary.json").read_text())
    plain = {r["row"]: r for r in S if r["gate"] and not r.get("route_sup")}
    rsup = {r["row"]: r for r in S if r["gate"] and r.get("route_sup") and not r.get("qubit_bands")}
    out = []
    for key in sorted(plain, key=lambda k: (ORDER.index(k.split("__")[0]) if k.split("__")[0] in ORDER else 9, k)):
        base, tag = key.split("__", 1)
        if "ens3" in tag:
            continue
        r = plain[key]; strat = tag.startswith("gatest")
        variant = tag.split("_", 1)[1].rsplit("_s", 1)[0] if "_" in tag else tag
        cell = lambda arm: (lambda a: fmt(a["mean"], a["sd"]) if a else "---")(
            (rsup.get(key, {}).get("arms", {}).get(arm)) or r["arms"].get(arm))
        gate = r["arms"].get("none")
        out.append(f"{NAME.get(base, base)}{'$^\\dagger$' if strat else ''} ({TRUNK.get(variant, variant)}) & {PUB.get(base, '---')} & "
                   f"{fmt(r['persistence_test'])} & {fmt(r['trunk0_test'])} & {fmt(r['anchor_test'])} & "
                   f"{fmt(gate['mean'], gate['sd']) if gate else '---'} & " + " & ".join(cell(a) for a in CORES) + r" \\")
    return out


def ft_rows():
    rows = []
    for f in sorted((ROOT / "runs/alloc/ft").glob("*.json")):
        d = json.loads(f.read_text())
        if "members" not in d and "dataset" in d:                # *_Eft.json: the registered ensemble, rows below; §29 divfree: own table
            rows.append(d)
    out = []
    for d in sorted(rows, key=lambda d: (ORDER.index(d["dataset"]) if d["dataset"] in ORDER else 9, d["variant"], d["loss"], d["head"])):
        sched = "pre" if d.get("gate_pretrain_n") else ("warm" if d.get("gate_warmup_s") else "joint")
        out.append(f"{NAME.get(d['dataset'], d['dataset'])} & {TRUNK.get(d['variant'], d['variant'])} & {d['loss']} & {d['head']} & {sched} & "
                   f"{d['valid_start']:.4f} & {d['valid_best']:.4f} & {d['strat_test']:.4f} & "
                   + (f"{d['full_test']:.4f}" if "full_test" in d else "---") + r" \\")
    for f in sorted((ROOT / "runs/alloc/ft").glob("*_Eft.json"), key=lambda f: ORDER.index(f.name[:-9]) if f.name[:-9] in ORDER else 9):
        e = json.loads(f.read_text())
        pick = "E-ft" if e["selected"] == "E-ft" else TRUNK.get(e["selected"].split("_vrmse")[0].split("_")[-1], e["selected"])
        out.append(f"{NAME.get(e['dataset'], e['dataset'])} & E-ft ({len(e['members'])} backbones, equal) & vrmse & gate & pre & --- & "
                   f"{e['valid']['E-ft']:.4f} & --- & " + (f"{e['full_test']:.4f}" if "full_test" in e else "---")
                   + (f" ({pick})" if pick != "E-ft" else "") + r" \\")
    return out


def band_rows():
    """Per-band decomposition of the route-supervised register arm (seed 1) after the gate: shares of the window's
    error before the gate; the chain captured_b <= span_b <= full_b is checked per window in every run."""
    out = []
    for key in ("shear_flow__gate_mr30ryd_s1337", "rayleigh_benard__gatest_mr30ryd_s1337"):
        f = ROOT / "runs/alloc/adrc_results" / f"{key}_adrc_ryd_gate_rs_s1.json"
        if not f.exists():
            continue
        b = json.loads(f.read_text())["bands_test"]; base = key.split("__")[0]
        for i in range(len(b["full"])):
            out.append(f"{NAME[base] if i == 0 else ''} & $b_{i}$ & {b['full'][i]:.4f} & {b['span'][i]:.4f} & "
                       f"{b['captured_qubit'][i]:.4f} & {b['captured_classical'][i]:.4f} & {100 * b['qubit_slot_share'][i]:.0f}\\% \\\\")
        out.append(f"& \\multicolumn{{6}}{{l}}{{max violation: captured$\\le$span {b['max_viol_cap_le_span']:.1e}, "
                   f"span$\\le$full {b['max_viol_span_le_full']:.1e}}} \\\\")
    return out


def fill(s, tag, body):
    return re.sub(rf"(%{tag}-BEGIN\n).*?(%{tag}-END)", lambda m: m.group(1) + body + "\n" + m.group(2), s, flags=re.S)


def main():
    g, f = "\n".join(gate_rows()), "\n".join(ft_rows())
    print(g); print(f)
    s = TEX.read_text()
    s = fill(s, "GATE-ROWS", g) if "%GATE-ROWS-BEGIN" in s else s
    s = fill(s, "FT-ROWS", f) if "%FT-ROWS-BEGIN" in s else s
    s = fill(s, "BAND-ROWS", "\n".join(band_rows())) if "%BAND-ROWS-BEGIN" in s else s
    TEX.write_text(s)


if __name__ == "__main__":
    main()
