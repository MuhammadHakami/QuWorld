"""§37 rayleigh_benard: collect every scored strategy on the fixed stratified VALID sample and compare it, per stratum
marginal, with the best single model (mr30ryd gated `_pre`) and with the E-ft ensemble (paired stratified bootstrap,
95% CI, population-weighted so 'all' estimates the full-VALID mean). CPU only.

    OMP_NUM_THREADS=8 .venv/bin/python paper/strat/rb/report.py   -> runs/strat/rb/strategies.json
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np

from analyze import OUT, load_strata, marginals, strat_boot, weights_for

KEY_MARGINALS = ["all", "pos=onset", "pos=developed", "Ra=1e+06", "Ra=1e+07", "Ra=1e+08", "Ra=1e+09", "Ra=1e+10",
                 "Pr=0.1", "Pr=0.2", "Pr=0.5", "Pr=1", "Pr=2", "Pr=5", "Pr=10",
                 "Ra=1e+10&onset", "Ra=1e+10&developed", "Ra=1e+06&onset", "Ra=1e+06&developed"]


def candidates():
    z = np.load(OUT / "diag_windows.npz")
    names = [str(n) for n in z["pred_names"]]
    cand = {n: z["vr"][:, i, :] for i, n in enumerate(names)}
    for f in sorted(glob.glob(str(OUT / "ft_*.npz"))):
        if "smoke" in f:
            continue
        y = np.load(f); tag = pathlib.Path(f).stem
        for k in ("final", "ema", "swa", "start"):
            if k in y:
                cand[f"{tag}:{k}"] = y[k]
    for f in sorted(glob.glob(str(OUT / "head_*.npz"))):
        if "smoke" in f:
            continue
        cand[pathlib.Path(f).stem] = np.load(f)["vr"]
    for f in sorted(glob.glob(str(OUT / "tta_*.npz"))):
        y = np.load(f); tag = pathlib.Path(f).stem
        for k in ("single", "flip2", "shift4", "tta8", "shift2", "fs2", "fs4"):
            if k in y:
                cand[f"{tag}:{k}"] = y[k]
    return cand, z["sid"]


def main():
    doc, _ = load_strata()
    cand, sid = candidates()
    w = weights_for(doc, sid); M = marginals(doc, sid)
    base_s, base_e = cand["single_mr30ryd"].mean(1), cand["ens"].mean(1)
    out = {"n_windows": int(len(sid)), "baselines": {"single": "single_mr30ryd", "ensemble": "ens"}, "strategies": {}}
    for name, v in cand.items():
        vm = v.mean(1); r = {}
        for mk in KEY_MARGINALS:
            m = M[mk]; wm = w[m] / w[m].sum()
            r[mk] = {"mean": float((wm * vm[m]).sum()),
                     "vs_single": strat_boot(vm[m] - base_s[m], base_s[m], sid[m], w[m], B=1000),
                     "vs_ens": strat_boot(vm[m] - base_e[m], base_e[m], sid[m], w[m], B=1000)}
        out["strategies"][name] = r
    (OUT / "strategies.json").write_text(json.dumps(out, indent=1))
    print(f"{'strategy':44s} {'all':>8s} {'vs single % [CI]':>26s} {'vs ens % [CI]':>26s} {'onset vs single':>18s}")
    for name, r in out["strategies"].items():
        a = r["all"]; s = a["vs_single"]; e = a["vs_ens"]; o = r["pos=onset"]["vs_single"]
        print(f"{name:44s} {a['mean']:8.5f} {s['rel'] * 100:+7.2f} [{s['rel_ci95'][0] * 100:+6.2f},{s['rel_ci95'][1] * 100:+6.2f}]"
              f" {e['rel'] * 100:+7.2f} [{e['rel_ci95'][0] * 100:+6.2f},{e['rel_ci95'][1] * 100:+6.2f}] {o['rel'] * 100:+8.2f}")


if __name__ == "__main__":
    main()
