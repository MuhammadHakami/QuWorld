"""§37 rayleigh_benard: CPU analysis of runs/strat/rb/diag_windows.npz (VALID stratified sample only).

Writes runs/strat/rb/diag_summary.json with, per marginal (overall, Ra, Pr, position, Ra x position):
  - population-weighted mean VRMSE (all fields and per field) of every prediction in the npz;
  - paired stratified-bootstrap deltas (95% CI) of: ensemble vs best single, each TTA variant vs single, persistence;
  - Krogh-Vedelsby shares (ambiguity / mean member MSE) per field;
  - band-resolved error (single, tta8, ensemble, target increment) and the share of the single-minus-ensemble error
    reduction in each band; y and x error profiles; plume error share.

    OMP_NUM_THREADS=8 .venv/bin/python paper/strat/rb/analyze.py [npz]
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "runs" / "strat" / "rb"


def load_strata():
    d = json.loads((OUT / "strata.json").read_text())
    sid = np.concatenate([[i] * len(s["windows"]) for i, s in enumerate(d["strata"])])
    return d, sid


def weights_for(doc, sid):
    pw = np.array([s["pop_weight"] for s in doc["strata"]]); n = np.bincount(sid, minlength=len(pw))
    return pw[sid] / n[sid]


def strat_boot(delta, base, sid, w, B=2000, seed=0):
    w = w / w.sum(); groups = [np.flatnonzero(sid == s) for s in np.unique(sid)]
    rng = np.random.default_rng(seed); d_bs, r_bs = np.empty(B), np.empty(B)
    for b in range(B):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        ww = w[idx] / w[idx].sum(); d = float((ww * delta[idx]).sum())
        d_bs[b] = d; r_bs[b] = d / float((ww * base[idx]).sum())
    d0 = float((w * delta).sum()); r0 = d0 / float((w * base).sum())
    return {"delta": d0, "ci95": [float(np.percentile(d_bs, 2.5)), float(np.percentile(d_bs, 97.5))],
            "rel": r0, "rel_ci95": [float(np.percentile(r_bs, 2.5)), float(np.percentile(r_bs, 97.5))],
            "n": int(len(delta)), "frac_improved": float((delta < 0).mean())}


def marginals(doc, sid):
    S = doc["strata"]
    ra = np.array([S[i]["rayleigh"] for i in sid]); pr = np.array([S[i]["prandtl"] for i in sid])
    pos = np.array([S[i]["position"] for i in sid])
    out = {"all": np.ones(len(sid), bool)}
    for q in ("onset", "developed"):
        out[f"pos={q}"] = pos == q
    for v in sorted(set(ra)):
        out[f"Ra={v:.0e}"] = ra == v
    for v in sorted(set(pr)):
        out[f"Pr={v:g}"] = pr == v
    for v in sorted(set(ra)):
        for q in ("onset", "developed"):
            out[f"Ra={v:.0e}&{q}"] = (ra == v) & (pos == q)
    for v in sorted(set(pr)):
        for q in ("onset", "developed"):
            out[f"Pr={v:g}&{q}"] = (pr == v) & (pos == q)
    return out


def wmean(v, w, m):
    ww = w[m] / w[m].sum()
    return float((ww[:, None] * v[m]).sum(0)) if v.ndim == 1 else (ww.reshape(-1, *[1] * (v.ndim - 1)) * v[m]).sum(0)


def main():
    fn = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "diag_windows.npz"
    z = np.load(fn, allow_pickle=False)
    doc, sid_all = load_strata()
    sid = z["sid"]; assert (sid == sid_all[:len(sid)]).all()
    w = weights_for(doc, sid)
    names = [str(n) for n in z["pred_names"]]; fields = [str(f) for f in z["fields"]]
    vr = z["vr"]                                           # (N, P, C)
    vm = vr.mean(2)                                        # (N, P): the metric (mean over fields)
    P = {n: i for i, n in enumerate(names)}
    single = "single_mr30ryd"
    M = marginals(doc, sid)
    comps = {"ens_vs_single": ("ens", single), "tta8_vs_single": ("tta8_mr30ryd", single),
             "flip2_vs_single": ("flip2_mr30ryd", single), "shift4_vs_single": ("shift4_mr30ryd", single),
             "tta8_vs_ens": ("tta8_mr30ryd", "ens"), "ens_tta8_vs_ens": ("ens_tta8", "ens"),
             "tta8_vanilla_vs_single_vanilla": ("tta8_vanilla", "single_vanilla"),
             "tta8_mlp30_vs_single_mlp30": ("tta8_mlp30", "single_mlp30"),
             "single_vs_persistence": (single, "persistence")}
    res = {"source": str(fn), "n_windows": int(len(sid)), "weighting": "population-weighted (estimates full VALID)",
           "fields": fields, "marginals": {}}
    for mk, m in M.items():
        if m.sum() == 0:
            continue
        wm = w[m] / w[m].sum()
        r = {"n": int(m.sum()), "mean": {n: float((wm * vm[m, P[n]]).sum()) for n in names},
             "per_field": {n: {f: float((wm * vr[m, P[n], c]).sum()) for c, f in enumerate(fields)}
                           for n in (single, "ens", "tta8_mr30ryd", "persistence", "trunk_raw_mr30ryd")},
             "median_single": float(np.median(vm[m, P[single]])), "deltas": {}}
        for ck, (a, b) in comps.items():
            r["deltas"][ck] = strat_boot(vm[m, P[a]] - vm[m, P[b]], vm[m, P[b]], sid[m], w[m], B=1000 if mk != "all" else 2000)
        # per-field ens-vs-single
        r["deltas_per_field_ens_vs_single"] = {
            f: strat_boot(vr[m, P["ens"], c] - vr[m, P[single], c], vr[m, P[single], c], sid[m], w[m], B=500)
            for c, f in enumerate(fields)}
        r["deltas_per_field_tta8_vs_single"] = {
            f: strat_boot(vr[m, P["tta8_mr30ryd"], c] - vr[m, P[single], c], vr[m, P[single], c], sid[m], w[m], B=500)
            for c, f in enumerate(fields)}
        mbar, amb, ens = z["kv_mbar"][m], z["kv_amb"][m], z["kv_ens"][m]
        r["kv_ambiguity_share"] = {f: float((wm * amb[:, c]).sum() / (wm * mbar[:, c]).sum()) for c, f in enumerate(fields)}
        band = z["band"][m]                                # (n, 4 diag, C, NB): single, tta8, ens, persistence
        bw = (wm[:, None, None, None] * band).sum(0)       # (4, C, NB)
        r["band_err"] = {dn: {f: bw[i, c].round(6).tolist() for c, f in enumerate(fields)}
                         for i, dn in enumerate(["single", "tta8", "ens", "increment"])}
        red = bw[0] - bw[2]                                # single - ens, per field x band (normalized mse units)
        r["ens_reduction_band_share"] = {f: (red[c] / max(red[c].sum(), 1e-30)).round(3).tolist() for c, f in enumerate(fields)}
        redt = bw[0] - bw[1]
        r["tta8_reduction_band_share"] = {f: (redt[c] / max(redt[c].sum(), 1e-30)).round(3).tolist() for c, f in enumerate(fields)}
        if mk in ("all", "pos=onset", "pos=developed", "Ra=1e+10", "Ra=1e+06", "Ra=1e+10&onset", "Ra=1e+10&developed"):
            yp = (wm[:, None, None, None] * z["yprof"][m]).sum(0); xp = (wm[:, None, None, None] * z["xprof"][m]).sum(0)
            r["yprof"] = {dn: yp[i].mean(0).round(6).tolist() for i, dn in enumerate(["single", "tta8", "ens", "persistence"])}
            r["xprof"] = {dn: xp[i].mean(0).round(6).tolist() for i, dn in enumerate(["single", "tta8", "ens", "persistence"])}
            pl = (wm[:, None, None] * z["plume"][m]).sum(0)
            r["plume_err_share"] = {dn: pl[i].round(4).tolist() for i, dn in enumerate(["single", "tta8", "ens", "persistence"])}
            r["plume_area_frac"] = float((wm * z["plume_frac"][m]).sum())
        res["marginals"][mk] = r
    # tail structure: share of the ens-vs-single reduction carried by the top decile of single-model error
    d = vm[:, P[single]] - vm[:, P["ens"]]
    q = vm[:, P[single]] >= np.quantile(vm[:, P[single]], 0.9)
    res["ens_gain_top_decile_share"] = float((w[q] * d[q]).sum() / (w * d).sum())
    res["tta8_gain_top_decile_share"] = float((w[q] * (vm[q, P[single]] - vm[q, P["tta8_mr30ryd"]])).sum()
                                             / (w * (vm[:, P[single]] - vm[:, P["tta8_mr30ryd"]])).sum())
    out = OUT / ("diag_summary.json" if fn.name == "diag_windows.npz" else fn.stem + "_summary.json")
    out.write_text(json.dumps(res, indent=1))
    a = res["marginals"]["all"]
    print(f"wrote {out}")
    print("overall (pop-weighted):", {k: round(v, 5) for k, v in a["mean"].items()})
    for ck, v in a["deltas"].items():
        print(f"  {ck}: {v['rel'] * 100:+.2f}% [{v['rel_ci95'][0] * 100:+.2f}, {v['rel_ci95'][1] * 100:+.2f}]")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()
