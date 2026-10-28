"""Doc 35 §37, MHD_64: offline analysis of the stratified shards written by eval_strat.py (CPU only, exact via Parseval).

1. Diagnosis on VALID: per-variant VRMSE overall and by stratum (Ma regime x time bin), by trajectory (Ma x Ms), channel
   and radial wavenumber band; Flowers vs CNextU-net per stratum; excess over the published 0.1165 per stratum.
2. Test-time augmentation and divergence-free B: deltas against the raw model, per stratum, with paired bootstraps.
3. TRAIN-fitted blends / regime-conditioned corrections (weights fit on the TRAIN sample only, applied to VALID):
     y = p_anchor + sum_k theta_k (p_k - p_anchor), theta per group (pooled | Ma | Ma x channel | Ma x channel x band),
   fit by least squares on sum over TRAIN windows of MSE_c / var_c (the squared VRMSE), solved exactly from the Gram.
Paired bootstrap 95% intervals: (i) windows resampled within the 8 strata (5,000 reps); (ii) trajectories resampled
within each Ma regime (5 per regime on VALID; coarse, reported for honesty about within-trajectory correlation).

    uv run --no-sync python paper/strat/mhd/analyze.py
"""
from __future__ import annotations

import glob
import json
import os
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "runs" / "strat" / "mhd"
BANDS = [[0], [1], [2, 3], list(range(4, 8)), list(range(8, 16)), list(range(16, 32)), list(range(32, 56))]
BAND_NAMES = ["k0", "k1", "k2-3", "k4-7", "k8-15", "k16-31", "k32+"]
CH = ["rho", "Bx", "By", "Bz", "vx", "vy", "vz"]
TARGET = 0.1165
NBOOT = 5000


def load(split: str, passes: str = "full"):
    fs = sorted(glob.glob(str(OUT / os.environ.get("STRAT_SHARD_DIR", "shards") / f"{split}_{passes}_*.npz")))
    if not fs:
        return None
    Gs, var, Ey, rows, feats = [], [], [], [], []
    names = None
    for f in fs:
        d = np.load(f)
        names = list(d["names"]) if names is None else names
        assert list(d["names"]) == names
        G = d["G"].astype(np.float64)
        Gs.append(np.stack([G[..., b].sum(-1) for b in BANDS], -1))            # (V, V, 7, 7 bands)
        var.append(d["var_y"]); Ey.append(np.stack([d["Ey"][:, b].sum(-1) for b in BANDS], -1))
        rows.append(json.loads(str(d["row"]))); feats.append(json.loads(str(d["feat"])))
    return {"names": names, "G": np.stack(Gs), "var": np.stack(var) + 1e-7, "Ey": np.stack(Ey), "rows": rows,
            "feats": feats, "idx": {n: i for i, n in enumerate(names)}}


def regime(D) -> np.ndarray:
    """Causal regime from the last input frame: mean B_x 1.0 (Ma 0.7) vs 0.1 (Ma 2) (§36 step 0)."""
    return np.array([0 if f["meanB"][0] > 0.5 else 1 for f in D["feats"]])


def mse_of(D, W) -> np.ndarray:
    """W: (n, V, 7, nb) affine weights (sum over V = 1). Returns per-window, per-channel MSE (n, 7)."""
    return np.einsum("nicb,nijcb,njcb->nc", W, D["G"], W)


def vr_from_mse(D, mse) -> np.ndarray:
    return np.sqrt(np.clip(mse, 0, None) / D["var"]).mean(1)                    # (n,) per-window VRMSE


def onehot(D, name) -> np.ndarray:
    n, V = D["G"].shape[0], D["G"].shape[1]
    W = np.zeros((n, V, 7, len(BANDS))); W[:, D["idx"][name]] = 1.0
    return W


def vr_variant(D, name) -> np.ndarray:
    return vr_from_mse(D, mse_of(D, onehot(D, name)))


def fit_theta(D, anchor, free, group) -> dict:
    """Least-squares theta per group key on D (TRAIN). group(window, channel, band) -> hashable key."""
    a, ks = D["idx"][anchor], [D["idx"][f] for f in free]
    G, w = D["G"], 1.0 / D["var"]
    n = G.shape[0]
    Gaa = G[:, a, a]; Gka = np.stack([G[:, k, a] for k in ks], 1)
    Gkl = np.stack([np.stack([G[:, k, l] for l in ks], 1) for k in ks], 1)    # (n, K, K, 7, nb)
    A_t = Gkl - Gka[:, :, None] - Gka[:, None, :] + Gaa[:, None, None]
    b_t = Gka - Gaa[:, None]
    keys = {}
    for i in range(n):
        for c in range(7):
            for bb in range(len(BANDS)):
                keys.setdefault(group(i, c, bb), []).append((i, c, bb))
    theta = {}
    for key, cells in keys.items():
        ii, cc, bb = map(np.array, zip(*cells))
        A = (w[ii, cc][:, None, None] * A_t[ii, :, :, cc, bb]).sum(0)
        b = (w[ii, cc][:, None] * b_t[ii, :, cc, bb]).sum(0)
        theta[key] = np.linalg.solve(A + 1e-12 * np.eye(len(ks)) * np.trace(A), -b)
    return {"anchor": anchor, "free": free, "theta": theta}


def apply_theta(D, fit, group) -> np.ndarray:
    W = onehot(D, fit["anchor"]); a = D["idx"][fit["anchor"]]
    n = W.shape[0]
    for i in range(n):
        for c in range(7):
            for bb in range(len(BANDS)):
                th = fit["theta"].get(group(i, c, bb))
                if th is None:
                    continue
                for t, f in zip(th, fit["free"]):
                    W[i, D["idx"][f], c, bb] += t; W[i, a, c, bb] -= t
    return W


def boot(D, base, new, reps=NBOOT, seed=0) -> dict:
    """Paired bootstrap of mean(new) - mean(base): windows within strata, and trajectories within Ma."""
    rng = np.random.default_rng(seed); d = new - base
    strata = np.array([2 * r["tbin"] + (0 if r["Ma"] < 1 else 8) for r in D["rows"]])
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    wb = []
    for _ in range(reps):
        ii = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]); wb.append(d[ii].mean())
    trajs = np.array([r["traj"] for r in D["rows"]]); ma = np.array([r["Ma"] for r in D["rows"]])
    tb = []
    tg = {m: np.unique(trajs[ma == m]) for m in np.unique(ma)}
    for _ in range(reps):
        ii = np.concatenate([np.where(trajs == t)[0] for m, ts in tg.items() for t in ts[rng.integers(0, len(ts), len(ts))]])
        tb.append(d[ii].mean())
    return {"delta": float(d.mean()), "rel_pct": float(100 * d.mean() / base.mean()),
            "ci95_window": [float(np.quantile(wb, 0.025)), float(np.quantile(wb, 0.975))],
            "ci95_traj": [float(np.quantile(tb, 0.025)), float(np.quantile(tb, 0.975))],
            "win_frac_better": float((d < 0).mean())}


def by_stratum(D, v) -> dict:
    rows = D["rows"]; out = {}
    for key, f in (("Ma", lambda r: f"Ma{r['Ma']}"), ("Ma_tbin", lambda r: f"Ma{r['Ma']}_t{r['tbin']}"),
                   ("Ma_Ms", lambda r: f"Ma{r['Ma']}_Ms{r['Ms']}"), ("tbin", lambda r: f"t{r['tbin']}")):
        g = {}
        for i, r in enumerate(rows):
            g.setdefault(f(r), []).append(v[i])
        out[key] = {k: float(np.mean(x)) for k, x in sorted(g.items())}
    return out


def stratum_boot(D, base, new, seed=0) -> dict:
    """Per Ma regime and per Ma x tbin: mean delta with a window-level bootstrap CI inside the stratum."""
    rng = np.random.default_rng(seed); d = new - base; out = {}
    for key, f in (("Ma", lambda r: f"Ma{r['Ma']}"), ("Ma_tbin", lambda r: f"Ma{r['Ma']}_t{r['tbin']}")):
        g = {}
        for i, r in enumerate(D["rows"]):
            g.setdefault(f(r), []).append(i)
        for k, ii in sorted(g.items()):
            ii = np.array(ii); bs = [d[ii[rng.integers(0, len(ii), len(ii))]].mean() for _ in range(2000)]
            out[k] = {"base": float(base[ii].mean()), "new": float(new[ii].mean()),
                      "rel_pct": float(100 * d[ii].mean() / base[ii].mean()),
                      "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
    return out


def channel_band(D, name, mask=None) -> dict:
    """Error energy per channel and band in units of var(y) (squared-VRMSE contributions), mean over windows."""
    i = D["idx"][name]; m = np.ones(len(D["rows"]), bool) if mask is None else mask
    E = D["G"][m, i, i] / D["var"][m][..., None]                                   # (n, 7, nb)
    return {"per_channel_vrmse": dict(zip(CH, np.sqrt(E.sum(-1)).mean(0).round(5).tolist())),
            "band_share": dict(zip(BAND_NAMES, (E.sum(1).mean(0) / E.sum(1).mean(0).sum()).round(4).tolist())),
            "band_energy": dict(zip(BAND_NAMES, E.sum(1).mean(0).round(6).tolist()))}


def main() -> None:
    V = load("valid"); T = load("train", "trainlite")
    assert V is not None
    res = {"n_valid": len(V["rows"]), "n_train": 0 if T is None else len(T["rows"]), "variants": V["names"]}
    # regime read vs file label
    rv = regime(V); res["regime_read_mismatch_valid"] = int((rv != np.array([0 if r["Ma"] < 1 else 1 for r in V["rows"]])).sum())
    vr = {n: vr_variant(V, n) for n in V["names"]}
    res["valid_mean"] = {n: float(v.mean()) for n, v in vr.items()}
    res["valid_by_stratum"] = {n: by_stratum(V, vr[n]) for n in ("P", "F_raw", "F_all30_df", "C_raw", "C_all30_df",
                                                                  "C_sh8_df", "F_g16_df", "F_shg8_df")}
    # 1. diagnosis
    ma2 = np.array([r["Ma"] > 1 for r in V["rows"]])
    diag = {}
    for n in ("F_raw", "C_raw", "F_all30_df", "C_all30_df", "P"):
        diag[n] = {"all": channel_band(V, n), "Ma0.7": channel_band(V, n, ~ma2), "Ma2": channel_band(V, n, ma2)}
    Eyb = (V["Ey"] / V["var"][..., None]).mean(0).sum(0)
    diag["target_band_share_of_var"] = dict(zip(BAND_NAMES, (Eyb / 7).round(4).tolist()))
    res["diagnosis"] = diag
    ratio = {}
    for n1, n2 in (("C_raw", "F_raw"), ("C_all30_df", "F_all30_df")):
        e1, e2 = V["G"][:, V["idx"][n1], V["idx"][n1]], V["G"][:, V["idx"][n2], V["idx"][n2]]
        ratio[f"{n1}/{n2}"] = {
            "band_energy_ratio": dict(zip(BAND_NAMES, ((e1 / V["var"][..., None]).sum((0, 1)) /
                                                     (e2 / V["var"][..., None]).sum((0, 1))).round(3).tolist())),
            "channel_vrmse_ratio": dict(zip(CH, (np.sqrt(e1.sum(-1) / V["var"]).mean(0) /
                                               np.sqrt(e2.sum(-1) / V["var"]).mean(0)).round(3).tolist()))}
    res["C_over_F"] = ratio
    ex = {}
    for n in ("F_raw", "F_all30_df"):
        s = by_stratum(V, vr[n])["Ma_tbin"]; exc = {k: v - TARGET for k, v in s.items()}
        tot = sum(max(v, 0) for v in exc.values())
        ex[n] = {"excess_over_0.1165": {k: round(v, 4) for k, v in exc.items()},
                 "share_of_positive_excess": {k: round(max(v, 0) / tot, 3) for k, v in exc.items()} if tot > 0 else {},
                 "uniform_factor_needed": float(TARGET / vr[n].mean())}
    res["excess"] = ex
    # per-pass sanity: mean VRMSE of each individual TTA pass
    pp = {}
    for m in ("F", "C"):
        acc = {}
        for f in V["feats"]:
            for g, fl, sw, sh, v in f["pass_vr"][m]:
                acc.setdefault(f"{g}|flip{''.join(map(str, fl))}|swap{int(sw)}|shift{sh[0]}", []).append(np.mean(v))
        pp[m] = {k: round(float(np.mean(v)), 5) for k, v in acc.items()}
    res["per_pass_valid"] = pp
    # 2. TTA / divfree deltas vs raw
    tta = {}
    for m in ("F", "C"):
        for vn in [n for n in V["names"] if n.startswith(m + "_") and n != f"{m}_raw"]:
            tta[vn] = {"overall": boot(V, vr[f"{m}_raw"], vr[vn]), "strata": stratum_boot(V, vr[f"{m}_raw"], vr[vn])}
    tta["F_raw_vs_C_raw"] = {"overall": boot(V, vr["C_raw"], vr["F_raw"]), "strata": stratum_boot(V, vr["C_raw"], vr["F_raw"])}
    tta["F_all30_df_vs_C_all30_df"] = {"overall": boot(V, vr["C_all30_df"], vr["F_all30_df"]),
                                       "strata": stratum_boot(V, vr["C_all30_df"], vr["F_all30_df"])}
    res["tta"] = tta
    # 3. TRAIN-fitted blends and regime corrections
    if T is not None:
        vrT = {n: vr_variant(T, n) for n in T["names"]}
        res["train_mean"] = {n: float(v.mean()) for n, v in vrT.items()}
        regT, regV = regime(T), regime(V)
        grp = {
            "pooled": lambda R: (lambda i, c, b: 0),
            "ch": lambda R: (lambda i, c, b: c),
            "Ma_ch": lambda R: (lambda i, c, b: (R[i], c)),
            "Ma_ch_band": lambda R: (lambda i, c, b: (R[i], c, b)),
            "ch_band": lambda R: (lambda i, c, b: (c, b)),
        }
        blends = {}
        for suf in ("raw", "shg8_df"):
            specs = {f"F+C[{suf}]": (f"F_{suf}", [f"C_{suf}"]), f"F+P[{suf}]": (f"F_{suf}", ["P"]),
                     f"F+C+P[{suf}]": (f"F_{suf}", [f"C_{suf}", "P"])}
            for sname, (anchor, free) in specs.items():
                for gname, gf in grp.items():
                    if anchor not in T["idx"] or any(f not in T["idx"] for f in free):
                        continue
                    fit = fit_theta(T, anchor, free, gf(regT))
                    WT, WV = apply_theta(T, fit, gf(regT)), apply_theta(V, fit, gf(regV))
                    vt, vv = vr_from_mse(T, mse_of(T, WT)), vr_from_mse(V, mse_of(V, WV))
                    th = {str(k): np.round(t, 3).tolist() for k, t in list(fit["theta"].items())[:14]}
                    blends[f"{sname}|{gname}"] = {"train": float(vt.mean()), "valid": float(vv.mean()),
                                                  "vs_anchor": boot(V, vr[anchor], vv),
                                                  "strata_vs_anchor": stratum_boot(V, vr[anchor], vv),
                                                  "theta_head": th, "n_groups": len(fit["theta"])}
        res["blends"] = blends
    (OUT / f"results{os.environ.get('STRAT_SHARD_DIR', '') and '_' + os.environ['STRAT_SHARD_DIR']}.json").write_text(json.dumps(res, indent=1))
    # console summary
    print(f"VALID n={res['n_valid']} TRAIN n={res['n_train']} regime mismatches {res['regime_read_mismatch_valid']}")
    for n in V["names"]:
        s = res["valid_by_stratum"].get(n, by_stratum(V, vr[n]))["Ma"]
        print(f"  {n:12s} {res['valid_mean'][n]:.4f}  " + " ".join(f"{k} {v:.4f}" for k, v in s.items())
              + (f"  TRAIN {res['train_mean'][n]:.4f}" if n in res.get("train_mean", {}) else ""))
    for k, t in res["tta"].items():
        o = t["overall"]
        print(f"  TTA {k:26s} {o['rel_pct']:+6.2f}%  CIwin [{o['ci95_window'][0]:+.4f},{o['ci95_window'][1]:+.4f}] "
              f"CItraj [{o['ci95_traj'][0]:+.4f},{o['ci95_traj'][1]:+.4f}] "
              + " ".join(f"{s} {v['rel_pct']:+.1f}%" for s, v in t["strata"].items() if "_" not in s))
    for k, b in res.get("blends", {}).items():
        o = b["vs_anchor"]
        print(f"  BLEND {k:32s} train {b['train']:.4f} valid {b['valid']:.4f} {o['rel_pct']:+6.2f}% "
              f"CIwin [{o['ci95_window'][0]:+.4f},{o['ci95_window'][1]:+.4f}] CItraj [{o['ci95_traj'][0]:+.4f},"
              f"{o['ci95_traj'][1]:+.4f}]")


if __name__ == "__main__":
    main()
