"""§37 TokaMark analysis on the fixed VALID sample (strata.json); TRAIN subsample fits, VALID selects. CPU only.

    OMP_NUM_THREADS=4 .venv-tokamark/bin/python paper/strat/tokamark/analyze.py
    -> runs/strat/tokamark/{diagnosis.json, strategies.json}

Metric = TokaMark task NRMSE exactly as tokamark.evaluator (window RMSE = sqrt(nanmean diff^2) over the window's
values; shot/feature RMSE = sqrt(mean over windows of window RMSE^2); NRMSE = RMSE / train-split std; shot score = mean
over the features present; task = mean over shots). check_metric() re-derives one member's numbers through
tokamark.evaluator.aggregate_windows_metrics. A stratum = (shot stratum, signal group): its score is the mean over the
stratum's sampled shots of the mean NRMSE over the group's features. Paired bootstrap: shots resampled within stratum.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "runs" / "strat" / "tokamark"
N_BOOT = 2000
PHASES = ("ramp_up", "flat_top", "ramp_down")
VALID_GROUPS = ("scratch", "ftvan", "ftneck", "ftq", "soupscr", "soupft", "ftcont")


# ---------------------------------------------------------------------------------------------------------------- data
def load_split(split: str, groups=("scratch", "ftvan", "ftneck", "ftq")) -> dict:
    """Merge the per-group dumps on (feature, shot, window); returns {members, features, std, data{f: (sid,win,true,pred)}}."""
    parts = [np.load(OUT / f"preds_{g}_{split}.npz", allow_pickle=False) for g in groups
             if (OUT / f"preds_{g}_{split}.npz").exists()]
    members = [str(m) for p in parts for m in p["members"]]
    feats = [str(f) for f in parts[0]["features"]]
    std = dict(zip(feats, parts[0]["std"].astype(float)))
    data = {}
    for f in feats:
        keyed = []
        for p in parts:
            key = p[f"{f}|sid"] * 1_000_000 + p[f"{f}|win"]
            o = np.argsort(key, kind="stable")
            keyed.append((key[o], p[f"{f}|true"][o], p[f"{f}|pred"][:, o]))
        common = keyed[0][0]
        for k, _, _ in keyed[1:]:
            common = np.intersect1d(common, k)
        sel = [np.searchsorted(k, common) for k, _, _ in keyed]
        true = keyed[0][1][sel[0]]
        pred = np.concatenate([pr[:, s] for (_, _, pr), s in zip(keyed, sel)], axis=0)
        data[f] = (common // 1_000_000, common % 1_000_000, true, pred)
    return {"members": members, "features": feats, "std": std, "data": data}


# -------------------------------------------------------------------------------------------------------------- metric
def window_mse(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(n,d) truth, (K,n,d) predictions -> (K,n) nanmean of squared error (NaN where the window has no finite truth)."""
    d2 = (pred - true[None]) ** 2
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(d2).sum(-1)
        return np.where(cnt > 0, np.nansum(d2, -1) / np.maximum(cnt, 1), np.nan)


def shot_nrmse(sid: np.ndarray, wmse: np.ndarray, std: float, shots: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """(K,n) window MSE -> (K,S) NRMSE per shot in `shots` order (NaN where the shot has no window)."""
    ok = np.isfinite(wmse[0]) if keep is None else (np.isfinite(wmse[0]) & keep)
    idx = np.searchsorted(shots, sid[ok])
    valid = (idx < len(shots)) & (shots[np.minimum(idx, len(shots) - 1)] == sid[ok])
    idx, w = idx[valid], wmse[:, ok][:, valid]
    s = np.zeros((wmse.shape[0], len(shots)))
    c = np.zeros(len(shots))
    np.add.at(c, idx, 1.0)
    for k in range(wmse.shape[0]):
        np.add.at(s[k], idx, w[k])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, np.sqrt(s / np.maximum(c, 1)) / std, np.nan)


def predictor_table(D: dict, weights: dict[str, dict[str, np.ndarray]], shots: np.ndarray,
                    phase_of: dict | None = None) -> dict:
    """weights: name -> {feature -> (M,) member weights}. Returns {'all': (K,S,F)} (+ per phase) NRMSE tables."""
    names = list(weights)
    out = {"names": names, "all": np.full((len(names), len(shots), len(D["features"])), np.nan)}
    if phase_of is not None:
        for p in PHASES:
            out[p] = np.full_like(out["all"], np.nan)
    for j, f in enumerate(D["features"]):
        sid, win, true, pred = D["data"][f]
        W = np.stack([weights[n][f] for n in names])                     # (K,M)
        P = np.tensordot(W, pred, axes=(1, 0))                           # (K,n,d)
        wm = window_mse(true, P)
        out["all"][:, :, j] = shot_nrmse(sid, wm, D["std"][f], shots)
        if phase_of is not None:
            ph = phase_of(sid, win)
            for pi, p in enumerate(PHASES):
                out[p][:, :, j] = shot_nrmse(sid, wm, D["std"][f], shots, keep=(ph == pi))
    return out


def shot_scores(tab: np.ndarray, fidx: list[int]) -> np.ndarray:
    """(K,S,F) -> (K,S) per-shot mean over the selected features present."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(tab[:, :, fidx], axis=2)


def boot_delta(a: np.ndarray, b: np.ndarray, rng) -> dict:
    """Paired shot bootstrap of mean(a) - mean(b) (negative = a better); relative to mean(b)."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    bd = d[idx].mean(1)
    rel = bd / b[idx].mean(1)
    return {"n": int(len(d)), "a": float(a.mean()), "b": float(b.mean()), "delta": float(d.mean()),
            "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
            "rel": float(d.mean() / b.mean()), "rel_ci95": [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))],
            "wins_a": int((d < 0).sum())}


# ------------------------------------------------------------------------------------------------------ verification
def check_metric(D: dict, member: int, shots: np.ndarray) -> dict:
    """Recompute one member's task NRMSE through tokamark.evaluator.aggregate_windows_metrics (their code)."""
    sys.path.insert(0, str(ROOT / "baseline" / "tokamark" / "src"))
    from tokamark.evaluator import aggregate_windows_metrics
    rows = []
    for f in D["features"]:
        sid, win, true, pred = D["data"][f]
        wm = window_mse(true, pred[member:member + 1])[0]
        mae = np.nanmean(np.abs(pred[member] - true), axis=-1)
        rows.append(pd.DataFrame({"shot_id": sid, "window_index": win, "feature_name": f, "RMSE": np.sqrt(wm), "MAE": mae,
                                  "nan_fraction": np.isnan(true).mean(-1)}))
    df = pd.concat(rows, ignore_index=True)
    df = df[df["shot_id"].isin(shots)]
    _, df_task_shots, _ = aggregate_windows_metrics(df=df)
    ours = predictor_table(D, {"m": {f: np.eye(len(D["members"]))[member] for f in D["features"]}}, shots)["all"]
    return {"theirs": float(df_task_shots["NRMSE"].mean()), "ours": float(np.nanmean(shot_scores(ours, list(range(len(D["features"]))))[0]))}


# ------------------------------------------------------------------------------------------------------- predictors
def base_weights(D: dict) -> dict[str, dict[str, np.ndarray]]:
    M, feats = D["members"], D["features"]
    e = {m: np.eye(len(M))[i] for i, m in enumerate(M)}

    def mean_of(tags):
        tags = [t for t in tags if t in e]
        return (sum(e[t] for t in tags) / len(tags)) if tags else None

    combos = {m: e[m] for m in M}
    combos.update({
        "E4_registered": mean_of(["scr_vanilla", "ft_vanilla_s1337", "ft_vanilla_s1338", "ft_vanilla_s1339"]),
        "scr_ens3": mean_of(["scr_vanilla", "scr_mlp30", "scr_mr30ryd"]),
        "scr_ens2_van_mlp": mean_of(["scr_vanilla", "scr_mlp30"]),
        "scr_ens2_van_mr": mean_of(["scr_vanilla", "scr_mr30ryd"]),
        "ft_van_ens3": mean_of(["ft_vanilla_s1337", "ft_vanilla_s1338", "ft_vanilla_s1339"]),
        "ft_all_ens": mean_of([m for m in M if m[:-6] in ("ft_vanilla", "ft_mlp30", "ft_mr30ryd")]),
        "scr3_ft9_ens": mean_of([m for m in M if m.startswith("scr_") or m[:-6] in ("ft_vanilla", "ft_mlp30", "ft_mr30ryd")]),
    })
    return {k: {f: v for f in feats} for k, v in combos.items() if v is not None}


def fit_group_selection(tab_tr: dict, feats: list[str], groups: dict, cands: list[str]) -> dict[str, str]:
    """Per signal group, the candidate with the lowest TRAIN group score."""
    pick = {}
    for g, fl in groups.items():
        fidx = [feats.index(f) for f in fl]
        sc = np.nanmean(shot_scores(tab_tr["all"], fidx), axis=1)
        order = [tab_tr["names"].index(c) for c in cands]
        pick[g] = cands[int(np.argmin(sc[order]))]
    return pick


def fit_blend(D_tr: dict, a: np.ndarray, b: np.ndarray, shots: np.ndarray) -> dict[str, float]:
    """Per feature, w in [0,1] minimising the TRAIN mean-over-shots NRMSE of w*A + (1-w)*B."""
    grid = np.linspace(0, 1, 21)
    out = {}
    for f in D_tr["features"]:
        sid, _, true, pred = D_tr["data"][f]
        W = np.stack([w * a + (1 - w) * b for w in grid])
        wm = window_mse(true, np.tensordot(W, pred, axes=(1, 0)))
        sc = np.nanmean(shot_nrmse(sid, wm, D_tr["std"][f], shots), axis=1)
        out[f] = float(grid[int(np.nanargmin(sc))])
    return out


def fit_nnls(D_tr: dict) -> dict[str, np.ndarray]:
    """Per feature, non-negative least squares of the window values on the member predictions (TRAIN windows)."""
    out = {}
    for f in D_tr["features"]:
        _, _, true, pred = D_tr["data"][f]
        y = true.reshape(-1)
        X = pred.reshape(pred.shape[0], -1).T
        ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        w, _ = nnls(X[ok], y[ok])
        out[f] = w
    return out


# ------------------------------------------------------------------------------------------------------------ report
def phase_lookup(meta_split: dict):
    ph = {int(s): np.asarray(v["phase"], dtype=np.int8) for s, v in meta_split.items() if "phase" in v}

    def phase_of(sid: np.ndarray, win: np.ndarray) -> np.ndarray:
        out = np.full(len(sid), -1, dtype=np.int8)
        for s in np.unique(sid):
            m = sid == s
            p = ph.get(int(s))
            if p is not None:
                w = win[m]
                out[m] = np.where(w < len(p), p[np.minimum(w, len(p) - 1)], -1)
        return out
    return phase_of


def cell_masks(strata: dict, shots: np.ndarray) -> dict[str, np.ndarray]:
    return {n: np.isin(shots, v) for n, v in strata["strata_shots"].items()}


def diagnosis(D, tab, strata, shots, meta_v, rng) -> dict:
    feats, names = D["features"], tab["names"]
    groups = strata["signal_groups"]
    allf = list(range(len(feats)))
    cm = cell_masks(strata, shots)
    key = [n for n in ("scr_vanilla", "scr_mlp30", "scr_mr30ryd", "scr_ens3", "ft_vanilla_s1337", "ft_van_ens3",
                       "ft_all_ens", "E4_registered") if n in names]
    ki = {n: names.index(n) for n in key}
    per_shot_all = shot_scores(tab["all"], allf)                      # (K,S)
    ref = ki["scr_vanilla"]
    out = {"overall": {n: float(np.nanmean(per_shot_all[i])) for n, i in ki.items()},
           "per_feature": {f: {n: float(np.nanmean(tab["all"][i, :, j])) for n, i in ki.items()} for j, f in enumerate(feats)},
           "per_group": {}, "per_shot_stratum": {}, "cells": {}, "phase": {}, "concentration": {}}
    for g, fl in groups.items():
        fidx = [feats.index(f) for f in fl]
        s = shot_scores(tab["all"], fidx)
        out["per_group"][g] = {n: float(np.nanmean(s[i])) for n, i in ki.items()}
    # contribution of each (shot stratum, group) cell to the overall reference score
    nf = np.sum(np.isfinite(tab["all"][ref]), axis=1)                 # features present per shot
    total = np.nansum(per_shot_all[ref])
    for c, m in cm.items():
        out["per_shot_stratum"][c] = {n: float(np.nanmean(per_shot_all[i][m])) for n, i in ki.items()}
        for g, fl in groups.items():
            fidx = [feats.index(f) for f in fl]
            s = shot_scores(tab["all"], fidx)
            contrib = float(np.nansum(np.nansum(tab["all"][ref][m][:, fidx], axis=1) / nf[m]) / total)
            best = min(ki, key=lambda n: np.nanmean(s[ki[n]][m]))
            out["cells"][f"{c}|{g}"] = {"share_of_ref_error": contrib, "best": best,
                                        **{n: float(np.nanmean(s[i][m])) for n, i in ki.items()},
                                        "scr_vs_ftens": boot_delta(s[ref][m], s[ki["ft_van_ens3"]][m], rng)
                                        if "ft_van_ens3" in ki else None}
    for p in PHASES:
        s = shot_scores(tab[p], allf)
        out["phase"][p] = {"n_shots": int(np.isfinite(s[ref]).sum()), **{n: float(np.nanmean(s[i])) for n, i in ki.items()},
                           "per_group_scr_vanilla": {g: float(np.nanmean(shot_scores(tab[p], [feats.index(f) for f in fl])[ref]))
                                                     for g, fl in groups.items()}}
    srt = np.sort(per_shot_all[ref][np.isfinite(per_shot_all[ref])])[::-1]
    out["concentration"] = {"top10pct_share": float(srt[: max(1, len(srt) // 10)].sum() / srt.sum()),
                            "median_shot": float(np.median(srt)), "p90_shot": float(np.percentile(srt, 90))}
    worst = np.argsort(-np.nan_to_num(per_shot_all[ref], nan=-1))[:10]
    out["worst_shots"] = [{"shot": int(shots[i]), "score": float(per_shot_all[ref][i]),
                           "duration": meta_v[str(int(shots[i]))]["duration"], "term_drop": meta_v[str(int(shots[i]))]["term_drop"],
                           "worst_feature": feats[int(np.nanargmax(tab["all"][ref, i]))]} for i in worst]
    return out


def score_strategies(tab, strata, shots, feats, rng, ref="scr_vanilla") -> dict:
    names = tab["names"]
    groups = strata["signal_groups"]
    cm = cell_masks(strata, shots)
    allf = list(range(len(feats)))
    ps = shot_scores(tab["all"], allf)
    r = names.index(ref)
    res = {}
    for i, n in enumerate(names):
        if n == ref:
            continue
        e = {"overall": boot_delta(ps[i], ps[r], rng), "groups": {}, "shot_strata": {}, "cells": {}}
        for g, fl in groups.items():
            s = shot_scores(tab["all"], [feats.index(f) for f in fl])
            e["groups"][g] = boot_delta(s[i], s[r], rng)
            for c, m in cm.items():
                e["cells"][f"{c}|{g}"] = boot_delta(s[i][m], s[r][m], rng)
        for c, m in cm.items():
            e["shot_strata"][c] = boot_delta(ps[i][m], ps[r][m], rng)
        res[n] = e
    return res


def seed_analysis(D: dict, shots: np.ndarray) -> dict:
    """Seed spread and ensemble-size curves on the VALID sample: FT-Base vanilla (3 seeds) and the three Scratch runs
    (one seed, three necks: a lower bound on seed diversity)."""
    import itertools
    M = D["members"]
    out = {}
    for fam, tags in {"ft_vanilla_seeds": ["ft_vanilla_s1337", "ft_vanilla_s1338", "ft_vanilla_s1339"],
                      "scratch_same_seed_necks": ["scr_vanilla", "scr_mlp30", "scr_mr30ryd"],
                      "ft_mlp30_seeds": ["ft_mlp30_s1337", "ft_mlp30_s1338", "ft_mlp30_s1339"],
                      "ft_mr30ryd_seeds": ["ft_mr30ryd_s1337", "ft_mr30ryd_s1338", "ft_mr30ryd_s1339"]}.items():
        if not all(t in M for t in tags):
            continue
        W = {}
        for k in (1, 2, 3):
            for c in itertools.combinations(tags, k):
                W["+".join(c)] = {f: sum(np.eye(len(M))[M.index(t)] for t in c) / k for f in D["features"]}
        tab = predictor_table(D, W, shots)
        sc = np.nanmean(shot_scores(tab["all"], list(range(len(D["features"])))), axis=1)
        res = dict(zip(tab["names"], sc.tolist()))
        by_k = {k: [v for n, v in res.items() if n.count("+") == k - 1] for k in (1, 2, 3)}
        out[fam] = {"scores": res, "single_mean": float(np.mean(by_k[1])), "single_sd": float(np.std(by_k[1], ddof=1)),
                    "ens2_mean": float(np.mean(by_k[2])), "ens3": float(by_k[3][0]),
                    "ens3_gain_vs_single_mean": float(by_k[3][0] / np.mean(by_k[1]) - 1)}
    return out


def tta_analysis(Dv: dict, strata: dict, rng) -> dict | None:
    """Test-time input dropout probe on the fixed 180-shot subset: average the Scratch prediction with its
    one-input-dropped predictions (the recipe trains with p_drop_inputs = 0.05)."""
    files = sorted(OUT.glob("preds_scratch_validsub_drop-*.npz"))
    if not files:
        return None
    sub = np.array(sorted(int(x) for v in strata["strata_shots"].values() for x in sorted(v)[:30]))
    i0 = Dv["members"].index("scr_vanilla")
    names = ["scr_vanilla"] + [f.stem.split("drop-")[1] for f in files]
    D = {"features": Dv["features"], "std": Dv["std"], "members": names, "data": {}}
    zs = [np.load(f) for f in files]
    for feat in Dv["features"]:
        sid, win, true, pred = Dv["data"][feat]
        key = sid * 1_000_000 + win
        keep = np.isin(sid, sub)
        key, true, base = key[keep], true[keep], pred[i0][keep]
        extra = []
        for z in zs:
            k2 = z[f"{feat}|sid"] * 1_000_000 + z[f"{feat}|win"]
            o = np.argsort(k2)
            pos = np.searchsorted(k2[o], key)
            ok = (pos < len(k2)) & (k2[o][np.minimum(pos, len(k2) - 1)] == key)
            e = np.full_like(base, np.nan)
            e[ok] = z[f"{feat}|pred"][0][o][pos[ok]]
            extra.append(e)
        D["data"][feat] = (key // 1_000_000, key % 1_000_000, true, np.stack([base] + extra))
    M = len(names)
    e = np.eye(M)
    W = {n: {f: e[i] for f in D["features"]} for i, n in enumerate(names)}
    W["tta_uniform"] = {f: np.ones(M) / M for f in D["features"]}
    W["tta_half"] = {f: np.r_[0.5, np.full(M - 1, 0.5 / (M - 1))] for f in D["features"]}
    tab = predictor_table(D, W, sub)
    ps = shot_scores(tab["all"], list(range(len(D["features"]))))
    r = tab["names"].index("scr_vanilla")
    return {"n_shots": int(len(sub)), "scores": {n: float(np.nanmean(ps[i])) for i, n in enumerate(tab["names"])},
            "vs_scr_vanilla": {n: boot_delta(ps[i], ps[r], rng) for i, n in enumerate(tab["names"]) if i != r}}


def main() -> None:
    rng = np.random.default_rng(0)
    strata = json.loads((OUT / "strata.json").read_text())
    meta = json.loads((OUT / "shot_meta.json").read_text())
    Dv = load_split("valid", groups=VALID_GROUPS)
    feats, groups = Dv["features"], strata["signal_groups"]
    shots_v = np.array(sorted(strata["valid_sample_shots"]))
    W = base_weights(Dv)

    def to_v(wdict: dict[str, float]) -> np.ndarray:
        """{member: weight} (TRAIN member names) -> weight vector over the VALID members."""
        v = np.zeros(len(Dv["members"]))
        for m, w in wdict.items():
            v[Dv["members"].index(m)] = w
        return v

    fits = {}
    if (OUT / "preds_scratch_train.npz").exists() and (OUT / "preds_ftvan_train.npz").exists():
        Dt = load_split("train", groups=("scratch", "ftvan", "ftneck"))
        shots_t = np.array(sorted(int(s) for s in meta["train_sub"]))
        Wt = base_weights(Dt)
        tab_t = predictor_table(Dt, Wt, shots_t)
        cands = [n for n in Wt if n in W]
        pick = fit_group_selection(tab_t, feats, groups, cands)
        fits["group_selection_train"] = pick
        W["groupsel_train"] = {f: W[pick[g]][f] for g, fl in groups.items() for f in fl}
        for nm, (a, b) in {"blend_scr3_ftall": ("scr_ens3", "ft_all_ens"),
                           "blend_scrvan_ftvan3": ("scr_vanilla", "ft_van_ens3"),
                           "blend_scr3_ftvan3": ("scr_ens3", "ft_van_ens3")}.items():
            if a in Wt and b in Wt and a in W and b in W:
                wb = fit_blend(Dt, Wt[a][feats[0]], Wt[b][feats[0]], shots_t)
                fits[nm + "_train_w"] = wb
                W[nm + "_train"] = {f: wb[f] * W[a][f] + (1 - wb[f]) * W[b][f] for f in feats}
        nn = fit_nnls(Dt)
        fits["nnls_train"] = {f: dict(zip(Dt["members"], np.round(w, 4).tolist())) for f, w in nn.items()}
        W["nnls_train"] = {f: to_v(dict(zip(Dt["members"], nn[f]))) for f in feats}
        st = shot_scores(tab_t["all"], list(range(len(feats))))
        fits["train_scores"] = {n: float(np.nanmean(st[i])) for i, n in enumerate(tab_t["names"])}
    tab_v = predictor_table(Dv, W, shots_v, phase_lookup(meta["valid"]))
    allf = list(range(len(feats)))
    sv = shot_scores(tab_v["all"], allf)
    valid_scores = {n: float(np.nanmean(sv[i])) for i, n in enumerate(tab_v["names"])}
    # VALID-selected per-group oracle (upper bound only: selected on the scored sample itself)
    orc = {g: min(W, key=lambda n: np.nanmean(shot_scores(tab_v["all"], [feats.index(f) for f in fl])[tab_v["names"].index(n)]))
           for g, fl in groups.items()}
    W_or = {"groupsel_VALIDORACLE": {f: W[orc[g]][f] for g, fl in groups.items() for f in fl}, "scr_vanilla": W["scr_vanilla"]}
    tab_or = predictor_table(Dv, W_or, shots_v)
    so = shot_scores(tab_or["all"], allf)
    diag = diagnosis(Dv, tab_v, strata, shots_v, meta["valid"], rng)
    diag["metric_check"] = check_metric(Dv, Dv["members"].index("scr_vanilla"), shots_v)
    diag["n_windows"] = {f: int(len(Dv["data"][f][0])) for f in feats}
    diag["seed_analysis"] = seed_analysis(Dv, shots_v)
    strat = score_strategies(tab_v, strata, shots_v, feats, rng)
    strat["groupsel_VALIDORACLE"] = {"picks": orc, "overall": boot_delta(so[0], so[1], rng)}
    (OUT / "diagnosis.json").write_text(json.dumps(diag, indent=1))
    strat_e3 = score_strategies(tab_v, strata, shots_v, feats, rng, ref="scr_ens3")
    tta = tta_analysis(Dv, strata, rng)
    if tta is not None:
        (OUT / "tta.json").write_text(json.dumps(tta, indent=1))
        print("TTA", {k: round(v, 5) for k, v in tta["scores"].items()})
    (OUT / "strategies.json").write_text(json.dumps({"valid_sample_scores": valid_scores, "fits": fits,
                                                     "vs_scr_vanilla": strat, "vs_scr_ens3": strat_e3}, indent=1))
    print("metric check", diag["metric_check"])
    for n in sorted(valid_scores, key=valid_scores.get):
        e = strat.get(n, {}).get("overall")
        print(f"{n:28s} {valid_scores[n]:.5f}" + (f"  d={e['delta']:+.5f} [{e['ci95'][0]:+.5f},{e['ci95'][1]:+.5f}]" if e else ""))


if __name__ == "__main__":
    main()
