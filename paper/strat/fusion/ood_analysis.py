"""Doc 35 §41x: out-of-distribution generalization of the Fusion pipeline and where the public-fold losses come from.

Reference model = HJl9's components (TRAIN-only: J1_s1-6 shape heads, QH_s1-6 q/b heads, TRAIN entry ensemble), whose public
per-scalar scores are known (Codabench 960275) and whose VALID + TEST (VT, 743 shots) scores are out-of-sample in distribution.
  pred   (GPU, inference only): head predictions on every cache row and every public row -> runs/strat/fusion/ood_pred.npz
  report (CPU): 1. OOD gap per term = public R2 - VT R2, split into error inflation vs label-spread change (the public label variance
            is estimated from the spread of the same model's predictions, calibrated on VT);
         2. shot-level domain classifiers (inputs-only descriptors + the heads' predicted scenario): TRAIN vs public, VT vs TRAIN
            (sanity, random split), AUC by 5-fold CV, permutation importance;
         3. importance-weighted VT R2 (shot weights p(public|x) / p(train|x)) = covariate-shift prediction of the public score,
            compared with the actual public numbers;
         4. per-shot error vs public-likeness (rank correlation), and which shot descriptors predict the per-shot error.
    .venv/bin/python paper/strat/fusion/ood_analysis.py pred | report
"""
from __future__ import annotations

import json
import sys

import numpy as np

import fx

SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
PRED = fx.OUT / "ood_pred.npz"
PUBLIC_HJL9 = {"R_axis": 0.983192, "Z_axis": 0.989752, "kappa": 0.975594, "tri_top": 0.993409, "tri_bot": 0.991841,
               "volume": 0.989382, "li": 0.960986, "q95": 0.996170, "betaN": 0.995577}
PUBLIC_F2 = {"R_axis": 0.984632, "Z_axis": 0.990677, "kappa": 0.979114, "tri_top": 0.993651, "tri_bot": 0.992538,
             "volume": 0.990621, "li": 0.967117, "q95": 0.996389, "betaN": 0.995573}


def pred() -> None:
    import torch
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import ens_select as ES
    import gpu_build as GB
    import submit_public as SP
    d = sota.Data(); pub = SP.PublicData(d.meta); allr = np.arange(len(d.X)); pr = np.arange(len(pub.X))
    J = "+".join(f"head_boundary_v8_J1_s{s}" for s in range(1, 7)); Q = "+".join(f"head_qb_QH_s{s}" for s in range(1, 7))
    sel = json.loads((GB.SOTA / "ens_select_mlp.json").read_text())
    def entry(data, shots):
        ent = [ES.member(str(fx.TRUNKS / m), data, shots) for m in sel["members"]]
        return np.concatenate([np.mean([m[s][1] for m in ent], 0) for s in shots])[:, :2]
    vt = np.nonzero(d.split != 0)[0]
    np.savez(PRED, j_all=GB.heads_gpu(d, allr, J)[:, :7].astype(np.float32), q_all=GB.heads_gpu(d, allr, Q)[:, :2].astype(np.float32),
             j_pub=GB.heads_gpu(pub, pr, J)[:, :7].astype(np.float32), q_pub=GB.heads_gpu(pub, pr, Q)[:, :2].astype(np.float32),
             e_vt=entry(d, list(np.unique(d.shot[vt]))).astype(np.float32), e_pub=entry(pub, pub.shots()).astype(np.float32),
             pub_shot=pub.shot)
    print("OOD_PRED_DONE", flush=True)


def shot_desc(X, fn, shot, P, Q, Xf):
    """Inputs-only shot descriptors + the heads' predicted scenario (means / spreads)."""
    col = {n: fn.index(n) for n in ("mag_plasma_current", "mag_bcoil", "mag_ECOILA", "t", "thomson_core_ne_mean", "thomson_core_Te_mean",
                                    "thomson_edge_Te_mean", "thomson_core_Te_count", "mag_F5A", "mag_F6A", "mag_F7A", "mag_F8A", "mag_F9A")}
    us, inv, cnt = np.unique(shot, return_inverse=True, return_counts=True)
    def agg(v, f):
        out = np.zeros(len(us))
        st = np.r_[0, np.cumsum(cnt)]
        for i in range(len(us)):
            out[i] = f(v[st[i]:st[i + 1]])
        return out
    ip = np.abs(Xf); D = {}
    D["ip_mean"], D["ip_max"], D["ip_std"] = agg(ip, np.mean), agg(ip, np.max), agg(ip, np.std)
    D["ip_sign"] = agg(Xf, lambda v: np.sign(np.mean(v)))
    D["bt_mean"] = agg(X[:, col["mag_bcoil"]], np.mean); D["ecoil_range"] = agg(X[:, col["mag_ECOILA"]], np.ptp)
    t = X[:, col["t"]]; D["t0"], D["dur"], D["nfr"] = agg(t, np.min), agg(t, np.ptp), cnt.astype(float)
    D["dt_med"] = agg(t, lambda v: np.median(np.diff(v)) if len(v) > 1 else 0.0)
    for n in ("thomson_core_ne_mean", "thomson_core_Te_mean", "thomson_edge_Te_mean"):
        D[n] = agg(X[:, col[n]], np.mean)
    D["ts_missing"] = agg(X[:, col["thomson_core_Te_count"]], lambda v: np.mean(v <= 0))
    for c in ("mag_F5A", "mag_F6A", "mag_F7A", "mag_F8A", "mag_F9A"):
        D[c + "_mean"] = agg(X[:, col[c]], np.mean)
    for j, n in enumerate(SC):
        D[f"p_{n}_mean"], D[f"p_{n}_std"] = agg(P[:, j], np.mean), agg(P[:, j], np.std)
    D["p_q95_mean"], D["p_betaN_mean"] = agg(Q[:, 0], np.mean), agg(Q[:, 1], np.mean)
    return us, inv, list(D), np.stack(list(D.values()), 1)


def report() -> None:
    import torch  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from scipy.stats import spearmanr
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    import submit_public as SP
    z = np.load(PRED); d = sota.Data(); pub = SP.PublicData(d.meta)
    xf_all = np.asarray(np.load(fx.OUT / "X_magfix_all.npy", mmap_mode="r")[:, d.fn.index("mag_plasma_current")])
    xf_pub = np.asarray(np.load(fx.OUT / "X_magfix_public.npy", mmap_mode="r")[:, d.fn.index("mag_plasma_current")])
    res = {}
    vt = np.nonzero(d.split != 0)[0]; tr = np.nonzero(d.split == 0)[0]
    lab = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); assert np.array_equal(lab["rows"], vt)
    Yvt = lab["y"][:, :7].astype(np.float64); Pvt = z["j_all"][vt].astype(np.float64); Ppub = z["j_pub"].astype(np.float64)
    qy = d.S[vt][:, [sota.LABELS.index("efit_q95"), sota.LABELS.index("efit_beta_n")]].astype(np.float64)
    qvt = 0.5 * (z["q_all"][vt] + z["e_vt"]).astype(np.float64); qpub = 0.5 * (z["q_pub"] + z["e_pub"]).astype(np.float64)
    # 1. OOD gap and its split
    print("== 1. OOD gap: public (HJl9) - VT (same TRAIN-only components) ==", flush=True)
    gap = {}
    for j, n in enumerate(SC + ["q95", "betaN"]):
        y, p, pp = (Yvt[:, j], Pvt[:, j], Ppub[:, j]) if j < 7 else (qy[:, j - 7], qvt[:, j - 7], qpub[:, j - 7])
        ok = np.isfinite(y) & np.isfinite(p); r2v = 1 - np.mean((y[ok] - p[ok]) ** 2) / np.var(y[ok])
        if n == "betaN":                                                  # VT shot 1252's -14.5 label frame (1 frame) dominates
            m = ok & (np.abs(y - p) < 5); r2v_trim = 1 - np.mean((y[m] - p[m]) ** 2) / np.var(y[m])
        vy, vp, vpp = np.var(y[ok]), np.var(p[ok]), np.var(pp[np.isfinite(pp)])
        var_pub = vy * vpp / vp                                         # public label variance, calibrated on VT
        mse_v = (1 - r2v) * vy; mse_p = (1 - PUBLIC_HJL9[n]) * var_pub
        gap[n] = {"vt": r2v, "public": PUBLIC_HJL9[n], "gap": PUBLIC_HJL9[n] - r2v, "var_ratio_pub_vt": var_pub / vy,
                  "mse_ratio_pub_vt": mse_p / mse_v, "rmse_vt": float(np.sqrt(mse_v)), "rmse_pub_est": float(np.sqrt(mse_p))}
        if n == "betaN":
            gap[n]["vt_trimmed"] = r2v_trim
        print(f"  {n:8s} VT {r2v:.5f} public {PUBLIC_HJL9[n]:.5f} gap {PUBLIC_HJL9[n] - r2v:+.5f} | label var pub/VT {var_pub / vy:.3f} "
              f"| error (MSE) pub/VT {mse_p / mse_v:.3f}" + (f" | VT trimmed {r2v_trim:.5f}" if n == "betaN" else ""), flush=True)
    res["gap"] = gap
    # 2. domain classifiers on shot descriptors
    print("== 2. shot-level domain classifiers ==", flush=True)
    Xa = np.asarray(d.X); Xp = np.asarray(pub.X)
    _, inv_a, names, Da = shot_desc(Xa, d.fn, d.shot, z["j_all"], z["q_all"], xf_all)
    _, _, _, Dp = shot_desc(Xp, d.fn, pub.shot, z["j_pub"], z["q_pub"], xf_pub)
    split_shot = np.array([d.split[np.nonzero(d.shot == s)[0][0]] for s in np.unique(d.shot)])
    Dtr, Dvt = Da[split_shot == 0], Da[split_shot != 0]
    clf = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0)  # noqa: E731
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for tag, A, B in (("public_vs_TRAIN", Dtr, Dp), ("VT_vs_TRAIN", Dtr, Dvt), ("public_vs_VT", Dvt, Dp)):
        X_ = np.concatenate([A, B]); y_ = np.r_[np.zeros(len(A)), np.ones(len(B))]
        p_ = cross_val_predict(clf(), X_, y_, cv=cv, method="predict_proba")[:, 1]; auc = roc_auc_score(y_, p_)
        m = clf().fit(X_, y_); pi = permutation_importance(m, X_, y_, scoring="roc_auc", n_repeats=5, random_state=0)
        top = sorted(zip(names, pi.importances_mean), key=lambda x: -x[1])[:8]
        res[f"auc_{tag}"] = auc; res[f"imp_{tag}"] = [(a, round(float(b), 4)) for a, b in top]
        print(f"  {tag}: AUC {auc:.3f} | top " + ", ".join(f"{a} {b:.3f}" for a, b in top), flush=True)
    # descriptor shifts (standardised mean difference public vs TRAIN)
    smd = (Dp.mean(0) - Dtr.mean(0)) / (Dtr.std(0) + 1e-12); o = np.argsort(-np.abs(smd))[:10]
    res["smd_public_vs_train"] = [(names[i], round(float(smd[i]), 3)) for i in o]
    print("  largest standardised mean shifts public vs TRAIN: " + ", ".join(f"{names[i]} {smd[i]:+.2f}" for i in o), flush=True)
    # 3. importance-weighted VT
    print("== 3. covariate-shift prediction of the public score (importance-weighted VT) ==", flush=True)
    m = clf().fit(np.concatenate([Dtr, Dp]), np.r_[np.zeros(len(Dtr)), np.ones(len(Dp))]); pv = np.clip(m.predict_proba(Dvt)[:, 1], 0.02, 0.98)
    w = pv / (1 - pv); w = w / w.mean(); ess = w.sum() ** 2 / np.sum(w ** 2); res["iw_ess"] = float(ess)
    vt_shots = np.unique(d.shot[vt]); sidx = np.searchsorted(vt_shots, d.shot[vt]); wf = w[sidx]
    pred_pub = {}
    for j, n in enumerate(SC + ["q95", "betaN"]):
        y, p = (Yvt[:, j], Pvt[:, j]) if j < 7 else (qy[:, j - 7], qvt[:, j - 7])
        ok = np.isfinite(y) & np.isfinite(p)
        if n == "betaN":
            ok &= np.abs(y - p) < 5
        ww = wf[ok]; mu = np.sum(ww * y[ok]) / ww.sum()
        r2w = 1 - np.sum(ww * (y[ok] - p[ok]) ** 2) / np.sum(ww * (y[ok] - mu) ** 2); pred_pub[n] = r2w
        print(f"  {n:8s} VT {gap[n]['vt'] if n != 'betaN' else gap[n]['vt_trimmed']:.5f} -> weighted {r2w:.5f} | actual public {PUBLIC_HJL9[n]:.5f}", flush=True)
    res["iw_vt"] = pred_pub; print(f"  importance-weight ESS {ess:.0f} of {len(w)} VT shots", flush=True)
    # 4. per-shot error vs public-likeness and its drivers
    print("== 4. per-shot error vs public-likeness; what predicts per-shot error (VT) ==", flush=True)
    for j, n in enumerate(SC):
        y, p = Yvt[:, j], Pvt[:, j]; ok = np.isfinite(y)
        se = np.bincount(sidx[ok], (y[ok] - p[ok]) ** 2, len(vt_shots)) / np.maximum(np.bincount(sidx[ok], minlength=len(vt_shots)), 1)
        rho = spearmanr(pv, se).correlation
        reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=15, random_state=0)
        ly = np.log(se + 1e-12); oof = cross_val_predict(reg, Dvt, ly, cv=5); r2s = 1 - np.mean((ly - oof) ** 2) / np.var(ly)
        mm = reg.fit(Dvt, ly); pi = permutation_importance(mm, Dvt, ly, n_repeats=5, random_state=0)
        top = sorted(zip(names, pi.importances_mean), key=lambda x: -x[1])[:5]
        res[f"shot_err_{n}"] = {"rho_public_likeness": float(rho), "oof_r2_log_mse": float(r2s), "drivers": [(a, round(float(b), 3)) for a, b in top]}
        print(f"  {n:8s} rho(err, public-likeness) {rho:+.3f} | shot-error predictability R2 {r2s:.2f} | drivers " + ", ".join(f"{a} {b:.2f}" for a, b in top), flush=True)
    (fx.OUT / "ood_analysis.json").write_text(json.dumps(res, indent=1, default=float)); print("OOD_REPORT_DONE", flush=True)


if __name__ == "__main__":
    pred() if sys.argv[1] == "pred" else report()
