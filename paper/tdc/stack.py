"""TDC ADMET: validation-selected blend of MapLight (CatBoost) and fine-tuned CheMeleon, per endpoint.

    .venv-tdc/bin/python paper/tdc/stack.py cm-valid [--arms vanilla mlp30 mr30ryd]     (GPU inference)
    .venv-tdc/bin/python paper/tdc/stack.py ml-valid [--features maplight maplight+chemeleon]  (CPU CatBoost)
    .venv-tdc/bin/python paper/tdc/stack.py blend    ->  runs/tdc_ensemble/stack.json + stack_table.txt

Protocol, declared on 2026-09-23 before any blended TEST score was computed:
- Seed s uses TDC's get_train_valid_split(seed=s). CheMeleon valid predictions come from that seed's best-on-valid
  checkpoint, identified by reproducing the saved TEST predictions; MapLight valid predictions come from MapLight refit
  on the train part only (same CatBoost params and seed).
- One weight w per (endpoint, MapLight feature set, CheMeleon arm) on {0, 0.1, ..., 1}, chosen by the valid metric
  averaged over the 5 seeds; TEST_s = w * MapLight_s (train_val fit) + (1 - w) * CheMeleon_s, scored once per seed by
  TDC's evaluator. w = 1 recovers MapLight and w = 0 recovers CheMeleon, so the blend is chosen, never tuned on TEST.
- Primary classical entry: MapLight+CheMeleon-features x CheMeleon-vanilla. Hybrid entry: x CheMeleon-mr30ryd, against
  the matched classical control x CheMeleon-mlp30 (same protocol, own w).
- Amendment (11:55, before any blended TEST score; after the MapLight+CheMeleon-feature TEST runs came out mixed): the
  MapLight feature set (MapLight vs MapLight+CheMeleon) is also chosen per endpoint on VALID ("valsel|<arm>"); the
  original primary is still reported. 12:00: MapLight+GNN (GIN port, paper/tdc/gin.py, validated 0.860 vs published
  0.859 on CYP2C9) and MapLight+GIN+CheMeleon join the VALID-chosen feature sets.
- Caveat: CheMeleon's valid split also drove its early stopping, so its valid score is mildly optimistic (w leans
  toward CheMeleon); reported, not corrected.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tdcdata as D  # noqa: E402

ENS = D.ROOT / "runs" / "tdc_ensemble"; VAL = ENS / "valid"
CM = D.ROOT / "runs" / "tdc_chemeleon"
ML_TEST = {f: D.ROOT / "runs" / d for f, d in {"maplight": "tdc_maplight", "maplight+chemeleon": "tdc_maplight_chemeleon",
                                                 "maplight+gin": "tdc_maplight_gin", "maplight+gin+chemeleon": "tdc_maplight_gin_chemeleon"}.items()}
SEEDS = (1, 2, 3, 4, 5)
GRID = np.round(np.linspace(0, 1, 11), 2)


def split(endpoint: str, seed: int):
    return D.group().get_train_valid_split(benchmark=endpoint, split_type="default", seed=seed)


def valid_metric(endpoint: str, y, p) -> float:
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score
    m = D.METRICS[endpoint]
    if m == "mae":
        return float(np.mean(np.abs(np.asarray(y) - p)))
    if m == "roc-auc":
        return float(roc_auc_score(y, p))
    if m == "pr-auc":
        return float(average_precision_score(y, p))
    if m == "spearman":
        return float(spearmanr(y, p)[0])
    raise ValueError(f"unknown metric {m} for {endpoint}")


def cm_valid(endpoint: str, arm: str, seed: int) -> str:
    """Valid predictions of the CheMeleon run whose checkpoint reproduces the saved TEST predictions."""
    dst = VAL / f"cm_{arm}_{endpoint}_s{seed}.npy"
    if dst.exists():
        return "cached"
    test_ref = np.load(CM / f"{endpoint}_{arm}_preds.npz")[f"s{seed}"]
    import lightning.pytorch as pl
    import torch
    from chemprop import data
    import run_chemeleon as RC
    model, _, va, te = RC.build_model(endpoint, arm, seed)
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False, enable_progress_bar=False)
    pred = lambda ds: torch.cat(trainer.predict(model, data.build_dataloader(ds, batch_size=256, num_workers=0, shuffle=False))).squeeze(-1).float().cpu().numpy()
    ckpts = sorted((CM / "ckpt" / f"{endpoint}_{arm}_s{seed}").glob("*.ckpt"), key=lambda f: f.stat().st_mtime, reverse=True)
    for ck in ckpts:
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"])
        if np.allclose(pred(te), test_ref, rtol=1e-4, atol=1e-4):
            np.save(dst, pred(va))
            return ck.name
    raise RuntimeError(f"no checkpoint of {endpoint}/{arm}/s{seed} reproduces its TEST predictions ({len(ckpts)} tried)")


def ml_valid(endpoint: str, feats: str, seed: int) -> None:
    dst = VAL / f"ml_{feats}_{endpoint}_s{seed}.npy"
    if dst.exists():
        return
    import run_maplight as RM
    tv = D.group().get(endpoint)["train_val"]
    X = RM.features(endpoint, tv["Drug"].tolist(), "trainval", feats)
    row = {s: i for i, s in enumerate(tv["Drug"])}
    tr, va = split(endpoint, seed)
    Xtr, Xva = X[[row[s] for s in tr["Drug"]]], X[[row[s] for s in va["Drug"]]]
    np.save(dst, RM.fit_predict(endpoint, Xtr, tr["Y"].values, Xva, seed))


def choose_w(endpoint: str, ml_v, cm_v, ys) -> tuple[float, list[float]]:
    sign = 1 if D.higher_is_better(endpoint) else -1
    scores = [float(np.mean([valid_metric(endpoint, y, w * a + (1 - w) * b) for a, b, y in zip(ml_v, cm_v, ys)])) for w in GRID]
    best = max(range(len(GRID)), key=lambda i: (sign * scores[i], -abs(GRID[i] - 0.5)))
    return float(GRID[best]), scores


def blend() -> None:
    g = D.group(); rows = {}
    for ep in D.endpoints():
        name, metric = g.get(ep)["name"], D.METRICS[ep]
        ys = [split(ep, s)[1]["Y"].values for s in SEEDS]
        row = {"metric": metric, "higher_is_better": D.higher_is_better(ep)}
        for feats, tdir in ML_TEST.items():
            ml_files = [VAL / f"ml_{feats}_{ep}_s{s}.npy" for s in SEEDS]
            if not (tdir / f"{ep}_preds.npz").exists() or not all(f.exists() for f in ml_files):
                continue
            ml_t, ml_v = np.load(tdir / f"{ep}_preds.npz"), [np.load(f) for f in ml_files]
            for arm in ("vanilla", "mlp30", "mr30ryd"):
                cm_files = [VAL / f"cm_{arm}_{ep}_s{s}.npy" for s in SEEDS]
                if not (CM / f"{ep}_{arm}_preds.npz").exists() or not all(f.exists() for f in cm_files):
                    continue
                cm_t, cm_v = np.load(CM / f"{ep}_{arm}_preds.npz"), [np.load(f) for f in cm_files]
                w, vs = choose_w(ep, ml_v, cm_v, ys)
                per = [float(g.evaluate({name: w * ml_t[f"s{s}"] + (1 - w) * cm_t[f"s{s}"]})[name][metric]) for s in SEEDS]
                row[f"{feats}|{arm}"] = {"w_maplight": w, "valid_by_w": vs, "valid_best": vs[list(GRID).index(w)], "test_per_seed": per,
                                          "test_mean": float(np.mean(per)), "test_sd": float(np.std(per))}
        sign = 1 if row["higher_is_better"] else -1
        for arm in ("vanilla", "mlp30", "mr30ryd"):     # feature set also chosen on VALID (added before any blend TEST score)
            cands = [k for k in row if isinstance(row[k], dict) and k.endswith(f"|{arm}")]
            if cands:
                pick = max(cands, key=lambda k: sign * row[k]["valid_best"])
                row[f"valsel|{arm}"] = dict(row[pick], chosen=pick)
        rows[ep] = row
        print(f"[stack] {ep}: " + " ".join(f"{k} w={v['w_maplight']:.1f} {v['test_mean']:.4f}" for k, v in row.items() if isinstance(v, dict)), flush=True)
    (ENS / "stack.json").write_text(json.dumps(rows, indent=1))
    table(rows)


def rank_on(lb: list, mean: float, hib: bool) -> int:
    return 1 + sum((e["mean"] > mean) if hib else (e["mean"] < mean) for e in lb)


def table(rows: dict) -> None:
    lb = json.loads((ENS / "leaderboard_2026-09-23.json").read_text())["endpoints"]
    cols = ["valsel|vanilla", "valsel|mlp30", "valsel|mr30ryd", "maplight+gin|vanilla", "maplight|vanilla"]
    lines = [f"{'endpoint':32s} {'metric':8s} {'leader':>22s} " + " ".join(f"{c.replace('maplight+chemeleon', 'ML+C').replace('maplight', 'ML'):>22s}" for c in cols)]
    for ep, r in rows.items():
        top = lb[ep]["leaderboard"][0]
        cells = []
        for c in cols:
            if c in r:
                v = r[c]; cells.append(f"{v['test_mean']:.3f}±{v['test_sd']:.3f} w{v['w_maplight']:.1f} #{rank_on(lb[ep]['leaderboard'], v['test_mean'], r['higher_is_better'])}")
            else:
                cells.append("—")
        lines.append(f"{ep:32s} {r['metric']:8s} {top['model'][:12]:>12s} {top['mean']:.3f}±{top['sd']:.3f} " + " ".join(f"{c:>22s}" for c in cells))
    lines.append("bioavailability_ma: the TDC page repeats the Pgp table; its ranks are not meaningful.")
    (ENS / "stack_table.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["cm-valid", "ml-valid", "blend"])
    ap.add_argument("--arms", nargs="+", default=["vanilla", "mlp30", "mr30ryd"])
    ap.add_argument("--features", nargs="+", default=list(ML_TEST))
    ap.add_argument("--endpoints", nargs="+", default=None)
    a = ap.parse_args(); VAL.mkdir(parents=True, exist_ok=True)
    eps = a.endpoints or D.endpoints()
    if a.stage == "cm-valid":
        for ep in eps:
            for arm in a.arms:
                if not (CM / f"{ep}_{arm}_preds.npz").exists():
                    continue
                for s in SEEDS:
                    try:
                        print(f"[cm-valid] {ep} {arm} s{s}: {cm_valid(ep, arm, s)}", flush=True)
                    except RuntimeError as e:
                        print(f"[cm-valid] MISS {e}", flush=True)
    elif a.stage == "ml-valid":
        for feats in a.features:
            for ep in eps:
                if (ML_TEST[feats] / f"{ep}_preds.npz").exists():
                    for s in SEEDS:
                        ml_valid(ep, feats, s)
                    print(f"[ml-valid] {feats} {ep} done", flush=True)
    else:
        blend()


if __name__ == "__main__":
    main()
