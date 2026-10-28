"""TDC ADMET — MapLight (Notwell & Wood 2023, arXiv 2310.00174) reproduced exactly on CPU CatBoost: the classical anchor.

    .venv-tdc/bin/python paper/tdc/run_maplight.py <endpoint|all> [--seeds 1 2 3 4 5]

Exactly MapLight's submission.ipynb: features = Morgan counts (r=2, 1024) + Avalon counts (1024) + ErG (315) + its
chosen RDKit descriptors; CatBoost defaults with random_strength=2, random_seed=seed, MAE loss (targets scaled by its
`scaler`, log10 where configured) or Logloss; trained on the official train_val, predicted on the official test,
evaluate_many over seeds 1-5. CPU by user exception (Sep 23, gradient-boosted trees only; the GPU build cannot run
Exact leaves + Plain boosting). Per-seed test predictions are saved for the ensembles (paper/tdc/ensemble.py, stack.py).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "baseline" / "MapLight-TDC"))
import tdcdata as D  # noqa: E402

OUT = D.ROOT / "runs" / "tdc_maplight"
FEAT = D.ROOT / "data" / "tdc_features"
CONFIG = {  # MapLight benchmark_config: (task, log-scale target)
    'caco2_wang': ('regression', False), 'bioavailability_ma': ('binary', False),
    'lipophilicity_astrazeneca': ('regression', False), 'solubility_aqsoldb': ('regression', False),
    'hia_hou': ('binary', False), 'pgp_broccatelli': ('binary', False), 'bbb_martins': ('binary', False),
    'ppbr_az': ('regression', False), 'vdss_lombardo': ('regression', True), 'cyp2c9_veith': ('binary', False),
    'cyp2d6_veith': ('binary', False), 'cyp3a4_veith': ('binary', False),
    'cyp2c9_substrate_carbonmangels': ('binary', False), 'cyp2d6_substrate_carbonmangels': ('binary', False),
    'cyp3a4_substrate_carbonmangels': ('binary', False), 'half_life_obach': ('regression', True),
    'clearance_hepatocyte_az': ('regression', True), 'clearance_microsome_az': ('regression', True),
    'ld50_zhu': ('regression', False), 'herg': ('binary', False), 'ames': ('binary', False), 'dili': ('binary', False)}


def fingerprints(smiles, tag: str) -> np.ndarray:
    """MapLight get_fingerprints (Morgan + Avalon + ErG + RDKit descriptors), cached per endpoint split."""
    dst = FEAT / f"maplight_{tag}.npy"
    if dst.exists():
        return np.load(dst)
    import pandas as pd
    from rdkit import Chem
    import maplight as M
    # RDKit 2026's DataStructs.ConvertToNumpyArray rejects MapLight's int8 buffer; same values via ToList, same int8 dtype
    M.count_to_array = lambda fp: np.asarray(fp.ToList(), dtype=np.int64).astype(np.int8)
    mols = pd.Series(smiles).apply(D.mol_of)
    X = np.concatenate([M.get_morgan_fingerprints(mols), M.get_avalon_fingerprints(mols),
                        M.get_erg_fingerprints(mols), M.get_rdkit_features(mols)], axis=1)
    FEAT.mkdir(parents=True, exist_ok=True); np.save(dst, X)
    return X


def chemeleon_embed(smiles, tag: str) -> np.ndarray:
    """Frozen CheMeleon (pretrained D-MPNN) mean-aggregated embedding, GPU inference, cached per endpoint split."""
    dst = FEAT / f"chemeleon_{tag}.npy"
    if dst.exists():
        return np.load(dst)
    import torch
    from chemprop import data, featurizers, nn as cnn
    import run_chemeleon as RC
    mp = RC.chemeleon_mp().cuda().eval(); agg = cnn.MeanAggregation()
    ds = data.MoleculeDataset([data.MoleculeDatapoint(D.mol_of(s)) for s in smiles], featurizers.SimpleMoleculeMolGraphFeaturizer())
    out = []
    with torch.no_grad():
        for b in data.build_dataloader(ds, batch_size=256, num_workers=0, shuffle=False):
            bmg = b.bmg; bmg.to("cuda")
            out.append(agg(mp(bmg), bmg.batch).cpu().numpy())
    X = np.concatenate(out); FEAT.mkdir(parents=True, exist_ok=True); np.save(dst, X)
    return X


FEATURES = "maplight"


def gin_embed(smiles, tag: str) -> np.ndarray:
    """MapLight+GNN's GIN (supervised masking) mean-pooled embedding, plain-PyTorch port on GPU (paper/tdc/gin.py)."""
    dst = FEAT / f"gin_{tag}.npy"
    if dst.exists():
        return np.load(dst)
    sys.path.insert(0, str(HERE.parent))
    import gpucap
    import gin
    gpucap.cap_from_env()
    X = gin.embed([D.mol_of(s) for s in smiles]); FEAT.mkdir(parents=True, exist_ok=True); np.save(dst, X)
    return X


FEATURE_SETS = {"maplight": "tdc_maplight", "maplight+chemeleon": "tdc_maplight_chemeleon",
                "maplight+gin": "tdc_maplight_gin", "maplight+gin+chemeleon": "tdc_maplight_gin_chemeleon"}


def features(endpoint: str, smiles, part: str, feats: str) -> np.ndarray:
    """MapLight fingerprints, then (MapLight+GNN's order) the GIN embedding, then the frozen CheMeleon embedding."""
    tag = f"{endpoint}_{part}"
    X = [fingerprints(smiles, tag)]
    if "gin" in feats.split("+"):
        X.append(gin_embed(smiles, tag))
    if "chemeleon" in feats.split("+"):
        X.append(chemeleon_embed(smiles, tag))
    return np.concatenate(X, 1)


def fit_predict(endpoint: str, Xtr, ytr, Xte, seed: int) -> np.ndarray:
    """Exact MapLight (their submission.ipynb): CatBoost defaults + random_strength=2 on CPU (user exception Sep 23:
    CPU allowed for gradient-boosted trees only; thread_count bounded so the GPU jobs keep cores)."""
    import catboost as cb
    import maplight as M
    task, log_scale = CONFIG[endpoint]
    params = dict(random_strength=2, random_seed=seed, verbose=0, thread_count=8)
    if task == "regression":
        sc = M.scaler(log=log_scale); sc.fit(ytr)
        model = cb.CatBoostRegressor(loss_function="MAE", **params)
        model.fit(Xtr, sc.transform(ytr).reshape(-1))
        return sc.inverse_transform(model.predict(Xte)).reshape(-1)
    model = cb.CatBoostClassifier(loss_function="Logloss", **params)
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def run(endpoint: str, seeds) -> dict:
    g = D.group(); bench = g.get(endpoint)
    train, test = bench["train_val"], bench["test"]
    Xtr = features(endpoint, train["Drug"].tolist(), "trainval", FEATURES)
    Xte = features(endpoint, test["Drug"].tolist(), "test", FEATURES)
    t0 = time.time()
    preds = [{bench["name"]: fit_predict(endpoint, Xtr, train["Y"].values, Xte, seed)} for seed in seeds]
    metric = D.METRICS[endpoint]
    per = [float(g.evaluate(p)[bench["name"]][metric]) for p in preds]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{endpoint}_preds.npz", **{f"s{s}": p[bench['name']] for s, p in zip(seeds, preds)})
    res = dict(endpoint=endpoint, model=f"MapLight CatBoost CPU exact [{FEATURES}]", metric=metric, higher_is_better=D.higher_is_better(endpoint),
               test_mean=float(np.mean(per)), test_sd=float(np.std(per)), test_per_seed=per,
               official_evaluate_many=g.evaluate_many(preds)[bench["name"]] if len(preds) >= 5 else None,
               seeds=list(seeds), n_features=int(Xtr.shape[1]), seconds=time.time() - t0)
    (OUT / f"{endpoint}.json").write_text(json.dumps(res, indent=1))
    print(f"[maplight] {endpoint} TEST {metric} {res['test_mean']:.4f} +- {res['test_sd']:.4f} ({res['seconds']:.0f}s)", flush=True)
    return res


def main() -> None:
    global OUT, FEATURES
    ap = argparse.ArgumentParser(); ap.add_argument("endpoint"); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--features", default="maplight", choices=list(FEATURE_SETS))
    a = ap.parse_args()
    FEATURES = a.features
    OUT = D.ROOT / "runs" / FEATURE_SETS[a.features]
    for ep in (list(CONFIG) if a.endpoint == "all" else [a.endpoint]):
        if (OUT / f"{ep}.json").exists():
            continue
        run(ep, a.seeds)


if __name__ == "__main__":
    main()
