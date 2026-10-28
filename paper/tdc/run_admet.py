"""One TDC ADMET endpoint x one arm x the official 5-seed protocol.

    uv run --python .venv-tdc/bin/python paper/tdc/run_admet.py <endpoint> <variant> [--seeds 1 2 3 4 5]

TDC's leaderboard protocol verbatim: for each seed, `get_train_valid_split(seed)` splits the official
train_val set, the model trains on train with early stopping on valid, then predicts the official test set;
`evaluate_many` over the five prediction sets returns the leaderboard metric's mean and standard deviation.
The test set is only ever read by TDC's own evaluator.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import tdcdata as D  # noqa: E402
from head import build  # noqa: E402
from model import _count  # noqa: E402

DEV = torch.device("cuda")
OUT = D.ROOT / "runs" / "tdc"


def _xy(df, emb: dict) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([emb[s] for s in df.Drug]).to(DEV)
    return x, torch.tensor(df.Y.values, dtype=torch.float32, device=DEV)


def _score(metric, y_true, y_pred) -> float:
    from tdc import Evaluator
    return float(Evaluator(name=metric)(np.asarray(y_true), np.asarray(y_pred)))


def train_one(endpoint: str, variant: str, seed: int, emb: dict, hp: dict) -> np.ndarray:
    """Train on the official seed split, return predictions for the official test set."""
    g = D.group()
    bench = g.get(endpoint)
    tr_df, va_df = g.get_train_valid_split(benchmark=endpoint, split_type="default", seed=seed)
    clf, metric = D.is_classification(endpoint), D.METRICS[endpoint]
    xtr, ytr = _xy(tr_df, emb); xva, yva = _xy(va_df, emb); xte, _ = _xy(bench["test"], emb)
    mu, sd = (0.0, 1.0) if clf else (ytr.mean().item(), ytr.std().item() + 1e-8)

    torch.manual_seed(seed); np.random.seed(seed)
    model = build(variant).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    loss_fn = nn.BCEWithLogitsLoss() if clf else nn.MSELoss()
    best, best_state, bad = -np.inf, None, 0
    n = xtr.shape[0]
    for epoch in range(hp["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, hp["batch"]):
            idx = perm[i:i + hp["batch"]]
            out = model(xtr[idx])
            loss = loss_fn(out, ytr[idx] if clf else (ytr[idx] - mu) / sd)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = model(xva)
            p = torch.sigmoid(p) if clf else p * sd + mu
        val = _score(metric, yva.cpu().numpy(), p.cpu().numpy())
        val = val if D.higher_is_better(endpoint) else -val
        if val > best:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p = model(xte)
        p = torch.sigmoid(p) if clf else p * sd + mu
    return p.cpu().numpy(), best, _count(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("endpoint"); ap.add_argument("variant")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--patience", type=int, default=40)
    a = ap.parse_args()
    hp = dict(lr=a.lr, wd=a.wd, epochs=a.epochs, batch=a.batch, patience=a.patience)
    OUT.mkdir(parents=True, exist_ok=True)
    emb = D.embeddings(a.endpoint)
    g = D.group()
    t0 = time.time(); preds, vals, params = [], [], 0
    for seed in a.seeds:
        p, val, params = train_one(a.endpoint, a.variant, seed, emb, hp)
        preds.append({a.endpoint: p}); vals.append(val)
        print(f"[tdc] {a.endpoint} {a.variant} seed {seed}: best valid {abs(val):.4f}", flush=True)
    metric = D.METRICS[a.endpoint]
    if len(preds) >= 5:                       # TDC's leaderboard aggregation (it refuses fewer than 5 runs)
        mean, sd = g.evaluate_many(preds)[a.endpoint][metric]
    else:                                     # smoke runs: same evaluator, aggregated here
        per = [g.evaluate(p)[a.endpoint][metric] for p in preds]
        mean, sd = float(np.mean(per)), float(np.std(per))
    summary = dict(endpoint=a.endpoint, variant=a.variant, metric=metric,
                   higher_is_better=D.higher_is_better(a.endpoint), test_mean=mean, test_sd=sd,
                   valid_per_seed=[abs(v) for v in vals], seeds=a.seeds, num_params=params,
                   hp=hp, seconds=time.time() - t0)
    (OUT / f"{a.endpoint}_{a.variant}.json").write_text(json.dumps(summary, indent=1))
    print(f"[tdc] {a.endpoint} {a.variant} TEST {metric} {mean} +- {sd} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
