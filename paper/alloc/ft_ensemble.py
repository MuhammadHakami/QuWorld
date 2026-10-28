"""Registered equal-weight ensemble of the two-stage gated fine-tuned trunks (doc 37, "E-ft").

    .venv/bin/python paper/alloc/ft_ensemble.py <dataset> [--full-test]

Without --full-test: scores every member and the equal-weight mean of their gated predictions on the stratified VALID
sample used for selection during fine-tuning, picks the VALID-best of {E-ft, best single member}, and records it.
With --full-test: scores the recorded selection once on the full TEST split.
"""
from __future__ import annotations

import argparse
import json

import torch

import finetune_gated as FG

VARIANTS = ("mr30ryd", "vanilla", "mlp30")


def members(dataset):
    """Per trunk, the two-stage cell: tagged `_pre` where a joint-only cell shares the name (RB), else the plain tag."""
    out = []
    for v in VARIANTS:
        for t in (f"{dataset}_{v}_vrmse_gate_s1_pre", f"{dataset}_{v}_vrmse_gate_s1"):
            f = FG.OUT / f"{t}.json"
            if f.exists() and (FG.OUT / f"{t}.pt").exists() and json.loads(f.read_text()).get("gate_pretrain_n"):
                out.append(t); break
    return out


def load(dataset, tag):
    variant = json.loads((FG.OUT / f"{tag}.json").read_text())["variant"]
    (train, valid, test), trunk = FG.load_data_and_trunk(dataset, variant)
    grid = tuple(train.S) if hasattr(train, "S") else (train.H, train.W)
    model = FG.GatedHead(trunk, train.C, grid, torch.ones(train.C, FG.NB), True).to(FG.DEV)
    model.load_state_dict(torch.load(FG.OUT / f"{tag}.pt", weights_only=True, map_location=FG.DEV)["model"])
    return model.eval(), (valid, test)


@torch.no_grad()
def score(models, split, pairs, bsz=4):
    """Per-window VRMSE of each member and of the equal-weight mean of their predictions."""
    per, ens = [[] for _ in models], []
    for x, y in FG.well2d.batched(split, pairs, bsz):
        x, y = x.to(FG.DEV), y.to(FG.DEV)
        preds = [m(x) for m in models]
        for i, p in enumerate(preds):
            per[i].append(FG.vrmse(p, y).cpu())
        ens.append(FG.vrmse(torch.stack(preds).mean(0), y).cpu())
    return [torch.cat(v) for v in per], torch.cat(ens)


@torch.no_grad()
def diversity(models, tags, split, pairs, bsz=4):
    """Descriptive only (not in any registered rule): mean over VALID windows of the pairwise Pearson correlation of
    the members' residual fields (prediction - target), all fields and grid points of a window together."""
    acc, n = {}, 0
    for x, y in FG.well2d.batched(split, pairs, bsz):
        x, y = x.to(FG.DEV), y.to(FG.DEV)
        r = [(m(x) - y).flatten(1) for m in models]
        r = [(v - v.mean(1, keepdim=True)) / (v.std(1, keepdim=True) + 1e-12) for v in r]
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                acc[(i, j)] = acc.get((i, j), 0.0) + float((r[i] * r[j]).mean(1).sum())
        n += y.shape[0]
    return {f"{tags[i].split('_vrmse')[0]} ~ {tags[j].split('_vrmse')[0]}": v / n for (i, j), v in acc.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--full-test", action="store_true")
    ap.add_argument("--diversity", action="store_true", help="descriptive residual correlations on stratified VALID")
    a = ap.parse_args()
    tags = members(a.dataset)
    if len(tags) < 2:
        raise SystemExit(f"[ens] {a.dataset}: need >= 2 two-stage members, found {tags}")
    loaded = [load(a.dataset, t) for t in tags]
    models, (valid, test) = [m for m, _ in loaded], loaded[0][1]
    out = FG.OUT / f"{a.dataset}_Eft.json"
    if a.diversity:
        div = diversity(models, tags, valid, FG.strat_pairs(valid, 2, 2))
        (FG.OUT.parent / f"eft_diversity_{a.dataset}.json").write_text(json.dumps(div, indent=1))
        print(f"[ens] {a.dataset}: residual correlation (VALID) {div}", flush=True)
        return
    if not a.full_test:
        per, ens = score(models, valid, FG.strat_pairs(valid, 2, 2))
        cand = {t: float(v.mean()) for t, v in zip(tags, per)} | {"E-ft": float(ens.mean())}
        best_single = min(tags, key=lambda t: cand[t])
        pick = "E-ft" if cand["E-ft"] < cand[best_single] else best_single
        out.write_text(json.dumps({"dataset": a.dataset, "members": tags, "valid": cand, "selected": pick}, indent=1))
        print(f"[ens] {a.dataset}: VALID {cand} -> selected {pick}", flush=True)
        return
    res = json.loads(out.read_text()); pick = res["selected"]
    idx = [i for i, t in enumerate(tags) if pick in ("E-ft", t)]
    per, ens = score([models[i] for i in idx], test, FG.well2d.all_test_windows(test))
    v = ens if pick == "E-ft" else per[0]
    with open(FG.OUT / f"{a.dataset}_Eft_test.csv", "w") as fh:            # per-window, the_well order (paired tests)
        fh.write("window,vrmse\n" + "".join(f"{i},{float(x)}\n" for i, x in enumerate(v)))
    if "full_test" in res and abs(res["full_test"] - float(v.mean())) > 1e-6:
        raise SystemExit(f"[ens] re-score mismatch {res['full_test']} vs {float(v.mean())}")
    res.update(full_test=float(v.mean()), full_test_median=float(v.median()), full_test_n=len(v))
    out.write_text(json.dumps(res, indent=1))
    print(f"[ens] {a.dataset}: {pick} FULL TEST {v.mean():.5f} (n={len(v)})", flush=True)


if __name__ == "__main__":
    main()
