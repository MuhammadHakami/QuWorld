"""Parameterized vs non-parameterized on-the-go tile assignment (TRL2D, GPU).

    .venv/bin/python paper/spectral/learned_router.py --steps 1500   ->  runs/spectral/trl2d_learned_router.json

A small per-tile scorer is trained on the TRAIN split to predict each wavelet tile's log error energy from
inference-time quantities only: Haar detail energies of the trunk's predicted change (p - x_t), the recent change
(x_t - x_{t-1}) and the input x_t, at the tile and at its parent tile one scale up (the wavelet-tree dependency).
Evaluated ONCE on the same TEST windows as router_capture.py: share of detail error energy captured by its top-K
tiles, against the oracle and the best zero-parameter router.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "phase_b")); sys.path.insert(0, str(ROOT / "paper" / "spectral"))
import cnext_hybrid  # noqa: E402
import well2d  # noqa: E402
from band_analysis import DEV, J, haar2d  # noqa: E402
from router_capture import KS, tiles  # noqa: E402

EPS = 1e-10


def level_feats(fields: list[torch.Tensor]):
    """Per level j: (B, 3*len(fields)*C, Hj, Wj) log detail energies, plus the level-J approximation energies."""
    per = [haar2d(f, J) for f in fields]
    feats = []
    for j in range(J):
        feats.append(torch.cat([torch.log(det[j][o].square() + EPS) for _, det in per for o in range(3)], dim=1))
    approx = torch.cat([torch.log(a.square() + EPS).repeat(1, 3, 1, 1) for a, _ in per], dim=1)
    return feats, approx


class TileScorer(nn.Module):
    """Shared across levels: tile features + parent features + level embedding -> 3*C scores per position."""

    def __init__(self, fin: int, out: int, hid: int = 64):
        super().__init__()
        self.level = nn.Embedding(J, 8)
        self.net = nn.Sequential(nn.Conv2d(2 * fin + 8, hid, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(hid, hid, 1), nn.GELU(), nn.Conv2d(hid, out, 1))

    def forward(self, feats, approx):
        B = feats[0].shape[0]; outs = []
        for j, f in enumerate(feats):
            parent = feats[j + 1] if j + 1 < J else approx
            parent = parent.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)[..., :f.shape[2], :f.shape[3]]
            lv = self.level.weight[j].view(1, -1, 1, 1).expand(B, -1, f.shape[2], f.shape[3])
            outs.append(self.net(torch.cat([f, parent, lv], dim=1)))
        return outs


def flatten_scores(outs, C: int) -> torch.Tensor:
    """Match tiles() order: level, orientation, channel, y, x."""
    B = outs[0].shape[0]
    return torch.cat([o.view(B, 3, C, o.shape[2], o.shape[3])[:, k].reshape(B, -1) for o in outs for k in range(3)], dim=1)


def batch_inputs(model, x, y, C):
    with torch.no_grad():
        p = model(x)
        xt = x[:, (well2d.N_HIST - 1) * C:]; xp = x[:, (well2d.N_HIST - 2) * C:(well2d.N_HIST - 1) * C]
        feats, approx = level_feats([p - xt, xt - xp, xt])
        target = torch.log(tiles(p - y).float() + EPS)
    return feats, approx, target, p


def prior(x, p, C):
    return torch.log(tiles(p - x[:, (well2d.N_HIST - 1) * C:]).float() + EPS)


def scores(kind, scorer, feats, approx, x, p, C):
    s = flatten_scores(scorer(feats, approx), C)
    return s + prior(x, p, C) if kind == "listnet_residual" else s


def objective(kind, scorer, feats, approx, target, x, p, C):
    s = scores(kind, scorer, feats, approx, x, p, C)
    if kind == "mse":
        return F.mse_loss(s, target)
    w = torch.softmax(target.double(), dim=1).float()            # = err / sum(err): energy-proportional target
    return -(w * torch.log_softmax(s, dim=1)).sum(1).mean()


def zero_init_last(scorer):
    nn.init.zeros_(scorer.net[-1].weight); nn.init.zeros_(scorer.net[-1].bias)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", default="mse", choices=["mse", "listnet_residual"],
                    help="mse: regress log tile energy; listnet_residual: zero-init residual on the zero-parameter "
                         "prior log E(trunk change), trained with a listwise loss whose target is proportional to "
                         "tile error energy (maximises expected captured energy)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    trunk = cnext_hybrid.build_hybrid("vanilla").to(DEV).eval()
    train = well2d.Split("turbulent_radiative_layer_2D", "train"); C = train.C
    scorer = TileScorer(3 * 3 * C, 3 * C).to(DEV)
    if a.objective == "listnet_residual":
        zero_init_last(scorer)                                   # starts exactly at the zero-parameter router
    opt = torch.optim.AdamW(scorer.parameters(), lr=2e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(a.seed)
    for step in range(a.steps):
        x, y = well2d.train_batch(train, a.bsz, gen, "cuda")
        feats, approx, target, p = batch_inputs(trunk, x, y, C)
        loss = objective(a.objective, scorer, feats, approx, target, x, p, C)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0:
            print(f"[router] step {step} loss {loss.item():.4f}", flush=True)
    train.close()
    # model selection happens on VALID; TEST is evaluated once per objective and reported regardless
    scorer.eval()
    res = {split: evaluate(trunk, scorer, split, a.objective, C) for split in ("valid", "test")}
    out = dict(objective=a.objective, K=KS, steps=a.steps, params=sum(q.numel() for q in scorer.parameters()), **res)
    (ROOT / "runs" / "spectral" / f"trl2d_learned_router_{a.objective}.json").write_text(json.dumps(out, indent=1))
    for split in ("valid", "test"):
        print(f"[{a.objective}] {split.upper()} ({res[split]['windows']} windows) capture of detail error energy:")
        print("K            " + " ".join(f"{k:>7d}" for k in KS))
        for r, v in res[split]["capture"].items():
            print(f"{r:12s} " + " ".join(f"{x:7.3f}" for x in v))


def evaluate(trunk, scorer, split, kind, C):
    data = well2d.Split("turbulent_radiative_layer_2D", split)
    cap = {r: torch.zeros(len(KS), device=DEV, dtype=torch.float64) for r in ("oracle", "learned", "trunk_change")}
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in well2d.batched(data, well2d.fixed_windows(data, 20), 16):
            x, y = x.to(DEV), y.to(DEV)
            feats, approx, _, p = batch_inputs(trunk, x, y, C)
            err = tiles(p - y)
            sc = {"oracle": err, "learned": scores(kind, scorer, feats, approx, x, p, C), "trunk_change": prior(x, p, C)}
            tot += err.sum().item(); n += x.shape[0]
            for r, s in sc.items():
                csum = err.gather(1, s.topk(max(KS), dim=1).indices).cumsum(1)
                cap[r] += torch.stack([csum[:, k - 1].sum() for k in KS])
    data.close()
    return dict(windows=n, capture={r: (v / tot).tolist() for r, v in cap.items()})


if __name__ == "__main__":
    main()
