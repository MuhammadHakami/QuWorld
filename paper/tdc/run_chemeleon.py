"""TDC ADMET with the CheMeleon foundation trunk (Chemprop >= 2.2) + our bottleneck arms, official 5-seed protocol.

    .venv-tdc/bin/python paper/tdc/run_chemeleon.py <endpoint> <variant> [--seeds 1 2 3 4 5] [--epochs 50]

Trunk: CheMeleon D-MPNN (pre-trained on Mordred descriptors of 1M PubChem molecules; Burns et al. 2025, arXiv 2506.15792),
fine-tuned end to end as in `chemprop train --from-foundation CheMeleon` (mean aggregation, fresh FFN head).
Arms: the bottleneck wraps the aggregation (pooled molecular embedding -> zero-init gated core -> FFN): vanilla | mlp30 |
mr30ryd | any Wave-G q-config. Reference = the reproducible leaders per the 2026 JCIM audit (MapLight, MapLight+GNN,
CaliciBoost). Protocol: TDC admet_group official split, get_train_valid_split(seed) per seed, early stopping on valid,
predict the official test set, evaluate_many.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import tdcdata as D  # noqa: E402
import gpucap  # noqa: E402

gpucap.cap_from_env()

CKPT = pathlib.Path.home() / ".chemprop" / "chemeleon_mp.pt"
OUT = D.ROOT / "runs" / "tdc_chemeleon"


def chemeleon_mp():
    from chemprop import nn as cnn
    if not CKPT.exists():
        CKPT.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve("https://zenodo.org/records/15460715/files/chemeleon_mp.pt", CKPT)
    ck = torch.load(CKPT, weights_only=True)
    mp = cnn.BondMessagePassing(**ck["hyper_parameters"]); mp.load_state_dict(ck["state_dict"])
    return mp


class AggWithBottleneck(nn.Module):
    """chemprop calls agg(H, batch) -> (B, d); our bottleneck acts on that pooled molecular embedding."""

    def __init__(self, agg, mid):
        super().__init__(); self.agg, self.mid = agg, mid
        self.hparams = agg.hparams          # chemprop's MPNN records each block's hparams

    def forward(self, H, batch):
        return self.mid(self.agg(H, batch))


def bottleneck(variant: str, d: int):
    from model import Bottleneck, MLPCore, _count, build_config_core, parse_config
    from qlayer import MultiRegister
    if variant == "vanilla":
        return None
    if variant == "mlp30":
        return Bottleneck(d, MLPCore(d, 30, _count(MultiRegister(d, 10, 3, 4, "hea"))), 30)
    if variant == "mr30ryd":
        return Bottleneck(d, MultiRegister(d, 10, 3, 4, "rydberg"), 30)
    core, out = build_config_core(parse_config(variant), d, 4)
    return Bottleneck(d, core, out)


def build_model(endpoint: str, variant: str, seed: int):
    """Datasets (train, valid, test) and the untrained MPNN of one (endpoint, arm, seed) run; shared with stack.py."""
    import lightning.pytorch as pl
    from chemprop import data, featurizers, models, nn as cnn
    g = D.group(); bench = g.get(endpoint)
    tr_df, va_df = g.get_train_valid_split(benchmark=endpoint, split_type="default", seed=seed)
    clf = D.is_classification(endpoint)
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mk = lambda df, with_y=True: data.MoleculeDataset(
        [data.MoleculeDatapoint(D.mol_of(s), [y] if with_y else None) for s, y in zip(df.Drug, df.Y)], feat)
    tr, va, te = mk(tr_df), mk(va_df), mk(bench["test"], with_y=False)
    pl.seed_everything(seed, workers=True)
    mp = chemeleon_mp(); agg = cnn.MeanAggregation()
    mid = bottleneck(variant, mp.output_dim)
    agg_mod = AggWithBottleneck(agg, mid) if mid is not None else agg
    if clf:
        ffn = cnn.BinaryClassificationFFN(input_dim=mp.output_dim)
    else:
        scaler = tr.normalize_targets(); va.normalize_targets(scaler)
        ffn = cnn.RegressionFFN(input_dim=mp.output_dim, output_transform=cnn.UnscaleTransform.from_standard_scaler(scaler))
    return models.MPNN(mp, agg_mod, ffn, batch_norm=False), tr, va, te


def run_seed(endpoint: str, variant: str, seed: int, epochs: int):
    import lightning.pytorch as pl
    from chemprop import data
    model, tr, va, te = build_model(endpoint, variant, seed)
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath=str(OUT / "ckpt" / f"{endpoint}_{variant}_s{seed}"))
    trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1, logger=False, enable_progress_bar=False,
                         callbacks=[ckpt_cb, pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)])
    trainer.fit(model, data.build_dataloader(tr, batch_size=64, num_workers=0),
                data.build_dataloader(va, batch_size=256, num_workers=0, shuffle=False))
    if ckpt_cb.best_model_path:  # best-on-valid weights back into the same module (keeps our aggregation wrapper)
        model.load_state_dict(torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False)["state_dict"])
    preds = torch.cat(trainer.predict(model, data.build_dataloader(te, batch_size=256, num_workers=0, shuffle=False)))
    return preds.squeeze(-1).float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("endpoint"); ap.add_argument("variant")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5]); ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); g = D.group(); metric = D.METRICS[a.endpoint]
    preds = [{a.endpoint: run_seed(a.endpoint, a.variant, s, a.epochs)} for s in a.seeds]
    np.savez(OUT / f"{a.endpoint}_{a.variant}_preds.npz", **{f"s{s}": p[a.endpoint] for s, p in zip(a.seeds, preds)})
    per = [float(g.evaluate(p)[a.endpoint][metric]) for p in preds]      # unrounded per-seed values (judge needs them)
    mean, sd = float(np.mean(per)), float(np.std(per))
    official = g.evaluate_many(preds)[a.endpoint] if len(preds) >= 5 else None  # TDC's own [mean, std], rounded to 3 dp
    res = dict(endpoint=a.endpoint, variant=a.variant, trunk="CheMeleon", metric=metric, higher_is_better=D.higher_is_better(a.endpoint),
               test_mean=mean, test_sd=sd, test_per_seed=per, official_evaluate_many=official,
               seeds=a.seeds, epochs=a.epochs, seconds=time.time() - t0)
    (OUT / f"{a.endpoint}_{a.variant}.json").write_text(json.dumps(res, indent=1))
    print(f"[chemeleon] {a.endpoint} {a.variant} TEST {metric} {mean} +- {sd} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
