"""RouteFinder EAL row (logistics): their pretrained RF-Transformer, their efficient-adapter-layers fine-tuning
to unseen variants, their data and gap-to-reference metric; our arms differ only by a bottleneck on the encoder
output (pool over nodes -> core -> zero-init residual broadcast back to every node), identical at step 0.

    cd baseline/routefinder && ../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py <variant> --setting md --seed 1

Protocol = run_eal.py (RouteFinder, TMLR 2025): eal-full, lr 3e-4, 10 epochs x 10k generated instances, batch
128, n = 100; model selection none (last epoch), then their test files for every unseen variant of the setting.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "paper"))
import rf_compat  # noqa: E402

rf_compat.install()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from lightning.pytorch import seed_everything  # noqa: E402
from rl4co.utils.trainer import RL4COTrainer  # noqa: E402

from model import Bottleneck, MLPCore, _count, build_config_core, parse_config  # noqa: E402
from qlayer import MultiRegister  # noqa: E402
from routed import RoutedNodeRegister  # noqa: E402
from routefinder.envs.mtdvrp import MTVRPEnv, MTVRPGenerator  # noqa: E402
from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingFull  # noqa: E402
from routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbeddingFull  # noqa: E402
from routefinder.models.finetuning.eal import efficient_adapter_layers  # noqa: E402
from routefinder.models.model import RouteFinderBase  # noqa: E402

sys.path.insert(0, str(ROOT / "baseline" / "routefinder"))
rf_compat.guard_dead_ends()
from run_eal import VARIANT_GEN_SETTINGS, finetune_variant_names  # noqa: E402

EMBED = 128


def bottleneck(variant: str) -> nn.Module | None:
    if variant == "vanilla":
        return None
    if variant == "mlp30":
        return Bottleneck(EMBED, MLPCore(EMBED, 30, _count(MultiRegister(EMBED, 10, 3, 4, "hea"))), 30)
    if variant.startswith("route"):                                   # doc 35 §16: route{K}[-mlp]
        return RoutedNodeRegister(EMBED, int(variant[5:].split("-")[0]), "mlp" if "-mlp" in variant else "ryd",  # route30-tw: §39d
                                  fgn=variant.endswith("-fgn"))
    if variant == "mr30ryd":
        return Bottleneck(EMBED, MultiRegister(EMBED, 10, 3, 4, "rydberg"), 30)
    cfg = parse_config(variant)
    if cfg is None:
        raise ValueError(f"unknown variant: {variant}")
    core, out = build_config_core(cfg, EMBED, 4)
    return Bottleneck(EMBED, core, out)


class EncoderWithBottleneck(nn.Module):
    """Wraps the RouteFinder encoder; the bottleneck acts on node embeddings [B, N, H] as a [B, H, N] field."""

    def __init__(self, encoder: nn.Module, mid: nn.Module, tw_gate: bool = False):
        super().__init__()
        self.encoder, self.mid, self.tw_gate = encoder, mid, tw_gate
        self.init_embedding = encoder.init_embedding  # EAL and the policy address it through the encoder

    def forward(self, td, mask=None):
        h, init_h = self.encoder(td, mask)
        if isinstance(self.mid, RoutedNodeRegister):                    # doc 35 §16: per-node, per-instance geometry
            out = self.mid(h, td["locs"])
            if self.tw_gate:                                            # doc 35 §39d: register only on time-window instances
                g = torch.isfinite(td["time_windows"][..., 1]).any(-1).to(h.dtype)[:, None, None]
                out = h + g * (out - h)
            return out, init_h
        return self.mid(h.transpose(1, 2)).transpose(1, 2), init_h


def build(ckpt: str, variant: str) -> RouteFinderBase:
    model = RouteFinderBase.load_from_checkpoint(ckpt, map_location="cpu")
    model = efficient_adapter_layers(model, init_embedding_cls=MTVRPInitEmbeddingFull,
                                     context_embedding_cls=MTVRPContextEmbeddingFull,
                                     init_embedding_num_new_feats=1, context_embedding_num_new_feats=3,
                                     adapter_only=False)
    with torch.random.fork_rng(devices=[]):          # amendment Sep 25 19:55: the bottleneck's init does not move the
        mid = bottleneck(variant)                     # CPU RNG, so every arm trains on vanilla's generated instances
    if mid is not None:
        model.policy.encoder = EncoderWithBottleneck(model.policy.encoder, mid, tw_gate=variant.endswith("-tw"))
    return model


def env_for(setting: str) -> tuple[MTVRPEnv, list[str]]:
    names = finetune_variant_names(setting)
    test = [f"{v}/test/100.npz" for v in names]
    gen = MTVRPGenerator(num_loc=100, variant_preset="all", **VARIANT_GEN_SETTINGS[setting])
    env = MTVRPEnv(gen, check_solution=False, data_dir="./data/", val_file=[t.replace("test", "val") for t in test],
                   test_file=test, val_dataloader_names=[v + "100" for v in names],
                   test_dataloader_names=[v + "100" for v in names])
    return env, names


def metrics(results: list[dict], names: list[str]) -> dict:
    """Per-variant gap (%) from Lightning's per-dataloader result dicts."""
    out = {}
    for name, r in zip(names, results):
        # Lightning keys look like "test/gap_to_bks/mdcvrp100[/dataloader_idx_k]"; one dict per dataloader
        gap = [v for k, v in r.items() if "gap_to_bks" in k and name in k] or [v for k, v in r.items() if "gap_to_bks" in k]
        out[name] = float(gap[0]) if gap else None
    if any(v is None for v in out.values()):
        raise RuntimeError(f"gap_to_bks missing for some dataloaders; keys seen: {sorted({k for r in results for k in r})[:6]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant"); ap.add_argument("--setting", default="md", choices=list(VARIANT_GEN_SETTINGS))
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--train-size", type=int, default=10_000); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--ckpt", default="checkpoints/100/rf-transformer.ckpt")
    ap.add_argument("--out", default=str(ROOT / "runs" / "routefinder"))
    a = ap.parse_args()
    seed_everything(a.seed, workers=True)
    model = build(a.ckpt, a.variant)
    env, names = env_for(a.setting)
    model.optimizer_kwargs["lr"] = a.lr
    model.env = env; model.setup()
    model.data_cfg.update(batch_size=128, val_batch_size=256, test_batch_size=256, train_data_size=a.train_size)
    # their protocol tests the LAST epoch (no checkpoint selection), so per-epoch validation cannot change the
    # result; validate once at the end only (saves ~1.5 h per run on 16 variant files)
    # precision: RouteFinder's default 16-mixed diverged (NaN action probabilities) on the vanilla arm on the GB10
    # (Sep 23); bf16-mixed keeps fp32's exponent range. Same for every arm; recorded as a stability deviation.
    trainer = RL4COTrainer(accelerator="gpu", devices=1, max_epochs=a.epochs, logger=False, precision=a.precision,
                           enable_checkpointing=False, enable_progress_bar=False,
                           check_val_every_n_epoch=a.epochs)
    t0 = time.time()
    zero_shot = metrics(trainer.validate(model), names)
    trainer.fit(model)
    test = metrics(trainer.test(model), names)
    gaps = [g for g in test.values() if g is not None]
    res = dict(setting=a.setting, variant=a.variant, seed=a.seed, epochs=a.epochs, train_size=a.train_size, lr=a.lr, precision=a.precision,
               num_params=_count(model.policy), zero_shot_val_gap=zero_shot, test_gap=test,
               test_gap_mean=sum(gaps) / len(gaps) if gaps else None, seconds=time.time() - t0,
               crn=True, dead_end_rows=rf_compat.dead_end_count())
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"eal_{a.setting}_{a.variant}_s{a.seed}.json").write_text(json.dumps(res, indent=1))
    torch.save({"policy": model.policy.state_dict(), "variant": a.variant, "setting": a.setting},
               out / f"eal_{a.setting}_{a.variant}_s{a.seed}.pt")        # doc 35 §23: kept for the route figure
    print(f"[rf-eal] {a.setting} {a.variant} s{a.seed}: mean test gap {res['test_gap_mean']} ({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
