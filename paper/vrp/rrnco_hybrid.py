"""RRNCO ACVRP row (doc 35 §41): the released RRNet (rcvrp, epoch 199) as the frozen classical part + a routed 30-atom
Rydberg register on its node latent (RoutedNodeRegister, the RouteFinder route30 design), trained with RRNCO's own
REINFORCE (multi-start shared baseline) on its lazy real-city generator. The register's write-back is zero-initialised,
so step 0 is exactly the released model; the same per-node correction is added to the row and column embeddings.
Test = test.py's protocol (dihedral x8 augmentation, 101 starts, best of all; fp16 autocast as their script), on the
in-distribution / cluster / out-of-distribution files, for both arms in this process (paired).

    cd baseline/rrnco && ../../.venv-vrp/bin/python ../../paper/vrp/rrnco_hybrid.py --seed 1 --epochs 10 --train-size 10000
        -> runs/routefinder/rrnco_hybrid_s<seed>.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RR = ROOT / "baseline" / "rrnco"
sys.path.insert(0, str(RR)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "paper"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from lightning.pytorch import seed_everything  # noqa: E402
from rl4co.data.utils import load_npz_to_tensordict  # noqa: E402
from rl4co.utils.ops import batchify, unbatchify  # noqa: E402
from rl4co.utils.trainer import RL4COTrainer  # noqa: E402

import test as rrtest  # noqa: E402  (RRNCO's test.py: augmentation, dataloader, torchrl patch)
from rrnco.envs.rcvrp import RCVRPEnv  # noqa: E402
from rrnco.models import RRNet  # noqa: E402
from routed import RoutedNodeRegister  # noqa: E402

FILES = {"in": "rcvrp_n100_seed3333_in_distribution.npz", "cluster": "rcvrp_n100_seed3333_in_distribution_cluster.npz",
         "ood": "rcvrp_n100_seed3333_out_of_distribution.npz"}
PUBLISHED = {"in": (72.145, 3.47), "cluster": (23.282, 3.57), "ood": (73.010, 3.22)}   # RRNCO paper: cost, gap %


class EncoderWithRegister(nn.Module):
    def __init__(self, encoder: nn.Module, mid: RoutedNodeRegister, safe: bool = False, H: int = 128):
        super().__init__()
        self.encoder, self.mid, self.safe, self.last_pen = encoder, mid, safe, None
        self.init_embedding = encoder.init_embedding
        if safe:                                   # §41c: per-instance gate starting nearly closed (sigmoid(-3) ~ 0.05)
            self.gate = nn.Linear(H, 1); nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -3.0)

    def forward(self, td, phase, mask=None):
        row, col = self.encoder(td, phase=phase, mask=mask)
        locs = td["locs"].float()
        if locs.shape[1] != row.shape[1]:
            raise RuntimeError(f"locs {tuple(locs.shape)} vs node latent {tuple(row.shape)}")
        new = self.mid(row, locs)
        if self.safe:
            new = row + torch.sigmoid(self.gate(row.mean(1)))[:, None] * (new - row)
            self.last_pen = ((new - row).float() ** 2).mean()
        return new, col + (new - row)


def env(train: bool) -> RCVRPEnv:
    gp = dict(num_loc=100, data_path=str(RR / "data" / "dataset"), file_name="splited_cities_list") if train else dict(num_loc=100)
    if train:
        from rrnco.envs.rcvrp.generator_lazy import LazyRCVRPGenerator
        return RCVRPEnv(generator=LazyRCVRPGenerator(**gp), check_solution=False)
    return RCVRPEnv(check_solution=False, generator_params=gp)


@torch.no_grad()
def evaluate(policy: nn.Module, name: str, batch_size: int = 32, limit: int | None = None) -> float:
    td = load_npz_to_tensordict(str(RR / "data" / "rcvrp" / FILES[name]))
    if limit:
        td = td[:limit]
    td["demand"] = td["demand"] / td["capacity"].unsqueeze(-1); td["capacity"] = torch.ones_like(td["capacity"])
    e = env(False); costs = []; dev = next(policy.parameters()).device
    with torch.autocast("cuda"), torch.inference_mode():
        for b in rrtest.get_dataloader(td, batch_size):
            r = e.reset(rrtest.augment(b)).to(dev)
            out = policy(r, e, return_actions=True, phase="val", calc_reward=False, num_starts=101)
            tb = batchify(r, 101)
            rew = e.get_reward(tb, out["actions"]); rew = rew[0] if isinstance(rew, tuple) else rew
            costs.append(unbatchify(rew, (8, 101)).max(-1)[0].max(-1)[0].mean().item())
    return -sum(costs) / len(costs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--train-size", type=int, default=10_000); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--precision", default="bf16-mixed"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--safe", action="store_true"); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--ckpt", default=str(RR / "checkpoints" / "rcvrp" / "epoch_199.ckpt"))
    a = ap.parse_args(); t0 = time.time()
    seed_everything(a.seed, workers=True)
    model = RRNet.load_from_checkpoint(a.ckpt, map_location="cpu", strict=False, load_baseline=False, weights_only=False)
    dev = torch.device("cuda"); lim = a.limit or None
    base = {k: evaluate(model.policy.to(dev).eval(), k, limit=lim) for k in FILES}
    print(f"[rrnco-h] classical (released, frozen): {base} ({time.time() - t0:.0f}s)", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        mid = RoutedNodeRegister(128, 30, "ryd")
    enc = EncoderWithRegister(model.policy.encoder, mid, safe=a.safe); model.policy.encoder = enc
    if a.safe:                                     # §41c: penalise the size of the write-back in the REINFORCE loss
        orig = model.calculate_loss

        def calc(td, batch, out, reward=None, log_likelihood=None):
            res = orig(td, batch, out, reward, log_likelihood)
            if enc.last_pen is not None:
                out["loss"] = out["loss"] + a.lam * enc.last_pen
            return res
        model.calculate_loss = calc
    model.env = env(True); model.optimizer_kwargs = dict(lr=a.lr, weight_decay=0.0)
    model.lr_scheduler = None; model.setup()
    model.data_cfg.update(batch_size=64, train_data_size=a.train_size)
    trainer = RL4COTrainer(accelerator="gpu", devices=1, max_epochs=a.epochs, logger=False, precision=a.precision,
                           enable_checkpointing=False, enable_progress_bar=False, limit_val_batches=0)
    trainer.fit(model)
    hyb = {k: evaluate(model.policy.to(dev).eval(), k, limit=lim) for k in FILES}
    ref = {k: c / (1 + g / 100) for k, (c, g) in PUBLISHED.items()}
    gap = lambda d: {k: round(100 * (d[k] / ref[k] - 1), 3) for k in d}
    res = dict(seed=a.seed, epochs=a.epochs, train_size=a.train_size, lr=a.lr, precision=a.precision, limit=a.limit,
               classical=base, hybrid=hyb, classical_gap=gap(base), hybrid_gap=gap(hyb),
               alpha_like=float(mid.up.weight.abs().mean()), seconds=round(time.time() - t0))
    out = ROOT / "runs" / "routefinder" / f"rrnco_hybrid_s{a.seed}_e{a.epochs}_lr{a.lr:g}{'_safe' + format(a.lam, 'g') if a.safe else ''}{'_lim' if a.limit else ''}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"RRNCO_HYBRID_DONE seed {a.seed}: classical gap {res['classical_gap']} | hybrid gap {res['hybrid_gap']} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
