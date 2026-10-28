"""End-to-end fine-tuning with a persistence-anchored, band-gated head and a metric-aligned loss (doc 37).

The evaluation audit (doc 36) showed that the Well trunks lose the mean VRMSE to persistence in a heavy tail of
low-variance windows. They were trained with plain MSE on normalized fields, which gives those windows almost no
weight, while the benchmark's VRMSE divides by each window's variance. Here the whole stack is trained end to end:
  y_hat = x_t + sum_b alpha_b(z) P_b(trunk(x) - x_t)
  - trunk: the fine-tuned CNextU-net (vanilla | mlp30 | mr30ryd = 30-atom register in the neck), all weights trainable;
  - gate (classical): alpha per field and band from CAUSAL features z (log band energies of p - x_t and dx normalized
    by var(x_t), plus global stats of x_t), initialised at the TRAIN-fitted blend of the matching ADR-C cache;
  - loss: `vrmse` = the benchmark metric itself (per window, per field sqrt(mse / var(y))), or `mse` (as before).
Selection on a stratified VALID sample (every trajectory x 8 time bins); the stratified TEST estimate is printed for
iteration only; the full TEST split is scored once for the VALID-selected configuration (eval mode --full-test).

    .venv/bin/python paper/alloc/finetune_gated.py <dataset> <variant> --loss vrmse --head gate --budget 2400 --seed 1
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "phase_b"))
import gpucap  # noqa: E402
import well2d  # noqa: E402
from well2d_backbone import load_checkpoint  # noqa: E402

gpucap.cap_from_env()
DEV = torch.device("cuda")
OUT = ROOT / "runs" / "alloc" / "ft"
NB = 5


def band_masks(shape, nb=NB):
    ks = [torch.fft.fftfreq(n, device=DEV).abs() * 2 for n in shape]
    kn = torch.stack(torch.meshgrid(*ks, indexing="ij")).amax(0)
    edges = [0.0] + [2.0 ** -(nb - 1 - b) for b in range(nb - 1)] + [float("inf")]
    return torch.stack([((kn >= lo) & (kn < hi)).float() for lo, hi in zip(edges[:-1], edges[1:])])   # (nb, *grid)


def fftn(t):
    return torch.fft.fftn(t.float(), dim=tuple(range(2, t.dim())))


def bandsum(K, masks):                                  # (B, C, *grid) x (nb, *grid) -> (B, C, nb)
    return (K.unsqueeze(2) * masks).flatten(3).sum(-1)


class GatedHead(nn.Module):
    def __init__(self, trunk, C, shape, alpha0, gated=True):
        super().__init__()
        self.trunk, self.C, self.gated = trunk, C, gated
        self.register_buffer("masks", band_masks(shape))
        self.register_buffer("alpha0", alpha0.float())
        fin = 2 * C * NB + 2 * C
        self.norm = nn.BatchNorm1d(fin, affine=False, momentum=None)                  # cumulative running stats
        self.net = nn.Sequential(nn.Linear(fin, 64), nn.GELU(), nn.Linear(64, C * NB))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def features(self, x, p):
        """Causal gate input z and the trunk increment's spectrum (both from inputs and the trunk output only)."""
        C = self.C; xt, xp = x[:, 3 * C:], x[:, 2 * C:3 * C]
        Dk, Xk = fftn(p - xt), fftn(xt - xp)
        vx = xt.flatten(2).var(-1, unbiased=False) + 1e-7                           # causal normalizer
        be = lambda K: bandsum(K.abs().square(), self.masks) / vx[..., None]
        z = torch.cat([be(Dk).clamp_min(1e-12).log().flatten(1), be(Xk).clamp_min(1e-12).log().flatten(1),
                       xt.flatten(2).mean(-1), vx.log()], 1)
        return z, Dk

    def alpha(self, z):
        return self.alpha0 + 1.5 * torch.tanh(self.net(self.norm(z).clamp(-10, 10))).reshape(-1, self.C, NB)

    def forward(self, x):
        C = self.C; xt = x[:, 3 * C:]
        p = self.trunk(x)
        if not self.gated:
            return p
        z, Dk = self.features(x, p); alpha = self.alpha(z)
        H = (alpha[..., None] * self.masks.flatten(1)).sum(2).reshape(*alpha.shape[:2], *self.masks.shape[1:])  # sum_b
        return xt + torch.fft.ifftn(H * Dk, dim=tuple(range(2, Dk.dim()))).real


@torch.no_grad()
def gate_stats(model, split, pairs, bsz=8):
    """Exact per-window band statistics for gate pre-training on the frozen trunk: z, gY, gYD, gDD (per field and band,
    spatial-energy units) and the VRMSE denominators Dv = npix (var(y) + 1e-7)."""
    model.eval(); out = {k: [] for k in ("z", "gY", "gYD", "gDD", "Dv")}; C = model.C
    for x, y in well2d.batched(split, pairs, bsz):
        x, y = x.to(DEV), y.to(DEV); xt = x[:, 3 * C:]; npx = y[0, 0].numel()
        z, Dk = model.features(x, model.trunk(x)); Yk = fftn(y - xt)
        out["z"].append(z); out["gY"].append(bandsum(Yk.abs().square(), model.masks) / npx)
        out["gYD"].append(bandsum((Yk * Dk.conj()).real, model.masks) / npx)
        out["gDD"].append(bandsum(Dk.abs().square(), model.masks) / npx)
        out["Dv"].append(npx * (y.flatten(2).var(-1, unbiased=False) + 1e-7))
    model.train()
    return {k: torch.cat(v) for k, v in out.items()}


def gate_vrmse(model, st, rows):
    """Exact VRMSE of the gated prediction for cached windows (Parseval; doc 36 §5, Lean gate_energy_exact)."""
    al = model.alpha(st["z"][rows])
    E = (st["gY"][rows] - 2 * al * st["gYD"][rows] + al.square() * st["gDD"][rows]).sum(-1).clamp_min(0)
    return (E / st["Dv"][rows]).clamp_min(1e-12).sqrt().mean(1)


def pretrain_gate(model, tr, va, epochs=60, lr=3e-3, bsz=64):
    opt = torch.optim.Adam(list(model.net.parameters()), lr=lr); N = len(tr["z"])
    best, state = float("inf"), None
    for ep in range(epochs):
        model.norm.train(); perm = torch.randperm(N, device=DEV)
        for i in range(0, N - bsz + 1, bsz):
            loss = gate_vrmse(model, tr, perm[i:i + bsz]).mean(); opt.zero_grad(); loss.backward(); opt.step()
        model.norm.eval()
        with torch.no_grad():
            v = gate_vrmse(model, va, torch.arange(len(va["z"]), device=DEV)).mean().item()
        if v < best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items() if not k.startswith("trunk.")}
    model.load_state_dict(state, strict=False)
    return best


@torch.no_grad()
def fit_alpha(trunk, train, seed, n=400, bsz=8):
    """TRAIN-fitted band blend (closed form, metric-weighted least squares; doc 36 §3) on n random TRAIN windows."""
    trunk.eval(); gen = torch.Generator().manual_seed(10_000 + seed); masks = None; num = den = 0
    for _ in range(n // bsz):
        x, y = well2d.train_batch(train, bsz, gen, DEV); C = train.C; xt = x[:, 3 * C:]
        masks = band_masks(tuple(y.shape[2:])) if masks is None else masks
        Dk, Yk = fftn(trunk(x) - xt), fftn(y - xt)
        w = 1 / (y.flatten(2).var(-1, unbiased=False) + 1e-7)
        num = num + (w[..., None] * bandsum((Yk * Dk.conj()).real, masks)).sum(0).double()
        den = den + (w[..., None] * bandsum(Dk.abs().square(), masks)).sum(0).double()
    trunk.train()
    alpha = (num / den.clamp_min(1e-30)).float().cpu()
    print(f"[ft] TRAIN-fitted blend alpha (field x band):\n{alpha.numpy().round(3)}", flush=True)
    return alpha


def vrmse(pred, y):                                     # the Well's VRMSE for any spatial rank (== vrmse_rows in 2-D)
    dims = tuple(range(2, y.dim()))
    mse = (pred - y).square().mean(dims); var = (y - y.mean(dims, keepdim=True)).square().mean(dims)
    return (mse / (var + 1e-7)).clamp_min(1e-12).sqrt().mean(1)   # clamp: finite gradient at mse -> 0 (review)


def load_data_and_trunk(dataset, variant):
    """Splits + the s1337 fine-tuned trunk of each row (Well 2-D rows, TRL2D campaign ckpts, MHD_64 3-D)."""
    if dataset == "MHD_64":
        sys.path.insert(0, str(HERE)); import cache as CA
        from cnext_hybrid import build_hybrid
        sp = [CA.WellND("MHD_64", s) for s in ("train", "valid", "test")]
        trunk = build_hybrid(variant, model_dir=str(ROOT / "data/cnext_mhd_ckpt"))
        st = torch.load(ROOT / "runs/phase_b_mhd" / f"{variant}_s1337.pt", weights_only=True, map_location="cpu")
        trunk.load_state_dict(st["model_state_dict"] if "model_state_dict" in st else st)
        return sp, trunk.to(DEV)
    sp = [well2d.Split(dataset, s) for s in ("train", "valid", "test")]
    ck = (ROOT / "runs/phase_b" / f"{variant}_s1337.pt" if dataset == "turbulent_radiative_layer_2D"
          else ROOT / "runs/well2d" / dataset / "cnext" / f"{variant}_s1337.pt")
    return sp, load_checkpoint(variant, dataset, str(ck), "cnext", DEV)


def strat_pairs(split: well2d.Split, k: int, seed: int):
    rng = np.random.default_rng(seed); bins = np.array_split(np.arange(3, split.T - 1), 8)
    return [(t, int(s)) for t in range(split.n) for b in bins for s in rng.choice(b, min(k, len(b)), replace=False)]


@torch.no_grad()
def evaluate(model, split, pairs, bsz=8):
    model.eval(); v = []
    for x, y in well2d.batched(split, pairs, bsz):
        v.append(vrmse(model(x.to(DEV)), y.to(DEV)).cpu())
    model.train()
    return torch.cat(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("variant")
    ap.add_argument("--loss", default="vrmse", choices=("vrmse", "mse")); ap.add_argument("--head", default="gate", choices=("gate", "plain"))
    ap.add_argument("--budget", type=float, default=2400); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--val-every", type=float, default=400); ap.add_argument("--valk", type=int, default=2)
    ap.add_argument("--full-test", action="store_true", help="score the saved best model once on the full TEST split")
    ap.add_argument("--testk", type=int, default=4, help="stratified TEST estimate: windows per trajectory x time bin")
    ap.add_argument("--gate-warmup", type=float, default=0, help="seconds of gate-only training (trunk frozen) first")
    ap.add_argument("--gate-pretrain", type=int, default=0, help="exact gate pre-training on N cached TRAIN windows (frozen trunk)")
    ap.add_argument("--tag-suffix", default="", help="appended to the run tag (e.g. _pre for the two-stage schedule)")
    a = ap.parse_args()
    tag = f"{a.dataset}_{a.variant}_{a.loss}_{a.head}_s{a.seed}{a.tag_suffix}"
    OUT.mkdir(parents=True, exist_ok=True); ck = OUT / f"{tag}.pt"
    (train, valid, test), trunk = load_data_and_trunk(a.dataset, a.variant)
    grid = tuple(train.S) if hasattr(train, "S") else (train.H, train.W)
    alpha0 = fit_alpha(trunk, train, a.seed) if a.head == "gate" else torch.ones(train.C, NB)   # same rule, every trunk
    model = GatedHead(trunk, train.C, grid, alpha0, a.head == "gate").to(DEV)
    if a.full_test:
        model.load_state_dict(torch.load(ck, weights_only=True, map_location=DEV)["model"])
        v = evaluate(model, test, well2d.all_test_windows(test))
        res = json.loads((OUT / f"{tag}.json").read_text()); res.update(full_test=float(v.mean()), full_test_median=float(v.median()), full_test_n=len(v))
        (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1)); print(f"[ft] {tag} FULL TEST {v.mean():.5f} (n={len(v)})", flush=True)
        return
    torch.manual_seed(a.seed); gen = torch.Generator().manual_seed(a.seed)
    vp, tp = strat_pairs(valid, a.valk, 2), strat_pairs(test, a.testk, 3)
    opt = torch.optim.AdamW([{"params": model.trunk.parameters(), "lr": a.lr}, {"params": model.net.parameters(), "lr": 1e-3 if a.gate_warmup > 0 else 3e-4}], weight_decay=0.01)
    pre = None
    if a.gate_pretrain and a.head == "gate":                                         # stage 1: exact, frozen trunk
        tp0 = time.time(); g = torch.Generator().manual_seed(20_000 + a.seed)
        trp = [(int(k), int(t)) for k, t in zip(torch.randint(0, train.n, (a.gate_pretrain,), generator=g),
                                                 torch.randint(3, train.T - 1, (a.gate_pretrain,), generator=g))]
        st_tr, st_va = gate_stats(model, train, trp), gate_stats(model, valid, vp)
        pre = pretrain_gate(model, st_tr, st_va)
        print(f"[ft] {tag}: exact gate pre-training on {len(trp)} TRAIN windows -> stratified VALID {pre:.5f} "
              f"({time.time() - tp0:.0f}s)", flush=True)
        del st_tr, st_va
    v0 = evaluate(model, valid, vp).mean().item(); best, curve, steps, t0, last = v0, [(0, v0)], 0, time.time(), time.time()
    torch.save({"model": model.state_dict(), "val": best, "step": 0}, ck)
    print(f"[ft] {tag}: start VALID(strat {len(vp)}) {v0:.5f}", flush=True)
    warm = a.gate_warmup > 0 and a.head == "gate"
    if warm:
        for q in model.trunk.parameters():
            q.requires_grad_(False)
    while time.time() - t0 < a.budget:
        if warm and time.time() - t0 > a.gate_warmup:                               # stage 2: joint, trunk unfrozen
            warm = False
            for q in model.trunk.parameters():
                q.requires_grad_(True)
            print(f"[ft] {tag} gate warm-up done at step {steps}", flush=True)
        x, y = well2d.train_batch(train, a.batch, gen, DEV)
        pred = model(x)
        loss = vrmse(pred, y).mean() if a.loss == "vrmse" else (pred - y).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); steps += 1
        if time.time() - last > a.val_every:
            v = evaluate(model, valid, vp).mean().item(); curve.append((steps, v)); last = time.time()
            print(f"[ft] {tag} step {steps} VALID {v:.5f} (best {min(best, v):.5f})", flush=True)
            if v < best:
                best = v; torch.save({"model": model.state_dict(), "val": best, "step": steps}, ck)
    v = evaluate(model, valid, vp).mean().item(); curve.append((steps, v))
    if v < best:
        best = v; torch.save({"model": model.state_dict(), "val": best, "step": steps}, ck)
    model.load_state_dict(torch.load(ck, weights_only=True, map_location=DEV)["model"])
    te = evaluate(model, test, tp)
    res = {"gate_warmup_s": a.gate_warmup, "gate_pretrain_n": a.gate_pretrain, "gate_pretrain_valid": pre, "tag": tag, "dataset": a.dataset, "variant": a.variant, "loss": a.loss, "head": a.head, "seed": a.seed, "lr": a.lr,
           "steps": steps, "valid_start": v0, "valid_best": best, "curve": curve, "strat_test": float(te.mean()),
           "strat_test_median": float(te.median()), "strat_test_n": len(te), "budget_s": a.budget}
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"[ft] {tag}: VALID {v0:.5f} -> {best:.5f} | strat-TEST {te.mean():.5f} (n={len(te)}) | {steps} steps", flush=True)


if __name__ == "__main__":
    main()
