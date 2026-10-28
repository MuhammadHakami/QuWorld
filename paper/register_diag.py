"""FW-B (Sep 19 sweep): zero-training register-configuration diagnostics.

For a register (3 x 10-qubit Rydberg blocks as used in every benchmark arm) and a batch of trunk latents h,
report per configuration:
  R_eff   effective rank of the readout feature matrix F (B x 30): exp(H(sigma/sum sigma))   [2607.09445]
  PR      basis coverage = mean participation ratio 1/sum_k p_k^2 of the register's basis distribution   [2608.25511]
  D_diag, D_off  butterfly response: mean |<Z_i>(psi0) - <Z_i>(X_j psi0)| for i=j / i!=j (frozen -> D_off ~ 0,
          scrambled -> D_off ~ D_diag); the onset window is the middle band                  [2608.25511, OTOC proxy]
Sweep axes (the per-benchmark "configuration only" knobs): input strength sigma_W (std of the detuning map
weights), evolution time tau, geometry scale g (positions x g).  Recommendation heuristic (documented, not tuned):
max R_eff subject to 0.2 <= D_off/D_diag <= 0.8.

Usage: python paper/register_diag.py [--variant mr30ryd] [--ckpt runs/x.pt] [--latents runs/h.npy] [--n 256]
                                     [--sigmas 0.03,0.1,0.3,1] [--taus 0.1,0.3,1] [--geos 0.7,1,1.4] [--out runs/fwa/register_diag]
Outputs <out>.json and <out>.md. GPU only. Forward passes only (no training).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))
from model import build_model  # noqa: E402
from qlayer import MultiRegister, QuantumBlock, _bits  # noqa: E402

DEV = torch.device("cuda")


def checkpoint_state(obj) -> dict:
    """Extract known containers; never treat checkpoint metadata as weights."""
    if isinstance(obj, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Unrecognized checkpoint container")


def load_arena_checkpoint(model, path: str) -> None:
    sd = checkpoint_state(torch.load(path, map_location="cpu", weights_only=True))
    if any(k.startswith("neck.") for k in sd):
        raise ValueError("Official CNext checkpoint cannot load into the arena trunk; "
                         "use a matching arena checkpoint. Previous permissive diagnostic loads are invalid.")
    model.load_state_dict(sd, strict=True)


def arena_latents(variant: str, n: int, ckpt: str | None, seed: int) -> tuple[torch.Tensor, MultiRegister]:
    """Pooled bottleneck inputs h (n, width) from the TRL2D arena trunk (random init or a checkpoint)."""
    from data import load  # noqa: E402  (arena cache)
    train, _, _, _ = load()
    torch.manual_seed(seed)
    model = build_model(variant, in_ch=train.shape[2]).to(DEV)
    if ckpt:
        load_arena_checkpoint(model, ckpt)
    g = torch.Generator().manual_seed(seed)
    ni = torch.randint(0, train.shape[0], (n,), generator=g)
    ti = torch.randint(0, train.shape[1], (n,), generator=g)
    with torch.no_grad():
        x = train[ni, ti].to(DEV)
        h = model.pre(model.lift(x)).mean(dim=(2, 3))
    core = model.mid.core
    assert isinstance(core, MultiRegister), type(core)
    return h.detach(), core


def effective_rank(F: torch.Tensor) -> float:
    s = torch.linalg.svdvals(F - F.mean(0, keepdim=True))
    # Constant features have zero usable rank; do not return exp(0) = 1.
    tol = max(F.shape) * torch.finfo(F.dtype).eps * F.abs().max().clamp_min(1)
    s = s[s > tol]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def butterfly(blk: QuantumBlock, h: torch.Tensor) -> tuple[float, float]:
    """Mean |<Z_i>(psi0) - <Z_i>(X_j psi0)| over the batch: diagonal (i=j) and off-diagonal (i!=j)."""
    n, b = blk.n, h.shape[0]
    base = blk.readout(blk.probs(h), h)                                   # (B, n)
    bits = _bits(n).to(h.device)
    D = torch.zeros(n, n, device=h.device)
    for j in range(n):
        # X_j |0...0> = basis state with only bit j set
        st = torch.zeros(b, 2**n, dtype=torch.complex64, device=h.device)
        st[:, 1 << (n - 1 - j)] = 1.0
        pert = blk.readout(blk.probs(h, init_state=st), h)
        D[:, j] = (base - pert).abs().mean(0)
    diag = D.diag().mean().item()
    off = (D.sum() - D.diag().sum()).item() / (n * (n - 1))
    return diag, off


@torch.no_grad()
def participation(blk: QuantumBlock, h: torch.Tensor) -> float:
    p = blk.probs(h)
    return float((1.0 / (p.square().sum(-1))).mean())


def configure(core: MultiRegister, sigma: float, tau: float, geo: float, seed: int) -> MultiRegister:
    c = copy.deepcopy(core)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for blk in c.regs:
            blk.cond.weight.copy_(torch.randn(blk.cond.weight.shape, generator=gen) * sigma)
            blk.cond.bias.zero_()
            blk.tau.fill_(tau)
            blk.pos.mul_(geo)
    return c


@torch.no_grad()
def diagnose(core: MultiRegister, h: torch.Tensor) -> dict:
    F = torch.cat([blk.readout(blk.probs(h), h) for blk in core.regs], -1)   # (B, 30)
    r = effective_rank(F)
    pr = float(np.mean([participation(blk, h) for blk in core.regs]))
    bd, bo = zip(*[butterfly(blk, h) for blk in core.regs])
    d, o = float(np.mean(bd)), float(np.mean(bo))
    return {"R_eff": r, "PR": pr, "D_diag": d, "D_off": o, "D_ratio": (o / d) if d > 1e-9 else float("nan"),
            "F_std": float(F.std())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="mr30ryd")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--latents", default=None, help=".npy (n, d_in) latents instead of the arena trunk")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--sigmas", default="0.03,0.1,0.3,1")
    ap.add_argument("--taus", default="0.1,0.3,1")
    ap.add_argument("--geos", default="0.7,1,1.4")
    ap.add_argument("--out", default="runs/fwa/register_diag")
    args = ap.parse_args()
    if args.latents and args.ckpt:
        ap.error("--latents with --ckpt was previously ignored; provide a matching arena "
                 "checkpoint without --latents, or use --latents for an explicitly fresh register")
    if args.n < 2:
        ap.error("--n must be at least 2")
    t0 = time.time()
    if args.latents:
        h = torch.from_numpy(np.load(args.latents)).float().to(DEV)[: args.n]
        torch.manual_seed(args.seed)
        core = MultiRegister(h.shape[1], 10, 3, 4, "rydberg").to(DEV)
    else:
        h, core = arena_latents(args.variant, args.n, args.ckpt, args.seed)
    rows = [{"config": "as_built", "sigma": float(core.regs[0].cond.weight.std()), "tau": float(core.regs[0].tau.mean()),
             "geo": 1.0, **diagnose(core, h)}]
    for sigma in map(float, args.sigmas.split(",")):
        for tau in map(float, args.taus.split(",")):
            for geo in map(float, args.geos.split(",")):
                c = configure(core, sigma, tau, geo, args.seed)
                rows.append({"config": "sweep", "sigma": sigma, "tau": tau, "geo": geo, **diagnose(c, h)})
    band = [r for r in rows[1:] if 0.2 <= r["D_ratio"] <= 0.8]
    best = max(band, key=lambda r: r["R_eff"]) if band else None
    out = {"variant": args.variant, "ckpt": args.ckpt, "n_latents": int(h.shape[0]), "d_in": int(h.shape[1]),
            "rows": rows, "recommended": best, "status": "heuristic_only" if best else "no_feasible_candidate",
            "checkpoint_load": "strict" if args.ckpt else "fresh_initialization",
            "rule": "max R_eff s.t. 0.2<=D_off/D_diag<=0.8 (unvalidated heuristic; no downstream improvement claim)",
           "wall_s": round(time.time() - t0, 1)}
    p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.with_suffix(".json").write_text(json.dumps(out, indent=1))
    md = ["| config | sigma_W | tau | geo | R_eff | PR | D_diag | D_off | D_off/D_diag | F_std |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['config']} | {r['sigma']:.3g} | {r['tau']:.3g} | {r['geo']:.3g} | {r['R_eff']:.2f} | {r['PR']:.1f} | "
                  f"{r['D_diag']:.3f} | {r['D_off']:.3f} | {r['D_ratio']:.2f} | {r['F_std']:.3f} |")
    if best:
        md.append(f"\nheuristic candidate: sigma_W={best['sigma']:.3g} tau={best['tau']:.3g} geo={best['geo']:.3g} "
                  f"(R_eff {best['R_eff']:.2f}, D_off/D_diag {best['D_ratio']:.2f}); rule: {out['rule']}")
    else:
        md.append("\nNo candidate satisfies the declared onset band; no recommendation.")
    p.with_suffix(".md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"[diag] wrote {p}.json/.md in {out['wall_s']} s")


if __name__ == "__main__":
    main()
