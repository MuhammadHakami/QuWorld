"""§37 rayleigh_benard, strategies 3d/3e: cheap single-model changes that need only an exact gate (re-)pre-training.

--mode beta   : band-gated linear extrapolation as a second regressor in the head of the best single model:
                  y_hat = x_t + sum_b alpha_b(z) P_b(trunk(x) - x_t) + sum_b beta_b(z) P_b(x_t - x_{t-1})
                The gate MLP gets C*NB extra outputs (zero-initialised, so the start equals the saved gate exactly) and is
                re-trained on the exact per-window VRMSE (Parseval, 2 regressors) of the same 3,000 TRAIN windows as
                doc 37 (generator 20000 + 1); epoch selection on doc 37's own stratified VALID sample (k = 1, seed 2),
                epoch 0 (= the saved gate) included. The trunk is frozen.
--mode soup   : uniform weight average ("model soup") of the three seed trunks of one variant (s1337/8/9, all
                fine-tuned from the released CNextU-net), then the doc 37 gate recipe unchanged (TRAIN-fitted alpha0,
                exact pre-training on the same 3,000 TRAIN windows, selection on doc 37's VALID sample).
Scored once on the fixed §37 stratified VALID sample; per-window per-field VRMSE -> runs/strat/rb/head_<tag>.npz.

    flock runs/strat/gpu.lock timeout 1800 env QW_GPU_GB=12 runs/capped.sh strat_rb 16 uv run --no-sync python paper/strat/rb/head_ext.py --modes beta,soup:vanilla
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

import common as CM

FG = CM.FG
NB, C = FG.NB, CM.C


class BetaHead(FG.GatedHead):
    def __init__(self, base: FG.GatedHead):
        super().__init__(base.trunk, base.C, tuple(base.masks.shape[1:]), base.alpha0, True)
        self.load_state_dict(base.state_dict())
        old = self.net[-1]; new = nn.Linear(old.in_features, 2 * C * NB).to(old.weight.device)
        with torch.no_grad():
            new.weight.zero_(); new.bias.zero_()
            new.weight[:C * NB].copy_(old.weight); new.bias[:C * NB].copy_(old.bias)
        self.net[-1] = new

    def ab(self, z):
        o = torch.tanh(self.net(self.norm(z).clamp(-10, 10)))
        return self.alpha0 + 1.5 * o[:, :C * NB].reshape(-1, C, NB), 1.5 * o[:, C * NB:].reshape(-1, C, NB)

    def forward(self, x):
        xt, xp = x[:, 3 * C:], x[:, 2 * C:3 * C]
        p = self.trunk(x)
        z, Dk = self.features(x, p); al, be = self.ab(z)
        Xk = FG.fftn(xt - xp); M = self.masks.flatten(1)
        H = lambda a: (a[..., None] * M).sum(2).reshape(*a.shape[:2], *self.masks.shape[1:])
        return xt + torch.fft.ifftn(H(al) * Dk + H(be) * Xk, dim=(2, 3)).real


@torch.no_grad()
def beta_stats(model, split, pairs, bsz=8):
    out = {k: [] for k in ("z", "gY", "gYD", "gDD", "gYX", "gDX", "gXX", "Dv")}
    for x, y in CM.well2d.batched(split, pairs, bsz):
        x, y = x.to(CM.DEV), y.to(CM.DEV); xt, xp = x[:, 3 * C:], x[:, 2 * C:3 * C]; npx = y[0, 0].numel()
        z, Dk = model.features(x, model.trunk(x)); Yk = FG.fftn(y - xt); Xk = FG.fftn(xt - xp)
        bs = lambda K: FG.bandsum(K, model.masks) / npx
        out["z"].append(z); out["gY"].append(bs(Yk.abs().square())); out["gYD"].append(bs((Yk * Dk.conj()).real))
        out["gDD"].append(bs(Dk.abs().square())); out["gYX"].append(bs((Yk * Xk.conj()).real))
        out["gDX"].append(bs((Dk * Xk.conj()).real)); out["gXX"].append(bs(Xk.abs().square()))
        out["Dv"].append(npx * (y.flatten(2).var(-1, unbiased=False) + 1e-7))
    return {k: torch.cat(v) for k, v in out.items()}


def beta_vrmse(model, st, rows):
    al, be = model.ab(st["z"][rows]); g = {k: v[rows] for k, v in st.items()}
    E = (g["gY"] - 2 * al * g["gYD"] - 2 * be * g["gYX"] + al.square() * g["gDD"] + 2 * al * be * g["gDX"]
         + be.square() * g["gXX"]).sum(-1).clamp_min(0)
    return (E / g["Dv"]).clamp_min(1e-12).sqrt().mean(1)


def pretrain(model, tr, va, lossf, epochs=60, lr=3e-3, bsz=64):
    opt = torch.optim.Adam(model.net.parameters(), lr=lr); N = len(tr["z"]); allv = torch.arange(len(va["z"]), device=CM.DEV)
    model.norm.eval()
    with torch.no_grad():
        best = lossf(model, va, allv).mean().item()
    v0, state, curve = best, {k: t.clone() for k, t in model.state_dict().items() if not k.startswith("trunk.")}, []
    for ep in range(epochs):
        perm = torch.randperm(N, device=CM.DEV)
        for i in range(0, N - bsz + 1, bsz):
            loss = lossf(model, tr, perm[i:i + bsz]).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = lossf(model, va, allv).mean().item()
        curve.append(v)
        if v < best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items() if not k.startswith("trunk.")}
    model.load_state_dict(state, strict=False)
    return v0, best, curve


def train_pairs(train, n=3000, seed=1):
    g = torch.Generator().manual_seed(20_000 + seed)
    return [(int(k), int(t)) for k, t in zip(torch.randint(0, train.n, (n,), generator=g),
                                             torch.randint(3, train.T - 1, (n,), generator=g))]


@torch.no_grad()
def score(model, valid, pairs, bsz=8):
    model.eval(); out = []
    for x, y in CM.well2d.batched(valid, pairs, bsz):
        out.append(CM.vrmse_fields(model(x.to(CM.DEV)), y.to(CM.DEV)).cpu())
    return torch.cat(out).numpy()


def soup_trunk(variant):
    from well2d_backbone import load_checkpoint
    sts = [torch.load(CM.trunk_ckpt(variant, s), weights_only=True, map_location="cpu")["model_state_dict"]
           for s in (1337, 1338, 1339)]
    avg = {k: (sum(st[k].double() for st in sts) / len(sts)).to(sts[0][k].dtype) if sts[0][k].is_floating_point()
           else sts[0][k] for k in sts[0]}
    trunk = load_checkpoint(variant, CM.DS, str(CM.trunk_ckpt(variant, 1337)), "cnext", CM.DEV)
    trunk.load_state_dict(avg)
    return trunk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--modes", default="beta,soup:vanilla")
    ap.add_argument("--npre", type=int, default=3000); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    doc, pairs, sid = CM.load_strata()
    train, valid = CM.train_split(), CM.valid_split()
    trp = train_pairs(train, 64 if a.smoke else a.npre)
    vp = FG.strat_pairs(valid, 1, 2)[: 64 if a.smoke else None]            # doc 37's selection sample
    if a.smoke:
        pairs, sid = pairs[:16], sid[:16]
    w = CM.weights_for(doc, sid)
    for mode in a.modes.split(","):
        t0 = time.time(); torch.cuda.reset_peak_memory_stats()
        if mode == "beta":
            model = BetaHead(CM.load_gated(CM.BEST_SINGLE, valid)).eval()
            st_tr, st_va = beta_stats(model, train, trp), beta_stats(model, valid, vp)
            v0, best, curve = pretrain(model, st_tr, st_va, beta_vrmse)
            tag = "beta_mr30ryd"
        elif mode.startswith("soup:"):
            variant = mode.split(":")[1]
            trunk = soup_trunk(variant)
            alpha0 = FG.fit_alpha(trunk, train, 1)
            model = FG.GatedHead(trunk, C, (valid.H, valid.W), alpha0, True).to(CM.DEV)
            st_tr, st_va = FG.gate_stats(model, train, trp), FG.gate_stats(model, valid, vp)
            model.train()
            v0, best, curve = None, FG.pretrain_gate(model, st_tr, st_va), []     # the doc 37 recipe, unchanged
            tag = f"soup_{variant}"
        else:
            raise SystemExit(f"unknown mode {mode}")
        del st_tr, st_va
        model.eval()
        v = score(model, valid, pairs)
        tag += "_smoke" if a.smoke else ""
        np.savez_compressed(CM.OUT / f"head_{tag}.npz", sid=sid, vr=v)
        if not a.smoke:
            torch.save({"model": model.state_dict()}, CM.OUT / f"head_{tag}.pt")
        meta = {"tag": tag, "mode": mode, "n_pretrain": len(trp), "select_valid_start": v0, "select_valid_best": best,
                "curve": curve, "pop_weighted_valid_sample": float((w / w.sum() * v.mean(1)).sum()),
                "seconds": time.time() - t0, "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        (CM.OUT / f"head_{tag}.json").write_text(json.dumps(meta, indent=1))
        print(f"[head] {tag}: selection VALID {v0} -> {best:.5f}; §37 sample pop-weighted "
              f"{meta['pop_weighted_valid_sample']:.5f} ({meta['seconds']:.0f}s, peak {meta['peak_gib']:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
