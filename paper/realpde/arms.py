"""RealPDE foil: classical+quantum low-mode correctors on a frozen released trunk (default U-Net-finetune).

    .venv/bin/python paper/realpde/arms.py --trunk unet --arms none lin mlp30 mr30ryd --seeds 1 2 3

Why low modes (runs/realpde/error_bands.json, VALID): the trunks' high-k error is shared across architectures
(coherence 0.80-0.93, PIV noise floor) while the model-specific error lives in the lowest (t, h, w) shells; the pure DC
shell holds only 0.8% of U-Net's error, so the pooled (DC-only) register insertion has a tiny ceiling. The corrector
therefore reads and writes a block of the lowest Fourier modes (the band generalisation of the DC bottleneck).
  features f = low-mode block of rfftn(P) and rfftn(X_last-window) (real + imag), standardised on TRAIN
  h = GELU(W_in f)  ->  core: none | lin (no hidden layer) | mlp30 (matched MLP) | mr30ryd (3 x 10-atom Rydberg register)
      as a zero-init gated residual on h (paper/model.py Bottleneck)  ->  W_out h = low-mode correction (zero-init)
  prediction = P + irfftn(correction placed in the low-mode block).
Trained on TRAIN windows (trunk predictions cached once), early-stopped on VALID rel-L2, TEST scored once per arm and
seed with RealPDEBench's metric (chunked, verified). Every arm shares W_in / W_out, so only the core differs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import common as C

sys.path.insert(0, str(C.ROOT / "paper"))
from model import Bottleneck, MLPCore, _count  # noqa: E402
from qlayer import MultiRegister  # noqa: E402

MT, MH, MW = 3, 4, 4                 # kept modes: t in [0, MT) U (-MT, 0); h likewise with MH; w in [0, MW) (rfft axis)
WIDTH = 256


def mode_index(t: int, h: int, w_r: int):
    ti = list(range(MT)) + list(range(t - MT + 1, t)); hi = list(range(MH)) + list(range(h - MH + 1, h))
    return torch.tensor(ti), torch.tensor(hi), torch.arange(MW)


@torch.no_grad()
def low_modes(A: torch.Tensor, idx) -> torch.Tensor:
    """(B, t, h, w, c) real -> (B, n_modes * c * 2) real features of the kept low-mode block."""
    F = torch.fft.rfftn(A.clone().to(C.DEV).double(), dim=(1, 2, 3))   # clone: GB10 turns mmap pages sent to the GPU into anonymous memory
    ti, hi, wi = (i.to(C.DEV) for i in idx)
    blk = F[:, ti][:, :, hi][:, :, :, wi]                                     # (B, mt, mh, mw, c) complex
    return torch.view_as_real(blk).flatten(1).float()


def place(corr: torch.Tensor, shape, idx) -> torch.Tensor:
    """Low-mode correction (B, n_modes * c * 2) -> spatial field (B, t, h, w, c)."""
    B, (t, h, w, c) = corr.shape[0], shape
    ti, hi, wi = (i.to(corr.device) for i in idx)
    blk = torch.view_as_complex(corr.double().reshape(B, len(ti), len(hi), len(wi), c, 2).contiguous())
    F = torch.zeros(B, t, h, w // 2 + 1, c, dtype=torch.complex128, device=corr.device)
    F[:, ti.view(-1, 1, 1), hi.view(1, -1, 1), wi.view(1, 1, -1)] = blk
    return torch.fft.irfftn(F, s=(t, h, w), dim=(1, 2, 3)).float()


class Corrector(nn.Module):
    def __init__(self, fin: int, fout: int, arm: str, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.arm = arm
        self.inp = nn.Linear(fin, WIDTH)
        if arm == "mlp30":
            self.mid = Bottleneck(WIDTH, MLPCore(WIDTH, 30, _count(MultiRegister(WIDTH, 10, 3, 4, "hea"))), 30)
        elif arm == "mr30ryd":
            self.mid = Bottleneck(WIDTH, MultiRegister(WIDTH, 10, 3, 4, "rydberg"), 30)
        elif arm.startswith("q"):                          # wave-G grammar, incl. doc 35 §23 q30-geo=3d[-mlp]
            from model import build_config_core, parse_config
            core, out = build_config_core(parse_config(arm), WIDTH, 4); self.mid = Bottleneck(WIDTH, core, out)
        else:
            self.mid = None
        self.out = nn.Linear(fin if arm == "lin" else WIDTH, fout)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, f):
        if self.arm == "lin":
            return self.out(f)
        h = nn.functional.gelu(self.inp(f))
        return self.out(self.mid(h) if self.mid is not None else h)


def features(X, P, idx, chunk=256):
    return torch.cat([torch.cat([low_modes(P[i:i + chunk], idx), low_modes(X[i:i + chunk], idx)], 1).cpu()
                      for i in range(0, len(P), chunk)])


def targets(Y, P, idx, chunk=256):
    return torch.cat([low_modes(Y[i:i + chunk] - P[i:i + chunk], idx).cpu() for i in range(0, len(P), chunk)])


class Corrected:
    """P + correction, indexable like a tensor (feeds official_metrics_chunked without materialising)."""

    def __init__(self, P, corr, idx):
        self.P, self.corr, self.idx, self.shape = P, corr, idx, P.shape

    def __getitem__(self, i):
        return self.P[i] + place(self.corr[i].to(C.DEV), self.shape[1:], self.idx).cpu()


@torch.no_grad()
def val_rel_l2(model, Fv, mu, sd, Pv, Yv, idx, chunk=256) -> float:
    tot = 0.0
    for i in range(0, len(Pv), chunk):
        corr = model(((Fv[i:i + chunk] - mu) / sd).to(C.DEV))
        p = Pv[i:i + chunk].clone().to(C.DEV) + place(corr, Pv.shape[1:], idx); y = Yv[i:i + chunk].clone().to(C.DEV)
        tot += ((p - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).sum().item()
    return tot / len(Pv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", default="unet"); ap.add_argument("--stage", default="finetune")
    ap.add_argument("--arms", nargs="+", default=["none", "lin", "mlp30", "mr30ryd"]); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--save-pred", action="store_true", help="cache corrected VALID/TEST predictions as model '<trunk>-<arm>-s<seed>' for ensemble.py")
    a = ap.parse_args()
    sp = {}
    for split in ("train", "val", "test"):
        X, Y, sims = C.load_windows(split)
        P = C.open_pred(C.CACHE / f"pred_{a.trunk}_{a.stage}_{split}.npy", Y.shape)
        sp[split] = (X, Y, P, sims)
    idx = mode_index(*sp["val"][1].shape[1:4])
    t0 = time.time()
    F = {s: features(sp[s][0], sp[s][2], idx) for s in sp}
    T = targets(sp["train"][1], sp["train"][2], idx)
    mu, sd = F["train"].mean(0), F["train"].std(0) + 1e-6
    tsd = T.std(0) + 1e-6
    print(f"[arms] features {F['train'].shape[1]} -> targets {T.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    import eval_trunk as E
    out = C.ROOT / "runs" / "realpde" / "arms"; out.mkdir(parents=True, exist_ok=True)
    for arm in a.arms:
        for seed in a.seeds:
            f_res = out / f"{a.trunk}_{arm}_s{seed}.json"
            tag = f"{a.trunk}-{arm}-s{seed}"
            if f_res.exists() and not (a.save_pred and not (C.CACHE / f"pred_{tag}_{a.stage}_test.done").exists()):
                continue
            Xv, Yv, Pv, _ = sp["val"]
            model = Corrector(F["train"].shape[1], T.shape[1], arm, seed).to(C.DEV)
            tsd_d = tsd.to(C.DEV)
            net = lambda f, m=model: m(f) * tsd_d          # targets are scaled (not centred): a zero head = no correction
            best, best_state, bad = val_rel_l2(net, F["val"], mu, sd, Pv, Yv, idx), None, 0
            base = best
            if arm != "none":
                opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
                g = torch.Generator().manual_seed(seed)
                Ft, Tt = ((F["train"] - mu) / sd), (T / tsd)
                for ep in range(a.epochs):
                    perm = torch.randperm(len(Ft), generator=g)
                    for i in range(0, len(Ft), a.batch):
                        b = perm[i:i + a.batch]
                        loss = (model(Ft[b].to(C.DEV)) - Tt[b].to(C.DEV)).square().mean()
                        opt.zero_grad(); loss.backward(); opt.step()
                    v = val_rel_l2(net, F["val"], mu, sd, Pv, Yv, idx)
                    if v < best - 1e-7:
                        best, best_state, bad = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
                    else:
                        bad += 1
                        if bad >= 8:
                            break
                if best_state is not None:
                    model.load_state_dict(best_state)
            Xt, Yt, Pt, st = sp["test"]
            with torch.no_grad():
                corr = torch.cat([net(((F["test"][i:i + 256] - mu) / sd).to(C.DEV)).cpu() for i in range(0, len(Pt), 256)])
            res = {"trunk": a.trunk, "arm": arm, "seed": seed, "valid_rel_l2_trunk": base, "valid_rel_l2": best,
                   "params_core": _count(model.mid) if model.mid is not None else 0, "test": {}}
            for sub, m in E.subsets(st).items():                  # not `tag`: that names the cached prediction files
                res["test"][sub] = C.official_metrics_chunked(Corrected(Pt, corr, idx), Yt, idx=torch.nonzero(m).flatten())
            f_res.write_text(json.dumps(res, indent=1))
            if a.save_pred:
                for split in ("val", "test"):
                    _, _, Ps, _ = sp[split]
                    with torch.no_grad():
                        cs = torch.cat([net(((F[split][i:i + 256] - mu) / sd).to(C.DEV)).cpu() for i in range(0, len(Ps), 256)])
                    dst = C.CACHE / f"pred_{tag}_{a.stage}_{split}.npy"; out_p = C.open_pred(dst, Ps.shape); cp = Corrected(Ps, cs, idx)
                    for i in range(0, len(Ps), 256):
                        out_p[i:i + 256] = cp[i:i + 256]
                    del out_p; dst.with_suffix(".done").touch()          # shared file mapping: written through on release
            print(f"[arms] {a.trunk} {arm} s{seed}: VALID {base:.5f} -> {best:.5f} | TEST all {res['test']['all']['rel_l2']:.5f} "
                  f"in {res['test']['in_dist']['rel_l2']:.5f} out {res['test']['out_dist']['rel_l2']:.5f}", flush=True)


if __name__ == "__main__":
    main()
