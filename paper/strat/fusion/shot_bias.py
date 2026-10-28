"""Doc 35 §38t: pooled shot-bias co-processor for the fusion shape heads (the paper's architecture on the competition
pipeline).

§38r-pre showed the 6-seed J1 heads' error on the 7 consistency scalars is mostly a per-shot offset (oracle: removing
each shot's mean residual lifts the sample's mean R2 0.98607 -> 0.99021), which frame-level linear models do not find.
A pooled insertion reads exactly that DC mode (Proposition 1), so the co-processor pools over the whole shot:

    frame encoder  e_t = MLP([corrected inputs_t, head outputs_t])            (F -> 256 -> 128, GELU)
    shot latent    h = mean_t e_t                                             (the shot's DC mode)
    offsets        o = W_c h + alpha * up(core(h))                            (7 scalars; alpha zero-init per target)
    core           none (arm "lin") | MLPCore matched to the register ("mlp") | MultiRegister 3 x 10 Rydberg ("reg")

Trained on TRAIN shots (targets = shot-mean residual of the heads vs the scorer labels; the heads underfit, TRAIN loss ~
VALID error), 10% of TRAIN shots held out for early stopping; scored once per arm on the 180-shot VALID sample: R2 of
head + offset per scalar against the official-extraction labels (boundary_labels_v8 y_valid).

    .venv/bin/python paper/strat/fusion/shot_bias.py --arm lin|mlp|reg [--seed 1] [--epochs 40]
        -> runs/strat/fusion/shot_bias_<arm>_s<seed>.{npz,pt,json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import fx  # noqa: I001
import gpu_build as GB
import gpu_pipe as GP
import strategies as ST

sys.path.insert(0, str(fx.ROOT / "paper"))
from model import MLPCore, _count  # noqa: E402
from qlayer import MultiRegister  # noqa: E402

HEADS = "+".join(f"head_boundary_v8_J1_s{s}" for s in range(1, 7))


class ShotBias(nn.Module):
    def __init__(self, f_in: int, arm: str, hid: int = 128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(f_in, 256), nn.GELU(), nn.Linear(256, hid), nn.GELU())
        self.lin = nn.Linear(hid, 7)
        self.arm = arm
        if arm != "lin":
            reg = MultiRegister(hid, 10, 3, 4, "rydberg")
            self.core = reg if arm == "reg" else MLPCore(hid, 30, _count(reg))
            self.up = nn.Linear(30, 7); self.alpha = nn.Parameter(torch.zeros(7))

    def forward(self, x, seg, n_seg):
        e = self.enc(x); h = torch.zeros(n_seg, e.shape[1], device=e.device, dtype=e.dtype).index_add_(0, seg, e)
        h = h / torch.bincount(seg, minlength=n_seg).clamp(min=1)[:, None].to(e.dtype)
        o = self.lin(h)
        return o + self.alpha * self.up(self.core(h)) if self.arm != "lin" else o


def frames(d, rows, ck):
    dd = GB.fixed(d)
    X = np.clip((dd.inputs(rows, ck["fm"], ck["edge"]) - ck["mu"]) / ck["sd"], -10, 10).astype(np.float32)
    H = GB.heads_gpu(d, rows, HEADS).astype(np.float32)
    return X, H


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="lin", choices=["lin", "mlp", "reg"])
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--epochs", type=int, default=40); a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); t0 = time.time(); dev = torch.device("cuda")
    sota, d = fx.data(); lab = np.load(fx.OUT / "boundary_labels_v8.npz")
    ck = torch.load(fx.OUT / "head_boundary_v8_J1_s1.pt", map_location="cpu", weights_only=False)
    cache = fx.OUT / "shot_bias_cache.npz"
    if cache.exists():
        z = np.load(cache); Xtr, Htr, Xs, Hs = z["Xtr"], z["Htr"], z["Xs"], z["Hs"]
    else:
        Xtr, Htr = frames(d, lab["rows_train"], ck); Xs, Hs = frames(d, lab["rows_valid"], ck)
        np.savez(cache, Xtr=Xtr, Htr=Htr, Xs=Xs, Hs=Hs)
    if os.environ.get("SB_OOF"):                         # §38u: TRAIN head outputs from the 5 out-of-fold J1 heads
        pos = {int(r): i for i, r in enumerate(lab["rows_train"])}; Htr = Htr.copy(); n = 0
        for k in range(5):
            z = np.load(fx.OUT / f"head_boundary_v8_J1F{k}_s1_oof.npz"); ii = [pos[int(r)] for r in z["rows_oof"]]
            Htr[ii] = z["pred_oof"]; n += len(ii)
        print(f"[sb] out-of-fold head outputs for {n} of {len(Htr)} TRAIN frames", flush=True)
    print(f"[sb] frames train {Xtr.shape} sample {Xs.shape} | {time.time() - t0:.0f}s", flush=True)
    Ytr, Ys = lab["y_train"][:, :7], lab["y_valid"][:, :7]
    Rtr = Ytr - Htr[:, :7]; ysd = np.nanstd(Rtr, 0) + 1e-9
    shot_tr = d.shot[lab["rows_train"]]; shot_s = d.shot[lab["rows_valid"]]
    us, seg = np.unique(shot_tr, return_inverse=True)
    T = np.zeros((len(us), 7)); M = np.zeros((len(us), 7))                   # shot-mean residual targets (NaN-aware)
    for j in range(7):
        ok = np.isfinite(Rtr[:, j])
        T[:, j] = np.bincount(seg[ok], Rtr[ok, j], len(us)) / np.maximum(np.bincount(seg[ok], minlength=len(us)), 1)
        M[:, j] = np.bincount(seg[ok], minlength=len(us)) > 0
    T = T / ysd
    rng = np.random.default_rng(a.seed); hold = rng.random(len(us)) < 0.1
    F = np.hstack([Xtr, Htr]); mu, sd = F.mean(0), F.std(0) + 1e-6
    Fg = torch.tensor((F - mu) / sd, dtype=torch.float32, device=dev); segg = torch.tensor(seg, device=dev)
    Tg = torch.tensor(T, dtype=torch.float32, device=dev); Mg = torch.tensor(M, dtype=torch.float32, device=dev)
    starts = np.r_[0, np.cumsum(np.bincount(seg, minlength=len(us)))]
    order = np.argsort(seg, kind="stable"); Fg, segg = Fg[torch.tensor(order, device=dev)], segg[torch.tensor(order, device=dev)]
    net = ShotBias(F.shape[1], a.arm).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    def run(ids, train):
        net.train(train); tot, n = 0.0, 0
        for b in np.array_split(ids, max(1, len(ids) // 64)):
            r = torch.cat([torch.arange(starts[i], starts[i + 1], device=dev) for i in b])
            loc = torch.repeat_interleave(torch.arange(len(b), device=dev), torch.tensor(starts[b + 1] - starts[b], device=dev))
            with torch.set_grad_enabled(train):
                o = net(Fg[r], loc, len(b)); bt = torch.tensor(b, device=dev)
                loss = ((o - Tg[bt]) ** 2 * Mg[bt]).sum() / Mg[bt].sum().clamp(min=1)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b); n += len(b)
        return tot / n
    tr_ids, ho_ids = np.nonzero(~hold)[0], np.nonzero(hold)[0]; best, state = np.inf, None
    for ep in range(a.epochs):
        ltr = run(rng.permutation(tr_ids), True); lho = run(ho_ids, False)
        if lho < best:
            best, state = lho, {k: v.detach().clone() for k, v in net.state_dict().items()}
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"[sb] {a.arm} ep {ep} train {ltr:.4f} holdout {lho:.4f} | {time.time() - t0:.0f}s", flush=True)
    net.load_state_dict(state); net.eval()
    usm, segs = np.unique(shot_s, return_inverse=True)
    with torch.no_grad():
        Fs = torch.tensor((np.hstack([Xs, Hs]) - mu) / sd, dtype=torch.float32, device=dev)
        off = net(Fs, torch.tensor(segs, device=dev), len(usm)).cpu().numpy() * ysd
    corr = Hs[:, :7] + off[segs]
    def r2(y, p):
        ok = np.isfinite(y) & np.isfinite(p); return float(1 - np.sum((y[ok] - p[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))
    head = [r2(Ys[:, j], Hs[:, j]) for j in range(7)]; new = [r2(Ys[:, j], corr[:, j]) for j in range(7)]
    rep = dict(arm=a.arm, seed=a.seed, holdout=best, head=head, corrected=new, mean_head=float(np.mean(head)), mean_corrected=float(np.mean(new)),
               params=_count(net), seconds=round(time.time() - t0))
    stem = fx.OUT / f"shot_bias_{a.arm}{'_oof' if os.environ.get('SB_OOF') else ''}_s{a.seed}"
    (stem.with_suffix(".json")).write_text(json.dumps(rep, indent=1)); np.savez(stem.with_suffix(".npz"), shots=usm, offsets=off)
    torch.save({"net": state, "mu": mu, "sd": sd, "ysd": ysd, "arm": a.arm, "f_in": F.shape[1]}, stem.with_suffix(".pt"))
    print(f"SB_RESULT {a.arm} s{a.seed}: mean R2 {rep['mean_head']:.5f} -> {rep['mean_corrected']:.5f} | "
          + " ".join(f"{n} {h:.4f}->{c:.4f}" for n, h, c in zip(GP.NAMES, head, new)), flush=True)


if __name__ == "__main__":
    main()
