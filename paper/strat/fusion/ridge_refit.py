"""Doc 35 §45b: ridge refit of a shape head's output layer (competitor evidence: Anulum refits its geometry heads by ridge regression on
each network's last-layer activations). Our heads end in out(norm(h)) + skip(x), trained by AdamW with dropout and a x3 edge weight;
here the 7 consistency outputs are refitted in closed form on Z = [norm(h), x, 1] over TRAIN frames (every 3rd), lambda chosen on 10% of
the TRAIN shots held out (never VT). VT read-out (shot-pooled R2 per scalar, as the scorer): original vs refitted head alone, and the
deployed TRAIN-only ensemble (F8 proportions) with the refitted head(s) in place of the originals. PRE-DECLARED rule (2026-10-16, before any
result): adopt iff the ensemble's VT mean-7 gain >= +0.0003 with a paired shot-bootstrap 95% CI > 0.
    .venv/bin/python paper/strat/fusion/ridge_refit.py STEM[,STEM...]
"""
from __future__ import annotations

import json
import pathlib
import sys

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"                                          # before fx (which hides the GPU) and gpucap's probe
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fx  # noqa: E402

OUT = fx.OUT; CONS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def feats(net, x):
    h = net.inp(x)
    for b in net.blocks:
        h = h + b(h)
    return torch.cat([net.norm(h), x, torch.ones(len(x), 1, device=x.device)], 1)


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    from gbdt_shape import DEPLOYED, head_inputs
    from head_boundary import Net
    stems = sys.argv[1].split(","); d = sota.Data()
    tr = np.load(OUT / "boundary_labels_v8.npz"); rtr = tr["rows_train"][::3]; ytr = tr["y_train"][::3, :7].astype(np.float64)
    vt = np.load(OUT / "boundary_labels_v8_vt.npz"); keep = d.split[vt["rows"]] != 0; rvt = vt["rows"][keep]; yvt = vt["y"][keep, :7].astype(np.float64)
    sh_tr = d.shot[rtr]; us = np.unique(sh_tr); hold = np.isin(sh_tr, np.random.default_rng(1).permutation(us)[: len(us) // 10])
    refit = {}
    for st in stems:
        ck = torch.load(OUT / f"{st}.pt", map_location="cpu", weights_only=False); c = ck["cfg"]
        net = Net(len(ck["mu"]), len(ck["ymu"]), c["width"], c["depth"], c["drop"]).cuda(); net.load_state_dict(ck["net"]); net.eval()
        nz = lambda r: torch.tensor(np.clip((head_inputs(d, r, st) - ck["mu"]) / ck["sd"], -10, 10), dtype=torch.float32, device="cuda")  # noqa: E731
        Yz = torch.tensor((ytr - ck["ymu"][:7]) / ck["ysd"][:7], device="cuda"); okn = np.isfinite(Yz.cpu().numpy()).all(1)
        X = nz(rtr); D = None; At = bt = Ah = bh = None; Zh, Yh = [], []                # streamed normal equations (float64)
        with torch.no_grad():
            for i in range(0, len(rtr), 32768):
                sl = slice(i, i + 32768); z = feats(net, X[sl]).double(); y = Yz[sl]; m = torch.tensor(okn[sl], device="cuda")
                h = torch.tensor(hold[sl], device="cuda")
                for sel, tag in ((m & ~h, "t"), (m & h, "h")):
                    zz, yy = z[sel], y[sel]; a_, b_ = zz.T @ zz, zz.T @ yy
                    if tag == "t":
                        At, bt = (a_, b_) if At is None else (At + a_, bt + b_)
                    else:
                        Ah, bh = (a_, b_) if Ah is None else (Ah + a_, bh + b_); Zh.append(zz); Yh.append(yy)
        Zh, Yh = torch.cat(Zh), torch.cat(Yh); I = torch.eye(At.shape[0], device="cuda", dtype=At.dtype); best = None
        for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
            Wt = torch.linalg.solve(At + lam * I, bt); e = ((Zh @ Wt - Yh) ** 2).mean().item()
            best = (e, lam) if best is None or e < best[0] else best
        W = torch.linalg.solve(At + Ah + best[1] * I, bt + bh)                       # final: all TRAIN rows at the chosen lambda
        del X, Zh, Yh
        with torch.no_grad():
            Zv = torch.cat([feats(net, x) for x in nz(rvt).split(32768)]).double()
        refit[st] = (Zv @ W).cpu().numpy() * ck["ysd"][:7] + ck["ymu"][:7]
        print(f"[ridge] {st}: lambda {best[1]} (held-out TRAIN mse {best[0]:.5f})", flush=True)
    sh = d.shot[rvt]; us_vt, inv = np.unique(sh, return_inverse=True)

    def r2_shots(p, idx=None):
        m = np.isfinite(yvt).all(1) & np.isfinite(p).all(1); y = yvt[m]; pp = p[m]; g = inv[m]
        if idx is not None:                                                     # bootstrap: shot multiplicities
            w = np.bincount(idx, minlength=len(us_vt))[g].astype(np.float64)
        else:
            w = np.ones(len(y))
        mu = (w[:, None] * y).sum(0) / w.sum()
        return 1 - (w[:, None] * (y - pp) ** 2).sum(0) / (w[:, None] * (y - mu) ** 2).sum(0)

    parts = []
    for st in DEPLOYED.split("+"):
        s, _, w = st.partition("*"); parts.append((s, float(w or 1)))
    P = {s: GB.heads_gpu(d, rvt, s)[:, :7] for s, _ in parts}
    ens = lambda use: np.average(np.stack([use.get(s, P[s]) for s, _ in parts]), axis=0, weights=[w for _, w in parts])  # noqa: E731
    base, new = ens({}), ens(refit); res = {}
    for st in stems:
        a, r = r2_shots(P[st]), r2_shots(refit[st]); res[st] = {"orig": a.tolist(), "refit": r.tolist(), "mean_gain": float((r - a).mean())}
        print(f"[ridge] {st} alone: mean-7 {a.mean():.5f} -> {r.mean():.5f} ({(r - a).mean():+.5f}) | " + " ".join(f"{n} {x:+.5f}" for n, x in zip(CONS, r - a)), flush=True)
    a, r = r2_shots(base), r2_shots(new); g = float((r - a).mean())
    rng = np.random.default_rng(0); bs = [float((r2_shots(new, i) - r2_shots(base, i)).mean()) for i in (rng.integers(0, len(us_vt), len(us_vt)) for _ in range(200))]
    lo, hi = np.percentile(bs, [2.5, 97.5]); ok = g >= 3e-4 and lo > 0
    res["ensemble"] = {"base": a.tolist(), "refit": r.tolist(), "gain": g, "ci": [float(lo), float(hi)], "adopt": bool(ok)}
    print(f"[ridge] ensemble (F8 proportions): mean-7 {a.mean():.5f} -> {r.mean():.5f} gain {g:+.5f} CI [{lo:+.5f}, {hi:+.5f}] "
          + " ".join(f"{n} {x:+.5f}" for n, x in zip(CONS, r - a)) + ("  ADOPT" if ok else "  REJECT"), flush=True)
    (OUT / "ridge_refit.json").write_text(json.dumps(res, indent=1)); print("RIDGE_REFIT_DONE", flush=True)


if __name__ == "__main__":
    main()
