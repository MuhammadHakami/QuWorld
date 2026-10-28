"""Doc 35 §38f: GPU screen of warp parameters on the 180-shot VALID sample (twin scorer; paired shot bootstrap of S vs
the frozen chain). Arms are JSON per-step overrides for gpu_pipe.post_all.

    .venv/bin/python paper/strat/fusion/warp_screen.py '<name>={"li": {"cap": 0.9}}' ...
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fx  # noqa: E402,I001
import strategies as ST  # noqa: E402
import axis_shift as AX  # noqa: E402
import gpu_pipe as GP  # noqa: E402
from gpu_post import GPUPost  # noqa: E402


def main() -> None:
    sota, d = fx.data(); shots, grp = ST.sample(); pe, nm = ST.load_entry(); f, comps, mean, H = ST.entry_preds(d, sota)
    iq, ib, ir_, iz_ = (H.index(h) for h in ("efit_q95", "efit_beta_n", "efit_r_axis", "efit_z_axis"))
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); pos = {int(r): i for i, r in enumerate(lab["rows_valid"])}
    heads = os.environ.get("STRAT_HEADS", "+".join(f"head_boundary_v8_s{s}" for s in range(1, 7)))
    Ph = np.mean([np.load(fx.OUT / f"{h}.npz")["pred_valid"] for h in heads.split("+")], 0).astype(np.float64)
    rows = np.concatenate([fx.valid_rows(d, s) for s in shots])
    C, S = zip(*(f(s) for s in shots)); C = np.concatenate([c.mean(0) for c in C]); S = np.concatenate([x.mean(0) for x in S])
    R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask); ref = GP.refs(d, shots)
    ph = torch.tensor(Ph[[pos[int(r)] for r in rows]], device=gp.dev); psi0 = torch.tensor((C @ comps + mean).reshape(-1, 65, 65), device=gp.dev)
    ax = torch.tensor(S[:, [ir_, iz_]], device=gp.dev, dtype=torch.float64); q, b = S[:, iq].astype(np.float64), S[:, ib].astype(np.float64)
    base_f = fx.OUT / "parts_GPU-frozen.pkl"; bkw = json.loads(os.environ.get("STRAT_BASE_KW", "{}"))    # §38n: base chain override
    bsteps = tuple(bkw.pop("_steps", ("shift", "bw", "kw", "li")))
    base = pickle.load(open(base_f, "rb")) if base_f.exists() and "STRAT_HEADS" not in os.environ and not bkw and len(bsteps) == 4 else None
    if base is None:
        base = GP.parts_all(gp, GP.post_all(gp, psi0, ax, ph[:, 7:], ph[:, 2], ph[:, 6], steps=bsteps, kw=bkw, tri_t=ph[:, 3:5]),
                            q, b, d, shots, ref=ref)
    sb = np.stack([fx.shot_sums(base[s]) for s in shots]); cb = fx.composite(sb, nm)
    print(f"[s38f] base (frozen chain, twin): S {cb['S']:.5f} cons {cb['consistency']:.5f} dlcfs {cb['dlcfs']:.5f}", flush=True)
    rep = {}
    for arg in sys.argv[1:]:
        name, js = arg.split("=", 1); kw = json.loads(js); t = time.time()
        steps = tuple(kw.pop("_steps", ("shift", "bw", "kw", "li")))
        parts = GP.parts_all(gp, GP.post_all(gp, psi0, ax, ph[:, 7:], ph[:, 2], ph[:, 6], steps=steps, kw=kw, tri_t=ph[:, 3:5]),
                             q, b, d, shots, ref=ref)
        if os.environ.get("STRAT_OFFICIAL") == "1":                         # official scorer on the arm's psi (CPU workers)
            psi = GP.post_all(gp, psi0, ax, ph[:, 7:], ph[:, 2], ph[:, 6], steps=steps, kw=kw, tri_t=ph[:, 3:5]).cpu().numpy().reshape(len(psi0), -1)
            off = np.cumsum([0] + [len(fx.valid_rows(d, s)) for s in shots]); sl = {s: slice(off[i], off[i + 1]) for i, s in enumerate(shots)}
            parts = fx.frame_parts(d, shots, lambda s: (psi[sl[s]], q[sl[s]], b[sl[s]]), np.asarray(nm["c_mean"]))
            base = pickle.load(open(fx.OUT / "parts_GPU-frozen-official.pkl", "rb")); sb = np.stack([fx.shot_sums(base[s]) for s in shots])
            cb = fx.composite(sb, nm); print(f"[s38f] official base (GPU-built frozen): S {cb['S']:.5f}", flush=True)
        sn = np.stack([fx.shot_sums(parts[s]) for s in shots]); cn = fx.composite(sn, nm)
        bt = fx.paired_boot(sb, sn, nm, groups=np.asarray([grp[s] for s in shots]) if isinstance(grp, dict) else grp)
        rep[name] = dict(kw=kw, S=cn["S"], boot=bt, cons_each=cn["cons_each"])
        print(f"[s38f] {name}: S {cn['S']:.5f} delta {bt['delta']:+.5f} [{bt['lo']:+.5f}, {bt['hi']:+.5f}] dlcfs {cn['dlcfs']:.5f} | "
              + " ".join(f"{k} {cn['cons_each'][k] - cb['cons_each'][k]:+.4f}" for k in GP.NAMES) + f" | {time.time() - t:.0f}s", flush=True)
    out = fx.OUT / f"s38f_{os.environ.get('STRAT_TAG', 'screen')}.json"
    out.write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
