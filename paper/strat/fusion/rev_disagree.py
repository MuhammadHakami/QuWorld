"""§45s (label-free): on reversed-Ip frames, how far each non-canonical component sits from the canonical MLP heads, public vs VT.
F16K showed that the VT router delta understated the public effect ~4x (public Z_axis +0.0026). The ratio public / VT of the mean squared
disagreement (canonical MLP vs non-canonical MLP) is the calibration of that; the same ratio for the register part tells whether routing
the register heads too is likely to pay on public. Inputs only; no labels used on the public set.
    .venv/bin/python paper/strat/fusion/rev_disagree.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

import fx
from vt_eval import SC

REG = "head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2+head_boundary_v8_J1QH_s1*4+head_boundary_v8_J1QH_s2*4+head_boundary_v8_J1QH_s3*4"


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import canon
    import gpu_build as GB
    import submit_public as SP
    d = sota.Data(); pub = SP.PublicData(d.meta)
    vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); vrows = vt["rows"]; nt = 7
    sv = np.asarray(canon.shot_sign(d))[vrows] < 0; sp = np.asarray(canon.shot_sign(pub, public=True)) < 0
    prow = np.nonzero(sp)[0]; vrow = vrows[sv]
    print(f"[rev] reversed frames: VT {len(vrow)} ({sv.mean():.3f}), public {len(prow)} ({sp.mean():.3f})", flush=True)
    sd = vt["y"][:, :nt].std(0)                                                            # scalar scale (VT labels; for normalisation only)
    sets = {"VT": (d, vrow, "head_boundary_v8_J1RL_s1", "head_boundary_v8_J1RLCS_s1"),
            "public": (pub, prow, "+".join(f"head_boundary_v8_J1RLHA_s{s}" for s in range(1, 9)),
                       "+".join(f"head_boundary_v8_J1RLHACS_s{s}" for s in range(1, 5)))}
    out = {}
    for name, (dd, rows, nc, cs) in sets.items():
        pc = GB.heads_gpu(dd, rows, cs)[:, :nt].astype(np.float64); pn = GB.heads_gpu(dd, rows, nc)[:, :nt].astype(np.float64)
        pr = GB.heads_gpu(dd, rows, REG)[:, :nt].astype(np.float64)
        msd = lambda a, b: (((a - b) / sd) ** 2).mean(0)                                    # noqa: E731
        out[name] = {"mlp_nc_vs_cs": msd(pn, pc).tolist(), "reg_vs_cs": msd(pr, pc).tolist(), "reg_vs_mlp_nc": msd(pr, pn).tolist()}
        print(f"[rev] {name}: mean sq. disagreement / sd^2 per scalar ({' '.join(SC)})", flush=True)
        for k, v in out[name].items():
            print(f"[rev]   {k:14s} " + " ".join(f"{x:.4f}" for x in v) + f"  mean {np.mean(v):.4f}", flush=True)
    for k in out["VT"]:
        r = np.array(out["public"][k]) / np.maximum(np.array(out["VT"][k]), 1e-12)
        print(f"[rev] ratio public / VT {k:14s} " + " ".join(f"{x:.2f}" for x in r) + f"  (of means {np.mean(out['public'][k]) / np.mean(out['VT'][k]):.2f})", flush=True)
    (fx.OUT / "rev_disagree.json").write_text(json.dumps(out, indent=1)); print("REV_DONE", flush=True)


if __name__ == "__main__":
    main()
