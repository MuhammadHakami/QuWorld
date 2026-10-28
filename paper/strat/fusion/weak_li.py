"""Why does the li warp miss the head on ~1% of frames? GPU, VALID sample: chain up to the elongation, then the li warp
with diagnostics; for the frames with the largest |li_final - head| report the warp state and the LCFS level change.

    .venv/bin/python paper/strat/fusion/weak_li.py
"""
from __future__ import annotations

import os

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
    ir_, iz_ = H.index("efit_r_axis"), H.index("efit_z_axis")
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); pos = {int(r): i for i, r in enumerate(lab["rows_valid"])}
    Ph = np.mean([np.load(fx.OUT / f"head_boundary_v8_s{s}.npz")["pred_valid"] for s in range(1, 7)], 0).astype(np.float64)
    rows = np.concatenate([fx.valid_rows(d, s) for s in shots]); ph = Ph[[pos[int(r)] for r in rows]]
    C, S = zip(*(f(s) for s in shots)); C = np.concatenate([c.mean(0) for c in C]); S = np.concatenate([x.mean(0) for x in S])
    gt = np.concatenate([pe[s]["cgt"] for s in shots])[:, 6]
    R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask); pht = torch.tensor(ph, device=gp.dev)
    psi = torch.tensor((C @ comps + mean).reshape(-1, 65, 65), device=gp.dev)
    ax = torch.tensor(S[:, [ir_, iz_]], device=gp.dev, dtype=torch.float64)
    pre = GP.post_all(gp, psi, ax, pht[:, 7:], pht[:, 2], pht[:, 6], steps=("shift", "bw", "kw"))
    keys = ["beta", "good", "li0", "li1", "li2", "lev", "lev_after", "li_final", "ok_after"]; diag = {k: [] for k in keys}
    for c in GP.chunks(len(pre), 4096):
        out = gp.li_warp(pre[c], pht[c, 6]); dg = gp.li_diag; o = gp.g.derive(out)
        for k in ("beta", "good", "li0", "li1", "li2", "lev"):
            diag[k].append(dg[k].cpu().numpy())
        diag["lev_after"].append(o["ext"][3].cpu().numpy()); diag["li_final"].append(o["li"].cpu().numpy()); diag["ok_after"].append(o["ok"].cpu().numpy())
    D = {k: np.concatenate(v) for k, v in diag.items()}; head = ph[:, 6]
    gap = np.abs(D["li_final"] - head); top = np.argsort(-np.nan_to_num(gap, nan=9))[: len(gap) // 100]
    dlev = np.abs(D["lev_after"] - D["lev"])
    print(f"[li] frames {len(gap)}; top-1% gap frames: median gap {np.nanmedian(gap[top]):.3f}")
    print(f"[li]   good=False {np.mean(~D['good'][top]):.2f} | |beta| at cap {np.mean(np.abs(D['beta'][top]) >= 0.5 - 1e-9):.2f} | "
          f"LCFS level moved >1e-4 {np.mean(dlev[top] > 1e-4):.2f} (all frames {np.mean(dlev > 1e-4):.3f}) | ok_after False {np.mean(~D['ok_after'][top]):.2f}")
    print(f"[li]   |li2 - head| (fixed-contour li after the warp) median {np.nanmedian(np.abs(D['li2'][top] - head[top])):.4f} vs re-extracted {np.nanmedian(gap[top]):.4f}")
    print(f"[li]   head - li0 median {np.nanmedian(np.abs(head[top] - D['li0'][top])):.3f} (all {np.nanmedian(np.abs(head - D['li0'])):.3f}); "
          f"|gt - head| median {np.nanmedian(np.abs(gt[top] - head[top])):.3f} (all {np.nanmedian(np.abs(gt - head)):.3f})")
    np.savez(fx.OUT / "weak_li.npz", top=top, **D)


if __name__ == "__main__":
    main()
