"""Doc 35 §45e diagnostic (label-free on public): why is our public Z_axis 0.9910 (every build since F2) while the top four reach
0.9946-0.9952 and our heads reach 0.9953 on VT? Three TRAIN-only head families that read different input blocks (J1RL: base + thz +
history; J1QPIpi: register PI-KAN, base inputs only; J1QH) predict every VT frame and every public frame. Per scalar: prediction sd
(public vs VT: a smaller public spread lowers R2 at equal error), the VT error of the 3-family mean, and the RMS pairwise disagreement
on VT vs public. A public-only input defect in one block raises that family's disagreement on public far above its VT level; a
harder public fold raises every pair's disagreement alike. Also by plasma-current sign and per public shot (top disagreement shots).
    .venv/bin/python paper/strat/fusion/z_public_diag.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"                                          # before fx (which hides the GPU) and gpucap's probe
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fx  # noqa: E402

CONS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
FAM = {"J1RL": "head_boundary_v8_J1RL_s1", "J1QPIpi": "head_boundary_v8_J1QPIpi_s1", "J1QH": "head_boundary_v8_J1QH_s1"}


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    import submit_public as SP
    d = sota.Data(); vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); keep = d.split[vt["rows"]] != 0
    rvt = vt["rows"][keep]; yvt = vt["y"][keep, :7].astype(np.float64); pub = SP.PublicData(d.meta); rp = np.arange(len(pub.X))
    P = {f: (GB.heads_gpu(d, rvt, st)[:, :7], GB.heads_gpu(pub, rp, st)[:, :7]) for f, st in FAM.items()}
    ip_col = d.fn.index("mag_plasma_current"); neg_v = d.X[rvt, ip_col] < 0; neg_p = pub.X[:, ip_col] < 0
    mv = np.mean([P[f][0] for f in FAM], 0); mp = np.mean([P[f][1] for f in FAM], 0); res = {"neg_frac": [float(neg_v.mean()), float(neg_p.mean())]}
    pairs = [("J1RL", "J1QPIpi"), ("J1RL", "J1QH"), ("J1QPIpi", "J1QH")]
    for j, nm in enumerate(CONS):
        ok = np.isfinite(yvt[:, j]); e = mv[ok, j] - yvt[ok, j]
        r = {"sd_true_vt": float(yvt[ok, j].std()), "sd_pred_vt": float(mv[:, j].std()), "sd_pred_pub": float(mp[:, j].std()),
             "rmse_vt": float(np.sqrt(np.mean(e ** 2))), "r2_vt": float(1 - np.mean(e ** 2) / yvt[ok, j].var())}
        for a, b in pairs:
            dv = P[a][0][:, j] - P[b][0][:, j]; dp = P[a][1][:, j] - P[b][1][:, j]
            r[f"dis_{a}_{b}"] = [float(np.sqrt(np.mean(dv ** 2))), float(np.sqrt(np.mean(dp ** 2)))]
            r[f"dis_{a}_{b}_neg"] = [float(np.sqrt(np.mean(dv[neg_v] ** 2))), float(np.sqrt(np.mean(dp[neg_p] ** 2)))]
        res[nm] = r
        print(f"[zdiag] {nm:>7}: sd true VT {r['sd_true_vt']:.4f} pred VT {r['sd_pred_vt']:.4f} pub {r['sd_pred_pub']:.4f} | VT rmse {r['rmse_vt']:.4f} "
              f"R2 {r['r2_vt']:.5f} | disagreement VT -> pub: " + " ".join(f"{a}-{b} {r[f'dis_{a}_{b}'][0]:.4f}->{r[f'dis_{a}_{b}'][1]:.4f} "
                                                                            f"(x{r[f'dis_{a}_{b}'][1] / r[f'dis_{a}_{b}'][0]:.2f})" for a, b in pairs), flush=True)
    j = CONS.index("Z_axis"); dz = np.abs(P["J1RL"][1][:, j] - P["J1QPIpi"][1][:, j]); per = []
    for s in np.unique(pub.shot):
        m = pub.shot == s; per.append((float(np.sum(dz[m] ** 2)), int(s), int(m.sum()), float(np.mean(neg_p[m])), float(mp[m, j].min()), float(mp[m, j].max())))
    per.sort(reverse=True); tot = sum(p[0] for p in per); res["Z_top_shots"] = per[:15]
    print(f"[zdiag] public Z disagreement (J1RL - J1QPIpi) SSE share of the top 5 / 15 shots: {sum(p[0] for p in per[:5]) / tot:.3f} / "
          f"{sum(p[0] for p in per[:15]) / tot:.3f}; top: " + "; ".join(f"shot {p[1]} n {p[2]} neg {p[3]:.0f} Z {p[4]:+.2f}..{p[5]:+.2f} share {p[0] / tot:.3f}"
                                                                     for p in per[:8]), flush=True)
    (fx.OUT / "z_public_diag.json").write_text(json.dumps(res, indent=1)); print("ZDIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
