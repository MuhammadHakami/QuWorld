"""§48n stage 0 (doc 35, registered before any result): label-side diagnostics on the calibration VT shots C = VT \\ (A150 u B150).

1. Test-time crop averaging for the QUOTA-Fusion temporal heads: tile stride 8 (deployed, crop / 2) vs 4 / 2 / 1 (QW_QF_STRIDE).
   psi: PSIV60 s1 + s2 coefficient SSE on components 0-31 (truth d.C); shape: V60 s1 + s2 R2 on the 7 main scalars (boundary_labels_v8_vt).
2. Coefficient shrinkage (§42k) for the F20K-equivalent mix PQmixV32: per-component slope / means fitted on one half of C and applied
   to the other (2-fold by shot), components >= 32 (MLP-only there) - out-of-fold SSE ratio vs the raw mix.
    .venv/bin/python paper/strat/fusion/qf_tta_probe.py   -> runs/strat/fusion/s48n_probe.json (+ coef_reliability_s48n.npz)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np
import torch

import fx
import gpu_build as GB

PSIH = "head_boundary_v8_PSIV60_s1+head_boundary_v8_PSIV60_s2"
SHAPEH = "head_boundary_v8_V60_s1+head_boundary_v8_V60_s2"
MLP = ["PQ2.pt", "LS_ctrl.pt"]
K32 = 32


def calib_shots(d) -> tuple[list[int], list[int], list[int]]:
    vt = [int(s) for s in np.unique(d.shot[d.split != 0])]
    a = [int(x) for x in re.search(r'--vt-shots "([0-9,]+)"', (fx.OUT / "s44h_jgn_end.sh").read_text()).group(1).split(",")]
    b = [int(x) for x in (fx.OUT / "vt_shots_B150.txt").read_text().strip().split(",")]
    return sorted(set(vt) - set(a) - set(b)), a, b


def mlp_coefs(d, shots, ES) -> np.ndarray:
    """Mean PCA coefficients of the PQmix MLP trunks on the shots (row order = concatenated shots), as gpu_build.members."""
    outs = []
    for p in MLP:
        cf = torch.load(GB.SOTA / p, map_location="cpu", weights_only=False)["cfg"]
        dd = GB.fixed(d, GB.mf_name(p, {})) if cf.get("inputs") == "magfix" else d
        dd = GB.canonical(dd) if cf.get("canon") else dd
        dd = GB.with_hfull(dd, "qphys2" if int(cf.get("qphys", 0)) == 2 else "qphys") if cf.get("qphys") else dd
        dd = GB.with_hfull(dd, "hfull") if cf.get("hfull") else dd
        m = ES.member(str(GB.SOTA / p), dd, shots); outs.append(np.concatenate([m[s][0][:, :64] for s in shots]))
    return np.mean(outs, 0)


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import ens_select as ES
    t0 = time.time(); d = sota.Data(); C, A, B = calib_shots(d)
    rows = np.concatenate([np.nonzero(d.shot == s)[0] for s in C]); shot_of = np.asarray(d.shot)[rows]
    ct = np.asarray(d.C[rows][:, :64], np.float64)
    z = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); pos = {r: i for i, r in enumerate(z["rows"])}
    ys = np.array([z["y"][pos[r]] if r in pos else np.full(z["y"].shape[1], np.nan) for r in rows], np.float64)[:, :7]
    out = {"stage": "48n-0", "calib_shots": len(C), "rows": int(len(rows)), "tta": {}}
    print(f"[s48n] C {len(C)} shots ({len(rows)} rows), A {len(A)}, B {len(B)}; setup {time.time() - t0:.0f}s", flush=True)
    for stride in (8, 4, 2, 1):
        os.environ["QW_QF_STRIDE"] = str(stride); t1 = time.time()
        pp = GB.heads_gpu(d, rows, PSIH)[:, :64]; sse = float(((pp[:, :K32] - ct[:, :K32]) ** 2).sum())
        ph = GB.heads_gpu(d, rows, SHAPEH)[:, :7]; ok = np.isfinite(ys)
        r2 = [float(1 - np.nansum(np.where(ok[:, j], (ph[:, j] - ys[:, j]) ** 2, np.nan)) / np.nansum((ys[ok[:, j], j] - ys[ok[:, j], j].mean()) ** 2))
              for j in range(7)]
        out["tta"][stride] = {"psi_sse_0_31": sse, "shape_r2": r2, "shape_r2_mean7": float(np.mean(r2)), "seconds": round(time.time() - t1)}
        print(f"[s48n] stride {stride}: psi SSE 0-31 {sse:.4e} | V60 R2 mean7 {np.mean(r2):.5f} " + " ".join(f"{x:.4f}" for x in r2)
              + f" ({time.time() - t1:.0f}s)", flush=True)
    os.environ.pop("QW_QF_STRIDE")
    b8 = out["tta"][8]
    for s_, v in out["tta"].items():
        v["psi_ratio_vs8"] = v["psi_sse_0_31"] / b8["psi_sse_0_31"]; v["shape_1mr2_ratio_vs8"] = (1 - v["shape_r2_mean7"]) / (1 - b8["shape_r2_mean7"])
    # 2. shrinkage of the F20K-equivalent mix (components >= 32 come from the MLP trunks alone)
    pm = mlp_coefs(d, C, ES); pv = GB.heads_gpu(d, rows, PSIH)[:, :64]
    mix = pm.copy(); mix[:, :K32] = (2 * pm[:, :K32] + 2 * pv[:, :K32]) / 4                         # PQ2, LS_ctrl, V s1, V s2 (nanmean)
    us = np.unique(shot_of); half = np.isin(shot_of, us[np.random.default_rng(2026).permutation(len(us))[: len(us) // 2]])
    def fit(m):                                                                              # per-component LS slope + means
        mp, mt = mix[m].mean(0), ct[m].mean(0); dc = mix[m] - mp
        return np.clip((dc * (ct[m] - mt)).sum(0) / np.maximum((dc ** 2).sum(0), 1e-30), 0, 1), mt, mp
    sse_raw, sse_shr = 0.0, 0.0
    for m in (half, ~half):
        sl, mt, mp = fit(~m); sl[:K32] = 1.0; sh = mt + sl * (mix[m] - mp); sh[:, :K32] = mix[m][:, :K32]
        sse_raw += float(((mix[m] - ct[m]) ** 2).sum()); sse_shr += float(((sh - ct[m]) ** 2).sum())
    hi_raw = float(((mix[:, K32:] - ct[:, K32:]) ** 2).sum())
    sl, mt, mp = fit(np.ones(len(rows), bool))
    r2c = 1 - ((mix - ct) ** 2).sum(0) / ((ct - ct.mean(0)) ** 2).sum(0)
    np.savez(fx.OUT / "coef_reliability_s48n.npz", r2=r2c, slope=sl, mean_true=mt, mean_pred=mp)
    out["shrink"] = {"oof_sse_ratio_all64": sse_shr / sse_raw, "sse_raw_all64": sse_raw, "sse_raw_hi32": hi_raw,
                     "hi_share_of_sse": hi_raw / sse_raw, "r2_blocks": {f"{a}-{a + 16}": float(r2c[a:a + 16].mean()) for a in range(0, 64, 16)},
                     "slope_blocks": {f"{a}-{a + 16}": float(sl[a:a + 16].mean()) for a in range(0, 64, 16)}}
    print(f"[s48n] shrink (comps >= 32, 2-fold OOF): SSE ratio {sse_shr / sse_raw:.5f}; comps 32-63 hold {hi_raw / sse_raw:.4f} of the mix SSE; "
          f"R2 blocks {out['shrink']['r2_blocks']}; slopes {out['shrink']['slope_blocks']}", flush=True)
    (fx.OUT / "s48n_probe.json").write_text(json.dumps(out, indent=1))
    print(f"S48N_PROBE_DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
