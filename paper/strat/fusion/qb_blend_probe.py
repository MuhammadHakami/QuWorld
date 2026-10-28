"""§48s stage 0 (doc 35, registered before any result): re-fit the q/b head weight (qbw, 0.8 since §42a on Oct 11) at label level on the
calibration VT shots C = VT \\ (A150 u B150), separately for q95 and betaN.
    sc = mean q95 / betaN of the TRAIN-only entry trunks (gpu_build's `ent`), qh = q/b heads; blend (1 - w) sc + w qh.
Two TRAIN-only head analogues: QH s1-6 (the q/b heads of every VT read) and route QHR2Hu s1 | QHR2CS s1 (the closest TRAIN-only twin of the
production route QHAR2Hu s1-3 | QHAR2HuCS s1-2). Read: R2 at w = 0.8 vs the least-squares w (clipped to [0, 1]), in-sample and 2-fold OOF by shot.
    .venv/bin/python paper/strat/fusion/qb_blend_probe.py   -> runs/strat/fusion/s48s_probe.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

import fx
import gpu_build as GB
from qf_tta_probe import calib_shots

ANALOGUES = {"QH6": "+".join(f"head_qb_QH_s{s}" for s in range(1, 7)),
             "R2Hu": "route:head_qb_QHR2Hu_s1|head_qb_QHR2CS_s1"}
W0 = 0.8


def r2(p, y) -> float:
    return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def best_w(sc, qh, y) -> float:
    """Least-squares w for y ~ sc + w (qh - sc), clipped to [0, 1]."""
    d = qh - sc
    return float(np.clip((d * (y - sc)).sum() / max((d ** 2).sum(), 1e-30), 0, 1))


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import ens_select as ES
    t0 = time.time(); d = sota.Data(); C, _, _ = calib_shots(d)
    rows = np.concatenate([np.nonzero(d.shot == s)[0] for s in C]); shot_of = np.asarray(d.shot)[rows]
    H = sota.HEADS; iq, ib = H.index("efit_q95"), H.index("efit_beta_n")
    y = np.asarray(d.S[rows][:, [sota.LABELS.index("efit_q95"), sota.LABELS.index("efit_beta_n")]], np.float64)   # S columns follow LABELS
    sel = json.loads((GB.SOTA / "ens_select_mlp.json").read_text())
    ent = [ES.member(str(fx.TRUNKS / m), d, C) for m in sel["members"]]
    sc = np.concatenate([np.mean([m[s][1] for m in ent], 0) for s in C])[:, [iq, ib]].astype(np.float64)
    us = np.unique(shot_of); half = np.isin(shot_of, us[np.random.default_rng(2026).permutation(len(us))[: len(us) // 2]])
    out = {"stage": "48s-0", "calib_shots": len(C), "rows": int(len(rows)), "w0": W0, "trunk_r2": [r2(sc[:, j], y[:, j]) for j in (0, 1)], "arms": {}}
    print(f"[s48s] C {len(C)} shots ({len(rows)} rows); trunk R2 q95 {out['trunk_r2'][0]:.5f} betaN {out['trunk_r2'][1]:.5f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    for nm, stems in ANALOGUES.items():
        qh = GB.heads_gpu(d, rows, stems)[:, :2].astype(np.float64); res = {}
        for j, tg in enumerate(("q95", "betaN")):
            s_, h_, y_ = sc[:, j], qh[:, j], y[:, j]; ok = np.isfinite(y_) & np.isfinite(s_) & np.isfinite(h_)
            s_, h_, y_, hf = s_[ok], h_[ok], y_[ok], half[ok]
            wa = best_w(s_, h_, y_); r_w0 = r2(s_ + W0 * (h_ - s_), y_); r_wa = r2(s_ + wa * (h_ - s_), y_)
            sse0 = sse1 = 0.0
            for m in (hf, ~hf):                                          # 2-fold OOF by shot: fit on one half, score the other
                w_ = best_w(s_[~m], h_[~m], y_[~m])
                sse0 += float(((s_[m] + W0 * (h_[m] - s_[m]) - y_[m]) ** 2).sum()); sse1 += float(((s_[m] + w_ * (h_[m] - s_[m]) - y_[m]) ** 2).sum())
            res[tg] = {"w_ls": wa, "r2_w0": r_w0, "r2_wls": r_wa, "r2_head": r2(h_, y_), "oof_sse_ratio": sse1 / sse0,
                       "folds_w": [best_w(s_[~m], h_[~m], y_[~m]) for m in (hf, ~hf)]}
            print(f"[s48s] {nm} {tg}: head R2 {res[tg]['r2_head']:.5f}; w0 {W0} R2 {r_w0:.5f} -> w* {wa:.3f} R2 {r_wa:.5f} "
                  f"(+{r_wa - r_w0:.5f}); fold w* {res[tg]['folds_w'][0]:.3f} / {res[tg]['folds_w'][1]:.3f}; OOF SSE ratio {sse1 / sse0:.4f}", flush=True)
        res["dS_insample"] = 0.15 * 0.5 * sum(res[t]["r2_wls"] - res[t]["r2_w0"] for t in ("q95", "betaN"))
        out["arms"][nm] = res; print(f"[s48s] {nm}: in-sample S gain (0.15 x mean dR2) {res['dS_insample']:+.6f}", flush=True)
    (fx.OUT / "s48s_probe.json").write_text(json.dumps(out, indent=1))
    print(f"S48S_PROBE_DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
