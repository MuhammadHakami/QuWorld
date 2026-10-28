"""Sophelio: where the Consistency residual comes from (VALID only). For saved trunks (mean-ensembled) on the NAS's fixed
40-shot VALID subset, re-derive every frame's seven psi-derived scalars exactly as the official scorer does, and split
each scalar's SS_res into (a) frames where the derivation FAILED on the prediction (the scorer substitutes the fold mean)
and (b) ordinary frames, plus the share held by the worst 1 % of ordinary frames.

    .venv/bin/python paper/sophelio/cons_diag.py --trunks runs/nas/sophelio/trunks/trunk1_t0.pt ...
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sota  # noqa: E402


def _frames(args):
    psi_pred, ref, psi_sign = args
    import train_hybrid  # noqa: F401  (puts the vendored scorer on sys.path in this spawned worker)
    import local_score as LS
    from common import AXIS_SIGN, CONS_SCALARS
    from derive import derive_frame
    from lcfs import extract_lcfs_with_sign
    mask = np.load(LS.HERE / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]; mc = mask["mask_coarse"].astype(bool); mf = mc.astype(np.float64)
    _, cons_gt, cmask, _ = ref
    axis_sign = AXIS_SIGN[LS.MACHINE] * psi_sign
    rows = []
    for k in range(len(psi_pred)):
        need = cmask[k]
        if not need.any():
            continue
        ex, _ = extract_lcfs_with_sign(psi_pred[k], R, Z, LS.MACHINE, mc, mf, n_iter=LS.N_ITER, axis_sign=axis_sign)
        vals = derive_frame(psi_pred[k], R, Z, LS.MACHINE, mc, mf, contour=ex, with_li=bool(need[LS.LI_IDX]), axis_sign=axis_sign)
        for j in np.nonzero(need)[0]:
            rows.append((k, int(j), float(cons_gt[k, j]), float(vals[CONS_SCALARS[j]]), ex is None))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--trunks", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--split", type=int, default=1)
    a = ap.parse_args()
    import train_hybrid as TH
    from common import CONS_SCALARS
    import postproc as PP
    data = sota.Data()
    shots = data.shots_of(a.split); pick = [shots[i] for i in np.random.default_rng(1).permutation(len(shots))[:a.n]]
    preds = [PP.load(p, data) for p in a.trunks]
    mask = np.load(TH.STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]; mc = mask["mask_coarse"].astype(bool); mf = mc.astype(np.float64)
    jobs = []
    for s in pick:
        rows = np.nonzero(data.shot == s)[0]
        psi = np.mean([f(rows)[0] for f in preds], axis=0).reshape(-1, 65, 65).astype(np.float64)
        gt = data.psi_of(rows).reshape(-1, 65, 65).astype(np.float32)
        ref = TH._ref_one((gt, data.meta["names"][s], str(sota.ROOT / "runs" / "sophelio_sota" / "ref_cache"), R, Z, mc, mf))
        jobs.append((psi, ref, 1))
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=TH.N_WORKERS, mp_context=mp.get_context("spawn")) as ex:
        per_shot = list(ex.map(_frames, jobs))
    flat = np.array([(si, len(jobs[si][0])) + r[:4] for si, rows in enumerate(per_shot) for r in rows], dtype=np.float64)
    np.save(sota.ROOT / "runs" / "sophelio_sota" / "cons_diag_frames.npy", flat)   # shot#, T, frame, scalar j, gt, pred
    out = {}
    for j, name in enumerate(CONS_SCALARS):
        g = np.array([r[2] for rows in per_shot for r in rows if r[1] == j]); p = np.array([r[3] for rows in per_shot for r in rows if r[1] == j])
        lcfs_fail = np.array([r[4] for rows in per_shot for r in rows if r[1] == j])
        fail = ~np.isfinite(p); p = np.where(fail, g.mean(), p)
        res = (g - p) ** 2; tot = ((g - g.mean()) ** 2).sum()
        ok = res[~fail]; top = np.sort(ok)[::-1][:max(1, len(ok) // 100)].sum()
        out[name] = dict(frames=int(len(g)), fails=int(fail.sum()), lcfs_fails=int(lcfs_fail.sum()), r2=float(1 - res.sum() / tot),
                         r2_ok_only=float(1 - ok.sum() / ((g[~fail] - g[~fail].mean()) ** 2).sum()),
                         share_fail=float(res[fail].sum() / res.sum()), share_top1pct_ok=float(top / res.sum()))
        print(f"[cons-diag] {name:8s} frames {len(g)} fails {fail.sum()} (lcfs {lcfs_fail.sum()}) R2 {out[name]['r2']:.4f} "
              f"R2(ok) {out[name]['r2_ok_only']:.4f} | SS_res share: fails {out[name]['share_fail']:.2f} top-1% {out[name]['share_top1pct_ok']:.2f}", flush=True)
    (sota.ROOT / "runs" / "sophelio_sota" / "cons_diag.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
