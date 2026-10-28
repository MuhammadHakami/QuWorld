"""Fusion Equilibrium Challenge: public DIII-D test predictions of the entry + the registered shape-head post-processing
(doc 35 §37n axis shift -> §37q boundary warp to the shape heads -> §37w elongation -> li warp toward the head li),
psirz stored float32. Run with STRAT_LABEL_SUFFIX=_v5 (the §37v heads that passed TEST).

Entry and member loader exactly as submit_public.py (ens_select_mlp.json, sigma 0). Shape heads: head_boundary{SUFFIX}
(3 seeds; SUFFIX from STRAT_LABEL_SUFFIX, "" = §37q heads, "_v2" = §37s). Float32 because float16 storage costs 0.0004 S
on VALID (fp16_check.py). The per-frame edits run in a process pool, one shot per task.

    STRAT_CUDA="" [STRAT_LABEL_SUFFIX=_v2] .venv/bin/python paper/sophelio/submit_public_v2.py [--check] [--out submission_v2]
--check: run the pooled pipeline on 3 local TEST shots and compare with the sequential path of bw_test.py (no files).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STRAT = os.path.join(HERE, "..", "strat", "fusion")
sys.path.insert(0, STRAT)
import fx  # noqa: E402,I001  (CPU only)
sys.path.insert(0, HERE)
import ens_select as ES  # noqa: E402
import sota  # noqa: E402
import submit_public as SP  # noqa: E402


def _post(args):
    """One shot: (psi (T,65,65) float64, trunk scalars (T,5), head outputs (T,7+n_ang)) -> edited psi."""
    psi, sc, ph = args
    sys.path.insert(0, STRAT); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import axis_shift as AX
    import boundary_warp as BW
    import kappa_warp as KW
    import li_warp as LW
    from common import AXIS_SIGN
    R, Z, mask = AX.grid(); sign = AXIS_SIGN["DIII-D"]
    shifter = AX.make_shifter(R, Z, mask, sign); liw = LW.make_li_warp(R, Z, mask, sign); bw = BW.make_boundary_warp(R, Z, mask, sign)
    kw = KW.make_kappa_warp(R, Z, mask, sign)
    H = sota.HEADS; ir_, iz_ = H.index("efit_r_axis"), H.index("efit_z_axis")
    out = np.empty_like(psi)
    for k in range(len(psi)):
        a, _ = shifter(psi[k], np.array([sc[k, ir_], sc[k, iz_]]), "RZ", "head")
        a, _ = bw(a, ph[k, 7:])
        a, _ = kw(a, ph[k, 2])                                                  # §37w elongation to the kappa head
        out[k] = liw(a, ph[k, 6], full=True)[0]
    return out


def predict_shots(data, shots, entry_members):
    from bw_test import head_predict
    comps, mean = data.meta["comps"][:ES.KMAX].numpy(), data.meta["mean"].numpy()
    jobs, scal = [], []
    for s in shots:
        rows = np.nonzero(data.shot == s)[0]
        pc = np.mean([mb[s][0] for mb in entry_members], axis=0); sc = np.mean([mb[s][1] for mb in entry_members], axis=0)
        jobs.append(((pc @ comps + mean).reshape(-1, 65, 65).astype(np.float64), sc, head_predict(data, rows))); scal.append(sc)
    with ProcessPoolExecutor(max_workers=fx.N_WORKERS, mp_context=mp.get_context("spawn")) as ex:
        psis = list(ex.map(_post, jobs))
    return psis, scal


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--out", default="submission_v2")
    a = ap.parse_args()
    sel = json.loads(SP.ENTRY.read_text()); vf = sel["valid_full"]; sigma = float(max(vf, key=lambda s: vf[s]["S"]))
    assert sigma == 0.0, "the post-processing was validated without temporal smoothing"
    trunks = sota.ROOT / "runs/nas/sophelio/trunks"
    import torch
    sota.DEV = torch.device("cpu")                                              # CPU inference (the GPU runs registered jobs)
    train = sota.Data()
    if a.check:
        shots = train.shots_of(2)[:3]
        members = [ES.member(str(trunks / m), train, shots) for m in sel["members"]]
        psis, _ = predict_shots(train, shots, members)
        from bw_test import head_predict
        comps, mean = train.meta["comps"][:ES.KMAX].numpy(), train.meta["mean"].numpy()
        for s, p in zip(shots, psis):
            rows = np.nonzero(train.shot == s)[0]
            pc = np.mean([mb[s][0] for mb in members], axis=0); sc = np.mean([mb[s][1] for mb in members], axis=0)
            ref = _post(((pc @ comps + mean).reshape(-1, 65, 65).astype(np.float64), sc, head_predict(train, rows)))
            print(f"[check] TEST shot {s}: max |pooled - sequential| = {np.abs(p - ref).max():.3e}", flush=True)
        return
    out = SP.OUT.parent / a.out; out.mkdir(parents=True, exist_ok=True)
    pub = SP.PublicData(train.meta); shots = pub.shots()
    members = [ES.member(str(trunks / m), pub, shots) for m in sel["members"]]
    psis, scal = predict_shots(pub, shots, members)
    H = sota.HEADS; preds = {}
    for i, psi, sc in zip(shots, psis, scal):
        T = int((pub.shot == i).sum())
        preds[f"shot_{i:04d}_psirz"] = np.asarray(psi, np.float32).reshape(T, 65, 65)
        preds[f"shot_{i:04d}_q95"] = np.asarray(sc[:, H.index("efit_q95")], np.float32)
        preds[f"shot_{i:04d}_betaN"] = np.asarray(sc[:, H.index("efit_beta_n")], np.float32)
        if not all(np.isfinite(preds[f"shot_{i:04d}_{k}"]).all() for k in ("psirz", "q95", "betaN")):
            raise RuntimeError(f"shot {i}: non-finite prediction")
    npz = out / "diii_d_public_test.npz"
    np.savez_compressed(npz, **preds)
    with zipfile.ZipFile(out / "submission.zip", "w", zipfile.ZIP_STORED) as z:
        z.write(npz, arcname=npz.name)
    info = {"entry": SP.ENTRY.name, "members": sel["members"], "sigma": sigma, "heads": f"head_boundary{os.environ.get('STRAT_LABEL_SUFFIX', '')}",
            "post": ["s37n RZ-head", "s37q boundary warp", "s37w elongation", "li warp (head li)"], "dtype": "float32", "shots": len(shots),
            "frames": int(len(pub.X)), "npz_bytes": npz.stat().st_size}
    (out / "build.json").write_text(json.dumps(info, indent=1)); print("[submit-v2] built", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
