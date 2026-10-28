"""Label-free comparison of two public submissions (doc 35, F17K vs F16K): per frame the 7 psi-derived consistency scalars of each,
rms difference / VT label sd, split by the shot's Ip sign (canon.shot_sign), plus the relative psi rms difference.
    .venv/bin/python paper/strat/fusion/pub_scalar_diff.py F17K F16K
"""
import sys
import numpy as np
import torch
ROOT = "/home/mohhakami/Documents/paper/quworld"
sys.path.insert(0, ROOT + "/paper/strat/fusion"); sys.path.insert(0, ROOT + "/paper/sophelio")
import fx, gpu_pipe as GP, axis_shift as AX
from gpu_post import GPUPost
import sota
sota.DEV = torch.device("cuda")
import canon, submit_public as SP
d = sota.Data(); pub = SP.PublicData(d.meta); sg = np.asarray(canon.shot_sign(pub, public=True))
vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); sd = np.nanstd(vt["y"][:, :7], 0)
A, B = (np.load(f"{ROOT}/runs/sophelio_sota/submission_{t}/diii_d_public_test.npz") for t in sys.argv[1:3])
R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask)
def der(x):
    p = torch.tensor(x.astype(np.float64), device="cuda"); out = []
    for c in GP.chunks(len(p), 4096):
        o = gp.g.derive(p[c]); out.append(torch.stack([o[k] for k in GP.NAMES], 1).cpu().numpy())
    return np.concatenate(out)
keys = sorted(k for k in B.files if k.endswith("_psirz")); da, db, sgn, dpsi = [], [], [], []
print("frames", len(sg), "keys", len(keys), [f.stem for f in pub.files[:2]], keys[:2], flush=True)
i = 0
for k in keys:
    a, b = A[k], B[k]; n = len(a); da.append(der(a)); db.append(der(b)); sgn.append(sg[i:i + n]); assert len(set(sg[i:i + n])) == 1, k; i += n
    dpsi.append(np.sqrt(((a - b) ** 2).reshape(n, -1).mean(1)) / (np.sqrt((b ** 2).reshape(n, -1).mean(1)) + 1e-12))
da, db, dpsi = np.concatenate(da), np.concatenate(db), np.concatenate(dpsi); sgn = np.concatenate(sgn)
print("sign array len", len(sg), "frames", len(da), "reversed frames", int((sgn < 0).sum()), flush=True)
for nm, m in (("all", np.ones(len(da), bool)), ("normal", sgn > 0), ("reversed", sgn < 0)):
    x = (da[m] - db[m]) / sd; ok = np.isfinite(x)
    rms = np.sqrt(np.nanmean(np.where(ok, x ** 2, np.nan), 0)); ch = (np.abs(np.nan_to_num(x)) > 1e-6).mean(0)
    print(f"[pubdiff] {nm:8s} n {m.sum():6d} rel psi rms {np.mean(dpsi[m]):.2e} | rms/sd " + " ".join(f"{q} {v:.4f}" for q, v in zip(GP.NAMES, rms))
          + " | changed " + " ".join(f"{v:.2f}" for v in ch), flush=True)
