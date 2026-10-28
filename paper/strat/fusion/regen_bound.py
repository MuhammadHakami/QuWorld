"""Bound |S(regenerated) - S(submitted)| for the F21K private fold without labels: scorer-derived LCFS + 7 scalars on both files."""
import sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
Q = "/home/mohhakami/Documents/paper/quworld"; FS = Q + "/baseline/sophelio-starter/fusion_scoring"
sys.path.insert(0, FS); sys.path.insert(0, Q + "/baseline/sophelio-starter")
A_P = Q + "/runs/sophelio_sota/submission_F21Kpriv/diii_d_private_test.npz"; B_P = Q + "/runs/sophelio_sota/submission_F21RPK/diii_d_private_test.npz"
NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]

def work(items):
    import local_score as LS
    from contour import symmetric_hausdorff
    from derive import derive_frame
    from lcfs import extract_lcfs_with_sign, major_radius
    m = np.load(FS + "/masks/d3d_envelope.npz"); R, Z = m["grid_R"], m["grid_Z"]; mc = m["mask_coarse"].astype(bool); mf = mc.astype(np.float64)
    A, B = np.load(A_P), np.load(B_P); sgn = LS.AXIS_SIGN[LS.MACHINE] * 1.0; out = []
    for key, k in items:
        res = []
        for F in (A, B):
            p = F[key][k].astype(np.float64).reshape(65, 65)
            ex, _ = extract_lcfs_with_sign(p, R, Z, LS.MACHINE, mc, mf, n_iter=LS.N_ITER, axis_sign=sgn)
            v = derive_frame(p, R, Z, LS.MACHINE, mc, mf, contour=ex, with_li=True, axis_sign=sgn)
            res.append((ex, np.array([v[n] for n in NAMES])))
        (ea, va), (eb, vb) = res
        if ea is None or eb is None: h = np.nan if (ea is None and eb is None) else 1.0
        else: h = symmetric_hausdorff(ea, eb) / major_radius(ea)
        out.append((h, va, vb))
    return out

if __name__ == "__main__":
    A, B = np.load(A_P), np.load(B_P); keys = [k for k in A.files if k.endswith("_psirz")]
    md, nfr = [], 0
    for key in keys:
        a, b = A[key], B[key]; d = np.abs(a.astype(np.float64) - b.astype(np.float64)).reshape(len(a), -1).max(1); nfr += len(a)
        md += [(float(x), key, i) for i, x in enumerate(d)]
    md.sort(reverse=True); dd = np.array([x for x, *_ in md])
    print(f"frames {nfr}; per-frame max|dpsi|: >1e-3 {int((dd>1e-3).sum())}, >1e-4 {int((dd>1e-4).sum())}, >1e-5 {int((dd>1e-5).sum())}, ==0 {int((dd==0).sum())}", flush=True)
    rng = np.random.default_rng(0); top = [(k, i) for _, k, i in md[:2000]]; rnd = [(md[j][1], md[j][2]) for j in rng.choice(len(md), 2000, replace=False)]
    with ProcessPoolExecutor(8) as ex:
        T = sum(ex.map(work, [top[i::8] for i in range(8)]), []); Rr = sum(ex.map(work, [rnd[i::8] for i in range(8)]), [])
    h = np.array([t[0] for t in T]); dv = np.array([t[1] - t[2] for t in T]); var = np.nanvar(np.array([t[1] for t in Rr]), 0)
    print(f"top-2000 frames: LCFS(regen vs sub)/R  mean {np.nanmean(h):.2e} max {np.nanmax(h):.2e}; frames with a contour lost on one side {int((h == 1.0).sum())}")
    # D_LCFS: mean over all frames; |dD_k| <= Hausdorff(regen, sub)/R by the triangle inequality; frames outside the top 2000 differ even less
    dD = np.nansum(h) / nfr + (nfr - 2000) * np.nanmin(h) / nfr
    print(f"bound |dD_LCFS| <= {dD:.2e} -> |dS_D| <= {0.10 * dD:.2e}")
    n_rest = int((dd > 0).sum()) - 2000; print(f"2000th largest per-frame max|dpsi| {dd[1999]:.2e}; other differing frames {n_rest}")
    one_minus_r2 = np.array([0.016, 0.01, 0.024, 0.01, 0.01, 0.01, 0.02])   # generous VT 1 - R2 per scalar (kappa 0.976, li 0.980, R_axis 0.984)
    tot = 0.0
    for j, n in enumerate(NAMES):
        q = (np.nansum(dv[:, j] ** 2) + n_rest * np.nanmin(np.abs(dv[:, j])) ** 2) / (nfr * var[j])   # top-2000 exact + the other differing frames at most the smallest top-2000 change
        b = 2 * np.sqrt(one_minus_r2[j] * q) + q; tot += b
        print(f"  {n:8s} max|dv| {np.nanmax(np.abs(dv[:, j])):.2e}  dR2 bound {b:.2e}  frames |dv|>1e-3: {int((np.abs(dv[:, j]) > 1e-3).sum())}")
    print(f"bound |dS_cons| <= 0.20 x mean = {0.20 * tot / 7:.2e}; |dS_psi| <= {0.55 * (2 * np.sqrt(1e-4 * 5.35e-10) + 5.35e-10):.2e}")
    print("REGEN_BOUND_DONE")
