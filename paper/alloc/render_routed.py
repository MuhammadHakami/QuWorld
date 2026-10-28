"""Fig. 2 rows for the routed co-processor (Table 1 top block): output fields of classical (gate + linear head), MLP 30,
register 30 and register 50 on one TEST window, from the per-window dumps of adrc.py (QW_DUMP_WINDOW).

Rebuilds the band-wise POD basis exactly as adrc_cache.py (same env: ADRC_BANDS / ADRC_BLEND / ADRC_CKPT / ADRC_VARIANT,
same TRAIN snapshots) and checks it: the projections of the window's blended residual must equal the cached test
coefficients (relative error printed; the row is written only below 2%). Output of an arm =
x_t + sum_b alpha_b P_b(p - x_t) + sum_m c_m phi_m, with that arm's gate alpha and corrections.

    env ADRC_BANDS=5 ADRC_BLEND=5 ADRC_TEST_ALL=1 ADRC_CKPT=... ADRC_VARIANT=vanilla \
        .venv/bin/python paper/alloc/render_routed.py <dataset>:<tag> <label>   -> paper/fig_cache/rows_s23/<dataset>_routed.npz
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adrc_cache as AC  # noqa: E402

ROOT = HERE.parents[1]
DUMP = ROOT / "runs/alloc/fig2_dump"


def window(row, i):
    """The i-th TEST window of an ADRC_TEST_ALL cache (gen_split order), with the same trunk: x_t, p, y (1, C, ...)."""
    ds = row.split(":")[0]
    import cache as CA
    if ds == "MHD_64":
        from cnext_hybrid import build_hybrid
        model = build_hybrid(os.environ.get("ADRC_VARIANT", "vanilla"), model_dir=str(ROOT / "data/cnext_mhd_ckpt"))
        if os.environ.get("ADRC_CKPT"):
            st = torch.load(ROOT / os.environ["ADRC_CKPT"], weights_only=True, map_location="cpu")
            model.load_state_dict(st["model_state_dict"] if "model_state_dict" in st else st)
        model = model.to(AC.PC.DEV).eval(); s = CA.WellND("MHD_64", "test")
    else:
        from well2d_backbone import build, load_checkpoint
        import well2d
        ck = os.environ.get("ADRC_CKPT")
        model = (load_checkpoint(os.environ.get("ADRC_VARIANT", "vanilla"), ds, str(ROOT / ck), "cnext", AC.PC.DEV) if ck
                 else build("vanilla", ds).to(AC.PC.DEV)).eval()
        s = well2d.Split(ds, "test")
    allp = [(k, t) for k in range(s.n) for t in range(3, s.T - 1)]           # ADRC_TEST_ALL: every window, sorted
    xs, ys = s.window(*allp[i]); C = s.C
    xb, yb = xs[None].to(AC.PC.DEV), ys[None].to(AC.PC.DEV)
    with torch.no_grad():
        return xb[:, 3 * C:], model(xb), yb


@torch.no_grad()
def main() -> None:
    row, label = sys.argv[1], sys.argv[2]; key = row.replace(":", "__"); nb = int(os.environ["ADRC_BANDS"])
    alpha_fit, H = AC.fit_blend(row, nb)
    snaps, shape = None, None
    for xt, dx, p, y, _ in AC.blended(AC.gen_split(row, "train", AC.N_SNAP), H):
        shape = tuple(y.shape[2:]); r = (y - p).flatten(2); var = y.flatten(2).var(-1, unbiased=False) + 1e-7
        w = (r / (var * r.shape[-1]).sqrt()[..., None]).half()
        snaps = [[] for _ in range(r.shape[1])] if snaps is None else snaps
        for c in range(r.shape[1]):
            snaps[c].append(w[:, c])
    masks = AC.band_masks(shape, nb)
    Phi = torch.stack([AC.band_pod(torch.cat(sc).float(), shape, masks)[0].half() for sc in snaps])        # (C, D, M)
    d0 = np.load(DUMP / f"{key}_classical.npz"); wi = int(d0["window"])
    xt, p, y = window(row, wi); dims = tuple(range(2, y.dim()))
    pb = xt + torch.fft.ifftn(H * torch.fft.fftn(p - xt, dim=dims), dim=dims).real
    a = torch.einsum("bcd,cdm->bcm", (y - pb).flatten(2).half(), Phi).float()[0].cpu()
    ac = torch.load(AC.OUT / key / "test.pt", weights_only=False, mmap=True)["a"][wi].float()
    rel = float((a - ac).norm() / ac.norm()); print(f"[render] {key} window {wi}: basis check rel err {rel:.4f}", flush=True)
    assert rel < 0.02, "rebuilt basis / window do not match the cache"
    Pk = torch.fft.fftn(p - xt, dim=dims); out = {"label": label, "metric": "VRMSE", "kind": "field", "sample": wi}
    c0 = 0; var = y.flatten(2).var(-1, unbiased=False)
    for arm in ("classical", "mlp", "qubits", "qubits50"):
        f = DUMP / f"{key}_{arm}.npz"
        if not f.exists():
            continue
        d = np.load(f); al = torch.tensor(d["alpha"], device=y.device, dtype=torch.float32)                   # (C, B)
        filt = sum(al[:, b].reshape(-1, *[1] * len(shape)) * mk for b, mk in enumerate(masks))
        blend = xt + torch.fft.ifftn(filt * Pk, dim=dims).real
        corr = torch.einsum("cdm,cm->cd", Phi.float(), torch.tensor(d["corr"], device=y.device)).reshape(1, -1, *shape)
        pred = blend + corr
        vr = float(((pred - y).flatten(2).pow(2).mean(-1) / var).sqrt().mean())
        sl = (pred[0, c0] if len(shape) == 2 else pred[0, c0, shape[0] // 2]).cpu().numpy()
        out[f"pred_{arm}"] = sl; out[f"m_{arm}"] = vr
        print(f"[render] {key} {arm}: window VRMSE {vr:.4f} (run TEST {float(d['test']):.4f})", flush=True)
    out["truth"] = (y[0, c0] if len(shape) == 2 else y[0, c0, shape[0] // 2]).cpu().numpy()
    dst = ROOT / "paper/fig_cache/rows_s23" / f"{row.split(':')[0]}_routed.npz"
    np.savez(dst, **{k: np.asarray(v) for k, v in out.items()}); print("wrote", dst, flush=True)


if __name__ == "__main__":
    main()
