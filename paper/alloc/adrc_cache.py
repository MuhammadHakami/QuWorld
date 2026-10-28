"""ADR-C v1 caches (35-analog-digital-router-configuration-losses.md §6): per-window POD components of the frozen
trunk's residual, per field, with causal features, so every co-processor loss and metric is exact by orthonormality.

    .venv/bin/python paper/alloc/adrc_cache.py <row>        # -> runs/alloc/adrc_cache/<row>/{basis,train,valid,test}.pt

Basis: top-M POD modes per field (method of snapshots on N_SNAP metric-weighted TRAIN residuals). Per window we store
  a  (C, M)     raw residual coefficients  <r_c, phi_cm>
  f  (C, M, 4)  raw projections of p - x_t, p, x_t, dx on the same modes (causal features)
  E  (C,)       ||r_c||^2 (full field, so energy outside the basis counts as unreachable)
  D  (C,)       metric denominators (VRMSE: npix * (var(y_c) + 1e-7))
  g  (8C,)      global statistics of d, p, x_t, dx (mean, std per channel)
Corrected error: E'_c = E_c - sum_m (a_cm^2 - (a_cm - c_cm)^2), exact.

v2 (doc 35 §8, env ADRC_BANDS=B): frequency-localized components. The metric-weighted residual is split by B dyadic
l-inf FFT shells (Shannon / Littlewood-Paley octaves: b = 0 holds |k|/k_nyq < 2^-(B-1) incl. DC, b = B-1 the top
octave); these are orthogonal projectors summing to I, so POD modes computed inside each band are orthogonal across
bands and the union is one orthonormal basis (every v1 identity stays exact). The M columns per field are the largest
band eigenvalues (greedy, at least M_FLOOR per band), ordered by eigenvalue; basis_meta stores each mode's band. Each
split additionally stores Eb (C, B) = ||P_b r_c||^2 (full field, Parseval), so per band: captured_b <= span_b <= Eb.

Band-gated persistence blend (doc 36, env ADRC_BLEND=B'): a classical, frequency-aware anchor. The trunk prediction is
replaced by p' = x_t + sum_b alpha_cb P_b(p - x_t), alpha per field and band by metric-weighted least squares on the
N_SNAP TRAIN windows (closed form, Parseval). alpha = 1 everywhere recovers the trunk, alpha = 0 persistence. Every
split then also stores E0 = ||y - p||^2 (the unblended trunk) so both anchors are reported.
Per-window gate statistics (ADRC_BLEND must equal ADRC_BANDS, so every POD mode lies inside one gate band): with
d0 = p - x_t, per field and band gY = ||P_b(y - x_t)||^2, gYD = <P_b(y - x_t), P_b d0>, gDD = ||P_b d0||^2 and the causal
gX = ||P_b dx||^2, plus ay = <y - x_t, phi> and fd0 = <d0, phi>. For ANY per-window gate alpha (C, B) the residual energy
is sum_b gY - 2 alpha gYD + alpha^2 gDD and the mode coefficients are ay - alpha_band(m) fd0, exactly (adrc.py --gate).
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import pod_capture as PC  # noqa: E402

M, N_SNAP, M_FLOOR = 128, 1000, 8
SIZES = {"train": 3000, "valid": 600, "test": 1200}
OUT = ROOT / "runs" / "alloc" / "adrc_cache"


def gen_split(row, split, n):
    import os
    """Like pod_capture.sampler but also yields x_t, dx, and supports 'valid' (seed 2). row may be 'dataset:tag' with
    env ADRC_CKPT / ADRC_VARIANT naming a fine-tuned trunk (e.g. the register-in-the-neck rayleigh_benard model)."""
    row = row.split(":")[0]
    if row == "pelzer":
        sys.path.insert(0, str(ROOT / "paper" / "geothermal"))
        import train_pelzer_unet as TP
        x, y, info, tr, va, te = TP.load_1000()
        model = TP.build("vanilla")
        model.load_state_dict(torch.load(ROOT / "runs/pelzer_unet/pretrain_lr0.001_s1337.pt", weights_only=True, map_location="cpu")["model_state_dict"])
        model = model.to(PC.DEV).eval()
        base = y[tr].mean(0, keepdim=True).to(PC.DEV)
        ids = {"train": tr, "valid": va, "test": te}[split][:n]
        for i in range(0, len(ids), 50):
            b = ids[i:i + 50]; xb, yb = x[b].to(PC.DEV), y[b].to(PC.DEV)
            yield base.expand(len(b), -1, -1, -1), xb[:, 1:2], model(xb), yb
        return
    import cache as CA
    if row == "MHD_64":
        import os
        from cnext_hybrid import build_hybrid
        model = build_hybrid(os.environ.get("ADRC_VARIANT", "vanilla"), model_dir=str(ROOT / "data/cnext_mhd_ckpt"))
        if os.environ.get("ADRC_CKPT"):                                   # fine-tuned MHD trunk (runs/phase_b_mhd)
            st = torch.load(ROOT / os.environ["ADRC_CKPT"], weights_only=True, map_location="cpu")
            model.load_state_dict(st["model_state_dict"] if "model_state_dict" in st else st)
        model = model.to(PC.DEV).eval()
        s = CA.WellND("MHD_64", split)
    else:
        from well2d_backbone import build, load_checkpoint
        import os
        import well2d
        ck = os.environ.get("ADRC_CKPT")                                   # comma-separated => mean of the members
        members = [load_checkpoint(os.environ.get("ADRC_VARIANT", "vanilla"), row, str(ROOT / c), "cnext", PC.DEV).eval()
                   for c in ck.split(",")] if ck else [build("vanilla", row).to(PC.DEV).eval()]
        model = members[0] if len(members) == 1 else (lambda xb: torch.stack([m(xb) for m in members]).mean(0))
        s = well2d.Split(row, split)
    C = s.C
    rng = np.random.default_rng({"train": 0, "valid": 2, "test": 3}[split])
    allp = [(k, t) for k in range(s.n) for t in range(3, s.T - 1)]
    pick = np.sort(rng.choice(len(allp), min(n, len(allp)), replace=False))
    strat = int(os.environ.get({"test": "ADRC_TEST_STRAT", "valid": "ADRC_VALID_STRAT"}.get(split, "_"), "0"))
    if strat:                                 # stratified: every trajectory x 8 equal time bins, `strat` windows each
        bins = np.array_split(np.arange(3, s.T - 1), 8)      # (the early-time heavy tail is represented pro rata)
        pick = np.sort([k * (s.T - 4) + int(t) - 3 for k in range(s.n) for b in bins
                        for t in rng.choice(b, min(strat, len(b)), replace=False)])
    bs = 2 if row == "MHD_64" else 8
    for i in range(0, len(pick), bs):
        xs, ys = zip(*(s.window(*allp[j]) for j in pick[i:i + bs]))
        xb, yb = torch.stack(xs).to(PC.DEV), torch.stack(ys).to(PC.DEV)
        xt, xp = xb[:, 3 * C:], xb[:, 2 * C:3 * C]
        yield xt, xt - xp, model(xb), yb


def band_masks(shape, nb):
    """Dyadic l-inf FFT shells over the spatial grid (bool, one per band); symmetric in k -> -k, so projections are real."""
    ks = [torch.fft.fftfreq(n, device=PC.DEV).abs() * 2 for n in shape]                 # 1 = Nyquist
    kn = torch.stack(torch.meshgrid(*ks, indexing="ij")).amax(0)
    edges = [0.0] + [2.0 ** -(nb - 1 - b) for b in range(nb - 1)] + [float("inf")]
    return [(kn >= lo) & (kn < hi) for lo, hi in zip(edges[:-1], edges[1:])]


def band_proj(S, shape, mask):
    """P_b applied to rows of S (N, D) living on grid `shape`."""
    dims = tuple(range(1, len(shape) + 1))
    X = torch.fft.fftn(S.reshape(len(S), *shape), dim=dims)
    return torch.fft.ifftn(X * mask, dim=dims).real.reshape(len(S), -1)


def band_pod(S, shape, masks):
    """Greedy band-wise POD: per band the top-M modes of P_b S, then the M largest eigenvalues overall (>= M_FLOOR per
    band). Returns (D, M) modes, (M,) mean energies, (M,) band ids, ordered by energy."""
    cand = []
    for b, mk in enumerate(masks):
        Sb = band_proj(S, shape, mk); ev, U = torch.linalg.eigh(Sb @ Sb.T)
        ev, U = ev.flip(0)[:M], U.flip(1)[:, :M]                       # rank(P_b S) <= band dimension
        keep = (torch.arange(len(ev), device=ev.device) < int(mk.sum())) & (ev > ev[0].clamp_min(1e-30) * 1e-6)
        ev, U = ev[keep], U[:, keep]
        cand.append(((Sb.T @ U) / ev.sqrt(), ev / len(S), torch.full((len(ev),), b, device=S.device)))
        del Sb
    Phi, en, bd = (torch.cat(t, -1 if i == 0 else 0) for i, t in enumerate(zip(*cand)))
    floor = torch.cat([(bd == b).nonzero().flatten()[:M_FLOOR] for b in range(len(masks))])
    rest = torch.ones(len(en), dtype=torch.bool, device=en.device); rest[floor] = False
    extra = rest.nonzero().flatten()[en[rest].argsort(descending=True)[:M - len(floor)]]
    pick = torch.cat([floor, extra]); pick = pick[en[pick].argsort(descending=True)]
    return Phi[:, pick], en[pick], bd[pick]


@torch.no_grad()
def fit_blend(row, nb):
    """alpha (C, nb) = argmin sum_w sum_c ||P_b(y - x_t) - alpha_cb P_b(p - x_t)||^2 / var(y_c), plus the per-field
    spectral filter H (C, *grid) = sum_b alpha_cb 1_b(k)."""
    num = den = masks = None
    for xt, dx, p, y in gen_split(row, "train", N_SNAP):
        dims = tuple(range(2, y.dim())); masks = masks or band_masks(tuple(y.shape[2:]), nb)
        w = 1 / (y.flatten(2).var(-1, unbiased=False) + 1e-7)
        Dk, Yk = torch.fft.fftn(p - xt, dim=dims), torch.fft.fftn(y - xt, dim=dims)
        n = torch.stack([((Yk * Dk.conj()).real * mk).flatten(2).sum(-1) for mk in masks], -1)
        d = torch.stack([(Dk.abs().square() * mk).flatten(2).sum(-1) for mk in masks], -1)
        n, d = (w[..., None].double() * n.double()).sum(0), (w[..., None].double() * d.double()).sum(0)
        num, den = (n, d) if num is None else (num + n, den + d)
    alpha = (num / den.clamp_min(1e-30)).float()
    H = sum(alpha[:, b].reshape(-1, *[1] * len(masks[0].shape)) * mk for b, mk in enumerate(masks))
    return alpha, H


def blended(gen, H):
    for xt, dx, p, y in gen:
        dims = tuple(range(2, y.dim()))
        yield xt, dx, xt + torch.fft.ifftn(H * torch.fft.fftn(p - xt, dim=dims), dim=dims).real, y, p


@torch.no_grad()
def main(row: str):
    dst = OUT / row.replace(":", "__"); dst.mkdir(parents=True, exist_ok=True)
    if os.environ.get("ADRC_TEST_ALL"):
        SIZES["test"] = 10 ** 9                                     # every test window (full-split metric)
    nb = int(os.environ.get("ADRC_BANDS", "0")); snaps, shape = None, None
    nbl = int(os.environ.get("ADRC_BLEND", "0")); alpha = H = None
    if nbl:
        assert nbl == nb, "gate bands must equal POD bands (ADRC_BLEND == ADRC_BANDS)"
        alpha, H = fit_blend(row, nbl)
        print(f"[adrc-cache] {row} blend alpha (field x band, low->high):\n{alpha.cpu().numpy().round(3)}", flush=True)
    gen = (lambda split, n: blended(gen_split(row, split, n), H)) if nbl else (lambda split, n: (t + (None,) for t in gen_split(row, split, n)))
    for xt, dx, p, y, _ in gen("train", N_SNAP):
        shape = tuple(y.shape[2:])
        r = (y - p).flatten(2); var = y.flatten(2).var(-1, unbiased=False) + 1e-7
        w = (r / (var * r.shape[-1]).sqrt()[..., None]).half()
        snaps = [[] for _ in range(r.shape[1])] if snaps is None else snaps
        for c in range(r.shape[1]):
            snaps[c].append(w[:, c])
    basis, energy, bands = [], [], []
    masks = band_masks(shape, nb) if nb else None
    for c, sc in enumerate(snaps):
        S = torch.cat(sc).float()
        if nb:
            Phi_c, en_c, bd_c = band_pod(S, shape, masks)
            basis.append(Phi_c.half()); energy.append(en_c); bands.append(bd_c)
        else:
            ev, U = torch.linalg.eigh(S @ S.T)
            ev, U = ev.flip(0)[:M], U.flip(1)[:, :M]
            basis.append(((S.T @ U) / ev.clamp_min(1e-12).sqrt()).half()); energy.append(ev / len(S))
        snaps[c] = None; del S
    meta = {"energy": torch.stack(energy).cpu(), "M": M}
    if nbl:
        meta.update(blend_alpha=alpha.cpu(), blend_bands=nbl)
    if nb:
        meta.update(band=torch.stack(bands).cpu(), n_bands=nb, band_rule="dyadic l-inf FFT shells")
        print(f"[adrc-cache] {row} modes per band (field 0): {torch.bincount(bands[0], minlength=nb).tolist()}", flush=True)
    torch.save(meta, dst / "basis_meta.pt")
    Phi = torch.stack(basis)                                          # (C, D, M) fp16, orthonormal columns
    for split, n in SIZES.items():
        rec = {k: [] for k in ("a", "f", "E", "D", "g", "Eb", "E0", "ay", "fd0", "gY", "gYD", "gDD", "gX")}
        for xt, dx, p, y, p0 in gen(split, n):
            B, C = y.shape[:2]
            proj = lambda t: torch.einsum("bcd,cdm->bcm", t.flatten(2).half(), Phi).float()
            r = y - p; d = p - xt
            rec["a"].append(proj(r).cpu())
            rec["f"].append(torch.stack([proj(d), proj(p), proj(xt), proj(dx)], -1).cpu())
            rec["E"].append(r.flatten(2).double().square().sum(-1).cpu())
            if p0 is not None:
                rec["E0"].append((y - p0).flatten(2).double().square().sum(-1).cpu())
                dims = tuple(range(2, y.dim())); npx = r[0, 0].numel()
                Yk, Dk, Xk = (torch.fft.fftn(t.float(), dim=dims) for t in (y - xt, p0 - xt, dx))
                bsum = lambda T: torch.stack([(T * mk).flatten(2).sum(-1) for mk in masks], -1).double().cpu() / npx
                rec["gY"].append(bsum(Yk.abs().square())); rec["gYD"].append(bsum((Yk * Dk.conj()).real))
                rec["gDD"].append(bsum(Dk.abs().square())); rec["gX"].append(bsum(Xk.abs().square()))
                rec["ay"].append(proj(y - xt).cpu()); rec["fd0"].append(proj(p0 - xt).cpu())
            if nb:                                                                  # Parseval: ||P_b r||^2 per field
                R = torch.fft.fftn(r.float(), dim=tuple(range(2, r.dim()))).abs().square()
                rec["Eb"].append(torch.stack([(R * mk).flatten(2).sum(-1) for mk in masks], -1).double().cpu() / r[0, 0].numel())
            npix = r[0, 0].numel()
            rec["D"].append((npix * (y.flatten(2).var(-1, unbiased=False) + 1e-7)).double().cpu())
            g = [torch.cat([t.flatten(2).mean(-1), t.flatten(2).std(-1)], 1) for t in (d, p, xt, dx)]
            rec["g"].append(torch.cat(g, 1).float().cpu())
        out = {k: torch.cat(v) for k, v in rec.items() if v}
        torch.save(out, dst / f"{split}.pt")
        a2 = (out["a"].double().square().sum(-1) / out["D"]).sum(1); tot = (out["E"] / out["D"]).sum(1)
        print(f"[adrc-cache] {row} {split}: N={len(out['a'])} basis share {float((a2 / tot).mean()):.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
