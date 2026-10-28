"""Where is the 'hard analog part'? FFT + wavelet decomposition of one-step errors on TRL2D (The Well).

    .venv/bin/python paper/spectral/band_analysis.py [--per-traj 20]   ->  runs/spectral/trl2d_bands.json

For every arm checkpoint (official CNextU-net, and vanilla / mlp30 / mr30ryd fine-tunes x 3 seeds, all 20k steps):
  * radial FFT spectra of the error e = pred - y and of the one-step change d = y - x_last (the content the model
    must predict); R(k) = E_e(k) / E_d(k) is the fraction of the change the model misses in wavenumber shell k;
  * orthonormal 2-D Haar wavelet energies of e and d per scale (J = 5) and orientation (Parseval holds exactly);
  * localisation: share of detail-coefficient error energy held by the top-30 and top-1% wavelet tiles;
  * register visibility: the bottleneck mean-pools the 672x8x24 neck latent, so the register only sees its DC
    (spatial-mean) component; we measure that component's share of latent energy.
All tensors on the GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "phase_b"))
import cnext_hybrid  # noqa: E402
import well2d  # noqa: E402

DEV = torch.device("cuda")
J = 5
TOPK_ATOMS = 30


def haar2d(x: torch.Tensor, levels: int):
    """Orthonormal multi-level 2-D Haar DWT. x (B,C,H,W) -> (approx, [(LH,HL,HH) per level, fine -> coarse])."""
    details = []
    a = x
    for _ in range(levels):
        p, q, r, s = a[..., 0::2, 0::2], a[..., 0::2, 1::2], a[..., 1::2, 0::2], a[..., 1::2, 1::2]
        details.append(((p - q + r - s) / 2, (p + q - r - s) / 2, (p - q - r + s) / 2))
        a = (p + q + r + s) / 2
    return a, details


def radial_bins(H: int, W: int):
    ky = torch.fft.fftfreq(H, device=DEV) * H
    kx = torch.fft.rfftfreq(W, device=DEV) * W
    k = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).round().long()
    w = torch.full_like(k, 2.0, dtype=torch.float32); w[:, 0] = 1.0            # rfft: double the half-plane
    if W % 2 == 0:
        w[:, -1] = 1.0
    return k, w, int(k.max()) + 1


def radial_energy(f: torch.Tensor, k, w, nk) -> torch.Tensor:
    """(B,C,H,W) -> (nk,) energy per wavenumber shell, summed over batch and channels (Parseval-normalised)."""
    F = torch.fft.rfft2(f, norm="ortho")
    e = (F.abs() ** 2 * w).sum(dim=(0, 1))
    return torch.zeros(nk, device=DEV).index_add_(0, k.flatten(), e.flatten())


class Acc:
    def __init__(self, nk):
        z = lambda *s: torch.zeros(*s, device=DEV, dtype=torch.float64)
        self.Ee, self.Ed, self.Ey = z(nk), z(nk), z(nk)
        self.We, self.Wd = z(J + 1, 3), z(J + 1, 3)          # row 0..J-1 details (LH,HL,HH), row J approx
        self.top30 = self.top1pct = self.detail_tot = 0.0
        self.lat_dc = self.lat_tot = 0.0
        self.n = 0

    def add_wavelets(self, W, f):
        a, det = haar2d(f, J)
        for j, (lh, hl, hh) in enumerate(det):
            for o, c in enumerate((lh, hl, hh)):
                W[j, o] += c.double().square().sum()
        W[J, 0] += a.double().square().sum()
        return det

    def localisation(self, det):
        """Per sample: share of detail error energy in its 30 / top-1% largest tiles (all scales, channels)."""
        B = det[0][0].shape[0]
        coeffs = torch.cat([c.reshape(B, -1) for lvl in det for c in lvl], dim=1).double().square()
        tot = coeffs.sum(1)
        srt = coeffs.sort(dim=1, descending=True).values
        k1 = max(1, int(0.01 * coeffs.shape[1]))
        self.top30 += (srt[:, :TOPK_ATOMS].sum(1)).sum().item()
        self.top1pct += (srt[:, :k1].sum(1)).sum().item()
        self.detail_tot += tot.sum().item()


def analyse(model, data, pairs, bsz, k, w, nk):
    acc = Acc(nk)
    lat = {}
    neck = model.neck.neck if hasattr(model.neck, "neck") else model.neck
    h = neck.register_forward_hook(lambda m, i, o: lat.__setitem__("z", o))
    C = data.C
    with torch.no_grad():
        for x, y in well2d.batched(data, pairs, bsz):
            x, y = x.to(DEV), y.to(DEV)
            p = model(x)
            e, d = p - y, y - x[:, (well2d.N_HIST - 1) * C:]
            acc.Ee += radial_energy(e, k, w, nk); acc.Ed += radial_energy(d, k, w, nk); acc.Ey += radial_energy(y, k, w, nk)
            det = acc.add_wavelets(acc.We, e); acc.add_wavelets(acc.Wd, d); acc.localisation(det)
            z = lat["z"].double()                                                 # (B, 672, 8, 24)
            dc = z.mean(dim=(2, 3), keepdim=True)
            acc.lat_dc += (dc.square() * z.shape[2] * z.shape[3]).sum().item(); acc.lat_tot += z.square().sum().item()
            acc.n += x.shape[0]
    h.remove()
    return acc


def summary(acc: Acc) -> dict:
    R = (acc.Ee / acc.Ed.clamp_min(1e-30)).tolist()
    Wr = (acc.We / acc.Wd.clamp_min(1e-30))
    return dict(n=acc.n, err_energy=acc.Ee.sum().item(), change_energy=acc.Ed.sum().item(),
                E_err_k=acc.Ee.tolist(), E_change_k=acc.Ed.tolist(), E_true_k=acc.Ey.tolist(), R_k=R,
                W_err=acc.We.tolist(), W_change=acc.Wd.tolist(), W_ratio=Wr.tolist(),
                top30_share=acc.top30 / acc.detail_tot, top1pct_share=acc.top1pct / acc.detail_tot,
                register_visible_latent_share=acc.lat_dc / acc.lat_tot)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--per-traj", type=int, default=20); ap.add_argument("--bsz", type=int, default=16)
    a = ap.parse_args()
    data = well2d.Split("turbulent_radiative_layer_2D", "test")
    pairs = well2d.fixed_windows(data, a.per_traj)
    k, w, nk = radial_bins(data.H, data.W)
    out = dict(dataset="turbulent_radiative_layer_2D", split="test", windows=len(pairs), H=data.H, W=data.W,
               fields=data.fields, levels=J, arms={})
    arms = [("official", None)] + [(f"{v}_s{s}", ROOT / f"runs/phase_b/{v}_s{s}.pt")
                                   for v in ("vanilla", "mlp30", "mr30ryd") for s in (1337, 1338, 1339)]
    for name, ck in arms:
        variant = "vanilla" if ck is None else name.rsplit("_s", 1)[0]
        model = (cnext_hybrid.build_hybrid("vanilla") if ck is None else cnext_hybrid.load_hybrid_checkpoint(variant, str(ck)))
        model = model.to(DEV).eval()
        out["arms"][name] = summary(analyse(model, data, pairs, a.bsz, k, w, nk))
        s = out["arms"][name]
        print(f"[bands] {name:14s} n={s['n']} err/change energy {s['err_energy']/s['change_energy']:.4f} "
              f"top30 {s['top30_share']:.3f} top1% {s['top1pct_share']:.3f} register-visible latent {s['register_visible_latent_share']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    dst = ROOT / "runs" / "spectral"; dst.mkdir(parents=True, exist_ok=True)
    (dst / "trl2d_bands.json").write_text(json.dumps(out))
    data.close()


if __name__ == "__main__":
    main()
