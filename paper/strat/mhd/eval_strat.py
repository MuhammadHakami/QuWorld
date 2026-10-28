"""Doc 35 §37, MHD_64: score the Flowers (§34, best-VALID copy) and gated CNextU-net models, their test-time-augmented
variants and persistence on the fixed stratified TRAIN/VALID samples (runs/strat/mhd/strata.json). TEST is never loaded.

For every window it stores exact second-order statistics, so any per-channel, per-regime or per-band linear blend of the
variants can be scored offline without re-running a model (Parseval):
  G[i, j, c, s] = sum over wavenumbers in radial shell s of Re(R_i R_j^*) / N, with R_v = FFT_ortho(pred_v - y)
  => mean_x (r_i r_j) for channel c = sum_s G[i, j, c, s]
plus var(y) per channel (VRMSE denominator), the target's shell spectrum and causal regime features of the input.

Test-time augmentation (TTA) passes per model (30): identity; 7 odd circular shifts (§29c ODD8); the 7 non-trivial
reflections of Z2^3 (B handled so that the guide field B_x > 0 is kept: an x-reflection maps B -> (Bx, -By, -Bz), the
axial rule; y and z reflections flip only By / Bz, the polar rule; both are symmetries of MHD since B -> -B is one);
the 8 elements y<->z swap x Z2^3; 7 reflection+odd-shift combinations. All sign flips act in physical units.
Variants (each also with the divergence-free projection of B, §29a): raw, sh8, g8, g16, shg8, all30.

    flock runs/strat/gpu.lock timeout 1800 env QW_GPU_GB=12 runs/capped.sh strat_mhd 16 uv run --no-sync python \
        paper/strat/mhd/eval_strat.py --split valid [--smoke] [--budget 1500] [--passes full|lite]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper"))
from gpucap import cap_from_env  # noqa: E402

cap_from_env()                                                        # before any CUDA allocation
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(ROOT / "paper" / "alloc")); sys.path.insert(0, str(ROOT / "paper" / "phase_b"))
sys.path.insert(0, str(ROOT / "baseline" / "flowers"))
import finetune_gated as FG  # noqa: E402  (its import-time cap_from_env is idempotent)
from cache import WellND  # noqa: E402
from mhd_divfree import ODD8, DivFreeB  # noqa: E402

DEV = torch.device("cuda")
SCR = pathlib.Path(os.environ.get("STRAT_FLOWER_CKPT", "/tmp/claude-1000/-home-mohhakami-Documents-paper-quworld/"
                                  "af4d80b6-5cb2-4675-b69d-81a19e2a7c31/scratchpad/flower_best_copy.pt"))  # copy of flower_s1337.pt (step 5694)
OUT = ROOT / "runs" / "strat" / "mhd"
C, NH, N = 7, 4, 64
PASS_BATCH = 2
NSHELL = 56                                                           # |k| rounded, 0..55 (max sqrt(3) * 32 = 55.4)
FLIPS = [(fx, fy, fz) for fx in (0, 1) for fy in (0, 1) for fz in (0, 1)]   # FLIPS[0] = identity


class Sym:
    """Symmetry element: reflections (fx, fy, fz), optional y<->z swap, then a circular shift. Acts on normalized
    tensors (..., 7, D, H, W) in physical units; spatial dims (D, H, W) = (x, y, z) components (checked by §29a)."""

    def __init__(self, flips=(0, 0, 0), swap=False, shift=(0, 0, 0)):
        self.flips, self.swap, self.shift = tuple(flips), bool(swap), tuple(shift)
        fx, fy, fz = self.flips
        sb = [1.0, 1.0, 1.0]; sv = [1.0, 1.0, 1.0]
        if fx:
            sb[1] *= -1; sb[2] *= -1; sv[0] *= -1        # axial rule on x: guide field kept
        if fy:
            sb[1] *= -1; sv[1] *= -1                     # polar rule on y, z
        if fz:
            sb[2] *= -1; sv[2] *= -1
        self.sign = torch.tensor([1.0] + sb + sv)

    def _flip(self, t, st):
        m, s = st
        d = t.dim()
        dims = [d - 3 + a for a, f in enumerate(self.flips) if f]
        sign = self.sign.to(t.device).view(C, 1, 1, 1)
        if not dims:
            return t
        phys = t * s + m
        return (torch.flip(phys, dims) * sign - m) / s

    def _swap(self, t, st):
        if not self.swap:
            return t
        m, s = st
        phys = t * s + m
        phys = phys[..., [0, 1, 3, 2, 4, 6, 5], :, :, :].transpose(-1, -2)
        return (phys - m) / s

    def fwd(self, t, st):
        t = self._swap(self._flip(t, st), st)
        return torch.roll(t, self.shift, dims=(-3, -2, -1)) if any(self.shift) else t

    def inv(self, t, st):
        if any(self.shift):
            t = torch.roll(t, tuple(-v for v in self.shift), dims=(-3, -2, -1))
        return self._flip(self._swap(t, st), st)


def pass_groups(kind: str):
    ident = [Sym()]
    shifts = [Sym(shift=s) for s in ODD8[1:]]
    refl = [Sym(flips=f) for f in FLIPS[1:]]
    swap = [Sym(flips=f, swap=True) for f in FLIPS]
    comb = [Sym(flips=f, shift=s) for f, s in zip(FLIPS[1:], ODD8[1:])]
    if kind == "lite":
        return {"id": ident, "shift": shifts, "refl": refl}
    if kind == "trainlite":                                            # TRAIN blend fits: raw and shg8 only
        return {"id": ident, "comb": comb}
    return {"id": ident, "shift": shifts, "refl": refl, "swap": swap, "comb": comb}


def variants(sums: dict) -> dict:
    v = {"raw": sums["id"]}
    if "shift" in sums:
        v["sh8"] = (sums["id"] + sums["shift"]) / 8
    if "refl" in sums:
        v["g8"] = (sums["id"] + sums["refl"]) / 8
    if "refl" in sums and "swap" in sums:
        v["g16"] = (sums["id"] + sums["refl"] + sums["swap"]) / 16
    if "comb" in sums:
        v["shg8"] = (sums["id"] + sums["comb"]) / 8
    if len(sums) == 5:
        v["all30"] = sum(sums.values()) / 30
    return v


def load_models():
    from cnext_hybrid import build_hybrid
    from flower_standalone import Flower
    trunk = build_hybrid("vanilla", model_dir=str(ROOT / "data/cnext_mhd_ckpt"))
    gated = FG.GatedHead(trunk, C, (N, N, N), torch.ones(C, FG.NB), True)
    gated.load_state_dict(torch.load(FG.OUT / "MHD_64_vanilla_vrmse_gate_s1.pt", weights_only=True,
                                     map_location="cpu")["model"])
    fl = Flower(dim_in=NH * C, dim_out=C, n_spatial_dims=3, spatial_resolution=[N, N, N],
                boundary_condition_types=["PERIODIC"])
    st = torch.load(SCR, weights_only=True, map_location="cpu")
    fl.load_state_dict(st["model_state_dict"])
    print(f"[load] flowers ckpt {SCR} step {st['step']} val {st['val']:.5f}", flush=True)
    return {"F": fl.to(DEV).eval(), "C": gated.to(DEV).eval()}, {"flower_step": st["step"], "flower_val": st["val"]}


def shell_index():
    k = torch.fft.fftfreq(N, d=1.0 / N)
    K = torch.stack(torch.meshgrid(k, k, k, indexing="ij"))
    return K.square().sum(0).sqrt().round().long().clamp_max(NSHELL - 1).flatten().to(DEV)


@torch.no_grad()
def run_window(models, split, row, groups, st_n, df, shells):
    x, y = split.window(row["traj"], row["t0"])
    x, y = x[None].to(DEV), y[None].to(DEV)                          # (1, 28, 64^3), (1, 7, 64^3)
    x6 = x.view(1, NH, C, N, N, N)
    preds = {"P": x6[:, -1]}                                          # persistence
    t_model, pass_vr = {}, {}
    var_y = (y - y.mean(dim=(-3, -2, -1), keepdim=True)).square().mean(dim=(-3, -2, -1))[0]
    for name, m in models.items():
        t0 = time.perf_counter(); sums = {g: 0 for g in groups}; pv = []
        flat = [(g, e) for g, elems in groups.items() for e in elems]
        for j in range(0, len(flat), PASS_BATCH):                       # batch size <= 2 at 64^3 (§37 rule)
            chunk = flat[j:j + PASS_BATCH]
            xin = torch.cat([e.fwd(x6, st_n) for _, e in chunk]).reshape(len(chunk), NH * C, N, N, N)
            outs = m(xin)
            for k, (g, e) in enumerate(chunk):
                out = e.inv(outs[k:k + 1], st_n)
                pv.append([g, list(e.flips), e.swap, list(e.shift),
                           ((out - y).square().mean(dim=(-3, -2, -1))[0] / (var_y + 1e-7)).sqrt().tolist()])
                sums[g] = sums[g] + out
        pass_vr[name] = pv
        for vn, p in variants(sums).items():
            preds[f"{name}_{vn}"] = p
            preds[f"{name}_{vn}_df"] = df(p)
        torch.cuda.synchronize(); t_model[name] = time.perf_counter() - t0
    names = list(preds)
    R = torch.fft.fftn(torch.cat([preds[k] for k in names]) - y, dim=(-3, -2, -1), norm="ortho").reshape(len(names), C, -1)
    V = len(names); G = torch.zeros(V, V, C, NSHELL, dtype=torch.float64, device=DEV)
    for i in range(V):
        prod = (R[i:i + 1].conj() * R[i:]).real.double() / R.shape[-1]
        g = torch.zeros(V - i, C, NSHELL, dtype=torch.float64, device=DEV).index_add_(2, shells, prod)
        G[i, i:] = g; G[i:, i] = g
    Y = torch.fft.fftn(y[0], dim=(-3, -2, -1), norm="ortho").reshape(C, -1)
    Ey = torch.zeros(C, NSHELL, dtype=torch.float64, device=DEV).index_add_(1, shells, Y.abs().square().double() / Y.shape[-1])
    m, s = st_n
    last = x6[0, -1] * s + m                                          # physical units, last input frame
    feat = {"meanB": last[1:4].mean(dim=(1, 2, 3)).tolist(), "vrms": float(last[4:7].square().sum(0).mean().sqrt()),
            "brms": float(last[1:4].square().sum(0).mean().sqrt()), "rho_mean": float(last[0].mean()), "pass_vr": pass_vr}
    return names, G.float().cpu().numpy(), Ey.float().cpu().numpy(), var_y.double().cpu().numpy(), feat, t_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("valid", "train"), required=True)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--budget", type=float, default=1500.0)
    ap.add_argument("--passes", choices=("full", "lite", "trainlite"), default="full")
    a = ap.parse_args()
    assert a.split in ("valid", "train")                              # TEST is never loaded
    torch.backends.cudnn.benchmark = False
    rows = json.loads((OUT / "strata.json").read_text())["samples"][a.split]
    shard_dir = OUT / ("shards_smoke" if a.smoke else os.environ.get("STRAT_SHARD_DIR", "shards")); shard_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(len(rows)) if not (shard_dir / f"{a.split}_{a.passes}_{i:04d}.npz").exists()]
    if a.smoke:
        todo = todo[:2]
    if not todo:
        print(f"[strat] {a.split} {a.passes}: nothing to do", flush=True); return
    split = WellND("MHD_64", a.split)
    models, meta = load_models()
    std = torch.tensor(split.std.reshape(C)).view(C, 1, 1, 1).to(DEV); mean = torch.tensor(split.mean.reshape(C)).view(C, 1, 1, 1).to(DEV)
    dfm = DivFreeB(torch.nn.Identity()).to(DEV)
    df = lambda p: torch.cat([p[:, :1], dfm.project(p[:, 1:4]), p[:, 4:]], 1)  # noqa: E731
    groups = pass_groups(a.passes); shells = shell_index()
    t_start = time.perf_counter(); done = 0
    for i in todo:
        if time.perf_counter() - t_start > a.budget:
            break
        tw = time.perf_counter()
        names, G, Ey, var_y, feat, tm = run_window(models, split, rows[i], groups, (mean, std), df, shells)
        np.savez(shard_dir / f"{a.split}_{a.passes}_{i:04d}.npz", G=G, Ey=Ey, var_y=var_y, names=np.array(names),
                 row=json.dumps(rows[i]), feat=json.dumps(feat), meta=json.dumps(meta))
        done += 1
        vr = {n: float(np.sqrt(G[k, k].sum(-1) / (var_y + 1e-7)).mean()) for k, n in enumerate(names)}
        print(f"[strat] {a.split} {i} traj={rows[i]['traj']} t0={rows[i]['t0']} Ma={rows[i]['Ma']} "
              f"F={vr['F_raw']:.4f} F_tta_df={vr.get('F_all30_df', vr.get('F_shg8_df', vr.get('F_g8_df'))):.4f} C={vr['C_raw']:.4f} "
              f"P={vr['P']:.4f} secs={time.perf_counter() - tw:.1f} tF={tm['F']:.1f} tC={tm['C']:.1f} "
              f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}G", flush=True)
    print(f"[strat] {a.split} {a.passes}: {done} windows in {time.perf_counter() - t_start:.0f}s, "
          f"left {len(todo) - done}", flush=True)


if __name__ == "__main__":
    main()
