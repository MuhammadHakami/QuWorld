"""Shared pieces for the §37 rayleigh_benard exploration (doc 35 §37). VALID/TRAIN only; TEST is never opened.

- the fixed stratified VALID sample (runs/strat/rb/strata.json, written by make_strata.py before any scoring);
- the gated fine-tuned models of doc 37 (two-stage `_pre` cells), loaded without constructing the TEST split;
- x-symmetry test-time transforms (x is periodic, y has walls): horizontal reflection with the u_x sign flip, and
  periodic x-shifts; both act on z-scored channels exactly (u_x -> -u_x is affine in normalized units);
- the stratified paired bootstrap used for every reported delta.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
for p in (ROOT / "paper", ROOT / "paper" / "phase_b", ROOT / "paper" / "alloc"):
    sys.path.insert(0, str(p))
import gpucap  # noqa: E402


def _mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


MIN_AVAIL_GB = 30.0                                        # §37 launch rule, checked when the job actually starts
if _mem_available_gb() < MIN_AVAIL_GB:
    raise SystemExit(f"[strat_rb] MemAvailable {_mem_available_gb():.1f} GB < {MIN_AVAIL_GB} GB: not launching")
gpucap.cap_from_env()                                     # before any CUDA allocation (the §37 rule)
import finetune_gated as FG  # noqa: E402  (its import also calls cap_from_env; idempotent)
import well2d  # noqa: E402
from well2d_backbone import load_checkpoint  # noqa: E402

DEV = FG.DEV
OUT = ROOT / "runs" / "strat" / "rb"
DS = "rayleigh_benard"
FIELDS = ("buoyancy", "pressure", "u_x", "u_y")
C = 4
UX = 2                                                     # u_x channel inside each 4-channel frame
MEMBERS = {"mr30ryd": "rayleigh_benard_mr30ryd_vrmse_gate_s1_pre",
           "vanilla": "rayleigh_benard_vanilla_vrmse_gate_s1_pre",
           "mlp30": "rayleigh_benard_mlp30_vrmse_gate_s1_pre"}
BEST_SINGLE = "mr30ryd"                                    # VALID-best single member (runs/alloc/ft/rayleigh_benard_Eft.json)


def load_strata():
    d = json.loads((OUT / "strata.json").read_text())
    pairs, sid = [], []
    for i, s in enumerate(d["strata"]):
        for traj, t0 in s["windows"]:
            pairs.append((traj, t0)); sid.append(i)
    return d, pairs, np.asarray(sid)


def valid_split():
    return well2d.Split(DS, "valid")


def train_split():
    return well2d.Split(DS, "train")


def trunk_ckpt(variant, seed=1337):
    return ROOT / "runs" / "well2d" / DS / "cnext" / f"{variant}_s{seed}.pt"


def load_gated(member: str, split_for_shape):
    """GatedHead(trunk) with the saved `_pre` state (trunk + gate); eval mode."""
    tag = MEMBERS[member]
    trunk = load_checkpoint(member, DS, str(trunk_ckpt(member)), "cnext", DEV)
    model = FG.GatedHead(trunk, C, (split_for_shape.H, split_for_shape.W), torch.ones(C, FG.NB), True).to(DEV)
    model.load_state_dict(torch.load(FG.OUT / f"{tag}.pt", weights_only=True, map_location=DEV)["model"])
    return model.eval()


class XSym:
    """Exact x-symmetries on z-scored RB tensors (B, k*4, H=x, W=y). flip: x -> -x with u_x -> -u_x."""

    def __init__(self, mean, std):
        m, s = float(mean[UX].squeeze()), float(std[UX].squeeze())
        self.off = -2.0 * m / s                            # n' = -n - 2m/s for the u_x channel

    def _flip(self, t):
        t = t.flip(2).clone()
        t[:, UX::C] = -t[:, UX::C] + self.off
        return t

    def apply(self, t, flip, shift):
        if shift:
            t = t.roll(shift, dims=2)
        return self._flip(t) if flip else t

    def invert(self, t, flip, shift):
        if flip:
            t = self._flip(t)                               # the flip is an involution
        return t.roll(-shift, dims=2) if shift else t


@torch.no_grad()
def tta_predict(model, x, sym: XSym, transforms):
    """Mean over transforms of T^-1(model(T(x))); transforms = [(flip, shift), ...]."""
    acc = None
    for flip, shift in transforms:
        p = sym.invert(model(sym.apply(x, flip, shift)), flip, shift)
        acc = p if acc is None else acc + p
    return acc / len(transforms)


def vrmse_fields(pred, y):
    """(B, C) per-field VRMSE (the Well; eps 1e-7) on any device."""
    dims = (2, 3)
    mse = (pred - y).square().mean(dims); var = (y - y.mean(dims, keepdim=True)).square().mean(dims)
    return (mse / (var + 1e-7)).clamp_min(1e-12).sqrt()


def weights_for(strata_doc, sid):
    """Per-window population weights: stratum population share / stratum sample size (sums to 1)."""
    pw = np.array([s["pop_weight"] for s in strata_doc["strata"]])
    n = np.bincount(sid, minlength=len(pw))
    return pw[sid] / n[sid]


def strat_boot(delta, base, sid, w, B=2000, seed=0, mask=None):
    """Stratified paired bootstrap (resample windows within each stratum) of the weighted mean delta and of the
    relative delta (weighted mean delta / weighted mean baseline). Returns point estimates and 95% percentile CIs."""
    delta, base, sid, w = map(np.asarray, (delta, base, sid, w))
    if mask is not None:
        delta, base, sid, w = delta[mask], base[mask], sid[mask], w[mask]
    w = w / w.sum()
    groups = [np.flatnonzero(sid == s) for s in np.unique(sid)]
    rng = np.random.default_rng(seed)
    d_bs, r_bs = np.empty(B), np.empty(B)
    for b in range(B):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        ww = w[idx] / w[idx].sum()
        d = float((ww * delta[idx]).sum()); d_bs[b] = d; r_bs[b] = d / float((ww * base[idx]).sum())
    d0 = float((w * delta).sum()); r0 = d0 / float((w * base).sum())
    return {"delta": d0, "delta_ci95": [float(np.percentile(d_bs, 2.5)), float(np.percentile(d_bs, 97.5))],
            "rel": r0, "rel_ci95": [float(np.percentile(r_bs, 2.5)), float(np.percentile(r_bs, 97.5))],
            "n": int(len(delta)), "frac_windows_improved": float((delta < 0).mean())}


def marginals(strata_doc, sid):
    """Named window masks: overall, per Rayleigh, per Prandtl, per position, per Ra x position."""
    S = strata_doc["strata"]
    ra = np.array([S[i]["rayleigh"] for i in sid]); pr = np.array([S[i]["prandtl"] for i in sid])
    pos = np.array([S[i]["position"] for i in sid])
    out = {"all": np.ones(len(sid), bool)}
    for v in sorted(set(ra)):
        out[f"Ra={v:.0e}"] = ra == v
    for v in sorted(set(pr)):
        out[f"Pr={v:g}"] = pr == v
    for v in ("onset", "developed"):
        out[f"pos={v}"] = pos == v
    for v in sorted(set(ra)):
        for q in ("onset", "developed"):
            out[f"Ra={v:.0e}&{q}"] = (ra == v) & (pos == q)
    return out
