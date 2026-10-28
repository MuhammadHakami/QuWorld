"""Doc 35 §29a: divergence-free magnetic field for the MHD_64 gated model (classical part of the hybrid).

The Well's MHD_64 fields are periodic and the true B is solenoidal in the spectral sense (longitudinal energy share
7.7e-6 on a VALID frame), while 5.9% of the trunk's B error is longitudinal (VALID diagnostic, 12 windows). This script
projects the predicted B onto its divergence-free part in physical units (official ZScore stats; the three components
have different std, so the projection cannot act on normalized channels) and scores the saved gated model with and
without it. Persistence is solenoidal and the gate blends trunk and persistence band by band in Fourier space, so
projecting the final output equals projecting the trunk's contribution; the gate's inputs are unchanged.

Rule (registered): VALID decides (stratified 80 windows, the gated fine-tune's own selection set); if the projection
lowers mean VALID VRMSE, the full TEST split is scored once with it.

    QW_GPU_GB=16 runs/capped.sh divfree 32 .venv/bin/python paper/alloc/mhd_divfree.py [--variant vanilla] [--tag-suffix ""]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
import torch.nn as nn
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import finetune_gated as FG  # noqa: E402

B_CH = slice(1, 4)                                    # [density, Bx, By, Bz, vx, vy, vz] (the_well order)


class DivFreeB(nn.Module):
    """Wraps a model whose output is (N, 7, 64, 64, 64) in normalized units; projects B spectrally in physical units."""

    def __init__(self, model: nn.Module, n: int = 64):
        super().__init__()
        self.model = model
        st = yaml.safe_load(open(FG.ROOT / "data/well/MHD_64/stats.yaml"))
        self.register_buffer("mean", torch.tensor(st["mean"]["magnetic_field"]).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor(st["std"]["magnetic_field"]).view(1, 3, 1, 1, 1))
        k = torch.fft.fftfreq(n) * n
        k[n // 2] = 0.0                                                          # Nyquist: keeps the projection real
        K = torch.stack(torch.meshgrid(k, k, k, indexing="ij"))
        self.register_buffer("K", K[None])                                       # (1, 3, n, n, n)
        self.register_buffer("K2", (K ** 2).sum(0).clamp_min(1.0)[None, None])

    def project(self, bn: torch.Tensor) -> torch.Tensor:
        B = bn.float() * self.std + self.mean
        Bh = torch.fft.fftn(B, dim=(2, 3, 4))
        Bh = Bh - self.K * (self.K * Bh).sum(1, keepdim=True) / self.K2          # remove the longitudinal part
        return ((torch.fft.ifftn(Bh, dim=(2, 3, 4)).real - self.mean) / self.std).to(bn.dtype)

    def forward(self, x):
        out = self.model(x)
        return torch.cat([out[:, :1], self.project(out[:, B_CH]), out[:, 4:]], 1)


# §29c: 8 circular shifts with odd offsets, fixed on a 12-window VALID diagnostic (aligned multiples of 8 gave less)
ODD8 = ((0, 0, 0), (5, 37, 21), (29, 11, 53), (43, 59, 3), (19, 27, 45), (61, 7, 33), (37, 49, 13), (11, 21, 59))


class ShiftAverage(nn.Module):
    """Mean of the model over circular shifts of the periodic input, each output shifted back (the trunk zero-pads and
    down-samples, so it is not shift-equivariant; the gate's band energies are shift-invariant)."""

    def __init__(self, model: nn.Module, shifts=ODD8):
        super().__init__()
        self.model, self.shifts = model, tuple(tuple(s) for s in shifts)

    def forward(self, x):
        acc = 0
        for s in self.shifts:
            acc = acc + torch.roll(self.model(torch.roll(x, s, dims=(2, 3, 4))), tuple(-v for v in s), dims=(2, 3, 4))
        return acc / len(self.shifts)


@torch.no_grad()
def channel_vrmse(model, split, pairs, bsz=2) -> torch.Tensor:
    """(N, 7) per-window, per-channel VRMSE (the Well's formula, as FG.vrmse before its channel mean)."""
    model.eval(); v = []
    for x, y in FG.well2d.batched(split, pairs, bsz):
        p, y = model(x.to(FG.DEV)), y.to(FG.DEV); dims = tuple(range(2, y.dim()))
        mse = (p - y).square().mean(dims); var = (y - y.mean(dims, keepdim=True)).square().mean(dims)
        v.append((mse / (var + 1e-7)).clamp_min(1e-12).sqrt().float().cpu())
    return torch.cat(v)


def summary(v: torch.Tensor) -> dict:
    """per-channel mean (our convention) and field-grouped mean (density, B, v; the published table's convention)."""
    g = torch.stack([v[:, 0], v[:, 1:4].mean(1), v[:, 4:7].mean(1)], 1).mean(1)
    return {"per_channel": float(v.mean()), "field_grouped": float(g.mean()), "channels": v.mean(0).tolist(), "n": len(v)}


def main_tta(a) -> None:
    """§29c: VALID decides between divergence-free alone and shift average + divergence-free; TEST once, both conventions."""
    tag = f"MHD_64_{a.variant}_vrmse_gate_s{a.seed}{a.tag_suffix}"
    (train, valid, test), trunk = FG.load_data_and_trunk("MHD_64", a.variant)
    model = FG.GatedHead(trunk, train.C, tuple(train.S), torch.ones(train.C, FG.NB), True).to(FG.DEV)
    model.load_state_dict(torch.load(FG.OUT / f"{tag}.pt", weights_only=True, map_location=FG.DEV)["model"])
    arms = {"raw": model, "divfree": DivFreeB(model), "tta_divfree": DivFreeB(ShiftAverage(model))}
    arms = {k: m.to(FG.DEV) for k, m in arms.items()}
    vp = FG.strat_pairs(valid, 1, 2)
    res = {"tag": tag, "shifts": ODD8, "valid": {k: summary(channel_vrmse(m, valid, vp)) for k, m in arms.items()}}
    print(f"[tta] {tag} VALID " + " | ".join(f"{k} {v['per_channel']:.5f}" for k, v in res["valid"].items()), flush=True)
    res["adopted"] = res["valid"]["tta_divfree"]["per_channel"] < res["valid"]["divfree"]["per_channel"]
    if res["adopted"]:                                                           # registered: TEST once
        tw = FG.well2d.all_test_windows(test)
        res["test"] = {k: summary(channel_vrmse(m, test, tw)) for k, m in arms.items()}
        print(f"[tta] {tag} FULL TEST " + " | ".join(f"{k} {v['per_channel']:.5f} (grouped {v['field_grouped']:.5f})"
                                                      for k, v in res["test"].items()), flush=True)
    (FG.OUT / f"{tag}_tta_divfree.json").write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="vanilla"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag-suffix", default=""); ap.add_argument("--tta", action="store_true", help="§29c")
    a = ap.parse_args()
    if a.tta:
        return main_tta(a)
    tag = f"MHD_64_{a.variant}_vrmse_gate_s{a.seed}{a.tag_suffix}"
    (train, valid, test), trunk = FG.load_data_and_trunk("MHD_64", a.variant)
    model = FG.GatedHead(trunk, train.C, tuple(train.S), torch.ones(train.C, FG.NB), True).to(FG.DEV)
    model.load_state_dict(torch.load(FG.OUT / f"{tag}.pt", weights_only=True, map_location=FG.DEV)["model"])
    proj = DivFreeB(model).to(FG.DEV)
    vp = FG.strat_pairs(valid, 1, 2)
    v_raw, v_proj = FG.evaluate(model, valid, vp).mean().item(), FG.evaluate(proj, valid, vp).mean().item()
    res = {"tag": tag, "valid_raw": v_raw, "valid_divfree": v_proj, "n_valid": len(vp), "adopted": v_proj < v_raw}
    print(f"[divfree] {tag} VALID {v_raw:.5f} -> {v_proj:.5f} ({100 * (v_proj / v_raw - 1):+.2f}%)", flush=True)
    if res["adopted"]:                                                           # registered: TEST once, only if VALID improves
        t = FG.evaluate(proj, test, FG.well2d.all_test_windows(test))
        res.update(test_divfree=float(t.mean()), test_divfree_median=float(t.median()), n_test=len(t))
        prev = json.loads((FG.OUT / f"{tag}.json").read_text()).get("full_test")
        res["test_raw"] = prev
        print(f"[divfree] {tag} FULL TEST {prev} -> {t.mean():.5f} (n={len(t)})", flush=True)
    (FG.OUT / f"{tag}_divfree.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
