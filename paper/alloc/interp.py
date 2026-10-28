"""Model interpretation for the allocation study (used by notebooks/19_models_and_interpretation.ipynb).

    .venv/bin/python paper/alloc/interp.py fit [bench ...]     # fit + save the interpretation models (GPU)

For each row, the register co-processor (and its matched-MLP twin) is refitted at the row's VALID-best tile strategy
with seed 1 and the study's exact settings (coproc.py), then saved to runs/alloc/interp/. The analysis functions
below read the saved weights and the frozen-trunk caches. Nothing here changes a reported number: the study's
numbers come from runs/alloc/results; these refits exist only to look inside the trained models.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import types

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import coproc as CP  # noqa: E402
from cores import TileRegister, _apply_1q, _rx  # noqa: E402

OUT = HERE.parents[1] / "runs" / "alloc" / "interp"
ROWS = ["turbulent_radiative_layer_2D", "rayleigh_benard", "shear_flow", "MHD_64", "realpde_unet", "realpde_fno", "pelzer"]
_SPLITS: dict = {}


def args(K=30):
    return types.SimpleNamespace(lr=3e-3, epochs=40, patience=6, batch=64, K=K)


def splits(bench: str):
    if bench not in _SPLITS:
        _SPLITS.clear()                                     # one row on the GPU at a time
        torch.cuda.empty_cache()
        _SPLITS[bench] = {s: CP.Split(bench, s) for s in ("train", "valid", "test")}
    return _SPLITS[bench]


def fit(bench: str, core: str, seed: int = 1):
    """Refit the co-processor at the row's VALID-best strategy (the study's rule) and save weights + scorer."""
    strategy = analyze.best(bench)
    f = OUT / f"{bench}_{strategy}_{core}_s{seed}.pt"
    if f.exists():
        return f
    sp = splits(bench)
    run = CP.Runner(args(), sp)
    if strategy == "learned":
        run.fit_scorer(lambda s, rows: ((s.sets["proxy"]["coef"][rows][:, :, 4].square()) / s.D[rows][:, None].float()).sum(-1), seed)
    if strategy == "gain":
        st1 = CP.build(run, "proxy", core, seed)
        run.train(st1, "proxy", 30, seed)
        run.fit_scorer(run.realized_gain_fn(st1), seed)
    model = CP.build(run, strategy, core, seed)
    info = run.train(model, strategy, 30, seed)
    te = run.evaluate(model, sp["test"], strategy, 30)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "scorer": None if run.scorer is None else run.scorer.state_dict(),
                "strategy": strategy, "core": core, "seed": seed, "train": info,
                "test": {k: v for k, v in te.items() if k != "per"}}, f)
    print(f"[interp] {bench} {strategy} {core}: TEST {sp['test'].metric_of(sp['test'].E).mean():.5f} -> {te['metric']:.5f}", flush=True)
    return f


def load(bench: str, core: str = "ryd", seed: int = 1):
    """-> (runner, model, strategy, saved dict) with the saved weights on the GPU."""
    sp = splits(bench)
    d = torch.load(fit(bench, core, seed), weights_only=False)
    run = CP.Runner(args(), sp)
    if d["scorer"] is not None:
        run.scorer = CP.Scorer(run.fz.F).to(CP.DEV)
        run.scorer.load_state_dict(d["scorer"]); run.scorer.eval()
    model = CP.build(run, d["strategy"], core, seed)
    model.load_state_dict(d["model"]); model.eval()
    return run, model, d["strategy"], d


# ------------------------------------------------------------------------------------------------ what was learned

def register_physics(model) -> dict:
    """Learned analog controls of the tile register (TileCoproc with a TileRegister core)."""
    r = model.core
    lam = r.log_lam.exp().detach().cpu().numpy()
    return {"blockade_scale_px": lam.round(2).tolist(),                 # pixels (frames) per blockade radius, x/y/z
            "scale_plane_dz": float(r.dz.detach()), "drive_omega": r.omega.detach().cpu().numpy().round(3).tolist(),
            "evolution_tau": r.tau.detach().cpu().numpy().round(3).tolist(),
            "detuning_weight_norm": float(r.w.weight.norm()),
            "base_detuning_spread": float(r.base.std())}


def readout_usage(model) -> dict:
    """Column norms of the zero-initialised readout head A: how much each observable feeds the correction."""
    a = model.A.weight.detach().abs().sum(0).cpu().numpy()
    names = ["Z", "X", "Y"] if len(a) == 3 else ["Z_a", "X_a", "Y_a", "Z_b", "X_b", "Y_b"]
    return dict(zip(names, (a / max(a.sum(), 1e-12)).round(3).tolist()))


@torch.no_grad()
def path_split(run, model, strategy: str, n: int = 4096) -> dict:
    """How the correction splits between the shared linear path W f and the core path A o, in metric-weighted energy
    (tiles weighted by 1/D, i.e. as they count in the error). `captured_without_core` zeroes A after joint training: it
    measures how much the trained linear path relies on the core, not the core's added value; the added value is the
    difference to the separately trained `lin` arm in runs/alloc/results."""
    sp = run.sp["test"]
    rows = torch.arange(min(n, sp.N), device=CP.DEV)
    idx, coef = run.select(sp, rows, 30, strategy)
    f, s, r, coords, key = run.fz.tiles(sp, rows, idx, coef)
    w = (1 / sp.D[rows].float())[:, None, :]
    lin = s * (torch.einsum("bkf,bkfc->bkc", f, model.W[key]) + model.b[key])
    core = model(f, s, coords, key) - lin
    e = lambda t: float((t.square() * w).sum())
    A = model.A.weight.clone(); Ab = model.A.bias.clone()
    cap_full = run.evaluate(model, sp, strategy, 30)["captured"]
    model.A.weight.zero_(); model.A.bias.zero_()
    cap_lin = run.evaluate(model, sp, strategy, 30)["captured"]
    model.A.weight.copy_(A); model.A.bias.copy_(Ab)
    return {"core_energy_share": e(core) / (e(lin) + e(core) + 1e-30), "captured_full": cap_full,
            "captured_without_core": cap_lin}


def splits_of(run):
    return run.sp


@torch.no_grad()
def register_state(reg: TileRegister, f, coords):
    """Replicates TileRegister.forward and returns the final statevector (G, 2^n)."""
    G, n = f.shape[0], reg.n
    det = reg.base.T.unsqueeze(0) + (reg.w(f) if reg.mode == "rydlin" else math.pi * torch.tanh(reg.w(f)))
    pos = reg.positions(coords)
    V = torch.cdist(pos, pos).clamp(min=reg.MIN_DIST).pow(-6)
    eng = (V[:, reg.iu[0], reg.iu[1]] @ reg.pairbits.T).to(torch.complex64)
    state = torch.zeros(G, 2 ** n, dtype=torch.complex64, device=f.device); state[:, 0] = 1.0
    for l in range(reg.layers):
        gx = _rx(reg.omega[l])
        for k in range(n):
            ph = torch.exp(-0.5j * det[:, k, l].to(torch.complex64))
            d = torch.stack([ph, ph.conj()], -1).unsqueeze(-1)
            state = _apply_1q(state, d * gx, k, n)
        state = state * torch.exp(-1j * reg.tau[l].to(torch.complex64) * eng)
    return state, V


@torch.no_grad()
def entanglement(run, model, strategy: str, n_windows: int = 64) -> dict:
    """Half-register von Neumann entropy (bits, 5|5 cut; max 5) and mean single-atom purity of the trained register on
    real test groups, plus the mean nearest-neighbour interaction V_ij of the allocated geometry."""
    sp = run.sp["test"]
    rows = torch.arange(min(n_windows, sp.N), device=CP.DEV)
    idx, coef = run.select(sp, rows, 30, strategy)
    f, s, r, coords, key = run.fz.tiles(sp, rows, idx, coef)
    order = CP.morton(coords[..., :3]).argsort(1)
    g = lambda t: t.gather(1, order[..., None].expand(-1, -1, t.shape[-1]))
    B, K = f.shape[:2]
    fg, cg = g(f).reshape(B * K // 10, 10, -1), g(coords).reshape(B * K // 10, 10, -1)
    state, V = register_state(model.core, fg, cg)
    psi = state.reshape(-1, 32, 32)
    sv = torch.linalg.svdvals(psi).square().clamp_min(1e-12)
    S = -(sv * sv.log2()).sum(-1)
    pur = []
    for q in range(10):
        m = state.reshape(-1, 2 ** q, 2, 2 ** (9 - q)).permute(0, 2, 1, 3).reshape(state.shape[0], 2, -1)
        rho = m @ m.conj().transpose(1, 2)
        pur.append((rho @ rho).diagonal(dim1=1, dim2=2).sum(-1).real)
    Vo = V.clone(); Vo.diagonal(dim1=1, dim2=2).fill_(0)
    return {"half_entropy_bits_mean": float(S.mean()), "half_entropy_bits_max": float(S.max()),
            "single_atom_purity_mean": float(torch.stack(pur).mean()),
            "nn_interaction_mean": float(Vo.max(-1).values.mean()), "groups": int(state.shape[0])}


@torch.no_grad()
def correction_vs_residual(run, model, strategy: str, n: int = 512):
    """Per allocated test tile: true residual vs predicted correction (both divided by the tile scale), flattened."""
    sp = run.sp["test"]
    rows = torch.arange(min(n, sp.N), device=CP.DEV)
    idx, coef = run.select(sp, rows, 30, strategy)
    f, s, r, coords, key = run.fz.tiles(sp, rows, idx, coef)
    c = model(f, s, coords, key)
    return (r / s).flatten().cpu().numpy(), (c / s).flatten().cpu().numpy()


@torch.no_grad()
def allocation(bench: str, window: int = 0, K: int = 30) -> dict:
    """Tile positions (x, y, z, level) chosen by each rule for one test window, and the level histogram of each rule's
    top-30 over all test windows."""
    sp = splits(bench)["test"]
    out = {}
    for rule in ("oracle", "static", "proxy", "random"):
        idx = sp.sets[rule]["idx"][:, :K]
        out[rule] = {"pos": sp.pos[idx[window]].cpu().numpy(), "level": sp.level[idx[window]].cpu().numpy(),
                     "levels_all": np.bincount(sp.level[idx].flatten().cpu().numpy(), minlength=int(sp.level.max()) + 1)}
    out["ceil"] = {k: float(v.mean()) for k, v in sp.ceil.items()}
    out["shape"] = sp.meta["shape"]
    return out


@torch.no_grad()
def one_group(run, model, strategy: str, window: int = 0, group: int = 0):
    """Atom positions (in blockade units), couplings V_ij, layer-1 detunings and <Z,X,Y> readout of one trained
    10-atom register on a real test input (for the close-up figure)."""
    sp = run.sp["test"]
    rows = torch.tensor([window], device=CP.DEV)
    idx, coef = run.select(sp, rows, 30, strategy)
    f, s, r, coords, key = run.fz.tiles(sp, rows, idx, coef)
    order = CP.morton(coords[..., :3]).argsort(1)
    g = lambda t: t.gather(1, order[..., None].expand(-1, -1, t.shape[-1]))
    fg, cg = g(f).reshape(3, 10, -1)[group:group + 1], g(coords).reshape(3, 10, -1)[group:group + 1]
    reg = model.core
    pos = reg.positions(cg)[0]
    det = (reg.base.T.unsqueeze(0) + math.pi * torch.tanh(reg.w(fg)))[0, :, 0]
    V = torch.cdist(pos, pos).clamp(min=reg.MIN_DIST).pow(-6)
    o = reg(fg, cg)[0]
    return pos.cpu().numpy(), V.cpu().numpy(), det.cpu().numpy(), o.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit"]); ap.add_argument("benches", nargs="*")
    a = ap.parse_args()
    for b in a.benches or ROWS:
        for core in ("ryd", "mlp"):
            fit(b, core)


if __name__ == "__main__":
    main()
