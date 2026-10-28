"""Doc 35 §33 stage A: mechanism diagnostics of a trained routed co-processor, on VALID only (TEST is never loaded).

From a checkpoint saved by adrc.py (QW_CKPT), rebuild the model and gate and report, over every VALID window:
  bypass   |A(o)| / |W_q(x_q)| per routed mode (how much of the routed correction goes through the core);
  ablation VALID VRMSE with the core's term A(o) removed (keep_q = 0), against the full model;
  config   per-input position spread (SD over windows, per atom, mean over atoms) and the per-input tethers actually used;
  swap     VALID VRMSE when each window gets another window's positions, another window's tethers (cyclic derangement
           within the batch), or the fixed grid (eval-only counterfactuals, modes and features unchanged);
  V        distribution of the pair interaction r^-6 with r clamped at 0.5 on the RAW layout, inside each group; V_eff the
           same on the register's own positions (core.positions: divided by exp(log_lam), as the forward pass), and the
           interaction phase max_l |tau_l| * V_eff (§33b: the regime check that sees a changed length scale);
  eff      correction-bearing atoms K - 2 * sum of chain tethers per input (a soft tether is a classical feed-forward
           strength; its fractional atom charge is a modelling convention, not a deployable discrete cost).
The checkpoint's QW_* environment (tether mode, 3-D, shift, loss, route target, ...) is restored before rebuilding.

    .venv/bin/python paper/alloc/diag_s33.py runs/alloc/ckpt_s33/<name>.pt [...]   ->  runs/alloc/s33_diag.jsonl
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adrc  # noqa: E402

OUT = adrc.ROOT / "runs/alloc/s33_diag.jsonl"
BS = 64


def rebuild(ck: dict):
    a = ck["args"]
    for k in [k for k in os.environ if k.startswith("QW_") and k not in ("QW_GPU_GB",)]:   # the run's exact QW_* behaviour
        del os.environ[k]
    os.environ.update(ck.get("env", {}))
    adrc.K, adrc.NG, adrc.G = ck["K"], min(10, ck["K"]), max(1, ck["K"] // 10)          # as adrc.py's --atoms
    adrc.DYN_TETHER = ck["dyn"]
    tr = adrc.Data(a["row"], "train")
    va = adrc.Data(a["row"], "valid", tr.stats)
    F = tr.phi(torch.arange(2, device=adrc.DEV))[0].shape[-1]
    model = adrc.ADRC(F, a["arm"], a["seed"]).to(adrc.DEV)
    model.load_state_dict(ck["model"]); model.eval()
    model.keep_cls, model.keep_q, model.trust, model.cf_perm = 1.0, 1.0, None, None     # adrc's credit / §33 hooks
    gate = None
    if a.get("gate"):
        gate = adrc.Gate(tr.z.shape[1], tr.alpha_fit).to(adrc.DEV)
        gate.load_state_dict(ck["gate"]); gate.eval()
    return model, gate, va, ck["route"].to(adrc.DEV)


@torch.no_grad()
def valid_vrmse(model, gate, va, route, perm_seed=None) -> float:
    ms = []
    for i in range(0, va.N, BS):
        rows = torch.arange(i, min(i + BS, va.N), device=adrc.DEV); x, s = va.phi(rows)
        if perm_seed is not None:                               # derangement within the batch (another window's config)
            n = len(rows); g = torch.Generator(device="cpu").manual_seed(perm_seed + i)
            shift = int(torch.randint(1, max(2, n), (1,), generator=g)) if n > 1 else 0     # cyclic shift: a derangement
            model.cf_perm = ((torch.arange(n) + shift) % n).to(adrc.DEV)
        al = gate(va.zin(rows)) if gate is not None else None
        ms.append(va.metric(rows, model(x, s, route)[0], al)[0])
    model.cf_perm = None
    return float(torch.cat(ms).mean())


@torch.no_grad()
def config_stats(model, va, route) -> dict:
    ratios, pos, teth, V, Ve = [], [], [], [], []
    rec = {}
    h1 = model.W_q.register_forward_hook(lambda m, i, o: rec.__setitem__("wq", o))
    h2 = model.A.register_forward_hook(lambda m, i, o: rec.__setitem__("ao", o))
    for i in range(0, va.N, BS):
        rows = torch.arange(i, min(i + BS, va.N), device=adrc.DEV); x, s = va.phi(rows)
        _, aux = model(x, s, route)
        if "ao" in rec:
            ratios.append((rec["ao"].abs() / rec["wq"].abs().clamp_min(1e-12)).flatten())
        if "pos" in aux:
            P = aux["pos"]; pos.append(P)
            B, K, _ = P.shape; Pg = P.reshape(B, K // adrc.NG, adrc.NG, -1)
            r = torch.cdist(Pg, Pg).clamp(min=0.5); off = ~torch.eye(adrc.NG, dtype=torch.bool, device=adrc.DEV)   # as the forward
            V.append(r.pow(-6)[..., off].flatten())
            if hasattr(model.core, "positions"):                     # effective geometry the register actually uses
                pe = model.core.positions(torch.cat([Pg, torch.zeros_like(Pg[..., :1])], -1))
                Ve.append(torch.cdist(pe, pe).clamp(min=model.core.MIN_DIST).pow(-6)[..., off].flatten())
        if "tethers" in aux:
            t = aux["tethers"]; teth.append(t if t.dim() == 2 else t[None].expand(len(rows), -1))
    h1.remove(); h2.remove()
    q = lambda t, ps: [round(float(v), 5) for v in torch.quantile(t.float()[:200000], torch.tensor(ps, device=t.device))]
    out = {}
    if ratios:
        out["bypass_ratio_q10_50_90"] = q(torch.cat(ratios), [0.1, 0.5, 0.9])
    if pos:
        P = torch.cat(pos); out["pos_sd_per_atom_xy"] = [round(float(v), 4) for v in P.std(0).mean(0)]
    if V:
        Vc = torch.cat(V); out["V_q10_50_90"] = q(Vc, [0.1, 0.5, 0.9]); out["V_frac_gt1"] = round(float((Vc > 1).float().mean()), 4)
    if Ve:
        Vc = torch.cat(Ve); ph = Vc * model.core.tau.detach().abs().max()
        out["V_eff_q10_50_90"] = q(Vc, [0.1, 0.5, 0.9]); out["V_eff_frac_gt1"] = round(float((Vc > 1).float().mean()), 4)
        out["phase_q50_90_99"] = q(ph, [0.5, 0.9, 0.99]); out["tau"] = [round(float(t), 4) for t in model.core.tau.detach()]
        out["log_lam"] = [round(float(t), 4) for t in model.core.log_lam.detach()]
    if teth:
        T = torch.cat(teth); out["tether_mean"] = [round(float(v), 4) for v in T.mean(0)]
        out["tether_frac_active"] = round(float((T > 1e-3).float().mean()), 4)
        out["tether_sd_over_inputs"] = [round(float(v), 4) for v in T.std(0)]
        eff = adrc.K - 2 * T[:, :max(0, adrc.G - 1)].sum(1)        # relay atoms: 2 per chain tether, charged by strength
        out["eff_atoms_mean_q10_50_90"] = [round(float(eff.mean()), 3)] + q(eff, [0.1, 0.5, 0.9])
    return out


def diagnose(path: pathlib.Path) -> dict:
    ck = torch.load(path, weights_only=False)
    model, gate, va, route = rebuild(ck)
    res = {"cell": path.parent.name, "ckpt": path.name, "arm": ck["args"]["arm"], "row": ck["args"]["row"],
           "seed": ck["args"]["seed"], "env": ck.get("env", {})}
    res["valid"] = valid_vrmse(model, gate, va, route)
    if res["arm"] != "cls_only":
        model.keep_q = 0.0; res["valid_no_core"] = valid_vrmse(model, gate, va, route); model.keep_q = 1.0
    if hasattr(model, "hyp") and res["arm"] not in ("cls_only", "static_lin"):
        model.cf_pos = True; res["valid_swap_pos"] = valid_vrmse(model, gate, va, route, perm_seed=1); model.cf_pos = False
        model.cf_tether = True; res["valid_swap_tether"] = valid_vrmse(model, gate, va, route, perm_seed=2); model.cf_tether = False
        model.cf_grid = True; res["valid_fixed_grid"] = valid_vrmse(model, gate, va, route); model.cf_grid = False
    res.update(config_stats(model, va, route))
    return res


def main() -> None:
    for p in sys.argv[1:]:
        r = diagnose(pathlib.Path(p))
        with OUT.open("a") as f:
            f.write(json.dumps(r) + "\n")
        print("[s33-diag]", json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
