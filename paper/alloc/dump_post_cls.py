"""Doc 35 §33 export: the residual LEFT after the strong classical part (gate + classical head, the cls_only arm), in the
band-POD coefficients, for TRAIN and VALID only (TEST is never loaded). Input to the reviewer's post-classical ceiling
rerun (paper/codex/, diagnostic only): which modes and how much error remain for any co-processor, per window.

Per split it saves, for every window: a (N, C*M) remaining mode coefficients, E (N, C) remaining full-field residual
energy, D (N, C) the metric denominators, band (C, M), and the classical-only VRMSE per window (mean over fields of
sqrt(E/D), the metric adrc.py reports). Mode features f and g stay in the source cache (same row order).
For a routed arm (adrc_ryd, adrc_mlp) it also saves idx (N, K): the qubit-routed modes each window selected, so a
perfect-cancellation ceiling can be scored on the route actually used (allocation loss vs correction loss), and
a_cls / E_cls: the residual under this arm's own gate and classical head applied to EVERY mode (the head's values that
the qubit path replaces at idx), i.e. what the qubit-routed modes faced; a_nocore / E_nocore / vrmse_nocore: the same
checkpoint with the core's term A(o) removed (keep_q = 0, as diag_s33's valid_no_core), so the fitted-core effect and
the route ceiling use the same co-trained gate and head.

    .venv/bin/python paper/alloc/dump_post_cls.py runs/alloc/ckpt_s33/A/<row>_cls_only_gate_s<seed>.pt
        -> runs/alloc/post_cls/<row>_s<seed>.pt            (routed arm: <row>_<arm>_s<seed>.pt)
"""
from __future__ import annotations

import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adrc  # noqa: E402
import diag_s33  # noqa: E402

OUT = adrc.ROOT / "runs/alloc/post_cls"
BS = 64


@torch.no_grad()
def post(model, gate, d, route) -> dict:
    A, E, V, I, AC, EC, AN, EN, VN = [], [], [], [], [], [], [], [], []
    for i in range(0, d.N, BS):
        rows = torch.arange(i, min(i + BS, d.N), device=adrc.DEV); x, s = d.phi(rows)
        al = gate(d.zin(rows)) if gate is not None else None
        corr, aux = model(x, s, route)
        if "idx" in aux:
            I.append(aux["idx"].cpu())
            ccls = (s.squeeze(-1) * model.W_cls(x).squeeze(-1)).double()               # classical head, all modes
            a0, e0 = d.resid(rows, al)
            g0 = (a0.square() - (a0 - ccls).square()).reshape(len(rows), d.C, d.M).sum(-1)
            AC.append((a0 - ccls).float().cpu()); EC.append((e0 - g0).clamp_min(0).cpu())
            model.keep_q = 0.0; cn = model(x, s, route)[0].double(); model.keep_q = 1.0         # same checkpoint, no core
            gn = (a0.square() - (a0 - cn).square()).reshape(len(rows), d.C, d.M).sum(-1)
            AN.append((a0 - cn).float().cpu()); EN.append((e0 - gn).clamp_min(0).cpu())
            VN.append(d.metric(rows, cn, al)[0].cpu())
        a, e = d.resid(rows, al); c = corr.double()
        gain = (a.square() - (a - c).square()).reshape(len(rows), d.C, d.M).sum(-1)
        A.append((a - c).float().cpu()); E.append((e - gain).clamp_min(0).cpu())
        V.append(d.metric(rows, corr, al)[0].cpu())
    return {"a": torch.cat(A), "E": torch.cat(E), "D": d.D.cpu(), "vrmse": torch.cat(V), **({"idx": torch.cat(I), "a_cls": torch.cat(AC), "E_cls": torch.cat(EC), "a_nocore": torch.cat(AN),
                                                   "E_nocore": torch.cat(EN), "vrmse_nocore": torch.cat(VN)} if I else {})}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in map(pathlib.Path, sys.argv[1:]):
        ck = torch.load(p, weights_only=False)
        model, gate, va, route = diag_s33.rebuild(ck)
        tr = adrc.Data(ck["args"]["row"], "train")                     # its own stats are the run's TRAIN stats
        res = {"train": post(model, gate, tr, route), "valid": post(model, gate, va, route),
               "band": va.band.cpu() if va.band is not None else None, "src": p.name, "env": ck.get("env", {})}
        arm = "" if ck["args"]["arm"] == "cls_only" else f"_{ck['args']['arm']}"
        out = OUT / f"{ck['args']['row']}{arm}_s{ck['args']['seed']}.pt"
        torch.save(res, out)
        print(f"[post-cls] {out.name}: valid VRMSE {float(res['valid']['vrmse'].mean()):.6f} "
              f"(N train {len(res['train']['a'])}, valid {len(res['valid']['a'])})", flush=True)


if __name__ == "__main__":
    main()
