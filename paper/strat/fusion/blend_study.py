"""Doc 35 §41s: multi-algorithm ensemble study - per-target stacking weights instead of the hand-set blends.

Members are TRAIN-only models evaluated on ALL VALID + TEST frames (743 shots; out-of-sample for every member):
  shape scalars (R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li): J1 MLP heads (seed means), the register heads (J1QPIpi, J1QPI),
    thz / thz+hfull / thp head variants when present, and the NAS entry ensemble's own scalar outputs (li, R_axis, Z_axis);
  q95 / betaN: QH MLP heads, older q/b heads, the QF trunks, the entry ensemble (the deployed blend is 0.5 QH + 0.5 entry).
Per target: non-negative weights summing to 1 (SSE-optimal, projected gradient on the simplex), fitted on 4/5 of the shots and scored
on the held-out 1/5 (5-fold GroupKFold) -> out-of-fold R2 of the stacked blend vs the deployed blend and vs the best single member;
also the full-data weights (for transfer to the all-data counterparts) and the member error correlations.

    .venv/bin/python paper/strat/fusion/blend_study.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

import fx

SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def simplex_ls(M, y, it=3000):
    """argmin_w ||M w - y||^2, w >= 0, sum w = 1 (accelerated projected gradient; M columns are member predictions)."""
    G, b = M.T @ M, M.T @ y; L = np.linalg.eigvalsh(G).max(); w = np.full(M.shape[1], 1 / M.shape[1]); z, t = w.copy(), 1.0
    def proj(v):
        u = np.sort(v)[::-1]; c = np.cumsum(u); r = np.nonzero(u * np.arange(1, len(v) + 1) > c - 1)[0][-1]
        return np.maximum(v - (c[r] - 1) / (r + 1), 0)
    for _ in range(it):
        wn = proj(z - (G @ z - b) / L); tn = (1 + np.sqrt(1 + 4 * t * t)) / 2; z = wn + (t - 1) / tn * (wn - w); w, t = wn, tn
    return w


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def study(name, y, members, deployed, shots, folds=5):
    ok = np.isfinite(y) & np.all(np.isfinite(np.stack(list(members.values()), 1)), 1)
    y, M, sh = y[ok], np.stack([members[k][ok] for k in members], 1), shots[ok]; keys = list(members)
    us = np.random.default_rng(0).permutation(np.unique(sh)); oof = np.zeros(len(y))
    for f in range(folds):
        te = np.isin(sh, us[f::folds]); w = simplex_ls(M[~te], y[~te]); oof[te] = M[te] @ w
    w_all = simplex_ls(M, y); single = {k: r2(y, M[:, i]) for i, k in enumerate(keys)}; dep = r2(y, deployed[ok])
    E = M - y[:, None]; C = np.corrcoef(E.T)
    out = {"oof_stack": r2(y, oof), "deployed": dep, "best_single": max(single.values()), "single": single,
           "weights_all": dict(zip(keys, np.round(w_all, 4).tolist())), "err_corr": np.round(C, 3).tolist(), "members": keys}
    print(f"[blend] {name:8s} deployed {dep:.5f} | best single {max(single, key=single.get)} {max(single.values()):.5f} | "
          f"OOF stack {out['oof_stack']:.5f} ({out['oof_stack'] - dep:+.5f}) | w " + " ".join(f"{k}:{v:.2f}" for k, v in out["weights_all"].items() if v > 0.005), flush=True)
    return out


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import ens_select as ES
    import gpu_build as GB
    d = sota.Data(); vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); rows, Y = vt["rows"], vt["y"][:, :7].astype(np.float64)
    assert np.all(np.diff(rows) > 0); shots = np.unique(d.shot[rows]); sh = d.shot[rows]
    H = sota.HEADS
    def heads(stems):
        return GB.heads_gpu(d, rows, "+".join(stems))
    have = lambda st: (fx.OUT / f"{st}.pt").exists()                       # noqa: E731
    sel = json.loads((GB.SOTA / "ens_select_mlp.json").read_text())
    ent = [ES.member(str(fx.TRUNKS / m), d, shots) for m in sel["members"]]
    esc = np.concatenate([np.mean([m[s][1] for m in ent], 0) for s in shots])    # rows are shot-sorted, shots ascending
    qf = []
    for p in GB.QF:
        cf = torch.load(GB.SOTA / p, map_location="cpu", weights_only=False)["cfg"]
        qf.append(ES.member(str(GB.SOTA / p), GB.fixed(d, GB.mf_name(p, {})) if cf.get("inputs") == "magfix" else d, shots))
    qsc = np.concatenate([np.mean([m[s][1] for m in qf], 0) for s in shots])
    shape = {"J1": heads([f"head_boundary_v8_J1_s{s}" for s in range(1, 7)])[:, :7],
             "QPIpi": heads([f"head_boundary_v8_J1QPIpi_s{s}" for s in range(1, 4)])[:, :7],
             "QPI": heads(["head_boundary_v8_J1QPI_s1"])[:, :7]}
    for tag in ("J1T", "J1THS", "J1THP"):
        if have(f"head_boundary_v8_{tag}_s1"):
            shape[tag] = heads([f"head_boundary_v8_{tag}_s1"])[:, :7]
    J = [f"head_boundary_v8_J1_s{s}" for s in range(1, 7)]; dep_shape = heads(J + J + ["head_boundary_v8_J1QPIpi_s1", "head_boundary_v8_J1QPIpi_s2"])[:, :7]
    res = {}
    for j, nm in enumerate(SC):
        mem = {k: v[:, j] for k, v in shape.items()}
        if nm in ("li", "R_axis", "Z_axis"):
            mem["entry"] = esc[:, H.index({"li": "efit_li", "R_axis": "efit_r_axis", "Z_axis": "efit_z_axis"}[nm])]
        res[nm] = study(nm, Y[:, j], mem, dep_shape[:, j], sh)
    yq = d.S[rows][:, [sota.LABELS.index("efit_q95"), sota.LABELS.index("efit_beta_n")]].astype(np.float64)
    qb = {"QH": heads([f"head_qb_QH_s{s}" for s in range(1, 7)])[:, :2], "qb_old": heads([f"head_qb_s{s}" for s in range(1, 4)])[:, :2]}
    if have("head_qb_QP_s1"):
        qb["QP"] = heads(["head_qb_QP_s1"])[:, :2]
    if have("head_qb_QHP_s1"):
        qb["QHP"] = heads(["head_qb_QHP_s1"])[:, :2]
    for j, nm in enumerate(["q95", "betaN"]):
        hk = H.index(("efit_q95", "efit_beta_n")[j]); mem = {k: v[:, j] for k, v in qb.items()}
        mem["entry"], mem["QF"] = esc[:, hk], qsc[:, hk]
        res[nm] = study(nm, yq[:, j], mem, 0.5 * (qb["QH"][:, j] + esc[:, hk]), sh)
    sm = lambda k: np.mean([res[n][k] for n in SC])                         # noqa: E731
    print(f"[blend] consistency-7 mean: deployed {sm('deployed'):.5f} OOF stack {sm('oof_stack'):.5f} ({sm('oof_stack') - sm('deployed'):+.5f}); "
          f"q/b mean: deployed {np.mean([res[n]['deployed'] for n in ('q95', 'betaN')]):.5f} OOF stack {np.mean([res[n]['oof_stack'] for n in ('q95', 'betaN')]):.5f}", flush=True)
    (fx.OUT / "blend_study.json").write_text(json.dumps(res, indent=1)); print("BLEND_DONE", flush=True)


if __name__ == "__main__":
    main()
