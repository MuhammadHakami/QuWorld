"""§37 fusion: exact additive decomposition of the entry's full-VALID 1 - S into shot stratum x frame phase x output
group (psi, q95, betaN, D_LCFS, the 7 consistency scalars), plus the Ip-live / Ip-dead split (diagnostic).

    .venv/bin/python paper/strat/fusion/loss_table.py   ->  runs/strat/fusion/loss_table.json

With SS_tot fixed at the full-VALID values, each R2 term's loss is a sum over frames; D_LCFS (mean over shots of the
per-shot frame mean) is split over frames as d_f / (n_frames_shot * n_shots). The cells sum to 1 - S exactly.
"""
from __future__ import annotations

import json
import pickle

import numpy as np

import fx

PH = ("ramp-up", "flat-top", "ramp-down")


def main() -> None:
    _, d = fx.data()
    e = pickle.load(open(fx.OUT / "parts_entry.pkl", "rb")); parts, nm = e["parts"], e["norms"]
    st = json.loads((fx.OUT / "strata.json").read_text())
    info = {int(k): v for k, v in st["valid_shot_info"].items()}
    cname = {v["cell"]: k for k, v in st["strata"].items()}
    valid = d.shots_of(1)
    ipcol = d.fn.index("mag_plasma_current")
    ntot = sum(len(parts[s]["psi_res"]) for s in valid)
    nq = sum(np.isfinite(parts[s]["q_gt"]).sum() for s in valid); nb = sum(np.isfinite(parts[s]["b_gt"]).sum() for s in valid)
    nc = sum(np.isfinite(parts[s]["cres"]).sum(0) for s in valid)
    groups = ["psi", "q95", "betaN", "dlcfs"] + fx.CONS
    cells = {}
    for s in valid:
        p = parts[s]; T = len(p["psi_res"])
        pos = d.edge_feats()[fx.valid_rows(d, s), 1]
        ph = np.where(pos < 0.1, 0, np.where(pos > 0.9, 2, 1))
        live = float(np.median(np.abs(np.asarray(d.X[fx.valid_rows(d, s), ipcol])))) > 100
        nd = np.isfinite(p["d"]).sum()
        per = np.zeros((T, len(groups)))
        per[:, 0] = fx.W_PSI * p["psi_res"] / (ntot * nm["npix"] * nm["psi_var"])
        per[:, 1] = np.nan_to_num(fx.W_QB * 0.5 * p["q_res"] / (nq * nm["q_var"]))
        per[:, 2] = np.nan_to_num(fx.W_QB * 0.5 * p["b_res"] / (nb * nm["b_var"]))
        per[:, 3] = np.nan_to_num(fx.W_LCFS * p["d"] / (max(nd, 1) * len(valid)))
        per[:, 4:] = np.nan_to_num(fx.W_CONS / 7 * p["cres"] / (nc * nm["c_var"]))
        for k in range(3):
            key = (cname[info[s]["cell"]], PH[k], "Ip-live" if live else "Ip-dead")
            cells.setdefault(key, np.zeros(len(groups) + 1))
            cells[key][:-1] += per[ph == k].sum(0); cells[key][-1] += (ph == k).sum()
    tot = sum(v[:-1].sum() for v in cells.values())

    def agg(fn):
        out = {}
        for k, v in cells.items():
            out.setdefault(fn(k), np.zeros(len(groups) + 1)); out[fn(k)] += v
        return {kk: dict(loss=float(v[:-1].sum()), share=float(v[:-1].sum() / tot), frame_share=float(v[-1] / ntot),
                         loss_per_frame_rel=float(v[:-1].sum() / tot / (v[-1] / ntot)),
                         by_group={g: float(x) for g, x in zip(groups, v[:-1])}) for kk, v in out.items()}
    res = dict(one_minus_S=float(tot), groups=groups,
               by_group={g: float(sum(v[i] for v in cells.values())) for i, g in enumerate(groups)},
               by_phase=agg(lambda k: k[1]), by_stratum=agg(lambda k: k[0]), by_ip=agg(lambda k: k[2]),
               by_ip_phase=agg(lambda k: f"{k[2]}|{k[1]}"))
    (fx.OUT / "loss_table.json").write_text(json.dumps(res, indent=1))
    print(f"1-S {tot:.5f}")
    print("by group:", {g: round(v, 5) for g, v in res["by_group"].items()})
    for key in ("by_phase", "by_ip", "by_ip_phase", "by_stratum"):
        for k, v in sorted(res[key].items(), key=lambda kv: -kv[1]["loss"]):
            top = sorted(v["by_group"].items(), key=lambda kv: -kv[1])[:4]
            print(f"{key:11s} {k:28s} share {v['share']:.3f} frames {v['frame_share']:.3f} rel/frame {v['loss_per_frame_rel']:.2f} "
                  f"top {[(g, round(x, 5)) for g, x in top]}")


if __name__ == "__main__":
    main()
