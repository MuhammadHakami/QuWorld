"""§37 fusion step 1: where is S lost? (VALID only; the submitted entry = ens_select_mlp.json, 7 members, sigma 0.)

    .venv/bin/python paper/strat/fusion/diag.py   ->  runs/strat/fusion/{pred_valid.npz, parts_entry.pkl, diag.json}

1. CPU inference of the 7 entry members on all 370 VALID shots (coefficients + scalar heads per member, saved).
2. Per-frame official-scorer parts for the entry (fx._frame_parts) on all VALID shots; check the full-VALID composite
   against the recorded official number (0.99127).
3. Exact decomposition of 1 - S by output group x shot stratum x frame phase (fixed full-VALID normalisers), the
   concentration of the loss over shots, and a descriptive split by whether the plasma-current input is live at the
   EFIT times (|Ip feature| > 100 kA; diagnostic, not a registered stratum).
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np

import fx


def phase_masks(d, s):
    rows = fx.valid_rows(d, s)
    pos = d.edge_feats()[rows, 1]
    return {"ramp-up": pos < 0.1, "flat-top": (pos >= 0.1) & (pos <= 0.9), "ramp-down": pos > 0.9}


def main() -> None:
    t0 = time.time()
    sota, d = fx.data()
    valid = d.shots_of(1)
    rows_by = {s: fx.valid_rows(d, s) for s in valid}
    pf = fx.OUT / "pred_valid.npz"
    if not pf.exists():
        preds = {m: fx.member_predict(fx.TRUNKS / m, d, rows_by) for m in fx.ENTRY}
        np.savez(pf, **{f"{i}_{s}_c": preds[m][s][0] for i, m in enumerate(fx.ENTRY) for s in valid},
                 **{f"{i}_{s}_s": preds[m][s][1] for i, m in enumerate(fx.ENTRY) for s in valid})
        print(f"[diag] member inference {time.time() - t0:.0f}s", flush=True)
    z = np.load(pf)
    comps = d.meta["comps"][:64].numpy().astype(np.float64); mean = d.meta["mean"].numpy().astype(np.float64)
    H = sota.HEADS

    def entry(s):
        c = np.mean([z[f"{i}_{s}_c"] for i in range(len(fx.ENTRY))], 0).astype(np.float64)
        sc = np.mean([z[f"{i}_{s}_s"] for i in range(len(fx.ENTRY))], 0).astype(np.float64)
        return c @ comps + mean, sc[:, H.index("efit_q95")], sc[:, H.index("efit_beta_n")]

    cons_mean = np.zeros(7)                                  # fails are ~0; exact fold mean filled from GT below
    parts = fx.frame_parts(d, valid, entry, cons_mean)
    nm = fx.norms(parts)
    print(f"[diag] per-frame scoring {time.time() - t0:.0f}s fails {sum(p['fails'] for p in parts.values())}", flush=True)
    with open(fx.OUT / "parts_entry.pkl", "wb") as f:
        pickle.dump(dict(parts=parts, norms=nm), f)
    st = json.loads((fx.OUT / "strata.json").read_text())
    info = {int(k): v for k, v in st["valid_shot_info"].items()}
    cell_name = {v["cell"]: k for k, v in st["strata"].items()}
    ipcol = d.fn.index("mag_plasma_current")
    live = {s: float(np.median(np.abs(np.asarray(d.X[rows_by[s], ipcol])))) > 100 for s in valid}

    full = np.stack([fx.shot_sums(parts[s]) for s in valid])
    overall = fx.composite(full, nm)
    out = dict(entry=fx.ENTRY, n_valid_shots=len(valid), full_valid=overall,
               recorded_official_full_valid=json.loads((fx.ROOT / "runs/sophelio_sota/ens_select_mlp.json").read_text())["valid_full"]["0.0"])

    # loss decomposition: share of 1 - S by output group and by consistency scalar
    L = overall["loss"]; tot = sum(L.values())
    cons_loss = {k: float(fx.W_CONS / 7 * (1 - v)) for k, v in overall["cons_each"].items()}
    out["loss_total"] = tot
    out["loss_share_group"] = {k: v / tot for k, v in L.items()}
    out["loss_cons_scalar"] = cons_loss

    def block(shots, fsel_fn=None):
        stt = np.stack([fx.shot_sums(parts[s], None if fsel_fn is None else fsel_fn(s)) for s in shots])
        r = fx.composite(stt, nm)
        # additive contribution of these shots/frames to the full-VALID 1 - S (psi, qb, cons exactly; D approx)
        s_ = stt.sum(0); fs = full.sum(0)
        contrib = dict(psi=fx.W_PSI * s_[0] / (fs[1] * nm["npix"] * nm["psi_var"]),
                       qb=fx.W_QB * 0.5 * (s_[2] / (fs[3] * nm["q_var"]) + s_[4] / (fs[5] * nm["b_var"])),
                       cons=fx.W_CONS / 7 * float(np.sum(s_[8:15] / (fs[15:22] * nm["c_var"]))),
                       dlcfs=fx.W_LCFS * float(np.sum(stt[stt[:, 7] > 0, 6] / stt[stt[:, 7] > 0, 7])) / len(valid))
        frames = int(s_[1])
        return dict(n_shots=len(shots), frames=frames, S_as_fold=r["S"], consistency=r["consistency"], r2_qb=r["r2_qb"],
                    dlcfs=r["dlcfs"], r2_psi=r["r2_psi"], cons_each=r["cons_each"],
                    contrib_to_1mS=contrib, contrib_total=float(sum(contrib.values())), frame_share=frames / float(fs[1]))

    out["by_stratum_allvalid"] = {cell_name[c]: block([s for s in valid if info[s]["cell"] == c]) for c in range(9)}
    out["by_phase_allvalid"] = {ph: block(valid, lambda s, ph=ph: phase_masks(d, s)[ph]) for ph in ("ramp-up", "flat-top", "ramp-down")}
    out["by_ip_live"] = {k: block([s for s in valid if live[s] == v]) for k, v in (("Ip-live", True), ("Ip-dead", False))}
    out["by_ip_live_x_phase"] = {f"{k}|{ph}": block([s for s in valid if live[s] == v], lambda s, ph=ph: phase_masks(d, s)[ph])
                                 for k, v in (("Ip-live", True), ("Ip-dead", False)) for ph in ("ramp-up", "flat-top", "ramp-down")}
    # concentration over shots: share of the loss held by the worst 5 / 10 % of shots
    per_shot = {s: block([s])["contrib_total"] for s in valid}
    v = np.sort(np.array(list(per_shot.values())))[::-1]
    out["shot_concentration"] = {"worst5pct_share": float(v[:len(v) // 20].sum() / v.sum()),
                                 "worst10pct_share": float(v[:len(v) // 10].sum() / v.sum()),
                                 "worst_shots": [(d.meta["names"][s], per_shot[s], info[s]) for s in
                                                 sorted(per_shot, key=per_shot.get, reverse=True)[:10]]}
    # frame concentration for consistency: worst 1 % of frames
    cres = np.concatenate([parts[s]["cres"] for s in valid]) / nm["c_var"]
    cf = np.nansum(cres, 1); cf = np.sort(cf[np.isfinite(cf)])[::-1]
    out["cons_frame_concentration"] = {"worst1pct_share": float(cf[:len(cf) // 100].sum() / cf.sum()),
                                       "worst5pct_share": float(cf[:len(cf) // 20].sum() / cf.sum())}
    out["ip_live_fraction_valid"] = float(np.mean(list(live.values())))
    (fx.OUT / "diag.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("full_valid", "loss_share_group", "loss_cons_scalar")}, indent=0, default=float)[:3000])
    for k in ("by_stratum_allvalid", "by_phase_allvalid", "by_ip_live", "by_ip_live_x_phase"):
        for n_, b in out[k].items():
            print(f"[diag] {k} {n_:28s} shots {b['n_shots']:3d} fr {b['frame_share']:.3f} S {b['S_as_fold']:.5f} cons {b['consistency']:.4f} "
                  f"qb {b['r2_qb']:.4f} dl {b['dlcfs']:.4f} | 1-S contrib {b['contrib_total']:.5f} {({kk: round(vv, 5) for kk, vv in b['contrib_to_1mS'].items()})}")
    print("[diag] concentration", out["shot_concentration"]["worst5pct_share"], out["shot_concentration"]["worst10pct_share"],
          out["cons_frame_concentration"], f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
