"""Doc 35 §37, MHD_64: print markdown tables from runs/strat/mhd/{results,flower_curve,swa_valid}.json (CPU only).

    uv run --no-sync python paper/strat/mhd/report_tables.py > runs/strat/mhd/tables.md
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[3] / "runs" / "strat" / "mhd"
CH = ["rho", "Bx", "By", "Bz", "vx", "vy", "vz"]


def f4(x):
    return f"{x:.4f}"


def ci(o, k):
    return f"[{o[k][0]:+.4f}, {o[k][1]:+.4f}]"


def main() -> None:
    R = json.loads((OUT / "results.json").read_text())
    print(f"VALID sample n = {R['n_valid']}, TRAIN sample n = {R['n_train']}, regime-read mismatches {R['regime_read_mismatch_valid']}\n")
    print("## Variants on the VALID sample (mean per-window VRMSE; Ma strata; TRAIN sample where scored)\n")
    print("| variant | VALID | Ma 0.7 | Ma 2 | t0 3-26 | 27-50 | 51-74 | 75-98 | TRAIN |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n, v in R["valid_mean"].items():
        s = R["valid_by_stratum"].get(n)
        if s is None:
            continue
        tb = s["tbin"]
        print(f"| {n} | {f4(v)} | {f4(s['Ma']['Ma0.7'])} | {f4(s['Ma']['Ma2.0'])} | " + " | ".join(f4(tb[f't{i}']) for i in range(4))
              + f" | {f4(R['train_mean'][n]) if n in R.get('train_mean', {}) else '-'} |")
    print("\n## Per trajectory (Ma x Ms), VALID sample\n")
    names = ["P", "C_raw", "F_raw", "C_all30_df", "F_all30_df"]
    print("| stratum | " + " | ".join(names) + " | F_raw/C_raw |"); print("|---" * (len(names) + 2) + "|")
    for k in R["valid_by_stratum"]["F_raw"]["Ma_Ms"]:
        vals = [R["valid_by_stratum"][n]["Ma_Ms"][k] for n in names]
        print(f"| {k} | " + " | ".join(f4(v) for v in vals) + f" | {vals[2] / vals[1]:.3f} |")
    print("\n## Per channel and band (VALID sample)\n")
    D = R["diagnosis"]
    for n in ("C_raw", "F_raw", "F_all30_df"):
        for reg in ("all", "Ma0.7", "Ma2"):
            d = D[n][reg]
            print(f"- {n} {reg}: channels " + ", ".join(f"{c} {v:.4f}" for c, v in d["per_channel_vrmse"].items())
                  + "; error-energy share by |k| band " + ", ".join(f"{b} {v:.3f}" for b, v in d["band_share"].items()))
    print(f"- target variance share by band: {D['target_band_share_of_var']}")
    for k, v in R["C_over_F"].items():
        print(f"- {k}: band energy ratio {v['band_energy_ratio']}; channel VRMSE ratio {v['channel_vrmse_ratio']}")
    print("\n## Excess over 0.1165 (Flowers paper TEST) by stratum, VALID sample\n")
    for n, e in R["excess"].items():
        print(f"- {n}: uniform factor needed {e['uniform_factor_needed']:.3f}; excess {e['excess_over_0.1165']}; "
              f"share {e['share_of_positive_excess']}")
    print("\n## TTA and divergence-free B vs the raw model (VALID sample; paired bootstrap 95% CI of the delta)\n")
    print("| variant | rel. delta | CI (windows in strata) | CI (trajectories in Ma) | Ma 0.7 | Ma 2 | frac. windows better |")
    print("|---|---|---|---|---|---|---|")
    for k, t in R["tta"].items():
        o, s = t["overall"], t["strata"]
        print(f"| {k} | {o['rel_pct']:+.2f}% | {ci(o, 'ci95_window')} | {ci(o, 'ci95_traj')} | {s['Ma0.7']['rel_pct']:+.1f}% "
              f"{ci(s['Ma0.7'], 'ci95')} | {s['Ma2.0']['rel_pct']:+.1f}% {ci(s['Ma2.0'], 'ci95')} | {o['win_frac_better']:.2f} |")
    print("\n## Per time bin, best TTA vs raw\n")
    for k in ("F_all30_df", "C_all30_df", "F_raw_vs_C_raw", "F_all30_df_vs_C_all30_df"):
        s = R["tta"][k]["strata"]
        print(f"- {k}: " + "; ".join(f"{kk} {v['rel_pct']:+.1f}% {ci(v, 'ci95')}" for kk, v in s.items() if "_t" in kk))
    print("\n## Individual TTA passes (mean VALID VRMSE)\n")
    for m, pp in R["per_pass_valid"].items():
        print(f"- {m}: " + ", ".join(f"{k} {v:.4f}" for k, v in pp.items()))
    if "blends" in R:
        print("\n## TRAIN-fitted blends and regime corrections, scored on VALID (delta vs the anchor)\n")
        print("| strategy / groups | TRAIN | VALID | rel. delta | CI windows | CI trajectories | Ma 0.7 | Ma 2 |")
        print("|---|---|---|---|---|---|---|---|")
        for k, b in R["blends"].items():
            o, s = b["vs_anchor"], b["strata_vs_anchor"]
            print(f"| {k} | {f4(b['train'])} | {f4(b['valid'])} | {o['rel_pct']:+.2f}% | {ci(o, 'ci95_window')} | "
                  f"{ci(o, 'ci95_traj')} | {s['Ma0.7']['rel_pct']:+.1f}% | {s['Ma2.0']['rel_pct']:+.1f}% |")
    fc = OUT / "flower_curve.json"
    if fc.exists():
        C = json.loads(fc.read_text())
        print(f"\n## Flowers VALID curve (log {C['log']}, {C['n_points']} points after warm-up, last epoch "
              f"{C['last_epoch']:.1f}, best {C['best']:.4f})\n")
        for k, v in C["fits"].items():
            print(f"- {k}: epoch 100 {v['epoch100']:.4f} (bootstrap 95% [{v['epoch100_ci95'][0]:.4f}, {v['epoch100_ci95'][1]:.4f}]),"
                  f" epoch 50 {v['epoch50']:.4f}, fit rmse {v['rmse']:.4f}")
    sw = OUT / "swa_valid.json"
    if sw.exists():
        S = json.loads(sw.read_text())
        per = [np.mean(p) for p in S["per_snapshot"]]
        print(f"\n## Weight averaging (VALID sample; steps {S['steps']})\n")
        print(f"- single snapshots: " + ", ".join(f"{s} {v:.4f}" for s, v in zip(S["steps"], per)))
        print(f"- prediction ensemble {np.mean(S['ensemble']):.4f}; SWA {np.mean(S['swa']):.4f}; SWA all30+df {np.mean(S['swa_all30_df']):.4f}")


if __name__ == "__main__":
    main()
