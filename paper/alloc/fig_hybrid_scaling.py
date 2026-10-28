"""Hybrid scaling with the qubit budget (doc 35 §13): fixed classical part (trunk + gate + classical head), dynamic
input-dependent register (positions, detunings, routing and tether strengths per input), K in {0, 5, 10, 20, 30}.
Writes paper/manuscript/iclr2027/fig_hybrid_scaling.pdf and runs/alloc/hybrid_scaling.json.

    .venv/bin/python paper/alloc/fig_hybrid_scaling.py [--trunk vanilla|mr30ryd]

--trunk vanilla (doc 35 §19, default): the fixed classical part is the never-register CNextU-net trunk on every row.
--trunk mr30ryd (doc 35 §13): the register-necked trunk on shear, RB and TRL2D (MHD_64 is vanilla in both); written
to fig_hybrid_scaling_mr30.pdf / hybrid_scaling_mr30.json as the trunk-sensitivity check.

Top: measured hybrid test VRMSE relative to the fixed classical part alone (K = 0, cls_only); mean over seeds, dots: seeds.
Bottom, in error-energy space (orthonormal modes add in energy): capture by the qubit-routed modes, eta(K) U(K), against
the coverage U(K) = share of the gated residual's energy in the best K modes, computed from the test cache for K up to
2,000 (no fitting), with the closed form U_inf [1 - (1 + K/K0)^(1 - a)] fitted to it. Extrapolation of the qubit
capture beyond K = 30: eta held at its K = 30 value (lower edge) or 1 (upper edge, the ceiling).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "runs/alloc/adrc_results"
TRUNKS = {
    "vanilla": [("shear_flow__gate_vanilla_s1337", "shear flow"), ("rayleigh_benard__gatest_vanilla_s1337", "convection$^\\dagger$"),
                ("turbulent_radiative_layer_2D__gate_vanilla_s1337", "TRL2D"), ("MHD_64__gate_vanilla_s1337", "MHD")],
    "mr30ryd": [("shear_flow__gate_mr30ryd_s1337", "shear_flow"), ("rayleigh_benard__gatest_mr30ryd_s1337", "rayleigh_benard$^\\dagger$"),
                ("turbulent_radiative_layer_2D__gate_mr30ryd_s1337", "TRL2D"), ("MHD_64__gate_vanilla_s1337", "MHD_64")]}
ROWS = TRUNKS["vanilla"]
OUT = ROOT / "paper/manuscript/iclr2027/fig_hybrid_scaling.pdf"
JOUT = ROOT / "runs/alloc/hybrid_scaling.json"
KS = (5, 10, 20, 30, 50, 100, 200)                                # doc 35 §25 adds 100 and 200
ARMS = (("adrc_ryd", "#d95f02", "register"), ("adrc_mlp", "#1b9e77", "similarly sized MLP"))
CLOSED = lambda K, Ui, K0, a: Ui * (1 - (1 + K / K0) ** (1 - a))


def runs(key, arm, k):
    sfx = f"_k{k}" if k != 30 else ""
    return [json.loads(f.read_text()) for f in sorted(RES.glob(f"{key}_{arm}_gate_rs{sfx}_dyn_s[123].json"))]   # §26 seeds 4-8 apart


def qubit_capture(r):
    return sum(max(v, 0) for v in r["bands_test"]["captured_qubit"])


def analyse(key):
    k0 = [json.loads(f.read_text()) for f in sorted(RES.glob(f"{key}_cls_only_gate_s*.json"))]
    out = {"V0": st.mean(d["test"] for d in k0) if k0 else None, "arms": {}, "U": {}}
    Us = {}
    for arm, _, _ in ARMS:
        a = {}
        for k in KS:
            rr = runs(key, arm, k)
            if len(rr) < 3:                                   # plot a budget only once all registered seeds are in
                continue
            a[k] = {"V": [d["test"] for d in rr], "q": [qubit_capture(d) for d in rr], "total": [d["test_captured"] for d in rr]}
            for d in rr:
                for kk, o in d.get("oracle", {}).items():
                    Us.setdefault(int(kk), []).append(o["U"])
        out["arms"][arm] = a
    out["U"] = {k: st.mean(v) for k, v in sorted(Us.items())}
    if len(out["U"]) >= 4:
        K = np.array(list(out["U"])); U = np.array(list(out["U"].values()))
        try:
            p, _ = curve_fit(CLOSED, K, U, p0=[U.max(), 8, 2], bounds=([0, 0.1, 1.01], [1, 1e4, 10]), maxfev=20000)
            out["fit"] = {"U_inf": p[0], "K0": p[1], "alpha": p[2], "max_resid": float(np.abs(CLOSED(K, *p) - U).max())}
        except Exception as e:                                                                   # noqa: BLE001
            out["fit"] = {"error": str(e)}
    for arm, a in out["arms"].items():
        eta = {k: st.mean(v["q"]) / out["U"][k] for k, v in a.items() if out["U"].get(k, 0) > 0}
        out[f"eta_{arm}"] = eta
        if 30 in eta and out["U"]:
            Umax = max(out["U"].values())
            out[f"reach30_{arm}"] = {"lower_eta30": st.mean(a[30]["q"]) / (eta[30] * Umax) if eta[30] > 0 else None,
                                     "upper_eta1": st.mean(a[30]["q"]) / Umax}
    return out


def main():
    plt.rcParams.update({"font.size": 6.5})
    fig, ax = plt.subplots(2, len(ROWS), figsize=(7.0, 3.6)); res = {}
    for j, (key, title) in enumerate(ROWS):
        r = analyse(key); res[key] = r
        top, bot = ax[0, j], ax[1, j]
        top.set_title(title, fontsize=7.5)
        for arm, col, lab in ARMS:
            a = r["arms"].get(arm, {})
            if not a or r["V0"] is None:
                continue
            ks = sorted(a); V0 = r["V0"]
            top.plot([0] + ks, [1.0] + [st.mean(a[k]["V"]) / V0 for k in ks], "o-", color=col, ms=2.5, lw=0.9, label=lab)
            for k in ks:
                top.scatter([k] * len(a[k]["V"]), [v / V0 for v in a[k]["V"]], s=3, color=col, alpha=0.5)
            bot.plot(ks, [st.mean(a[k]["q"]) for k in ks], "o-", color=col, ms=2.5, lw=0.9)
            if 30 in a and r[f"eta_{arm}"].get(30, 0) >= 0.1:
                KK = [k for k in r["U"] if k >= 30]
                bot.fill_between(KK, [r[f"eta_{arm}"][30] * r["U"][k] for k in KK], [r["U"][k] for k in KK], color=col, alpha=0.08)
        if r["U"]:
            bot.plot(list(r["U"]), list(r["U"].values()), "k-", lw=0.8, label="coverage $U(K)$")
            if "U_inf" in r.get("fit", {}):
                Kf = np.geomspace(5, 2000, 60); f = r["fit"]
                bot.plot(Kf, CLOSED(Kf, f["U_inf"], f["K0"], f["alpha"]), "k:", lw=0.8, label="closed form")
        top.set_ylim(0.78, 1.01); top.axhline(1.0, color="0.6", lw=0.5, ls="--"); top.set_xticks([0, 50, 100, 200]); top.set_xlabel("qubits $K$", labelpad=1)
        bot.set_xscale("log"); bot.set_xlabel("qubits $K$ (log)", labelpad=1)
        if j == 0:
            top.set_ylabel("hybrid VRMSE / classical part"); bot.set_ylabel("share of error energy")
            top.legend(fontsize=5.5, frameon=False); bot.legend(fontsize=5.3, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3); fig.savefig(OUT, bbox_inches="tight")
    JOUT.write_text(json.dumps(res, indent=1, default=float)); print("wrote", OUT)
    for key, r in res.items():
        a = r["arms"].get("adrc_ryd", {}); m = r["arms"].get("adrc_mlp", {})
        print(key.split("__")[0], "V0", r["V0"] and round(r["V0"], 5),
              "reg V", {k: round(st.mean(v["V"]), 5) for k, v in a.items()}, "mlp V", {k: round(st.mean(v["V"]), 5) for k, v in m.items()},
              "eta_reg", {k: round(v, 2) for k, v in r.get("eta_adrc_ryd", {}).items()}, "fit", {k: round(v, 3) for k, v in r.get("fit", {}).items() if isinstance(v, float)},
              "reach30", r.get("reach30_adrc_ryd"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trunk", default="vanilla", choices=list(TRUNKS))
    t = ap.parse_args().trunk
    ROWS = TRUNKS[t]
    if t != "vanilla":
        OUT = OUT.with_name("fig_hybrid_scaling_mr30.pdf"); JOUT = JOUT.with_name("hybrid_scaling_mr30.json")
    main()
