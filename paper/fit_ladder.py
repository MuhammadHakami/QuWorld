"""Qubit-count scaling fit + figure for the WM-PAI paper (2026-09-06).

Reads VRMSE@8 selection values from the wave-1.5 logs (0q trunk, 30q Rydberg,
matched-MLP control) and the ladder logs ({5,10,20}q), fits two closed-form
scaling ansatze to the per-count means (weighted by 1/std^2), picks the better
by AIC, writes paper/manuscript/fig_scaling.pdf, and prints the LaTeX macro
values to paste into main4p.tex.

Usage: uv run paper/fit_ladder.py
"""

import glob
import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = [1337, 1338, 1339]

LOGS = {
    0: [ROOT / f"runs/ag15_none_k4_s{s}.log" for s in SEEDS],
    5: [ROOT / f"runs/ladder_ryd5_k4_s{s}.log" for s in SEEDS],
    10: [ROOT / f"runs/ladder_mr10ryd_k4_s{s}.log" for s in SEEDS],
    20: [ROOT / f"runs/ladder_mr20ryd_k4_s{s}.log" for s in SEEDS],
    30: [ROOT / f"runs/ag15_mr30ryd_k4_s{s}.log" for s in SEEDS],
}
MLP_LOGS = [ROOT / f"runs/ag15_mlp30_k4_s{s}.log" for s in SEEDS]

VAL_RE = re.compile(r"val_vrmse_8:\s+([0-9.]+)")


def read_val(path: pathlib.Path) -> float:
    text = path.read_text()
    m = VAL_RE.search(text)
    if not m:
        raise RuntimeError(f"no val_vrmse_8 in {path}")
    return float(m.group(1))


def stats(paths):
    vals = [read_val(p) for p in paths]
    arr = np.array(vals)
    return arr.mean(), arr.std(ddof=1), vals


def main() -> None:
    counts = sorted(LOGS)
    means, stds, per_seed = {}, {}, {}
    for n in counts:
        means[n], stds[n], per_seed[n] = stats(LOGS[n])
        print(f"n={n:>2}q  {means[n]:.4f} +/- {stds[n]:.4f}   seeds={['%.4f' % v for v in per_seed[n]]}")
    mlp_mean, mlp_std, mlp_vals = stats(MLP_LOGS)
    print(f"mlp30  {mlp_mean:.4f} +/- {mlp_std:.4f}")

    x = np.array(counts, dtype=float)
    y = np.array([means[n] for n in counts])
    w = 1.0 / np.array([max(stds[n], 1e-3) for n in counts]) ** 2

    def fit(model, p0, bounds):
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(model, x, y, p0=p0, sigma=1.0 / np.sqrt(w),
                            bounds=bounds, maxfev=20000)
        resid = y - model(x, *popt)
        chi2 = float(np.sum(w * resid ** 2))
        k = len(popt)
        aic = chi2 + 2 * k
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return popt, aic, 1 - ss_res / ss_tot

    def exp_sat(n, e_inf, e0, nstar):
        return e_inf + (e0 - e_inf) * np.exp(-n / nstar)

    def pow_sat(n, e_inf, e0, gamma):
        return e_inf + (e0 - e_inf) * (1.0 + n) ** (-gamma)

    results = {}
    for name, model, p0, bounds in [
        ("exp", exp_sat, (0.8, 0.96, 10.0), ([0, 0, 0.1], [2, 2, 200])),
        ("pow", pow_sat, (0.8, 0.96, 0.5), ([0, 0, 0.01], [2, 2, 10])),
    ]:
        try:
            results[name] = fit(model, p0, bounds)
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"fit {name} FAILED: {exc}")
    if not results:
        sys.exit("no fit succeeded")
    best = min(results, key=lambda k: results[k][1])
    popt, aic, r2 = results[best]
    for name, (p, a, r) in results.items():
        print(f"fit {name}: params={np.round(p, 4)} AIC={a:.2f} R2={r:.3f}"
              + ("   <== selected" if name == best else ""))

    e_inf, e0, rate = popt
    if best == "exp":
        eq = (rf"\mathrm{{VRMSE@8}}(n) = {e_inf:.3f} + {e0 - e_inf:.3f}\,"
              rf"e^{{-n/{rate:.1f}}}")
        model = exp_sat
    else:
        eq = (rf"\mathrm{{VRMSE@8}}(n) = {e_inf:.3f} + {e0 - e_inf:.3f}\,"
              rf"(1+n)^{{-{rate:.2f}}}")
        model = pow_sat

    print("\n=== LaTeX macro values ===")
    for n, macro in [(5, "QLfive"), (10, "QLten"), (20, "QLtwenty")]:
        print(f"\\newcommand{{\\{macro}}}{{${means[n]:.4f} \\pm {stds[n]:.4f}$}}")
    print(f"\\newcommand{{\\ScalingEq}}{{${eq}$ (R$^2$={r2:.2f}, "
          f"{'exponential' if best == 'exp' else 'power-law'} saturation)}}")

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#1F2937", "#6B7280"
    import os
    validated = os.environ.get("QW_PALETTE") == "validated"   # dataviz-validated set (ICLR version)
    c_q, c_ctrl = ("#2a78d6", "#eb6834") if validated else ("#1D4ED8", "#B45309")
    fig, ax = plt.subplots(figsize=(5.4, 3.1), dpi=200)
    nn = np.linspace(0, 31, 200)
    ax.plot(nn, model(nn, *popt), lw=1.4, color=c_q, alpha=0.55, zorder=2)
    for n in counts:  # individual seeds, jittered
        ax.scatter([n - 0.6, n, n + 0.6], per_seed[n], s=10, color=c_q,
                   alpha=0.4, linewidths=0, zorder=2.5)
    ax.errorbar(x, y, yerr=[stds[n] for n in counts], fmt="o", ms=5.5,
                color=c_q, ecolor=c_q, elinewidth=1.2, capsize=3, zorder=3,
                label="Rydberg register(s)")
    ax.errorbar([30], [mlp_mean], yerr=[mlp_std], fmt="s", ms=5.5,
                color=c_ctrl, ecolor=c_ctrl, elinewidth=1.2, capsize=3,
                zorder=3, label="matched MLP (same trainable parameters)")
    ax.set_xlabel("simulated qubits $n$", fontsize=9, color=ink)
    ax.set_ylabel("VRMSE@8 (lower is better)", fontsize=9, color=ink)
    ax.set_xticks(counts)
    ax.tick_params(labelsize=8, colors=ink)
    ax.grid(True, axis="y", lw=0.4, color="#E5E7EB", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(muted)
    ax.axhline(e_inf, lw=0.8, ls="--", color=muted, zorder=1)
    ax.annotate(f"plateau $\\approx$ {e_inf:.3f} (saturation $<$ 5q)",
                xy=(15, e_inf), xytext=(12.5, e_inf - 0.055),
                fontsize=8, color=muted)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    out = (pathlib.Path(os.environ["QW_OUT_DIR"]) / "fig_scaling.pdf") if os.environ.get("QW_OUT_DIR") else ROOT / "paper/manuscript/fig_scaling.pdf"
    fig.savefig(out)
    print(f"figure written: {out}")


if __name__ == "__main__":
    main()
