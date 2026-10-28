"""Read-only scaling audit: existing logs, null models, held-out register counts.

No model training/inference. Run with uv run --no-sync python paper/scaling/audit_existing.py.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))
from fit_ladder import LOGS, MLP_LOGS, SEEDS

OUT = ROOT / "runs/scaling"


def read_rows():
    rows = []
    for qubits, paths in list(LOGS.items()) + [(None, MLP_LOGS)]:
        for seed, path in zip(SEEDS, paths):
            raw = path.read_bytes()
            text = raw.decode()
            def field(key):
                match = re.search(rf"^{key}:\s+([0-9.eE+-]+)", text, re.M)
                if match is None:
                    raise ValueError(f"Missing {key}: {path}")
                value = float(match.group(1))
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite {key}: {path}")
                return value
            steps = int(field("num_steps"))
            if steps != 2000 or field("rollout_k") != 4:
                raise ValueError(f"Unmatched optimization protocol: {path}")
            rows.append(dict(seed=seed, qubits=qubits, arm="mlp" if qubits is None else
                             ("vanilla" if qubits == 0 else "quantum"),
                             vrmse8=field("val_vrmse_8"), steps=steps,
                             allocated_params=int(field("num_params")),
                             train_seconds=field("train_seconds"),
                             source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest()))
    return rows


def fit_predict(kind, x, y, test_x):
    """Descriptive fits on q>=5 only: baseline and signed trends; no forced win."""
    if kind == "constant":
        return np.full_like(test_x, y.mean()), [float(y.mean())]
    if kind == "log_trend":
        design = np.column_stack([np.ones_like(x), np.log(x)])
        p = np.linalg.lstsq(design, y, rcond=None)[0]
        return np.column_stack([np.ones_like(test_x), np.log(test_x)]) @ p, p.tolist()
    def model(p, z):
        return p[0] + p[1] * np.exp(-z / p[2])
    fits = [least_squares(lambda p: model(p, x) - y, [y.mean(), 0, scale],
                         bounds=([0, -2, 0.1], [3, 2, 200]), max_nfev=3000)
            for scale in [1, 5, 20, 100]]
    best = min(fits, key=lambda f: np.sum(f.fun ** 2))
    return model(best.x, test_x), best.x.tolist()


def main():
    rows = read_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "existing_ladder.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    quantum = [r for r in rows if r["arm"] == "quantum"]
    x = np.array([r["qubits"] for r in quantum], dtype=float)
    y = np.array([r["vrmse8"] for r in quantum])
    scores = {}
    for kind in ["constant", "log_trend", "exp_saturation"]:
        residuals = []
        for held_out in sorted(set(x)):
            train = x != held_out
            predicted, _ = fit_predict(kind, x[train], y[train], x[~train])
            residuals.extend((predicted - y[~train]).tolist())
        _, params = fit_predict(kind, x, y, x)
        scores[kind] = dict(leave_one_count_out_rmse=float(np.sqrt(np.mean(np.square(residuals)))),
                            full_fit_parameters=params)
    groups = []
    for q in [0, 5, 10, 20, 30, None]:
        group = [r for r in rows if r["qubits"] == q]
        vals = [r["vrmse8"] for r in group]
        groups.append(dict(qubits=q, mean=float(np.mean(vals)), sample_sd=float(np.std(vals, ddof=1)),
                           allocated_params=sorted({r["allocated_params"] for r in group}), values=vals))
    report = dict(
        status="exploratory audit; joint classical-quantum law not identified",
        rows=rows, groups=groups, models_on_positive_qubits=scores,
        limitations=[
            "One classical trunk size, one data budget, one optimizer-step budget: joint exponents unidentifiable.",
            "Qubit count changes both register width and register count; allocated parameter counts also change.",
            "Only three paired seeds; leave-one-count-out scores are descriptive, not an independent confirmation.",
            "Validation VRMSE@8, not official-test scaling evidence.",
            "MLP control exists at 30-dimensional output only, not at every quantum budget.",
            "Wall times came from different runs/hardware conditions; they are not calibrated compute measurements.",
            "No per-run utilization/effective-rank measurements; quantum total count is not effective capacity.",
        ])
    (OUT / "existing_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Existing scaling evidence audit", "", report["status"], "",
             "| Qubits / arm | Mean ± sample SD VRMSE@8 | Allocated parameters |",
             "|---|---|---|"]
    for g in groups:
        lines.append(f"| {g['qubits'] if g['qubits'] is not None else 'MLP'} | "
                     f"{g['mean']:.4f} ± {g['sample_sd']:.4f} | {g['allocated_params']} |")
    lines += ["", "Positive-qubit leave-one-count-out prediction RMSE (lower is better):", ""]
    for kind, score in scores.items():
        lines.append(f"- {kind}: {score['leave_one_count_out_rmse']:.6f}")
    lines += ["", "Limitations:", ""] + [f"- {s}" for s in report["limitations"]]
    (OUT / "existing_audit.md").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for g in groups[:-1]:
        axes[0].errorbar(g["qubits"], g["mean"], yerr=g["sample_sd"], fmt="o", color="#2365a8", capsize=3)
        axes[0].scatter([g["qubits"]] * 3, g["values"], alpha=.4, color="#2365a8", s=15)
    axes[0].axhline(y.mean(), ls="--", color="#555555", label="constant fit, q ≥ 5")
    axes[0].set(xlabel="Total simulated qubits", ylabel="Validation VRMSE@8", title="One classical trunk; three seeds")
    axes[0].legend(fontsize=8)
    sizes = np.array([5, 10, 20, 30])
    entries = np.array([32, 1024, 2048, 3072])
    axes[1].semilogy(sizes, entries, "o-", label="actual independent-register entries")
    axes[1].semilogy(sizes, 2.0 ** sizes, "--", label="hypothetical monolithic state")
    axes[1].set(xlabel="Total simulated qubits", ylabel="Amplitude entries (not total memory)", title="Register partition changes cost")
    axes[1].legend(fontsize=7)
    fig.savefig(OUT / "existing_audit.pdf")
    fig.savefig(OUT / "existing_audit.png", dpi=160)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
