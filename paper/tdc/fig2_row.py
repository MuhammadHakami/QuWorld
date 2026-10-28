"""Fig. 2 materials row (user, Oct 3): TDC ADMET Caco-2 permeability (regression), official test set; seed 1 of the
classical head, + matched MLP (30), + 30-atom and + 50-atom registers; predictions saved as a scatter row in the uniform
row format of paper/alloc/fig_outputs_all.py.

    .venv-tdc/bin/python paper/tdc/fig2_row.py      -> paper/fig_cache/rows_s23/tdc.npz
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_admet as RA  # noqa: E402

EP = "caco2_wang"


def main() -> None:
    hp = dict(lr=1e-3, wd=1e-4, epochs=300, batch=256, patience=40)
    emb = RA.D.embeddings(EP); g = RA.D.group(); y = np.asarray(g.get(EP)["test"]["Y"], float)
    row = {"label": "TDC Caco-2", "metric": "MAE", "kind": "scatter", "truth": y}
    for arm, v in (("classical", "vanilla"), ("mlp", "mlp30"), ("qubits", "mr30ryd"), ("qubits50", "mr50ryd")):
        p, val, _ = RA.train_one(EP, v, 1, emb, hp); p = np.asarray(p, float).reshape(-1)
        row[f"pred_{arm}"] = p; row[f"m_{arm}"] = float(np.mean(np.abs(p - y)))
        print(f"[tdc-fig2] {v}: test MAE {row[f'm_{arm}']:.4f}", flush=True)
    out = HERE.parents[1] / "paper/fig_cache/rows_s23/tdc.npz"
    np.savez(out, **{k: np.asarray(v) for k, v in row.items()}); print("wrote", out)


if __name__ == "__main__":
    main()
