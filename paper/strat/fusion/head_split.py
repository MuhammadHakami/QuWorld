"""Fusion §41q: per-scalar R2 of head stems on the VALID sample rows, all / normal-Ip / reversed-Ip shots (ip_sign.json).

    .venv/bin/python paper/strat/fusion/head_split.py head_boundary_v8_J1_s1 head_boundary_v8_J1CN_s1 ...
"""
from __future__ import annotations

import json
import sys

import numpy as np

import fx  # noqa: F401,I001
sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio"))
import sota  # noqa: E402

NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def main() -> None:
    d = sota.Data(); lab = np.load(fx.OUT / "boundary_labels_v8.npz"); va, Y = lab["rows_valid"], lab["y_valid"].astype(np.float64)
    sg = json.loads((fx.OUT / "ip_sign.json").read_text()); rev = np.array([sg[str(int(s))] < 0 for s in d.shot[va]])
    for stem in sys.argv[1:]:
        P = np.mean([np.load(fx.OUT / f"{s}.npz")["pred_valid"] for s in stem.split("+")], 0).astype(np.float64)
        out = []
        for nm, m in (("all", np.ones(len(va), bool)), ("normal", ~rev), ("reversed", rev)):
            r2 = []
            for j in range(7):
                ok = m & np.isfinite(Y[:, j]); var = np.var(Y[np.isfinite(Y[:, j]), j])          # pooled variance (as the composite)
                r2.append(1 - np.mean((Y[ok, j] - P[ok, j]) ** 2) / var)
            ok = m[:, None] & np.isfinite(Y[:, 7:]); rays = np.sqrt(np.mean(((Y[:, 7:] - P[:, 7:]) ** 2)[ok])) * 1000
            out.append(f"{nm} mean {np.mean(r2):.5f} [" + " ".join(f"{n} {v:.4f}" for n, v in zip(NAMES, r2)) + f"] rays {rays:.2f} mm")
        print(f"[split] {stem}:\n  " + "\n  ".join(out), flush=True)


if __name__ == "__main__":
    main()
