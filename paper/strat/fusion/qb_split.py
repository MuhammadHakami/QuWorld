"""Fusion §41q: q95 / betaN R2 of q/b head stems on the VALID sample rows (pooled variance), all / normal / reversed-Ip shots.

    .venv/bin/python paper/strat/fusion/qb_split.py head_qb_QH_s1 head_qb_QHM_s1 ...
"""
from __future__ import annotations

import json
import sys

import numpy as np

import fx  # noqa: F401,I001
sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio"))
import sota  # noqa: E402


def main() -> None:
    d = sota.Data(); va = np.load(fx.OUT / "boundary_labels_v8.npz")["rows_valid"]
    Y = np.stack([d.S[va, sota.LABELS.index(h)] for h in ("efit_q95", "efit_beta_n")], 1).astype(np.float64)
    sg = json.loads((fx.OUT / "ip_sign.json").read_text()); rev = np.array([sg[str(int(s))] < 0 for s in d.shot[va]])
    for stem in sys.argv[1:]:
        P = np.mean([np.load(fx.OUT / f"{s}.npz")["pred_valid"] for s in stem.split("+")], 0).astype(np.float64); out = []
        for nm, m in (("all", np.ones(len(va), bool)), ("normal", ~rev), ("reversed", rev)):
            r = [1 - np.nanmean((Y[m, j] - P[m, j]) ** 2) / np.nanvar(Y[:, j]) for j in range(2)]
            out.append(f"{nm} q95 {r[0]:.5f} betaN {r[1]:.5f} mean {np.mean(r):.5f}")
        print(f"[qb] {stem}: " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
