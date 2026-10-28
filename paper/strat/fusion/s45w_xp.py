"""§45w (2): X-point heads XP1CS_s1 vs XP1_s1 on VT frames split by Ip sign; rmse (cm) per component against xp_labels (finite rows).
    .venv/bin/python paper/strat/fusion/s45w_xp.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

import fx


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import canon
    import gpu_build as GB
    d = sota.Data(); vrows = np.load(fx.OUT / "boundary_labels_v8_vt.npz")["rows"]; sg = np.asarray(canon.shot_sign(d))[vrows]
    xl = np.load(fx.OUT / "xp_labels.npz")["y"]; out = {}
    for part, rows in (("normal", vrows[sg > 0]), ("reversed", vrows[sg < 0])):
        y = xl[rows].astype(np.float64); ok = np.isfinite(y)
        for lab in ("XP1", "XP1CS"):
            p = GB.heads_gpu(d, rows, f"head_xp_{lab}_s1")[:, :4].astype(np.float64)
            e = np.sqrt(np.nanmean(np.where(ok, (p - y) ** 2, np.nan), 0)) * 100; out[f"{part}_{lab}"] = e.tolist()
            print(f"[s45w] VT {part:8s} {lab:5s} X-point rmse (cm) R_l {e[0]:.3f} Z_l {e[1]:.3f} R_u {e[2]:.3f} Z_u {e[3]:.3f} | mean {e.mean():.3f} | "
                  f"frames {len(rows)} finite {ok.all(1).sum()}", flush=True)
        r = np.mean(out[f"{part}_XP1CS"]) / np.mean(out[f"{part}_XP1"]); out[f"{part}_ratio"] = float(r)
        print(f"[s45w] VT {part} XP1CS / XP1 mean rmse ratio {r:.4f} (rule <= 1.05)", flush=True)
    (fx.OUT / "s45w_xp.json").write_text(json.dumps(out, indent=1)); print("S45W_XP_DONE", flush=True)


if __name__ == "__main__":
    main()
