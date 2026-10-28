"""Doc 35 §42z: clamp the submitted q95 / betaN to the labelled range (qb_clamp.json: q95 label min / max, betaN quantiles 1e-4 / 1 - 1e-4).
EFIT fails on a handful of last frames (betaN -14.5 / -12.5 / -7.7 / -6.5, q95 1.3-3.9 after ~1.2 / ~7.5 one frame earlier); the q/b heads
learn that signature from 4 frames and fire on look-alike frames with extreme values of either sign (VT: +143 against a -14.5 label;
public F6L: -11.1). The clamp is free on normal VT frames (betaN R2 0.999621 -> 0.999620) and removes the blow-ups. psi copied unchanged.
    .venv/bin/python paper/strat/fusion/post_clamp.py --src F7 --tag F7C
"""
from __future__ import annotations

import argparse
import json
import zipfile

import numpy as np

import fx


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", default="public"); a = ap.parse_args()
    B = json.loads((fx.OUT / "qb_clamp.json").read_text()); base = fx.ROOT / "runs" / "sophelio_sota"
    name = f"diii_d_{a.fold}_test.npz"; src = np.load(base / f"submission_{a.src}" / name); preds = {}; n = {"q95": 0, "betaN": 0}
    for k in src.files:
        v = src[k]
        for s in ("q95", "betaN"):
            if k.endswith("_" + s):
                lo, hi = B[s]; c = np.clip(v, lo, hi); n[s] += int((c != v).sum())
                if (c != v).any():
                    print(f"[clamp] {k}: " + ", ".join(f"frame {i}/{len(v)} {v[i]:.3f} -> {c[i]:.3f}" for i in np.nonzero(c != v)[0]), flush=True)
                v = c
        if not np.isfinite(v).all():
            raise RuntimeError(f"{k}: non-finite prediction")
        preds[k] = v
    od = base / f"submission_{a.tag}"; od.mkdir(parents=True, exist_ok=True); npz = od / name; np.savez_compressed(npz, **preds)
    with zipfile.ZipFile(od / "submission.zip", "w", zipfile.ZIP_STORED) as z:
        z.write(npz, arcname=npz.name)
    info = {"src": a.src, "clamp": B, "clamped_frames": n}; (od / "build.json").write_text(json.dumps(info, indent=1))
    print("POST_CLAMP built", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
