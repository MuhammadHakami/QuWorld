"""Doc 35 §38v-pre: how do the public test shots differ from our local splits? (inputs only; no labels involved)

Local consistency gains (VALID, TEST) have not transferred to the public board twice (§38p, §38s). This compares the
raw input features of the public test frames with TRAIN / VALID / TEST: frames per shot, per-feature standardized mean
difference (SMD, in TRAIN standard deviations) and the share of frames outside the TRAIN 0.1-99.9% range per feature.

    .venv/bin/python paper/strat/fusion/shift_probe.py   -> runs/strat/fusion/shift_probe.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

import fx

sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio"))


def stats(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    return dict(smd=(X.mean(0) - mu) / sd, out=((X < lo) | (X > hi)).mean(0))


def main() -> None:
    import submit_public as SP
    sota, d = fx.data(); fn = list(d.fn); X = np.asarray(d.X, np.float64)
    pub = SP.PublicData(d.meta); P = np.asarray(pub.X, np.float64)
    tr = d.split == 0; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    lo, hi = np.percentile(X[tr], 0.1, axis=0), np.percentile(X[tr], 99.9, axis=0)
    rep = {}
    for name, A, shot in [("valid", X[d.split == 1], d.shot[d.split == 1]), ("test", X[d.split == 2], d.shot[d.split == 2]),
                          ("public", P, pub.shot)]:
        s = stats(A, mu, sd, lo, hi); fps = np.bincount(shot)[np.unique(shot)]
        top = np.argsort(-np.abs(s["smd"]))[:12]; topo = np.argsort(-s["out"])[:12]
        rep[name] = dict(frames=len(A), shots=len(fps), frames_per_shot=[float(np.percentile(fps, q)) for q in (10, 50, 90)],
                         mean_abs_smd=float(np.abs(s["smd"]).mean()), out_frac=float(s["out"].mean()),
                         top_smd={fn[i]: round(float(s["smd"][i]), 3) for i in top},
                         top_out={fn[i]: round(float(s["out"][i]), 4) for i in topo})
        print(f"[shift] {name}: {len(A)} frames, {len(fps)} shots, frames/shot p10/50/90 {rep[name]['frames_per_shot']}, "
              f"mean|SMD| {rep[name]['mean_abs_smd']:.3f}, out-of-range {rep[name]['out_frac']:.4f}", flush=True)
        print(f"   top SMD {rep[name]['top_smd']}", flush=True)
        print(f"   top out {rep[name]['top_out']}", flush=True)
    (fx.ROOT / "runs/strat/fusion/shift_probe.json").write_text(json.dumps(rep, indent=1))
    print("SHIFT_DONE", flush=True)


if __name__ == "__main__":
    main()
