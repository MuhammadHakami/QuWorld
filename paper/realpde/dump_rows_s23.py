"""Figure 2 row for RealPDEBench (airfoil, doc 35 §23): the released U-Net (classical) and its low-mode corrector with the
matched 3-D MLP and the 3-D tethered register (seed 1), middle TEST window fixed in advance, first field at the window's
middle time step. Reads the cached predictions written by arms.py --save-pred; writes paper/fig_cache/rows_s23/realpde.npz.

    .venv/bin/python paper/realpde/dump_rows_s23.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C  # noqa: E402

ARMS = (("classical", "unet"), ("mlp", "unet-q30-geo=3d-mlp-s1"), ("qubits", "unet-q30-geo=3d-s1"))


def rel_l2(p, t):
    return float(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-12))


def main() -> None:
    X, Y, sims = C.load_windows("test")
    w = Y.shape[0] // 2; y = np.asarray(Y[w], dtype=np.float32)            # (T, H, W, C) window
    tm = y.shape[0] // 2
    out = {"label": "RealPDE (foil)", "metric": "rel L2", "kind": "field", "sample": f"test window {w}",
           "truth": y[tm, :, :, 0]}
    for name, model in ARMS:
        P = C.open_pred(C.CACHE / f"pred_{model}_finetune_test.npy", Y.shape)
        p = np.asarray(P[w], dtype=np.float32)
        out[f"pred_{name}"] = p[tm, :, :, 0]; out[f"m_{name}"] = rel_l2(p, y)
        print(name, round(out[f"m_{name}"], 5))
    dst = C.ROOT / "paper/fig_cache/rows_s23"; dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / "realpde.npz", **out); print("wrote", dst / "realpde.npz")


if __name__ == "__main__":
    main()
