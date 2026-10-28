"""Fusion-equilibrium outputs for Figure 2: poloidal flux psi(R, Z) on the middle frame of the middle TEST shot, fixed in
advance, from the starter U-Net (classical) and the same model with the matched MLP / 30-atom register (seed 1337,
runs/sophelio, same data split and scaler as train_hybrid.py).

    QW_GPU_GB=8 runs/capped.sh fpsi 24 .venv/bin/python paper/sophelio/dump_fields_psi.py
"""
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_hybrid as H  # noqa: E402

RUNS = H.ROOT / "runs" / "sophelio"
import os  # noqa: E402
S23 = bool(os.environ.get("S23"))                                   # doc 35 §23 set: 3-D MLP and 3-D qubits
VARIANTS = ("vanilla", "mlp30", "mr30ryd")
FILE = {"vanilla": "vanilla", "mlp30": "q30-geo=3d-mlp" if S23 else "mlp30", "mr30ryd": "q30-geo=3d" if S23 else "mr30ryd"}


class _NoScalars:
    def predict(self, X):
        return np.zeros((len(X), len(H.EFIT_SCALAR_LABELS)))


@torch.no_grad()
def main() -> None:
    thomson = bool(json.loads((RUNS / "vanilla_s1337.json").read_text()).get("thomson", False))
    H.INCLUDE_THOMSON = thomson
    config = H.ExperimentConfig(data_source="local", local_data_dir=H.DATA_DIR, n_shots=200, random_state=H.DATA_SEED,
                                epochs=1, batch_size=256, learning_rate=1e-3, device="cuda", output_dir=RUNS,
                                include_thomson=thomson)
    shots = H.load_shots(config)
    X, Y, S, shot_ids = H.build_feature_matrix(shots, config.include_thomson)
    tr, va, te, test_shot_idx, _ = H.split_by_shot(shot_ids, config)
    scaler = H.StandardScaler().fit(X[tr]); n_feat = int(X.shape[1])
    k = int(test_shot_idx[len(test_shot_idx) // 2]); shot = next(s for s in shots if s["shot_index"] == k)
    out = {}
    for v in VARIANTS:
        model = H.wrap(H.UNetLite(n_feat), FILE[v])
        model.load_state_dict(torch.load(RUNS / f"{FILE[v]}_s1337.pt", weights_only=True, map_location="cpu")["model_state_dict"])
        dev = torch.device("cuda"); model = model.to(dev).eval()
        gt, pr = H.shot_predictions(shot, model, scaler, _NoScalars(), dev)
        frames = np.where(np.isfinite(pr["psirz"]).all(axis=(1, 2)))[0]; f = int(frames[len(frames) // 2])
        p, t = pr["psirz"][f], gt["psi"][f]
        out[f"{v}_pred"] = p.astype(np.float32); out["truth"] = t.astype(np.float32)
        out[f"{v}_nrmse"] = float(np.sqrt(np.mean((p - t) ** 2)) / (np.std(t) + 1e-12))
        print(v, "shot", k, "frame", f, "NRMSE", round(out[f"{v}_nrmse"], 5))
    dst = H.ROOT / "paper" / "fig_cache" / ("fig_fields_psi_s23.npz" if S23 else "fig_fields_psi.npz")
    np.savez(dst, shot=k, **out); print("wrote", dst)


if __name__ == "__main__":
    main()
