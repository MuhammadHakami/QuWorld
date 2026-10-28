"""Figure 2 row for MHD_64 with the doc 35 §23 models (classical, matched 3-D MLP, 3-D tethered register; seed 1337):
mid-plane density one step ahead on validation trajectory 0 (the MHD cache has no test split), same rollout code as
paper/figs_fields_mhd.py. Writes paper/fig_cache/rows_s23/MHD_64.npz.

    QW_GPU_GB=28 runs/capped.sh rowsmhd 60 .venv/bin/python paper/alloc/dump_rows_mhd_s23.py
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))
import figs_fields_mhd as F  # noqa: E402

ARMS = (("classical", "vanilla"), ("mlp", "q30-geo=3d-mlp"), ("qubits", "q30-geo=3d"))


def main() -> None:
    data = F.load_valid_traj0()
    out = {"label": "MHD (val.)", "metric": "VRMSE", "kind": "field", "sample": "valid trajectory 0, lead 1",
           "truth": data[F.N_HIST, 0, :, :, F.MIDZ].numpy()}
    for name, v in ARMS:
        slices, vr = F.rollout(str(ROOT / "runs/phase_b_mhd" / f"{v}_s1337.pt"), data)
        out[f"pred_{name}"] = slices[1]; out[f"m_{name}"] = float(vr[1])
        print(name, round(vr[1], 5), flush=True)
    dst = ROOT / "paper/fig_cache/rows_s23"; dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / "MHD_64.npz", **out); print("wrote", dst / "MHD_64.npz")


if __name__ == "__main__":
    main()
