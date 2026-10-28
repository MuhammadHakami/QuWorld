"""Data for notebook 20 (competitions + geothermal hybrid, diagram-first). Writes notebooks/data/nb20.npz and nb20.json.

Fusion: one VALID-sample frame (truth psi, X8 psi, HJl9 psi, both LCFS levels), per-scalar consistency on the sample for
X8 / X8axH / HJ / HJl9 (official parts pickles), the R_top grid-column histogram (official truth contours), D_LCFS anatomy.
Learn2Design: best-feasible curves of the v3b submission runs saved on the Spark. Geothermal: one TEST sample (inputs,
truth, classical and register predictions; paper/fig_cache/fig_fields_1hp.npz).

    .venv/bin/python notebooks/nb20_data.py
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import pickle
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FUS = ROOT / "paper" / "strat" / "fusion"
sys.path.insert(0, str(FUS))
import fx  # noqa: E402,I001
import gpu_build as GB  # noqa: E402
import gpu_pipe as GP  # noqa: E402
import strategies as ST  # noqa: E402
import axis_shift as AX  # noqa: E402
from gpu_post import GPUPost  # noqa: E402

OUT = ROOT / "notebooks" / "data"
H = "+".join(f"head_boundary_v8_J1_s{s}" for s in range(1, 7))
ARMS = {"X8": ('{"_steps":["shift","bw","kw","xp","li"],"xp":{"sigma":0.15,"cap":0.10}}', "trunk"),
        "HJl9": ('{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}', "head")}


def fusion_frame(d, sota, ES):
    """Sample shot 0..9: the frame where HJl9 moves kappa toward the truth the most (chosen to show the change)."""
    shots, _ = ST.sample(); shots = shots[:10]; R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask); psi = {}
    class A: pass
    for nm, (kw, axis) in ARMS.items():
        a = A(); a.psi, a.qb, a.heads, a.kw, a.axis = "psiF", "qh50", H, kw, axis; a.qbheads = "+".join(f"head_qb_QH_s{s}" for s in range(1, 7))
        P = GB.predict(d, shots, a, sota, ES); psi[nm] = np.concatenate([P[s][0] for s in shots])
    rows = np.concatenate([np.nonzero(d.shot == s)[0] for s in shots]); truth = d.psi_of(rows).reshape(-1, 65, 65).astype(np.float64)
    ex = {k: gp.g.derive(torch.tensor(v, device="cuda")) for k, v in {"truth": truth, **psi}.items()}
    k = {n: ex[n]["kappa"].cpu().numpy() for n in ex}
    i = int(np.nanargmax(np.abs(k["X8"] - k["truth"]) - np.abs(k["HJl9"] - k["truth"])))
    lev = {n: float(ex[n]["ext"][3][i]) for n in ex}
    sc = {n: {s: float(ex[n][s][i]) for s in GP.NAMES} for n in ex}
    return dict(R=R, Z=Z, psi_truth=truth[i], psi_X8=psi["X8"][i], psi_HJl9=psi["HJl9"][i]), dict(level=lev, scalars=sc, frame=i)


def fusion_scalars():
    shots, _ = ST.sample(); pe, nm = ST.load_entry(); rep = {}
    for t in ("X8", "X8axH", "HJ", "HJl9"):
        p = pickle.load(open(fx.OUT / f"parts_GPUS-{t}.pkl", "rb")); c = fx.composite(np.stack([fx.shot_sums(p[s]) for s in shots]), nm)
        rep[t] = dict(S=float(c["S"]), consistency=float(c["consistency"]), dlcfs=float(c["dlcfs"]), cons_each={a: float(b) for a, b in c["cons_each"].items()})
    return rep


def rtop_hist():
    R, Z, mask = AX.grid(); dR = R[1] - R[0]; fr = []
    for f in sorted(glob.glob(str(fx.REF / "ref_*.npz")))[:60]:
        for c in np.load(f, allow_pickle=True)["contours"]:
            if c is not None:
                c = np.asarray(c, float); fr.append(((c[c[:, 1].argmax(), 0] - R[0]) / dR) % 1)
    return np.asarray(fr)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import ens_select as ES
    d = sota.Data(); arr, meta = fusion_frame(d, sota, ES)
    meta["sample_scores"] = fusion_scalars(); arr["rtop_frac"] = rtop_hist()
    meta["dlcfs_anatomy"] = json.loads((fx.OUT / "dlcfs_anatomy.json").read_text())
    for r in (2, 5, 6):                                                  # Learn2Design v3b (submission 413) runs on the Spark
        z = np.load(ROOT / "runs" / "l2d" / f"hist_v3b_run{r}.npz"); L = np.where(z["feas"] & np.isfinite(z["loss"]), z["loss"], np.inf)
        arr[f"l2d_v3b_run{r}"] = np.minimum.accumulate(L)
    g = np.load(ROOT / "paper" / "fig_cache" / "fig_fields_1hp.npz"); sys.path.insert(0, str(ROOT / "paper" / "geothermal"))
    import train_pelzer_unet as T
    x, y, info, tr, va, te = T.load_1000(); i = int(g["sample"])
    arr["geo_inputs"] = x[i].numpy(); arr["geo_truth"] = g["truth"]; arr["geo_classical"] = g["vanilla_pred"]; arr["geo_register"] = g["mr30ryd_pred"]
    meta["geo"] = dict(sample=i, inputs=sorted((k for k in info if isinstance(info[k], dict) and "index" in info[k]), key=lambda k: info[k]["index"]),
                       mae_classical=float(g["vanilla_mae"]), mae_register=float(g["mr30ryd_mae"]))
    np.savez_compressed(OUT / "nb20.npz", **arr); (OUT / "nb20.json").write_text(json.dumps(meta, indent=1, default=float))
    print("NB20_DATA_DONE", flush=True)


if __name__ == "__main__":
    main()
