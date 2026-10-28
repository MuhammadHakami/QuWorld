"""Figure 2 row for TokaMark task 1-1 (doc 35 §23 models on the TokaMind trunk): the middle test shot of the dumps (its
windows in order), every predicted signal normalised by its ground-truth mean and std and stacked into a signals x time image.
Reads runs/tokamark/results/dumps/{vanilla,q30-geo=3d-mlp,q30-geo=3d}_s1337.npz (TOKAMIND_DUMP_TRUTH=1).

    python paper/tokamark/dump_rows_s23.py
"""
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "runs/tokamark/results/dumps"
ARMS = (("classical", "vanilla"), ("mlp", "q30-geo=3d-mlp"), ("qubits", "q30-geo=3d"))


def image(recs: dict, keys, names, truth: bool, ref: dict | None = None):
    """(signal rows, time) image of one shot: each dump record is one window with one value per signal (two for the
    X-point signals), so the time axis is the shot's windows in order."""
    rows = []
    for n in names:
        a = np.stack([np.asarray(recs[k][n + "#true"] if truth else recs[k][n], dtype=float).reshape(-1) for k in keys]).T
        m, s = ref[n] if ref else (np.nanmean(a), np.nanstd(a) + 1e-12)
        rows.append((a - m) / s)
    return np.concatenate(rows, 0)


def main() -> None:
    recs = {name: np.load(D / f"{v}_s1337.npz", allow_pickle=True)["rec"].item() for name, v in ARMS}
    keys = set.intersection(*[set(r) for r in recs.values()])
    shots = sorted({int(k.split("|")[0]) for k in keys})
    shot = shots[len(shots) // 2]                                        # middle test shot, fixed in advance
    wins = sorted((k for k in keys if int(k.split("|")[0]) == shot), key=lambda k: int(k.split("|")[1]))
    base = recs["classical"]
    names = sorted(n for n in base[wins[0]] if not n.endswith("#true") and n + "#true" in base[wins[0]])
    ref = {}
    for n in names:
        t = np.stack([np.asarray(base[k][n + "#true"], dtype=float).reshape(-1) for k in wins]); ref[n] = (np.nanmean(t), np.nanstd(t) + 1e-12)   # EFIT truth has missing entries
    truth = image(base, wins, names, True, ref)
    out = {"label": "TokaMark", "metric": "NRMSE", "kind": "spacetime", "sample": f"test shot {shot}, {len(wins)} windows", "truth": truth}
    for name, _ in ARMS:
        p = image(recs[name], wins, names, False, ref)
        out[f"pred_{name}"] = p; ok = np.isfinite(truth) & np.isfinite(p)
        out[f"m_{name}"] = float(np.sqrt(np.mean((p[ok] - truth[ok]) ** 2)) / (truth[ok].std() + 1e-12))
        print(name, round(out[f"m_{name}"], 5))
    dst = ROOT / "paper/fig_cache/rows_s23"; dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / "tokamark.npz", **out); print("wrote", dst / "tokamark.npz", "signals", names)


if __name__ == "__main__":
    main()
