"""Doc 35 §43a: exact shape labels for the TRAIN rows (v9). boundary_labels.py derived the TRAIN labels from the PCA-256 reconstruction
(the cache keeps only PCA coefficients for TRAIN), which costs kappa R2 0.9953 and tri 0.998 against the true psi, and the heads inherit
that error (VT slope 0.64 for kappa). Here every TRAIN frame is labelled from its full psi (the per-shot extracted npz files) with the
same official-scorer code (boundary_labels._one, 7 scalars + 64 rays). Output has the v8 layout: rows_train / y_train (exact),
rows_valid / y_valid copied from v8 (already from the full psi). CPU workers (the scorer's derivation is CPU code).
    STRAT_NANG=64 .venv/bin/python paper/strat/fusion/boundary_labels_exact.py     -> runs/strat/fusion/boundary_labels_v9.npz
"""
from __future__ import annotations

import os

os.environ.setdefault("STRAT_NANG", "64")
import numpy as np  # noqa: E402

import boundary_labels as BL  # noqa: E402
import fx  # noqa: E402


def chunks(d, sota, tr, size=2000):
    """(rows, full psi) chunks of whole TRAIN shots, frames in cache-row order (rows of shot j are contiguous)."""
    names = d.meta["names"]; want = np.zeros(len(d.shot), bool); want[tr] = True; rows_b, psi_b, n = [], [], 0
    for j in np.unique(d.shot[tr]):
        rows = np.nonzero(d.shot == j)[0]; psi = np.load(sota.PROC / f"{names[j]}.npz")["psi"].reshape(-1, 4225)
        assert len(psi) == len(rows), (names[j], len(psi), len(rows))
        k = want[rows]; rows_b.append(rows[k]); psi_b.append(psi[k].astype(np.float32)); n += int(k.sum())
        if n >= size:
            yield np.concatenate(rows_b), np.concatenate(psi_b); rows_b, psi_b, n = [], [], 0
    if rows_b:
        yield np.concatenate(rows_b), np.concatenate(psi_b)


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    assert BL.N_ANG == 64, BL.N_ANG
    sota, d = fx.data(); old = np.load(fx.OUT / "boundary_labels_v8.npz"); tr = old["rows_train"]
    assert (d.split[tr] == 0).all() and len(np.unique(tr)) == len(tr)
    order = np.full(len(d.shot), -1, np.int64); order[tr] = np.arange(len(tr)); Y = np.full(old["y_train"].shape, np.nan, np.float32)
    gen = chunks(d, sota, tr); done = 0
    with ProcessPoolExecutor(max_workers=fx.N_WORKERS, mp_context=mp.get_context("spawn")) as ex:
        while True:
            batch = [c for _, c in zip(range(2 * fx.N_WORKERS), gen)]       # bounded memory: ~2 chunks per worker
            if not batch:
                break
            for (r, _), o in zip(batch, ex.map(BL._one, [(p,) for _, p in batch])):
                Y[order[r]] = o
            done += sum(len(r) for r, _ in batch)
            print(f"[labels-exact] {done}/{len(tr)} frames", flush=True)
    assert done == len(tr), (done, len(tr))
    for j, nm in ((2, "kappa"), (3, "tri_top"), (4, "tri_bot"), (6, "li")):
        m = np.isfinite(Y[:, j]) & np.isfinite(old["y_train"][:, j]); e = Y[m, j] - old["y_train"][m, j]
        print(f"[labels-exact] {nm} exact vs v8 (PCA-256): R2 {1 - np.mean(e ** 2) / np.var(Y[m, j]):.5f}, rms {np.sqrt(np.mean(e ** 2)):.5f}; "
              f"nan-frac exact {np.isnan(Y[:, j]).mean():.4f} v8 {np.isnan(old['y_train'][:, j]).mean():.4f}", flush=True)
    np.savez(fx.OUT / "boundary_labels_v9.npz", rows_train=tr, y_train=Y, rows_valid=old["rows_valid"], y_valid=old["y_valid"],
             names=old["names"], n_ang=old["n_ang"])
    print("LABELS_EXACT_DONE", flush=True)


if __name__ == "__main__":
    main()
