"""RealPDE foil (RealPDEBench, ICLR 2026) — the classical trunk anchor on RealPDEBench's own protocol.

    .venv/bin/python paper/realpde/eval_trunk.py [--models unet fno cno] [--stage finetune]  ->  runs/realpde/trunk_<model>_<stage>.json

Trunk = RealPDEBench's released foil checkpoints (HF AI4Science-WestlakeU/RealPDEBench-models; stage finetune = its
"real-world finetuning" paradigm, the published best row). The competition checkpoints are NOT used: the competition
release processes the same experiments differently (u mean 0.046 vs 0.280 on (3750, 0)), so they are off-distribution.
Protocol = RealPDEBench eval.py: real test windows (test_mode "all", 20 -> 20, N_autoregressive 1, 64 x 128),
GaussianNormalizer, eval_metrics on the whole set as one batch (computed chunked, verified equal to 3e-7), measured
channels u, v. Normaliser choice per model is made on VALID windows only. TEST is also reported without the windows of
the 2 flow conditions also in the competition release and per RealPDEBench partition (in_dist / out_dist / seen). Predictions are cached for the arm trainers.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import common as C

OUT = C.ROOT / "runs" / "realpde"
BATCH = {"unet": 8, "fno": 16, "cno": 16, "transolver": 4}          # from runs/realpde/smoke2.log peaks (batch 4)


def select(model, stage, Xv, Yv, norm, n_sel: int):
    sub = torch.linspace(0, len(Xv) - 1, n_sel).long()
    scores = []
    for g in (True, False):
        net = C.build(model, Xv.shape[1:], Yv.shape[1:], stage=stage)
        r = C.rel_l2(C.predict(net, Xv[sub], norm, g, batch=BATCH[model]), Yv[sub])
        scores.append((r, g)); print(f"[realpde] {model}/{stage} VALID-select gaussian={g}: rel_l2 {r:.4f}", flush=True)
        del net; torch.cuda.empty_cache()
    return min(scores)


def subsets(sims):
    """RealPDEBench's test windows cover all 98 real trajectories (a time split); its test_mode filters are the
    trajectory partitions in_dist / out_dist (unseen parameters) and remain ("seen")."""
    part = {k: C.sim_ids(k) for k in ("in_dist", "out_dist", "remain")}
    mask = lambda keep: torch.tensor([keep(s) for s in sims])
    return {"all": mask(lambda s: True), "not_in_comp_release": mask(lambda s: s not in C.LEAKED),
            "in_dist": mask(lambda s: s in part["in_dist"]), "out_dist": mask(lambda s: s in part["out_dist"]),
            "seen": mask(lambda s: s in part["remain"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["unet", "fno", "cno"]); ap.add_argument("--stage", default="finetune")
    ap.add_argument("--n-select", type=int, default=256)
    a = ap.parse_args()
    norm = C.normaliser()                     # CUDA context first
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    Xv, Yv, sv = C.load_windows("val"); Xt, Yt, st = C.load_windows("test")   # disk-backed (mmap)
    print(f"[realpde] VALID {len(Xv)} windows ({len(set(sv))} traj), TEST {len(Xt)} ({len(set(st))} traj; "
          f"{sum(s in C.LEAKED for s in st)} windows from the 2 flow conditions also in the competition release), "
          f"loaded in {time.time()-t0:.0f}s", flush=True)
    for model in a.models:
        sel_f = OUT / f"select_{model}_{a.stage}.json"
        if sel_f.exists():
            r_sel, g = json.loads(sel_f.read_text())
        else:
            r_sel, g = select(model, a.stage, Xv, Yv, norm, a.n_select); sel_f.write_text(json.dumps([r_sel, g]))
        net, P, peak = None, {}, 0.0
        for split, X in (("val", Xv), ("test", Xt)):
            path = C.CACHE / f"pred_{model}_{a.stage}_{split}.npy"; done = path.with_suffix(".done")
            if not done.exists():
                path.unlink(missing_ok=True)
                net = net or C.build(model, Xv.shape[1:], Yv.shape[1:], stage=a.stage)
                torch.cuda.reset_peak_memory_stats(); ts = time.time()
                C.predict(net, X, norm, g, batch=BATCH[model], out=C.open_pred(path, X.shape))
                peak = max(peak, torch.cuda.max_memory_allocated() / 2**30); done.touch()
                print(f"[realpde] {model}/{a.stage} {split} predicted in {time.time()-ts:.0f}s", flush=True)
            P[split] = C.open_pred(path, X.shape)
        del net; torch.cuda.empty_cache()
        res = {"model": model, "stage": a.stage, "gaussian": g, "valid_select_rel_l2": r_sel, "peak_gib_inference": peak,
               "valid": C.official_metrics_chunked(P["val"], Yv), "test": {}}
        for tag, m in subsets(st).items():
            res["test"][tag] = C.official_metrics_chunked(P["test"], Yt, idx=torch.nonzero(m).flatten())
        (OUT / f"trunk_{model}_{a.stage}.json").write_text(json.dumps(res, indent=1))
        T = res["test"]
        print(f"[realpde] {model}/{a.stage}: VALID rel_l2 {res['valid']['rel_l2']:.4f} | TEST all {T['all']['rel_l2']:.4f} "
              f"rmse {T['all']['rmse']:.4f} | excl-2-cond {T['not_in_comp_release']['rel_l2']:.4f} | in {T['in_dist']['rel_l2']:.4f} "
              f"out {T['out_dist']['rel_l2']:.4f} seen {T['seen']['rel_l2']:.4f} | rel_high_f {T['all']['rel_high_f']:.3f}", flush=True)


if __name__ == "__main__":
    main()
