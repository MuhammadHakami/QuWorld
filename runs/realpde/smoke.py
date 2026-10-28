# RealPDE loader equivalence + per-trunk peak GPU memory (batch 8) before any full evaluation (Sep 23 OOM lesson).
import json, sys, time, torch
sys.path.insert(0, "paper/realpde")
import common as C
t = time.time(); X, Y, s = C.load_windows("val", limit=300)
print("loaded", tuple(X.shape), len(set(s)), "traj", f"{time.time()-t:.1f}s", flush=True)
from datasets import load_from_disk
from realpdebench.data.fluid_hf_dataset import FoilHFDataset
ds = object.__new__(FoilHFDataset); hf = C.DATA / "foil" / "hf_dataset"
ds._indices = json.loads((hf / "val_index_real.json").read_text())[:300]; ds.trajectories = load_from_disk(str(hf / "real"))
ds._sim_id_to_idx = {x: i for i, x in enumerate(ds.trajectories["sim_id"])}
ds.sub_s, ds.horizon, ds.in_step, ds.dataset_type, ds.noise_scale, ds.mask_prob, ds.condition_on_para = 2, 40, 20, "real", 0.0, 0.0, False
for i in (0, 150, 299):
    x, y = ds[i]; print("getitem match", i, torch.equal(x, X[i]), torch.equal(y, Y[i]), flush=True)
norm = C.normaliser()
for m in ["fno", "transolver", "cno"]:
    for g in (True, False):
        torch.cuda.reset_peak_memory_stats()
        net = C.build(m, X.shape[1:], Y.shape[1:], dict(gaussian=g))
        P = C.predict(net, X[:32], norm, g, batch=8)
        print(f"{m} gaussian={g} rel_l2 {C.rel_l2(P, Y[:32]):.4f} peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
        del net; torch.cuda.empty_cache()
