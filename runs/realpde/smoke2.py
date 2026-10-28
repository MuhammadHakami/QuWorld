# RealPDEBench's own foil checkpoints: VALID rel-L2 + peak GPU memory on 32 windows (batch 4), before any full run.
import sys, time, torch
sys.path.insert(0, "paper/realpde")
import common as C
t = time.time(); X, Y, s = C.load_windows("val", limit=32); print("loaded", tuple(X.shape), f"{time.time()-t:.0f}s", flush=True)
norm = C.normaliser()
for m in ["unet", "fno", "cno", "transolver"]:
    for stage in ["finetune", "real"]:
        for g in (True, False):
            try:
                torch.cuda.reset_peak_memory_stats()
                net = C.build(m, X.shape[1:], Y.shape[1:], dict(gaussian=g), stage=stage)
                P = C.predict(net, X, norm, g, batch=4)
                print(f"{m} {stage} gaussian={g} rel_l2 {C.rel_l2(P, Y):.4f} peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
            except Exception as e:
                print(f"{m} {stage} gaussian={g} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
            finally:
                net = None; torch.cuda.empty_cache()
