# chunked metric == eval_metrics(batch = N) on a 192-window VALID subset (FNO finetune predictions)
import sys, torch
sys.path.insert(0, "paper/realpde")
import common as C
X, Y, s = C.load_windows("val", limit=192)
net = C.build("fno", X.shape[1:], Y.shape[1:], stage="finetune")
P = C.predict(net, X, C.normaliser(), True, batch=16); del net; torch.cuda.empty_cache()
a = C.official_metrics(P, Y); b = C.official_metrics_chunked(P, Y, chunk=50)
for k in C.METRIC_NAMES:
    print(f"{k:11s} official {a[k]:.6g}  chunked {b[k]:.6g}  rel.diff {abs(a[k]-b[k])/max(abs(a[k]),1e-12):.1e}")
