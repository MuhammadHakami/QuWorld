"""TORAX-Bench v1 adapter arms (GPU when available): 1-D spectral trunk + swappable bottleneck.

Same protocol family as paper/train.py (sprint arena): variants
  none | mlp30 | mr30ryd   (bottleneck = paper/model.py Bottleneck; cores from
  paper/model.py MLPCore / paper/qlayer.py MultiRegister, parameter-matched at the core)
State x_t = [T_e, T_i, n_e, q] on the 27-cell rho grid (q interpolated from the
26 face points); conditioning = next-step actuators (3) + scenario (12) broadcast
over rho. Objective: k-step pushforward MSE (k=4). Metric: rollout VRMSE@8 on
the frozen TEST split (baselines.py definition: per (discharge, frame, channel)
sqrt(mean_x err^2 / var_x(true)), averaged), from strided starts.
Usage: uv run paper/torax_bench/train_torax.py <variant> <seed> <steps> <k> [out_dir]
"""
import json, pathlib, sys, time
import numpy as np, torch, torch.nn as nn
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))
from model import Bottleneck, MLPCore, _count, build_config_core, parse_config  # noqa: E402
from qlayer import MultiRegister  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "none"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1337
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
K = int(sys.argv[4]) if len(sys.argv) > 4 else 4
OUT = pathlib.Path(sys.argv[5]) if len(sys.argv) > 5 else ROOT / "runs" / "torax"
DATA = ROOT / "data" / "torax_bench_v1"; SPLITS = json.load(open(ROOT / "paper/torax_bench/splits.json"))
CH = ["T_e", "T_i", "n_e", "q"]; H = 8; BATCH = 16; LR = 1e-3; WIDTH = 24; MODES = 8
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED)

def load(ids):
    X, C = [], []
    for i in ids:
        with np.load(DATA / f"discharge_{i:05d}.npz") as d:
            q = np.stack([np.interp(d["rho_norm"], d["rho_face_norm"], row) for row in d["q"]])
            x = np.stack([d["T_e"], d["T_i"], d["n_e"] / 1e19, q], 1).astype(np.float64)   # (T,4,27); n_e in 1e19 m^-3 (float32-safe)
            act = np.repeat((d["actuators"] / np.array([1e6, 1e6, 1e22]))[:, :, None], 27, 2).astype(np.float64)  # MA, MW, 1e22/s
            sc = np.repeat(np.repeat(d["scenario"][None, :, None], x.shape[0], 0), 27, 2).astype(np.float64)
            X.append(x); C.append(np.concatenate([act, sc], 1))
    return np.stack(X), np.stack(C)  # (N,T,4,27), (N,T,15,27)

def zs(a, m, s): return (a - m) / s
Xtr, Ctr = load(SPLITS["train"]); Xte, Cte = load(SPLITS["test"])
mx, sx = Xtr.mean((0, 1, 3), keepdims=True), Xtr.std((0, 1, 3), keepdims=True) + 1e-6
mc, sc_ = Ctr.mean((0, 1, 3), keepdims=True), Ctr.std((0, 1, 3), keepdims=True) + 1e-6
Xtr_t, Ctr_t = torch.tensor(zs(Xtr, mx, sx), dtype=torch.float32).to(DEV), torch.tensor(zs(Ctr, mc, sc_), dtype=torch.float32).to(DEV)
Xte_t, Cte_t = torch.tensor(zs(Xte, mx, sx), dtype=torch.float32).to(DEV), torch.tensor(zs(Cte, mc, sc_), dtype=torch.float32).to(DEV)
assert np.isfinite(sx).all() and np.isfinite(sc_).all() and (sx > 0).all(), "normalization stats degenerate"
T = Xtr_t.shape[1]

class Spec1d(nn.Module):
    def __init__(s, ch, m): super().__init__(); s.m = m; s.w = nn.Parameter((1 / (ch * ch)) * torch.randn(ch, ch, m, 2))
    def forward(s, x):
        xf = torch.fft.rfft(x, norm="ortho"); out = torch.zeros_like(xf)
        out[:, :, :s.m] = torch.einsum("bim,iom->bom", xf[:, :, :s.m], torch.view_as_complex(s.w))
        return torch.fft.irfft(out, n=x.shape[-1], norm="ortho")
class Blk(nn.Module):
    def __init__(s, ch, m): super().__init__(); s.spec, s.skip, s.act = Spec1d(ch, m), nn.Conv1d(ch, ch, 1), nn.GELU()
    def forward(s, x): return s.act(s.spec(x) + s.skip(x))
def build(variant):
    if variant == "none": mid = nn.Identity()
    elif variant == "mlp30": mid = Bottleneck(WIDTH, MLPCore(WIDTH, 30, _count(MultiRegister(WIDTH, 10, 3, 4, "hea"))), 30)
    elif variant == "mr30ryd": mid = Bottleneck(WIDTH, MultiRegister(WIDTH, 10, 3, 4, "rydberg"), 30)
    elif parse_config(variant) is not None:                        # wave-G grammar (q30-link=shared, ...)
        core, out = build_config_core(parse_config(variant), WIDTH, 4); mid = Bottleneck(WIDTH, core, out)
    else: raise ValueError(variant)
    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.lift = nn.Conv1d(4 + 15, WIDTH, 1); s.pre = nn.Sequential(Blk(WIDTH, MODES), Blk(WIDTH, MODES))
            s.mid = mid; s.post = nn.Sequential(Blk(WIDTH, MODES), Blk(WIDTH, MODES)); s.proj = nn.Sequential(nn.Conv1d(WIDTH, 64, 1), nn.GELU(), nn.Conv1d(64, 4, 1))
        def forward(s, x, c): return x + s.proj(s.post(s.mid(s.pre(s.lift(torch.cat([x, c], 1))))))  # delta prediction
    return Net()

def vrmse(true, pred):  # (B,4,27) tensors; baselines.py convention
    mse = ((true - pred) ** 2).mean(-1); var = true.var(-1, unbiased=False)
    return torch.sqrt(mse / (var + 1e-12)).mean().item()
@torch.no_grad()
def evaluate(model, X, C):
    model.eval(); one, roll = [], []
    for t0 in range(0, T - 1, 8):
        x = X[:, t0]; one.append(vrmse(X[:, t0 + 1], model(x, C[:, t0 + 1])))
    for t0 in range(0, T - 1 - H, 16):
        x = X[:, t0]
        for j in range(H): x = model(x, C[:, t0 + 1 + j])
        roll.append(vrmse(X[:, t0 + H], x))
    model.train(); return float(np.mean(one)), float(np.mean(roll))

def main():
    global BATCH
    model = build(VARIANT).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=LR); n_params = _count(model)
    print(f"device {DEV} variant {VARIANT} seed {SEED} params {n_params}", flush=True)
    N = Xtr_t.shape[0]; t_start = time.time(); last = None
    for step in range(STEPS):
        b = torch.randint(0, N, (BATCH,)); t0 = torch.randint(0, T - 1 - K, (BATCH,))
        x = Xtr_t[b, t0]; loss = 0.0
        for j in range(K):
            x = model(x, Ctr_t[b, t0 + 1 + j]); loss = loss + ((x - Xtr_t[b, t0 + 1 + j]) ** 2).mean()
        loss = loss / K; opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
        if step % 200 == 0: print(f"step {step} loss {last:.5f} elapsed {time.time()-t_start:.0f}s", flush=True)
    v1, v8 = evaluate(model, Xte_t, Cte_t); OUT.mkdir(parents=True, exist_ok=True)
    res = dict(variant=VARIANT, seed=SEED, steps=STEPS, k=K, test_vrmse_1=v1, test_vrmse_8=v8, train_loss=last, num_params=n_params, train_seconds=time.time() - t_start, device=str(DEV))
    (OUT / f"{VARIANT}_k{K}_s{SEED}.json").write_text(json.dumps(res, indent=1))
    torch.save({"model_state_dict": model.state_dict()}, OUT / f"{VARIANT}_k{K}_s{SEED}.pt")   # doc 35 §23: kept for figures
    with torch.no_grad():                                         # Figure 2 row: middle TEST discharge, rollout from t0 = 0
        model.eval(); i = Xte_t.shape[0] // 2; L = min(32, T - 1); x = Xte_t[i:i + 1, 0]; pr = [x]
        for j in range(L): x = model(x, Cte_t[i:i + 1, j + 1]); pr.append(x)
        pr = torch.cat(pr).cpu().numpy() * sx[0, 0] + mx[0, 0]; tr_ = Xte_t[i, :L + 1].cpu().numpy() * sx[0, 0] + mx[0, 0]
        np.savez(OUT / f"{VARIANT}_k{K}_s{SEED}_fig.npz", truth=tr_[:, 0], pred=pr[:, 0], discharge=int(SPLITS["test"][i]),
                 vrmse8=v8)                                        # channel 0 = T_e (keV), (time, rho)
    for k_, v_ in res.items(): print(f"{k_}:{' '*max(1,16-len(k_))}{v_}")

if __name__ == "__main__":
    main()
