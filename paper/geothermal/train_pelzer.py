"""Pelzer 1HP heat-plume adapter arms (GPU when available, field track): 2-D spectral trunk + swappable bottleneck.

Inputs (4,256,16) -> temperature label (1,256,16), MSE on normalized fields (1HP_NN
convention); reported: test MAE [C] (denormalized), PAT@0.1C. Frozen 70/20/10 split
(paper/geothermal/splits.json, seed 20260816). Arms: none | mlp30 | mr30ryd (cores from
paper/model.py / paper/qlayer.py, parameter-matched at the core). Fixed protocol across
arms: steps, batch, LR, seed.
Usage: uv run paper/geothermal/train_pelzer.py <variant> <seed> <steps> [out_dir]
"""
import json, pathlib, sys, time
import torch, torch.nn as nn
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "geothermal"))
from model import FNOBlock, Bottleneck, MLPCore, _count  # noqa: E402
from qlayer import MultiRegister  # noqa: E402
import data as D  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "none"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1337
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
OUT = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "runs" / "pelzer"
WIDTH, MODES, BATCH, LR = 24, (12, 4), 8, 1e-3
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4); torch.manual_seed(SEED)

x_all, y_all = D.load_fields(); sp = D.split(); info = D.load_info()
tr, va, te = [torch.tensor(sp[k]) for k in ("train", "val", "test")] if "val" in sp else [torch.tensor(sp[k]) for k in ("train", "valid", "test")]
x_all, y_all = torch.nan_to_num(x_all.float()).to(DEV), torch.nan_to_num(y_all.float()).to(DEV)
tr, va, te = tr.to(DEV), va.to(DEV), te.to(DEV)

def build(variant):
    if variant == "none": mid = nn.Identity()
    elif variant == "mlp30": mid = Bottleneck(WIDTH, MLPCore(WIDTH, 30, _count(MultiRegister(WIDTH, 10, 3, 4, "hea"))), 30)
    elif variant == "mr30ryd": mid = Bottleneck(WIDTH, MultiRegister(WIDTH, 10, 3, 4, "rydberg"), 30)
    else: raise ValueError(variant)
    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.lift = nn.Conv2d(4, WIDTH, 1); s.pre = nn.Sequential(FNOBlock(WIDTH, *MODES), FNOBlock(WIDTH, *MODES))
            s.mid = mid; s.post = nn.Sequential(FNOBlock(WIDTH, *MODES), FNOBlock(WIDTH, *MODES)); s.proj = nn.Sequential(nn.Conv2d(WIDTH, 64, 1), nn.GELU(), nn.Conv2d(64, 1, 1))
        def forward(s, x): return s.proj(s.post(s.mid(s.pre(s.lift(x)))))
    return Net()

@torch.no_grad()
def evaluate(model, idx):
    model.eval(); p = model(x_all[idx]).cpu(); t = y_all[idx].cpu(); model.train()
    pc, tc = D.denormalize_temperature(p, info), D.denormalize_temperature(t, info)
    return float(D.mae_celsius(p, t, info)), float(D.pat(pc, tc)), float((pc - tc).abs().amax(dim=(1, 2, 3)).mean())

def main():
    model = build(VARIANT).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=LR); n_params = _count(model)
    print(f"device {DEV} variant {VARIANT} seed {SEED} params {n_params}", flush=True)
    t0 = time.time(); last = None
    for step in range(STEPS):
        b = tr[torch.randint(0, len(tr), (BATCH,))]
        loss = ((model(x_all[b]) - y_all[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
        if step % 250 == 0: print(f"step {step} loss {last:.5f} elapsed {time.time()-t0:.0f}s", flush=True)
    va_mae, va_pat, va_max = evaluate(model, va); te_mae, te_pat, te_max = evaluate(model, te)
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(variant=VARIANT, seed=SEED, steps=STEPS, val_mae_c=va_mae, test_mae_c=te_mae, test_pat_0p1=te_pat, test_max_err_c=te_max, train_loss=last, num_params=n_params, train_seconds=time.time() - t0, n_train=len(tr), n_val=len(va), n_test=len(te), device=str(DEV))
    (OUT / f"{VARIANT}_s{SEED}.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items(): print(f"{k}:{' '*max(1,16-len(k))}{v}")

if __name__ == "__main__":
    main()
