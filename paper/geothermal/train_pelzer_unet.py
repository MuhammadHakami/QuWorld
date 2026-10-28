"""Pelzer 1HP heat-plume — SoTA-architecture track: 1HP_NN's OWN U-Net (Pelzer et al.,
baseline/1HP_NN processing/networks/unet.py: init_features 32, depth 3, kernel 5) on the
1000-run set prepared with their pipeline (paper/geothermal/prepare_1000dp.py, inputs gksi,
27.5-year label, 256x16), their loss (MSE on Rescale-normalized fields), their split ratio
0.7/0.2/0.1 (torch random_split, seed 1 as in 1HP_NN main.py).

Two stages (mirrors the Well protocol: pretrained SoTA checkpoint -> arms):
  stage pretrain: vanilla U-Net from scratch -> runs/pelzer_unet/pretrain_lr{lr}_s{seed}.pt
                  (selection on val MAE; this is the SoTA reproduction, target 0.046 C)
  stage arm:      load the pretrained U-Net, wrap the deepest encoder (256 ch, 32x2) with the
                  zero-init gated bottleneck {vanilla | mlp30 | mr30ryd}, fine-tune ALL params
                  for a fixed step budget, single-shot test MAE [C] + PAT@0.1C.
Usage:
  uv run paper/geothermal/train_pelzer_unet.py pretrain <lr> <seed> <steps>
  uv run paper/geothermal/train_pelzer_unet.py arm <variant> <seed> <steps> <lr> <pretrain_ckpt>
Env: BATCH (default 100 = 1HP_NN), OUT (default runs/pelzer_unet)
"""
import json, os, pathlib, sys, time, types
import torch, torch.nn as nn, yaml
ROOT = pathlib.Path(__file__).resolve().parents[2]
HP = ROOT / "baseline" / "1HP_NN"
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "geothermal")); sys.path.insert(0, str(HP))
sys.modules.setdefault("pynvml", types.ModuleType("pynvml"))
from model import Bottleneck, MLPCore, _count, build_config_core, parse_config  # noqa: E402
from qlayer import MultiRegister  # noqa: E402
from processing.networks.unet import UNet  # noqa: E402  (1HP_NN's architecture)
import data as D  # noqa: E402

PREP = ROOT / "data" / "geothermal" / "prepared_1000dp_gksi"
OUT = pathlib.Path(os.environ.get("OUT", ROOT / "runs" / "pelzer_unet"))
BATCH = int(os.environ.get("BATCH", "100"))
SPLIT_SEED = 1  # 1HP_NN main.py: random_split(..., generator=torch.Generator().manual_seed(1))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 1000


def load_1000():
    info = yaml.safe_load(open(PREP / "info.yaml"))
    x = torch.stack([torch.load(PREP / "Inputs" / f"RUN_{i}.pt", weights_only=True) for i in range(N)])
    y = torch.stack([torch.load(PREP / "Labels" / f"RUN_{i}.pt", weights_only=True) for i in range(N)])
    assert x.shape == (N, 4, 256, 16) and y.shape == (N, 1, 256, 16), (x.shape, y.shape)
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(N, generator=g)
    tr, va, te = perm[:700], perm[700:900], perm[900:]
    return torch.nan_to_num(x).float(), torch.nan_to_num(y).float(), info, tr, va, te


class EncoderWithBottleneck(nn.Module):
    def __init__(self, enc, bottleneck):
        super().__init__(); self.enc, self.bottleneck = enc, bottleneck
    def forward(self, x): return self.bottleneck(self.enc(x))


def build(variant: str, q_layers: int = 4) -> UNet:
    net = UNet(in_channels=4, out_channels=1, init_features=32, depth=3, kernel_size=5)
    if variant == "vanilla":
        return net
    ch = 32 * 2 ** 3  # deepest encoder output: 256 channels at 32x2
    if variant == "mlp30":
        core, out = MLPCore(ch, 30, _count(MultiRegister(ch, 10, 3, q_layers, "hea"))), 30
    elif variant == "mr30ryd":
        core, out = MultiRegister(ch, 10, 3, q_layers, "rydberg"), 30
    elif parse_config(variant) is not None:                       # wave-G grammar (q30-link=shared, ...)
        core, out = build_config_core(parse_config(variant), ch, q_layers)
    else:
        raise ValueError(variant)
    net.encoders[-1] = EncoderWithBottleneck(net.encoders[-1], Bottleneck(ch, core, out))
    return net


@torch.no_grad()
def evaluate(model, x, y, idx, info):
    model.eval()
    p = torch.cat([model(x[idx[i:i + 200]]) for i in range(0, len(idx), 200)]).cpu(); t = y[idx].cpu(); model.train()
    pc, tc = D.denormalize_temperature(p, info), D.denormalize_temperature(t, info)
    return dict(mae_c=float(D.mae_celsius(p, t, info)), pat_0p1=float(D.pat(pc, tc)),
                max_err_c=float((pc - tc).abs().amax(dim=(1, 2, 3)).mean()), mse_norm=float(((p - t) ** 2).mean()))


def train(model, x, y, tr, va, info, steps, lr, seed, tag):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # 1HP_NN solver defaults (Adam, wd 1e-4, MSE)
    loss_fn = nn.MSELoss(); t0 = time.time(); best = (float("inf"), None, -1); last = None
    for step in range(1, steps + 1):
        b = tr[torch.randint(0, len(tr), (BATCH,), generator=g)]
        loss = loss_fn(model(x[b]), y[b]); opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
        if step % 100 == 0 or step == steps:
            v = evaluate(model, x, y, va, info)
            if v["mae_c"] < best[0]:
                best = (v["mae_c"], {k: t.detach().clone() for k, t in model.state_dict().items()}, step)
            print(f"[{tag}] step {step} loss {last:.6f} val_mae {v['mae_c']:.4f}C pat {v['pat_0p1']:.3f} best {best[0]:.4f}@{best[2]} {time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(best[1])
    return dict(best_val_mae_c=best[0], best_step=best[2], train_loss=last, train_seconds=time.time() - t0)


def main():
    stage = sys.argv[1]; OUT.mkdir(parents=True, exist_ok=True)
    x, y, info, tr, va, te = load_1000(); x, y = x.to(DEV), y.to(DEV)
    if stage == "pretrain":
        lr, seed, steps = float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        model = build("vanilla").to(DEV)
        res = train(model, x, y, tr, va, info, steps, lr, seed, f"pretrain lr{lr} s{seed}")
        ck = OUT / f"pretrain_lr{lr:g}_s{seed}.pt"; torch.save({"model_state_dict": model.state_dict(), **res}, ck)
        res.update(stage="pretrain", lr=lr, seed=seed, steps=steps, num_params=_count(model), device=str(DEV), checkpoint=str(ck),
                   val=evaluate(model, x, y, va, info))
        (OUT / f"pretrain_lr{lr:g}_s{seed}.json").write_text(json.dumps(res, indent=1))
    elif stage == "arm":
        variant, seed, steps, lr, ck = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
        model = build(variant)
        st = torch.load(ck, weights_only=True, map_location="cpu")["model_state_dict"]
        # remap the deepest encoder's keys into the wrapper (encoders.3.* -> encoders.3.enc.*) for the adapter arms
        if variant != "vanilla":
            st = {(k.replace("encoders.3.", "encoders.3.enc.", 1) if k.startswith("encoders.3.") else k): v for k, v in st.items()}
        missing, unexpected = model.load_state_dict(st, strict=False)
        assert not unexpected and all(m.startswith("encoders.3.bottleneck") for m in missing), (missing[:5], unexpected[:5])
        model = model.to(DEV)
        pre = evaluate(model, x, y, te, info)  # sanity: alpha=0 -> identical to the pretrained model on test (recorded, not selected on)
        res = train(model, x, y, tr, va, info, steps, lr, seed, f"arm {variant} s{seed}")
        res.update(stage="arm", variant=variant, seed=seed, steps=steps, lr=lr, pretrain=ck, num_params=_count(model), device=str(DEV),
                   test_at_start=pre, test=evaluate(model, x, y, te, info), val=evaluate(model, x, y, va, info))
        (OUT / f"{variant}_s{seed}.json").write_text(json.dumps(res, indent=1))
        torch.save({"model_state_dict": model.state_dict()}, OUT / f"{variant}_s{seed}.pt")
    else:
        raise SystemExit("stage must be pretrain|arm")
    for k, v in res.items():
        if not isinstance(v, dict): print(f"{k}:{' '*max(1,16-len(k))}{v}")
    for k in ("val", "test", "test_at_start"):
        if k in res: print(f"{k}: " + " ".join(f"{a}={b:.4f}" for a, b in res[k].items()))


if __name__ == "__main__":
    main()
