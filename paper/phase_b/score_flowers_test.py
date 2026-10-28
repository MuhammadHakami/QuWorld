"""Doc 35 §34 evaluation: score the Flowers MHD_64 best-VALID checkpoint once on the full TEST split, with the
registered conventions: per-channel mean VRMSE (Table 1) and field-grouped VRMSE (density, B, v; the published one).

Two checks run before TEST is opened, and the script stops if either fails:
  1. the checkpoint reproduces its logged VALID score on the training script's stride-2 VALID windows (data_ram_mhd);
  2. the TEST reader (cache.WellND, the reader behind every MHD TEST number in the paper) and data_ram_mhd give the
     same normalised VALID window (max abs difference < 1e-4).
Arms (doc 35 §37e, registered before TEST): raw (the §34 registered number); dfc = raw + divergence-free B (§29a)
+ the last input frame's spatial means for density and B (both exactly conserved on the periodic domain), single pass,
the headline; shg8_dfc and all30_dfc = symmetry averages over 8 / 30 passes (paper/strat/mhd/eval_strat.py groups) with
the same projections, daggered rows (8x / 30x inference).

    QW_GPU_GB=16 runs/capped.sh s34test 32 uv run --no-sync python paper/phase_b/score_flowers_test.py [--check-only]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "alloc"))
sys.path.insert(0, str(ROOT / "baseline" / "flowers"))
from gpucap import cap_from_env  # noqa: E402
import data_ram_mhd as data_ram  # noqa: E402
from flower_standalone import Flower  # noqa: E402
from cache import WellND  # noqa: E402

DEV = "cuda"
OUT = ROOT / "runs" / "phase_b_mhd"


@torch.no_grad()
def valid_score(model, valid, stride_t: int = 2, chunk: int = 5) -> float:
    """train_flowers_mhd.validate, verbatim metric (mean over windows of the per-window channel-mean VRMSE)."""
    errs = []
    for x, y in data_ram.valid_windows(valid, stride_t):
        for i in range(0, len(x), chunk):
            pred, yd = model(x[i:i + chunk].to(DEV)), y[i:i + chunk].to(DEV)
            mse = (pred - yd).square().mean(dim=(2, 3, 4))
            var = (yd - yd.mean(dim=(2, 3, 4), keepdim=True)).square().mean(dim=(2, 3, 4))
            errs.append((mse / (var + 1e-7)).sqrt().mean(dim=1))
    return torch.cat(errs).mean().item()


@torch.no_grad()
def channel_vrmse(model, split: WellND, pairs, bsz: int = 2) -> torch.Tensor:
    """(N, 7) per-window, per-channel VRMSE (the formula of paper/alloc/mhd_divfree.channel_vrmse)."""
    v = []
    for i in range(0, len(pairs), bsz):
        xs, ys = zip(*(split.window(k, t) for k, t in pairs[i:i + bsz]))
        p, y = model(torch.stack(xs).to(DEV)), torch.stack(ys).to(DEV)
        mse = (p - y).square().mean((2, 3, 4)); var = (y - y.mean((2, 3, 4), keepdim=True)).square().mean((2, 3, 4))
        v.append((mse / (var + 1e-7)).clamp_min(1e-12).sqrt().float().cpu())
    return torch.cat(v)


def dfc(p: torch.Tensor, x: torch.Tensor, proj) -> torch.Tensor:
    """Divergence-free B, then density and B spatial means set to the last input frame's (conserved quantities)."""
    p = torch.cat([p[:, :1], proj.project(p[:, 1:4]), p[:, 4:]], 1)
    last = x[:, -7:]
    fix = (last[:, :4].mean((2, 3, 4), keepdim=True) - p[:, :4].mean((2, 3, 4), keepdim=True))
    return torch.cat([p[:, :4] + fix, p[:, 4:]], 1)


@torch.no_grad()
def arms_vrmse(model, split: WellND, pairs, st, proj) -> dict:
    """(N, 7) per-window per-channel VRMSE for every arm; one window at a time, 30 symmetry passes each."""
    import eval_strat as ES
    groups = ES.pass_groups("full"); out = {k: [] for k in ("raw", "dfc", "shg8_dfc", "all30_dfc")}
    for k, t in pairs:
        x, y = (v.unsqueeze(0).to(DEV) for v in split.window(k, t))
        xs = x.view(1, 4, 7, *x.shape[-3:])
        sums = {}
        for g, syms in groups.items():
            acc = 0
            for s in syms:
                acc = acc + s.inv(model(s.fwd(xs, st).reshape(1, 28, *x.shape[-3:])), st)
            sums[g] = acc
        v = {"raw": sums["id"], "shg8": (sums["id"] + sums["comb"]) / 8, "all30": sum(sums.values()) / 30}
        preds = {"raw": v["raw"], "dfc": dfc(v["raw"], x, proj), "shg8_dfc": dfc(v["shg8"], x, proj),
                 "all30_dfc": dfc(v["all30"], x, proj)}
        for a, p in preds.items():
            mse = (p - y).square().mean((2, 3, 4)); var = (y - y.mean((2, 3, 4), keepdim=True)).square().mean((2, 3, 4))
            out[a].append((mse / (var + 1e-7)).clamp_min(1e-12).sqrt().float().cpu())
    return {a: torch.cat(v) for a, v in out.items()}


def summary(v: torch.Tensor) -> dict:
    g = torch.stack([v[:, 0], v[:, 1:4].mean(1), v[:, 4:7].mean(1)], 1).mean(1)
    return {"per_channel": float(v.mean()), "field_grouped": float(g.mean()), "channels": v.mean(0).tolist(),
            "per_channel_median": float(v.mean(1).median()), "n": len(v)}


def test_stats():
    """(mean, std) as (7, 1, 1, 1) arrays in the loaders' channel order (official stats.yaml, min std 1e-4)."""
    m, s = data_ram._stats()
    return m, s


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--check-only", action="store_true", help="run both checks, never open TEST")
    ap.add_argument("--ckpt", default="", help="checkpoint name under runs/phase_b_mhd (default flower_s<seed>.pt; §37i: flower_vft_s1337.pt)")
    a = ap.parse_args()
    cap_from_env(); torch.backends.cudnn.benchmark = True
    ck = OUT / (a.ckpt or f"flower_s{a.seed}.pt"); stem = ck.stem
    st = torch.load(ck, weights_only=True, map_location=DEV)
    model = Flower(dim_in=28, dim_out=7, n_spatial_dims=3, spatial_resolution=[64, 64, 64],
                   boundary_condition_types=["PERIODIC"]).to(DEV)
    model.load_state_dict(st["model_state_dict"]); model.eval()
    res = {"ckpt": str(ck.relative_to(ROOT)), "step": st["step"], "logged_valid": st["val"]}

    _, valid = data_ram.load()
    v = valid_score(model, valid)
    res["valid_repro"] = v
    ok1 = abs(v - st["val"]) < 5e-4
    wv = WellND("MHD_64", "valid")
    x_w, y_w = wv.window(0, data_ram.N_HIST - 1)
    x_r = valid[0, 0:data_ram.N_HIST].reshape(-1, *valid.shape[-3:]); y_r = valid[0, data_ram.N_HIST]
    dmax = max((x_w - x_r).abs().max().item(), (y_w - y_r).abs().max().item())
    res["reader_max_abs_diff"] = dmax
    ok2 = dmax < 1e-4
    del valid
    print(f"[s34test] step {st['step']} VALID logged {st['val']:.6f} repro {v:.6f} ({'ok' if ok1 else 'MISMATCH'}); "
          f"reader diff {dmax:.2e} ({'ok' if ok2 else 'MISMATCH'})", flush=True)
    if not (ok1 and ok2) or a.check_only:
        (OUT / f"{stem}_testcheck.json").write_text(json.dumps(res, indent=1)); return

    sys.path.insert(0, str(ROOT / "paper" / "strat" / "mhd"))
    from mhd_divfree import DivFreeB
    proj = DivFreeB(torch.nn.Identity()).to(DEV)
    st = (torch.tensor(test_stats()[0], device=DEV), torch.tensor(test_stats()[1], device=DEV))
    test = WellND("MHD_64", "test")                                              # registered: TEST once
    pairs = [(k, t0) for k in range(test.n) for t0 in range(data_ram.N_HIST - 1, test.T - 1)]
    arms = arms_vrmse(model, test, pairs, st, proj)
    res["test"] = summary(arms["raw"])
    res["test_arms"] = {a: summary(v) for a, v in arms.items()}
    (OUT / f"{stem}_test.json").write_text(json.dumps(res, indent=1))
    print(f"S34_TEST per_channel={res['test']['per_channel']:.5f} field_grouped={res['test']['field_grouped']:.5f} "
          f"n={res['test']['n']} | " + " ".join(f"{k} {v['per_channel']:.5f}" for k, v in res["test_arms"].items()), flush=True)


if __name__ == "__main__":
    main()
