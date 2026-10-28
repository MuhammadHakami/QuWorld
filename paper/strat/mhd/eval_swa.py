"""Doc 35 §37, MHD_64: weight averaging probe on the live §34 Flowers run (read-only snapshots, VALID sample only).

Models: the best-VALID copy (step 5694) and every read-only snapshot of flower_s1337_latest.pt taken by snap_latest.py
(scratch/snaps/flower_step<step>.pt). Scored per window on the fixed stratified VALID sample (strata.json):
  each snapshot (single pass), SWA = uniform average of the weights of all snapshots (GroupNorm only, so no
  BatchNorm re-estimation is needed), the prediction ensemble of the snapshots, and SWA with the all30 TTA + div-free B.
Writes runs/strat/mhd/swa_valid.json (per-window per-channel VRMSE for every model).

    flock runs/strat/gpu.lock timeout 1800 env QW_GPU_GB=12 runs/capped.sh strat_mhd 16 uv run --no-sync python \
        paper/strat/mhd/eval_swa.py <snap_dir> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gpucap import cap_from_env  # noqa: E402

cap_from_env()
import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_strat as E  # noqa: E402
from flower_standalone import Flower  # noqa: E402

DEV = E.DEV


def vr(p, y):
    var = (y - y.mean(dim=(-3, -2, -1), keepdim=True)).square().mean(dim=(-3, -2, -1))
    return ((p - y).square().mean(dim=(-3, -2, -1)) / (var + 1e-7)).sqrt()[0].tolist()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("snaps"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    files = [E.SCR] + sorted(pathlib.Path(a.snaps).glob("flower_step*.pt"), key=lambda p: int(p.stem[11:]))
    sds, steps = [], []
    for f in files:
        st = torch.load(f, weights_only=True, map_location="cpu"); sds.append(st["model_state_dict"]); steps.append(int(st["step"]))
    order = np.argsort(steps); sds = [sds[i] for i in order]; steps = [steps[i] for i in order]
    swa = {k: torch.stack([sd[k].float() for sd in sds]).mean(0).to(sds[0][k].dtype) for k in sds[0]}
    mk = lambda sd: Flower(dim_in=E.NH * E.C, dim_out=E.C, n_spatial_dims=3, spatial_resolution=[E.N] * 3,  # noqa: E731
                           boundary_condition_types=["PERIODIC"]).eval().to(DEV).requires_grad_(False)
    net = mk(None)
    rows = json.loads((E.OUT / "strata.json").read_text())["samples"]["valid"]
    if a.smoke:
        rows = rows[:2]
    split = E.WellND("MHD_64", "valid")
    std = torch.tensor(split.std.reshape(E.C)).view(E.C, 1, 1, 1).to(DEV); mean = torch.tensor(split.mean.reshape(E.C)).view(E.C, 1, 1, 1).to(DEV)
    dfm = E.DivFreeB(torch.nn.Identity()).to(DEV)
    df = lambda p: torch.cat([p[:, :1], dfm.project(p[:, 1:4]), p[:, 4:]], 1)  # noqa: E731
    groups = E.pass_groups("full"); flat = [e for el in groups.values() for e in el]
    xs, ys = [], []
    for r in rows:
        x, y = split.window(r["traj"], r["t0"]); xs.append(x); ys.append(y)
    out = {"steps": steps, "rows": rows, "per_snapshot": [], "ensemble": [], "swa": [], "swa_all30_df": []}
    ens = [0] * len(rows); t0 = time.perf_counter()
    with torch.no_grad():
        for sd, s in zip(sds, steps):
            net.load_state_dict(sd); res = []
            for i in range(len(rows)):
                p = net(xs[i][None].to(DEV)); res.append(vr(p, ys[i][None].to(DEV))); ens[i] = ens[i] + p.cpu()
            out["per_snapshot"].append(res)
            print(f"[swa] step {s}: {np.mean(res):.5f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        out["ensemble"] = [vr(ens[i].to(DEV) / len(sds), ys[i][None].to(DEV)) for i in range(len(rows))]
        net.load_state_dict(swa)
        for i in range(len(rows)):
            x6 = xs[i][None].to(DEV).view(1, E.NH, E.C, E.N, E.N, E.N); y = ys[i][None].to(DEV)
            acc = 0; raw = None
            for j in range(0, len(flat), E.PASS_BATCH):
                ch = flat[j:j + E.PASS_BATCH]
                o = net(torch.cat([e.fwd(x6, (mean, std)) for e in ch]).reshape(len(ch), E.NH * E.C, E.N, E.N, E.N))
                for k, e in enumerate(ch):
                    q = e.inv(o[k:k + 1], (mean, std)); acc = acc + q
                    raw = q if raw is None else raw
            out["swa"].append(vr(raw, y)); out["swa_all30_df"].append(vr(df(acc / len(flat)), y))
        print(f"[swa] ensemble {np.mean(out['ensemble']):.5f} swa {np.mean(out['swa']):.5f} "
              f"swa_all30_df {np.mean(out['swa_all30_df']):.5f} ({time.perf_counter() - t0:.0f}s) "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f}G", flush=True)
    if not a.smoke:
        (E.OUT / "swa_valid.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
