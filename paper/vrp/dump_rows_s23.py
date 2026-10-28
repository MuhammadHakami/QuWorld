"""Figure 2 row for RouteFinder (doc 35 §23): the first test instance of the first unseen multi-depot variant, routes of
the EAL-fine-tuned classical policy, its matched 3-D MLP and its 3-D tethered register (seed 1, greedy decoding), drawn
as images (depots, customers, one colour per route) with each tour's cost and gap to the reference.

    cd baseline/routefinder && ../../.venv-vrp/bin/python ../../paper/vrp/dump_rows_s23.py
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_rf_eal as E  # noqa: E402  (installs rf_compat)

import numpy as np  # noqa: E402
import torch  # noqa: E402

ARMS = (("classical", "vanilla"), ("mlp", "q30-geo=3d-mlp"), ("qubits", "q30-geo=3d"))
RES = 160


def raster(locs: np.ndarray, actions: np.ndarray, n_depot: int) -> np.ndarray:
    """Draw tours into a RES x RES image: value = route index + 1 along each segment, depots = -1."""
    img = np.zeros((RES, RES)); pts = np.clip((locs * (RES - 1)).round().astype(int), 0, RES - 1)
    route, prev = 1, 0
    for a in list(actions) + [0]:
        a = int(a)
        if a < n_depot and prev < n_depot:
            route += 1; prev = a; continue
        p, q = pts[prev], pts[a]
        for t in np.linspace(0, 1, 2 * RES):
            x, y = (p + t * (q - p)).round().astype(int); img[y, x] = route
        prev = a
    for d in range(n_depot):
        x, y = pts[d]; img[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = -1
    return img


@torch.no_grad()
def main() -> None:
    env, names = E.env_for("md")
    td0 = env.load_data(f"./data/{names[0]}/test/100.npz")[:1].to("cuda")
    out = {"label": f"routing ({names[0]})", "metric": "tour cost", "kind": "route", "sample": f"{names[0]} test instance 0"}
    for name, v in ARMS:
        model = E.build("checkpoints/100/rf-transformer.ckpt", v)
        st = torch.load(ROOT / "runs/routefinder/s23" / f"eal_md_{v}_s1.pt", weights_only=False, map_location="cpu")
        model.policy.load_state_dict(st["policy"]); model = model.to("cuda").eval()
        td = env.reset(td0.clone())
        o = model.policy(td, env, phase="test", decode_type="greedy", return_actions=True)
        locs = td["locs"][0].cpu().numpy(); n_dep = int(td["num_depots"][0]) if "num_depots" in td.keys() else 1
        out[f"pred_{name}"] = raster(locs, o["actions"][0].cpu().numpy(), n_dep); out[f"m_{name}"] = float(-o["reward"][0])
        print(name, v, "cost", round(out[f"m_{name}"], 4), flush=True)
    pts = np.clip((locs * (RES - 1)).round().astype(int), 0, RES - 1); inst = np.zeros((RES, RES))
    for k, (x, y) in enumerate(pts):                                      # the instance: customers (+1), depots (-1)
        inst[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = -1 if k < n_dep else 1
    out["truth"] = inst                                                   # no reference tour is released, only its cost
    dst = ROOT / "paper/fig_cache/rows_s23"; dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / "routefinder.npz", **out); print("wrote", dst / "routefinder.npz")


if __name__ == "__main__":
    main()
